"""
sweep_runner.py  — модуль гиперпараметрического перебора SmartPVD
Python ≥ 3.10

Правьте только блок НАСТРОЙКИ в начале файла, остальной код оставляйте без изменений.
"""

from __future__ import annotations

# ─────────────── БЛОК НАСТРОЙКИ ────────────────────────────────────────────
QUICK_MODE   = False    # True → quick (4 ранних × 70 trials), False → полный (96 × 500)
JOBS         = 8        # число процессов Parallel (лучше os.cpu_count())
QUICK_EARLY  = 10        # «ранних» конфигов в quick
QUICK_TRIALS = 300       # trials в quick
FULL_TRIALS  = 1300      # trials в full
SEED         = 42       # фиксированный seed для воспроизводимости TPE
# ──────────────────────────────────────────────────────────────────────────

import os
# Каждая копия процесса займёт ровно 1 ядро для BLAS
os.environ["OMP_NUM_THREADS"]      = "1"
os.environ["MKL_NUM_THREADS"]      = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import logging
import optuna
from optuna.logging import set_verbosity, WARNING
set_verbosity(WARNING)
optuna.logging.disable_default_handler()
_lg = logging.getLogger("optuna")
_lg.setLevel(logging.WARNING)
_lg.handlers.clear()
_lg.propagate = False

import contextlib
import hashlib
import itertools
import time
import threading
from threading import Event
from multiprocessing import Manager
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import pandas as pd

# ─── SmartPVD core ────────────────────────────────────────────────────────
import sys
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

import config, preprocess, events_PPD, well_pairs, oil_windows  # noqa: E402
import metrics, correl, final_mix                               # noqa: E402

# ─── Пары сеток ───────────────────────────────────────────────────────────
EARLY_GRID = {
    "PPD_REL_THRESH":    [0.15, 0.20, 0.25, 0.30],
    "LAG_DAYS":          [0, 1],
    "OIL_CHECK_DAYS":    [7, 10, 14],
    "OIL_DELTA_P_THRESH":[3, 4, 5, 6],
}
DIST_LIMITS = [600, 800, 900, 1100]

CSV_PATH  = "parameters_sweeping.csv"
CACHE_DIR = ROOT / ".cache_sweep"
CACHE_DIR.mkdir(exist_ok=True)

mem_l1 = joblib.Memory(CACHE_DIR / "l1", verbose=0)
mem_l2 = joblib.Memory(CACHE_DIR / "l2", verbose=0)
mem_l3 = joblib.Memory(CACHE_DIR / "l3", verbose=0)

# ─── patch config+correl ───────────────────────────────────────────────────
@contextlib.contextmanager
def patch_cfg(**kw):
    old_cfg  = {k: getattr(config, k) for k in kw}
    old_corr = {k: getattr(correl, k) for k in kw if hasattr(correl, k)}
    for k, v in kw.items():
        setattr(config, k, v)
        if k in old_corr:
            setattr(correl, k, v)
    try:
        yield
    finally:
        for k, v in old_cfg.items():
            setattr(config, k, v)
        for k, v in old_corr.items():
            setattr(correl, k, v)

# ─── L1–L3 кеши ───────────────────────────────────────────────────────────
@mem_l1.cache
def _clean():
    return preprocess.build_clean_data(save_csv=False)

@mem_l2.cache
def _ppd_events(rel: float):
    ppd_df, oil_df, coords_df = _clean()
    with patch_cfg(PPD_REL_THRESH=rel):
        return events_PPD.detect_ppd_events(ppd_df=ppd_df, save_csv=False)

@mem_l3.cache
def _oil_windows(rel: float, lag: int, check: int, delta_p: float):
    ppd_df, oil_df, coords_df = _clean()
    ppd_ev = _ppd_events(rel)
    pairs_df = well_pairs.build_pairs(
        coords_df=coords_df, oil_df=oil_df, ppd_df=ppd_df, save_csv=False
    )
    with patch_cfg(LAG_DAYS=lag,
                   OIL_CHECK_DAYS=check,
                   OIL_DELTA_P_THRESH=delta_p):
        oil_win = oil_windows.build_oil_windows(
            ppd_events_df=ppd_ev,
            oil_df=oil_df,
            pairs_df=pairs_df,
            save_csv=False
        )
    return oil_win, ppd_ev, oil_df, pairs_df, ppd_df

# ─── GT-оценка ─────────────────────────────────────────────────────────────
def evaluate(final_df: pd.DataFrame):
    gt = Path("start_data/ground_truth.csv")
    if not gt.exists():
        total = len(final_df)
        return 0, 0, total, 0.0
    gt_df = pd.read_csv(gt, dtype=str)
    if "oil_well" not in gt_df.columns and "well" in gt_df.columns:
        gt_df = gt_df.rename(columns={"well": "oil_well"})
    merged = final_df.merge(
        gt_df[["oil_well", "ppd_well", "expected", "acceptable"]],
        on=["oil_well", "ppd_well"], how="right"
    )
    exact  = int((merged["final_cat"] == merged["expected"]).sum())
    nearby = int(merged.apply(
        lambda r: str(r["final_cat"]) in str(r["acceptable"]).split(";")
                  if pd.notna(r["acceptable"]) else False, axis=1
    ).sum())
    miss = len(merged) - exact - nearby
    acc  = (exact + nearby) / len(merged) if len(merged) else 0.0
    return exact, nearby, miss, acc

# ─── Объект-фабрика с RAM-кешем L4/L5 и Pruner ────────────────────────────
def make_objective(early: Dict[str, Any], counter, lock):
    oil_win, ppd_ev, oil_df, pairs_df, ppd_df = _oil_windows(
        early["PPD_REL_THRESH"],
        early["LAG_DAYS"],
        early["OIL_CHECK_DAYS"],
        early["OIL_DELTA_P_THRESH"]
    )

    ci_cache:   Dict[Tuple[Any, ...], pd.DataFrame] = {}
    corr_cache: Dict[Tuple[Any, ...], pd.DataFrame] = {}
    mix_cache:  Dict[Tuple[Any, ...], pd.DataFrame] = {}

    def _objective(trial: optuna.trial.Trial):
        dq = trial.suggest_int("divider_q", 2, 6)
        dp = trial.suggest_int("divider_p", 2, 6)
        wq = trial.suggest_categorical("w_q", [0.4, 0.5, 0.6])
        wp = round(1 - wq, 3)

        ci_low, ci_high = map(float, trial.suggest_categorical(
            "CI_THRESHOLDS", ["0.5,3","1,3","1.5,3.5","2,4","3,5"]
        ).split(","))
        corr_low, corr_high = map(float, trial.suggest_categorical(
            "CORR_THRESHOLDS", ["0.05,0.30","0.10,0.30","0.05,0.40","0.10,0.40"]
        ).split(","))
        mp = trial.suggest_categorical("MIN_POINTS_CCF", [30, 45, 60])

        late_cfg = dict(
            divider_q=dq, divider_p=dp,
            w_q=wq, w_p=wp,
            CI_THRESHOLDS=(ci_low, ci_high),
            CORR_THRESHOLDS=(corr_low, corr_high),
            MIN_POINTS_CCF=mp
        )
        gt_df = pd.read_csv('start_data/ground_truth.csv', dtype=str)
        # CI-кеш
        key_ci = (dq, dp, wq, ci_low, ci_high)
        if key_ci in ci_cache:
            ci_df = ci_cache[key_ci]
        else:
            with patch_cfg(**late_cfg):
                _, ci_df = metrics.compute_ci(
                    (oil_win, ppd_ev, oil_df, pairs_df, gt_df), save_csv=False, filter_by_gt=False
                )
            ci_cache[key_ci] = ci_df

        # Corr-кеш
        key_corr = (mp, corr_low, corr_high)
        if key_corr in corr_cache:
            corr_df = corr_cache[key_corr]
        else:
            with patch_cfg(**late_cfg):
                corr_df = correl.calc_corr(
                    (ppd_df, oil_df, ppd_ev, oil_win, pairs_df),
                    save_csv=False
                )
            corr_cache[key_corr] = corr_df

        best_loss = float("inf")
        rows = []
        for i_dist, dist in enumerate(DIST_LIMITS):
            # final_mix-кеш
            key_mix = key_ci + key_corr + (dist,)
            if key_mix in mix_cache:
                final = mix_cache[key_mix]
            else:
                with patch_cfg(FUSION_DIST_LIMIT=dist):
                    final = final_mix.run_final_mix(
                        corr_df=corr_df,
                        ci_df=ci_df,
                        pairs_df=pairs_df,
                        dist_limit=dist,
                    )
                mix_cache[key_mix] = final

            exact, nearby, miss, acc = evaluate(final)

            loss = -exact + 0.001 * miss
            # отчёт и pruner
            trial.report(loss, step=i_dist)
            if trial.should_prune():
                raise optuna.TrialPruned()

            if loss < best_loss:
                best_loss = loss

            # атомарно считаем прогресс
            with lock:
                counter.value += 1

            rows.append(dict(
                exact=exact, nearby=nearby, miss=miss, accuracy=round(acc,4),
                ppd_rel=early["PPD_REL_THRESH"],
                lag=early["LAG_DAYS"],
                check=early["OIL_CHECK_DAYS"],
                delta_p=early["OIL_DELTA_P_THRESH"],
                div_q=dq, div_p=dp, w_q=wq,
                ci_low=ci_low, ci_high=ci_high,
                corr_low=corr_low, corr_high=corr_high,
                min_pts=mp, dist=dist,
                early_hash=early["EARLY_HASH"],
                trial_id=trial.number
            ))
        trial.set_user_attr("rows", rows)
        return best_loss

    return _objective

# ─── Работа по одному процессу ────────────────────────────────────────────
def run_one(idx: int, ec: dict, trials: int, counter, lock):
    import logging, optuna
    from optuna.samplers import TPESampler
    from optuna.pruners  import MedianPruner
    from optuna.logging  import set_verbosity, WARNING

    set_verbosity(WARNING)
    optuna.logging.disable_default_handler()
    _lg = logging.getLogger("optuna")
    _lg.setLevel(logging.WARNING)
    _lg.handlers.clear()
    _lg.propagate = False

    sampler = TPESampler(
        n_startup_trials=10,
        n_ei_candidates=128,
        seed=None
    )
    pruner  = MedianPruner(
        n_startup_trials=10,
        n_warmup_steps=2
    )

    ec = ec.copy()
    ec["EARLY_HASH"] = hashlib.md5(
        str(sorted(ec.items())).encode()
    ).hexdigest()[:8]

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner
    )
    study.optimize(
        make_objective(ec, counter, lock),
        n_trials=trials,
        n_jobs=1,
        show_progress_bar=False
    )

    all_rows: List[dict] = []
    for t in study.trials:
        all_rows.extend(t.user_attrs.get("rows", []))
    return all_rows

# ─── Генерация «ранних» конфигов ─────────────────────────────────────────
def gen_early(quick: bool):
    keys   = list(EARLY_GRID)
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*(EARLY_GRID[k] for k in keys))]
    return combos[:QUICK_EARLY] if quick else combos

# ─── Монитор прогресса в потоке ──────────────────────────────────────────
def monitor_thread(counter, total, done_evt: Event):
    start = time.perf_counter()
    logging.info("… %d/%d параметров выполнено, прошло %d с", 0, total, 0)
    last = 0
    while True:
        time.sleep(30)
        if done_evt.is_set():
            break
        done = counter.value
        if done >= total:
            break
        if done != last:
            elapsed = int(time.perf_counter() - start)
            logging.info("… %d/%d параметров выполнено, прошло %d с", done, total, elapsed)
            last = done

# ─── Main ─────────────────────────────────────────────────────────────────
def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s | %(message)s",
        datefmt="%H:%M:%S"
    )

    quick  = QUICK_MODE
    trials = QUICK_TRIALS if quick else FULL_TRIALS
    early  = gen_early(quick)
    total  = len(early) * trials * len(DIST_LIMITS)

    logging.info(
        "Старт: quick=%s | процессов=%d | ранних=%d | trials=%d | params=%d",
        quick, JOBS, len(early), trials, total
    )

    _clean()  # прогрев L1–L3 кешей

    mgr     = Manager()
    counter = mgr.Value('i', 0)
    lock    = mgr.Lock()

    done_evt = Event()
    thr = threading.Thread(target=monitor_thread, args=(counter, total, done_evt), daemon=True)
    thr.start()

    start = time.perf_counter()
    rows_nested = joblib.Parallel(n_jobs=JOBS, backend="loky")(
        joblib.delayed(run_one)(i+1, ec, trials, counter, lock)
        for i, ec in enumerate(early)
    )
    done_evt.set()
    thr.join()

    elapsed = time.perf_counter() - start
    elapsed_int = int(elapsed)
    done = counter.value

    # логируем оба показателя: номинальные и реальные
    speed_nominal = total / elapsed if elapsed > 0 else 0
    speed_real    = done  / elapsed if elapsed > 0 else 0
    logging.info("… %d/%d параметров выполнено, прошло %d с", done, total, elapsed_int)
    logging.info(
        "Скорость: nominal=%.2f п./с; real=%.2f п./с (отработано %d)",
        speed_nominal, speed_real, done
    )

    flat = [r for sub in rows_nested for r in sub]
    pd.DataFrame(flat).sort_values(
        ["exact", "nearby", "miss"], ascending=[False, False, True]
    ).to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    logging.info("✓ CSV сохранён → %s | rows=%d", CSV_PATH, len(flat))


if __name__ == "__main__":
    main()
