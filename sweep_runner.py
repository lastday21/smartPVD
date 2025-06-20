"""
sweep_runner.py — перебор гиперпараметров SmartPVD
Python ≥ 3.10

Редактируйте ТОЛЬКО блок НАСТРОЙКИ — остальное не трогайте.
"""

from __future__ import annotations

# ── НАСТРОЙКИ ПРОГОНА ────────────────────────────────────────────────────
QUICK_MODE   = True   # True  → quick-режим, False → полный
JOBS         = 8      # число параллельных ПРОЦЕССОВ (os.cpu_count() для 100%)
QUICK_EARLY  = 4       # сколько ранних конфигов, если QUICK_MODE = True
QUICK_TRIALS = 50      # trials на одну раннюю конфу в quick
FULL_TRIALS  = 500     # trials на одну раннюю конфу в full
# ─────────────────────────────────────────────────────────────────────────

import contextlib, hashlib, itertools, logging, os, time
from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd

# ── Optuna (тихий) ───────────────────────────────────────────────────────
try:
    import optuna

    try:                       # >=3.5
        from optuna.logging import set_verbosity, WARNING as OT_WARNING
        set_verbosity(OT_WARNING)
    except ImportError:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
except ModuleNotFoundError:
    # суррогат, чтобы файл импортировался без Optuna
    import random, types

    class _DTrial:             # noqa: D401
        def __init__(self, n): self.number = n
        def suggest_int(self, *_): return random.randint(2, 6)
        def suggest_categorical(self, _, choices): return random.choice(choices)

    class _DStudy:
        def __init__(self): self.best_value = 0
        def optimize(self, fn, n_trials, **_):
            for i in range(n_trials): fn(_DTrial(i))

    optuna = types.SimpleNamespace(
        samplers=types.SimpleNamespace(TPESampler=lambda seed=None: None),
        create_study=lambda direction, sampler: _DStudy(),
    )

# ── SmartPVD core ────────────────────────────────────────────────────────
import sys
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

import config, preprocess, events_PPD, well_pairs, oil_windows  # noqa: E402
import metrics, correl, final_mix                               # noqa: E402

# ── Глобальные списки параметров ─────────────────────────────────────────
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

# ── patch config + correl ────────────────────────────────────────────────
@contextlib.contextmanager
def patch_cfg(**kw):
    old_cfg  = {k: getattr(config, k) for k in kw}
    old_corr = {k: getattr(correl, k) for k in kw if hasattr(correl, k)}
    for k, v in kw.items():
        setattr(config, k, v)
        if k in old_corr: setattr(correl, k, v)
    try: yield
    finally:
        for k, v in old_cfg.items():  setattr(config, k, v)
        for k, v in old_corr.items(): setattr(correl, k, v)

# ── L1-L3 кеши ───────────────────────────────────────────────────────────
@mem_l1.cache
def _clean():
    return preprocess.build_clean_data(save_csv=False)

@mem_l2.cache
def _ppd_events(rel):
    ppd, _, _ = _clean()
    with patch_cfg(PPD_REL_THRESH=rel):
        return events_PPD.detect_ppd_events(ppd_df=ppd, save_csv=False)

@mem_l3.cache
def _oil_windows(rel, lag, chk, dlt):
    ppd, oil, crd = _clean()
    ppd_ev = _ppd_events(rel)
    pairs  = well_pairs.build_pairs(coords_df=crd, oil_df=oil, ppd_df=ppd, save_csv=False)
    with patch_cfg(LAG_DAYS=lag, OIL_CHECK_DAYS=chk, OIL_DELTA_P_THRESH=dlt):
        wins = oil_windows.build_oil_windows(
            ppd_events_df=ppd_ev, oil_df=oil, pairs_df=pairs, save_csv=False
        )
    return wins, ppd_ev, oil, pairs, ppd

# ── GT-метрики ───────────────────────────────────────────────────────────
def evaluate(final_df: pd.DataFrame):
    gt = Path("start_data/ground_truth.csv")
    if not gt.exists():              # нет эталона — все miss
        tot = len(final_df)
        return 0, 0, tot, 0.0
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
    miss   = int(len(merged) - exact - nearby)
    acc    = (exact + nearby) / len(merged) if len(merged) else 0.0
    return exact, nearby, miss, acc

# ── objective фабрика ────────────────────────────────────────────────────
def make_objective(early: Dict[str, Any]):
    wins, ppd_ev, oil_df, pairs_df, ppd_df = _oil_windows(
        early["PPD_REL_THRESH"], early["LAG_DAYS"],
        early["OIL_CHECK_DAYS"], early["OIL_DELTA_P_THRESH"]
    )

    def obj(trial):
        divider_q = trial.suggest_int("divider_q", 2, 6)
        divider_p = trial.suggest_int("divider_p", 2, 6)
        w_q       = trial.suggest_categorical("w_q", [0.4, 0.5, 0.6])
        w_p       = round(1 - w_q, 3)

        ci_low, ci_high = map(float, trial.suggest_categorical(
            "CI_THRESHOLDS", ["0.5,3", "1,3", "1.5,3.5", "2,4", "3,5"]).split(','))

        corr_low, corr_high = map(float, trial.suggest_categorical(
            "CORR_THRESHOLDS", ["0.05,0.30", "0.10,0.30", "0.05,0.40", "0.10,0.40"]).split(','))

        min_pts = trial.suggest_categorical("MIN_POINTS_CCF", [30, 45, 60])

        late = dict(divider_q=divider_q, divider_p=divider_p,
                    w_q=w_q, w_p=w_p,
                    CI_THRESHOLDS=(ci_low, ci_high),
                    CORR_THRESHOLDS=(corr_low, corr_high),
                    MIN_POINTS_CCF=min_pts)

        with patch_cfg(**late):
            _, ci_df = metrics.compute_ci((wins, ppd_ev, oil_df, pairs_df), save_csv=False)
            corr_df  = correl.calc_corr((ppd_df, oil_df, ppd_ev, wins, pairs_df), save_csv=False)

        best = float('inf')
        rows = []
        for dist in DIST_LIMITS:
            with patch_cfg(FUSION_DIST_LIMIT=dist):
                t0 = time.perf_counter()
                final = final_mix.run_final_mix(corr_df, ci_df, pairs_df, dist_limit=dist)
                rt  = int((time.perf_counter() - t0) * 1000)

            exact, nearby, miss, acc = evaluate(final)
            loss = -exact + 0.001 * miss
            best = min(best, loss)

            rows.append(dict(
                exact=exact, nearby=nearby, miss=miss, accuracy=round(acc,4),
                ppd_rel=early["PPD_REL_THRESH"], lag=early["LAG_DAYS"],
                check=early["OIL_CHECK_DAYS"], delta_p=early["OIL_DELTA_P_THRESH"],
                div_q=divider_q, div_p=divider_p, w_q=w_q,
                ci_low=ci_low, ci_high=ci_high,
                corr_low=corr_low, corr_high=corr_high,
                min_pts=min_pts, dist=dist,
                early_hash=early["EARLY_HASH"], trial_id=trial.number,
                runtime_ms=rt
            ))
        trial.set_user_attr("rows", rows)
        return best
    return obj

# ── одна ранняя конфигурация (работает в ПРОЦЕССЕ) ───────────────────────
def run_one(idx: int, early_cfg: dict, trials: int):
    early_cfg = early_cfg.copy()        # локальная копия
    early_cfg["EARLY_HASH"] = hashlib.md5(
        str(sorted(early_cfg.items())).encode()).hexdigest()[:8]

    logging.info("► proc %d | hash %s", idx, early_cfg["EARLY_HASH"])

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler())  # без seed
    study.optimize(make_objective(early_cfg), n_trials=trials, n_jobs=1, show_progress_bar=False)

    # вытаскиваем все накопленные строки CSV из trials
    all_rows = []
    for t in study.trials:
        all_rows.extend(t.user_attrs.get("rows", []))
    return all_rows

# ── генерация ранних ─────────────────────────────────────────────────────
def gen_early(quick: bool):
    keys  = list(EARLY_GRID)
    combos = [dict(zip(keys, vals))
              for vals in itertools.product(*(EARLY_GRID[k] for k in keys))]
    return combos[:QUICK_EARLY] if quick else combos

# ── главный запуск ───────────────────────────────────────────────────────
def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s | %(message)s",
        datefmt="%H:%M:%S"
    )

    quick  = QUICK_MODE
    trials = QUICK_TRIALS if quick else FULL_TRIALS
    early  = gen_early(quick)

    total_params = len(early)*trials*len(DIST_LIMITS)
    logging.info("Старт: quick=%s | процессов=%d | ранних=%d | trials=%d | params=%d",
                 quick, JOBS, len(early), trials, total_params)

    _clean()   # прогрев кеша

    start = time.perf_counter()
    rows_nested = joblib.Parallel(n_jobs=JOBS, backend="loky")(
        joblib.delayed(run_one)(i+1, ec, trials)
        for i, ec in enumerate(early)
    )
    elapsed = int(time.perf_counter() - start)
    logging.info("Все процессы завершили работу за %d с", elapsed)

    # собираем и сохраняем CSV
    flat_rows: List[dict] = [r for sub in rows_nested for r in sub]
    df = (pd.DataFrame(flat_rows)
            .sort_values(["exact","nearby","miss"], ascending=[False,False,True])
            .reset_index(drop=True))
    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    logging.info("✓ CSV сохранён → %s | rows=%d", CSV_PATH, len(df))

if __name__ == "__main__":
    main()
