"""
correl.py — корреляционный этап пайплайна SmartPVD-MVP.

Для каждой пары (oil_well → ppd_well) рассчитывает категорию связи corr_cat ∈
{'none','weak','medium','strong'} на основе двух метрик:
  • event_corr — Spearman ρ между ΔQ_inj и ΔQ_liq по окнам событий
  • ccf_corr   — максимальный |ρ| кросс-корреляции Δ-рядов при лагах ±MAX_LAG дней

Затем базовая категория по |ρ| понижается за:
  – отрицательный знак ρ  → PENALTY_NEG
  – ровно 1 событие      → PENALTY_ONE_EVT
  – ровно 2 события      → PENALTY_TWO_EVT

Выходной CSV/DF содержит колонки:
    oil_well, ppd_well, corr_cat, abs_corr,
    n_events, event_corr, ccf_corr, expected, acceptable
и строку TOTALS с подсчетом exact/miss/all/accuracy/off.

Режимы работы:
  • Импорт в пайплайн:
        from correl import calc_corr
        corr_df = calc_corr(ppd_clean, oil_clean, ppd_events, oil_windows, pairs_df)
  • CLI:
        python correl.py    # создаст clean_data/corr_results.csv
"""
from __future__ import annotations
import warnings
from pathlib import Path
from typing import Dict, Tuple, List

import pandas as pd

# ── Константы из config.py ─────────────────────────────────────────────
from config import (
    CORR_THRESHOLDS,  # thresholds for |ρ|
    MAX_LAG,          # ± days for CCF
    PENALTY_NEG,      # penalty for negative ρ
    PENALTY_ONE_EVT,  # penalty for exactly 1 event
    PENALTY_TWO_EVT,  # penalty for exactly 2 events
    MIN_POINTS_CCF,   # minimum points for CCF
)

# ── Пути и вспомогательные переменные ───────────────────────────────────
BASE_DIR     = Path("clean_data")
OUT_CSV      = BASE_DIR / "corr_results.csv"
GROUND_TRUTH = Path("start_data") / "ground_truth.csv"

LEVELS: Tuple[str, ...] = ("none", "weak", "medium", "strong")
L2I:    Dict[str, int]  = {lvl: i for i, lvl in enumerate(LEVELS)}

# Подавляем warnings о парсинге дат pandas
warnings.filterwarnings(
    "ignore",
    message="Parsing dates in .* format when dayfirst=.* was specified"
)


# ── Вспомогательные функции ────────────────────────────────────────────
def _cat_by_abs(r: float) -> str:
    """Категория по |ρ| и порогам CORR_THRESHOLDS."""
    t1, t2, t3 = CORR_THRESHOLDS
    if r < t1:
        return "none"
    if r < t2:
        return "weak"
    if r < t3:
        return "medium"
    return "strong"


def _best_ccf(inj: pd.Series, liq: pd.Series) -> float:
    """
    Максимальный |ρ| кросс-корреляции Δ-рядов при лагах ±MAX_LAG.
    Возвращает 0.0, если длина ряда < MIN_POINTS_CCF.
    """
    df = pd.concat([inj, liq], axis=1, keys=["inj", "liq"]).dropna()
    if len(df) < MIN_POINTS_CCF:
        return 0.0
    inj_d = df["inj"].diff().dropna()
    liq_d = df["liq"].diff().dropna()
    df2 = pd.concat([inj_d, liq_d], axis=1).dropna()

    best = 0.0
    for lag in range(-MAX_LAG, MAX_LAG + 1):
        r = df2["inj"].corr(df2["liq"].shift(lag), method="spearman")
        if pd.notna(r) and abs(r) > abs(best):
            best = r
    return best


def _to_datetime(col: pd.Series) -> pd.Series:
    """
    Парсит столбец дат:
      • сначала стандартный ISO / pandas-парсер (YYYY-MM-DD и др.)
      • затем при необходимости явный формат DD.MM.YYYY
    """
    dt = pd.to_datetime(col, dayfirst=True, errors="coerce")
    mask = dt.isna()
    if mask.any():
        dt.loc[mask] = pd.to_datetime(
            col[mask], format="%d.%m.%Y", dayfirst=True, errors="coerce"
        )
    return dt


# ── Основная функция для пайплайна ─────────────────────────────────────
def calc_corr(
    ppd_clean:   pd.DataFrame,
    oil_clean:   pd.DataFrame,
    ppd_events:  pd.DataFrame,
    oil_windows: pd.DataFrame,
    pairs:       pd.DataFrame,
) -> pd.DataFrame:
    """
    Принимает загруженные DataFrame’ы, возвращает итоговый DataFrame
    с колонками [oil_well, ppd_well, corr_cat, abs_corr,
                  n_events, event_corr, ccf_corr, expected, acceptable]
    и строку TOTALS.
    """

    # 1) Унитаризуем имена скважин и приводим их к str
    ppd_events = ppd_events.rename(columns={"well": "ppd_well"}, errors="ignore")
    oil_windows = oil_windows.rename(columns={"well": "oil_well"}, errors="ignore")
    for df, c1, c2 in (
        (pairs,     "oil_well", "ppd_well"),
        (oil_clean, "well",     None),
        (ppd_clean, "well",     None),
    ):
        df[c1] = df[c1].astype(str)
        if c2:
            df[c2] = df[c2].astype(str)

    # 2) Парсим даты
    ppd_clean["date"] = _to_datetime(ppd_clean["date"])
    oil_clean["date"] = _to_datetime(oil_clean["date"])

    # 3) Строим словари суточных рядов
    inj = {
        w: g.set_index("date")["q_ppd"].astype(float).sort_index()
        for w, g in ppd_clean.groupby("well")
    }
    oil_clean["q_liq"] = oil_clean["q_oil"] / (1 - oil_clean["watercut"] / 100)
    liq = {
        w: g.set_index("date")["q_liq"].astype(float).sort_index()
        for w, g in oil_clean.groupby("well")
    }

    # 4) Вычисляем ΔQ по событиям
    ppd_events["event_start"]  = _to_datetime(ppd_events["start_date"])
    oil_windows["event_start"] = _to_datetime(oil_windows["ppd_start"])
    ppd_events["delta_q_inj"]  = (
        ppd_events["baseline_during"] - ppd_events["baseline_before"]
    )
    oil_windows["delta_q_liq"] = (
        oil_windows["q_end"] - oil_windows["q_start"]
    )
    ev = (
        oil_windows[["oil_well", "ppd_well", "event_start", "delta_q_liq"]]
        .merge(
            ppd_events[["ppd_well", "event_start", "delta_q_inj"]],
            on=["ppd_well", "event_start"],
        )
    )

    # 5) Собираем статистику по событиям
    ev_stats = (
        ev.groupby(["oil_well", "ppd_well"], as_index=False)
        .agg(
            n_events=("delta_q_inj", "size"),
            event_corr=(
                "delta_q_inj",
                lambda s: s.corr(ev.loc[s.index, "delta_q_liq"], method="spearman"),
            ),
        )
        .astype({"oil_well": str, "ppd_well": str})
    )

    # 6) Строим DataFrame ccf_corr
    ccf_df = pd.DataFrame(
        [
            {
                "oil_well": o,
                "ppd_well": p,
                "ccf_corr": _best_ccf(inj.get(p, pd.Series(dtype=float)),
                                      liq.get(o, pd.Series(dtype=float))),
            }
            for o, p in pairs[["oil_well", "ppd_well"]].drop_duplicates().values
        ]
    )

    # 7) Объединяем все признаки
    feat = (
        pairs[["oil_well", "ppd_well"]]
        .drop_duplicates()
        .merge(ev_stats, on=["oil_well", "ppd_well"], how="left")
        .merge(ccf_df,   on=["oil_well", "ppd_well"], how="left")
        .fillna({"n_events": 0, "event_corr": 0.0, "ccf_corr": 0.0})
    )

    # 8) Присваиваем финальную категорию с учётом штрафов
    def _assign_cat(r: pd.Series) -> str:
        best = (
            r["event_corr"]
            if abs(r["event_corr"]) >= abs(r["ccf_corr"])
            else r["ccf_corr"]
        )
        lvl = L2I[_cat_by_abs(abs(best))]
        if best < 0:
            lvl -= PENALTY_NEG
        if r["n_events"] == 1:
            lvl -= PENALTY_ONE_EVT
        elif r["n_events"] == 2:
            lvl -= PENALTY_TWO_EVT
        return LEVELS[max(0, lvl)]

    feat["corr_cat"] = feat.apply(_assign_cat, axis=1)
    feat["abs_corr"] = feat[["event_corr", "ccf_corr"]].abs().max(axis=1)

    # 9) Подхватываем ground_truth, если он есть
    if GROUND_TRUTH.exists():
        gt = pd.read_csv(GROUND_TRUTH, dtype=str)
        if "well" in gt.columns:
            gt = gt.rename(columns={"well": "oil_well"}, errors="ignore")
        gt[["oil_well", "ppd_well"]] = gt[["oil_well", "ppd_well"]].astype(str)
        feat = feat.merge(
            gt[["oil_well", "ppd_well", "expected", "acceptable"]],
            on=["oil_well", "ppd_well"],
            how="left",
        )
        feat["expected"]   = feat["expected"].fillna("")
        feat["acceptable"] = feat["acceptable"].fillna("")
    else:
        feat["expected"]   = ""
        feat["acceptable"] = ""

    # 10) Итоговый DataFrame + строка TOTALS
    out_cols = [
        "oil_well", "ppd_well", "corr_cat", "abs_corr",
        "n_events", "event_corr", "ccf_corr", "expected", "acceptable",
    ]
    corr_df = feat[out_cols]

    total = len(corr_df)
    # точные совпадения
    exact = int(((corr_df["expected"] != "") & (corr_df["corr_cat"] == corr_df["expected"])).sum())
    # «nearby» – допустимые (acceptable), но не точные
    acc_lists = corr_df["acceptable"].fillna("").apply(
        lambda s: [x.strip() for x in s.split(";")] if s else []
    )
    nearby = int(sum(
        (exp != "") and (cat != exp) and (cat in acc)
        for exp, cat, acc in zip(corr_df["expected"], corr_df["corr_cat"], acc_lists)
    ))
    # промахи
    miss = total - exact - nearby
    # точность учитывает exact + nearby
    accuracy = (exact + nearby) / total if total else 0.0

    totals = pd.DataFrame([{
        "oil_well": "TOTALS",
        "ppd_well": "",
        "corr_cat": "",
        "abs_corr": f"exact={exact}",
        "n_events": f"nearby={nearby}",
        "event_corr": f"miss={miss}",
        "ccf_corr": f"all={total}",
        "expected": f"accuracy={accuracy:.2f}",
        "acceptable": ""
    }])

    return pd.concat([corr_df, totals], ignore_index=True)


# ── CLI: читаем clean_data/*, рассчитываем и сохраняем ─────────────────
def _load_csv() -> Tuple[pd.DataFrame, ...]:
    return (
        pd.read_csv(BASE_DIR / "ppd_clean.csv"),
        pd.read_csv(BASE_DIR / "oil_clean.csv"),
        pd.read_csv(BASE_DIR / "ppd_events.csv"),
        pd.read_csv(BASE_DIR / "oil_windows.csv"),
        pd.read_csv(BASE_DIR / "pairs_oil_ppd.csv"),
    )


if __name__ == "__main__":
    print("[CORREL] ▶ расчёт…")
    df = calc_corr(*_load_csv())
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[CORREL] ✔ {len(df)-1} пар → {OUT_CSV}")
