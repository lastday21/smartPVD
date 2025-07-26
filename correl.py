"""
Модуль рассчитывает категорию связи между закачкой (ppd-скважины) и добычей
(oil-скважины) на основе корреляционного анализа (событийный Spearman ρ и
кросс-корреляция ∆-рядов). Логика вынесена в _calc_corr_df ― чистую
функцию-ядро, свободную от I/O. Для использования в пайплайне (импорт),
CLI и unit-тестах предоставлен универсальный интерфейс calc_corr.

Структура:
    • _calc_corr_df  ― ядро, принимает 5 DataFrame и возвращает итоговый
      DataFrame (вкл. строку TOTALS).
    • calc_corr      ― публичная оболочка с универсальной сигнатурой
      (df | csv_path, save_csv), обеспечивающая загрузку/сохранение CSV.
    • CLI-блок       ― воспроизводит поведение исходного correl.py.

"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, Tuple, Union, Mapping, Sequence

import pandas as pd

# ── Константы из config.py ─────────────────────────────────────────────
from config import (
    CORR_THRESHOLDS,  # thresholds for |ρ|
    MAX_LAG,          # ± days for CCF
    PENALTY_NEG,      # penalty for negative ρ
    PENALTY_ONE_EVT,  # penalty for exactly 1 event
    MIN_POINTS_CCF,   # minimum points for CCF
)

# ── Пути и файлы ───────────────────────────────────────────────────────
BASE_DIR: Path = Path("clean_data")
OUT_CSV: Path = BASE_DIR / "corr_results.csv"
GROUND_TRUTH: Path = Path("start_data") / "ground_truth.csv"

LEVELS: Tuple[str, ...] = ("none", "weak", "impact")
L2I: Dict[str, int] = {lvl: i for i, lvl in enumerate(LEVELS)}

# Подавляем ворнинги pandas о парсинге дат (специфично для ДД.ММ.ГГГГ)
warnings.filterwarnings(
    "ignore",
    message="Parsing dates in .* format when dayfirst=.* was specified",
)

# ───────────────────────────────────────────────────────────────────────
#                              ВСПОМОГАТЕЛЬНОЕ
# ───────────────────────────────────────────────────────────────────────
def _cat_by_abs(r: float) -> str:
    """Возвращает категорию ('none'…'impact') по абсолютному значению |ρ|."""
    t1, t2 = CORR_THRESHOLDS
    if r < t1:
        return "none"
    if r < t2:
        return "weak"
    return "impact"


def _best_ccf(inj: pd.Series, liq: pd.Series) -> float:
    """
    Вычисляет максимальный |ρ| кросс-корреляции ∆-рядов при лагах ±MAX_LAG.

    Параметры
    ----------
    inj, liq : pd.Series
        Ряды суточных объёмов закачки и добычи соответственно
        (индекс ― pd.DatetimeIndex).

    Returns
    -------
    float
        Максимальное по модулю значение Spearman ρ; 0.0, если точек < MIN_POINTS_CCF.
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
    Универсальный парсер дат:
      • сначала стандартный парсер pandas (ISO-форматы, YYYY-MM-DD),
      • затем при необходимости формат «ДД.ММ.ГГГГ».
    """
    dt = pd.to_datetime(col, dayfirst=True, errors="coerce")
    mask = dt.isna()
    if mask.any():
        dt.loc[mask] = pd.to_datetime(
            col[mask], format="%d.%m.%Y", dayfirst=True, errors="coerce"
        )
    return dt


# ───────────────────────────────────────────────────────────────────────
#                              ЧИСТОЕ ЯДРО
# ───────────────────────────────────────────────────────────────────────
def _calc_corr_df(
    ppd_clean: pd.DataFrame,
    oil_clean: pd.DataFrame,
    ppd_events: pd.DataFrame,
    oil_windows: pd.DataFrame,
    pairs: pd.DataFrame,
) -> pd.DataFrame:
    """
    Ядро корреляционного этапа (без I/O).

    Параметры
    ----------
    ppd_clean : pd.DataFrame
        Сырые суточные объёмы закачки (ppd) с колонками ['well', 'date', 'q_ppd'].
    oil_clean : pd.DataFrame
        Сырые суточные объёмы добычи (oil) с колонками
        ['well', 'date', 'q_oil', 'watercut'].
    ppd_events : pd.DataFrame
        Моменты начала событий интенсивной закачки (ppd) с колонками
        ['ppd_well', 'start_date', 'baseline_before', 'baseline_during'].
    oil_windows : pd.DataFrame
        Окна отклика добычи с колонками
        ['oil_well', 'ppd_well', 'ppd_start', 'q_start', 'q_end'].
    pairs : pd.DataFrame
        Все пары (oil_well → ppd_well), которые требуется оценить, с колонками
        ['oil_well', 'ppd_well'].

    Returns
    -------
    pd.DataFrame
        Итоговый DataFrame со столбцами:
            ['oil_well', 'ppd_well', 'corr_cat', 'abs_corr',
             'n_events', 'event_corr', 'ccf_corr', 'expected', 'acceptable']
        + строка 'TOTALS' с агрегированной статистикой.
    """
    # 1) Приводим названия скважин к str и унифицируем имена столбцов ------- #
    ppd_events = ppd_events.rename(columns={"well": "ppd_well"}, errors="ignore")
    oil_windows = oil_windows.rename(columns={"well": "oil_well"}, errors="ignore")

    for df, c1, c2 in (
        (pairs, "oil_well", "ppd_well"),
        (oil_clean, "well", None),
        (ppd_clean, "well", None),
    ):
        df[c1] = df[c1].astype(str)
        if c2:
            df[c2] = df[c2].astype(str)

    # 2) Парсим даты -------------------------------------------------------- #
    ppd_clean["date"] = _to_datetime(ppd_clean["date"])
    oil_clean["date"] = _to_datetime(oil_clean["date"])

    # 3) Собираем суточные ряды -------------------------------------------- #
    inj = {
        w: g.set_index("date")["q_ppd"].astype(float).sort_index()
        for w, g in ppd_clean.groupby("well")
    }
    oil_clean["q_liq"] = oil_clean["q_oil"]
    liq = {
        w: g.set_index("date")["q_liq"].astype(float).sort_index()
        for w, g in oil_clean.groupby("well")
    }

    # 4) ∆Q по событиям ----------------------------------------------------- #
    ppd_events["event_start"] = _to_datetime(ppd_events["start_date"])
    oil_windows["event_start"] = _to_datetime(oil_windows["ppd_start"])

    ppd_events["delta_q_inj"] = (
        ppd_events["baseline_during"] - ppd_events["baseline_before"]
    )
    oil_windows["delta_q_liq"] = oil_windows["q_end"] - oil_windows["q_start"]

    ev = (
        oil_windows[["oil_well", "ppd_well", "event_start", "delta_q_liq"]]
        .merge(
            ppd_events[["ppd_well", "event_start", "delta_q_inj"]],
            on=["ppd_well", "event_start"],
        )
    )

    # 5) Сводная статистика по событиям ------------------------------------ #
    ev_stats = (
        ev.groupby(["oil_well", "ppd_well"], as_index=False)
        .agg(
            n_events=("delta_q_inj", "size"),
            event_corr=(
                "delta_q_inj",
                lambda s: s.corr(
                    ev.loc[s.index, "delta_q_liq"],
                    method="spearman",
                ),
            ),
        )
        .astype({"oil_well": str, "ppd_well": str})
    )

    # 6) Кросс-корреляция ∆-рядов ------------------------------------------ #
    ccf_df = pd.DataFrame(
        [
            {
                "oil_well": o,
                "ppd_well": p,
                "ccf_corr": _best_ccf(
                    inj.get(p, pd.Series(dtype=float)),
                    liq.get(o, pd.Series(dtype=float)),
                ),
            }
            for o, p in pairs[["oil_well", "ppd_well"]].drop_duplicates().values
        ]
    )

    # 7) Объединяем признаки ---------------------------------------------- #
    feat = (
        pairs[["oil_well", "ppd_well"]]
        .drop_duplicates()
        .merge(ev_stats, on=["oil_well", "ppd_well"], how="left")
        .merge(ccf_df, on=["oil_well", "ppd_well"], how="left")
        .fillna({"n_events": 0, "event_corr": 0.0, "ccf_corr": 0.0})
    )

    # 8) Финальная категория с учётом штрафов ------------------------------ #
    def _assign_cat(r: pd.Series) -> str:
        best = r["event_corr"] if abs(r["event_corr"]) >= abs(r["ccf_corr"]) else r[
            "ccf_corr"
        ]
        lvl = L2I[_cat_by_abs(abs(best))]
        if best < 0:
            lvl -= PENALTY_NEG
        if r["n_events"] == 1:
            lvl -= PENALTY_ONE_EVT
        return LEVELS[max(0, lvl)]

    feat["corr_cat"] = feat.apply(_assign_cat, axis=1)
    feat["abs_corr"] = feat[["event_corr", "ccf_corr"]].abs().max(axis=1)

    # 9) Добавляем ground_truth (если есть) -------------------------------- #
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
        feat["expected"] = feat["expected"].fillna("")
        feat["acceptable"] = feat["acceptable"].fillna("")
    else:
        feat["expected"] = ""
        feat["acceptable"] = ""

    # 10) Итоговый DataFrame + строка TOTALS ------------------------------ #
    out_cols = [
        "oil_well",
        "ppd_well",
        "corr_cat",
        "abs_corr",
        "n_events",
        "event_corr",
        "ccf_corr",
        "expected",
        "acceptable",
    ]
    corr_df = feat[out_cols]

    total = len(corr_df)
    exact = int(
        ((corr_df["expected"] != "") & (corr_df["corr_cat"] == corr_df["expected"])).sum()
    )

    acc_lists = corr_df["acceptable"].fillna("").apply(
        lambda s: [x.strip() for x in s.split(";")] if s else []
    )
    nearby = int(
        sum(
            (exp != "") and (cat != exp) and (cat in acc)
            for exp, cat, acc in zip(
                corr_df["expected"], corr_df["corr_cat"], acc_lists
            )
        )
    )

    miss = total - exact - nearby
    accuracy = (exact + nearby) / total if total else 0.0

    totals = pd.DataFrame(
        [
            {
                "oil_well": "TOTALS",
                "ppd_well": "",
                "corr_cat": "",
                "abs_corr": f"exact={exact}",
                "n_events": f"nearby={nearby}",
                "event_corr": f"miss={miss}",
                "ccf_corr": f"all={total}",
                "expected": f"accuracy={accuracy:.2f}",
                "acceptable": "",
            }
        ]
    )

    return pd.concat([corr_df, totals], ignore_index=True)


# ───────────────────────────────────────────────────────────────────────
#                       УНИВЕРСАЛЬНЫЙ ИНТЕРФЕЙС
# ───────────────────────────────────────────────────────────────────────
def calc_corr(
    df: Union[
        Mapping[str, pd.DataFrame],
        Sequence[pd.DataFrame],
        pd.DataFrame,
        None,
    ] = None,
    csv_path: Union[str, Path, None] = None,
    save_csv: bool = True,
) -> pd.DataFrame:
    """
    Универсальный интерфейс «DataFrame или CSV-директория → DataFrame».

    Параметры
    ----------
    df : Mapping[str, pd.DataFrame] | Sequence[pd.DataFrame] | None, optional
        • Если передан словарь, ожидаются ключи:
          {'ppd_clean', 'oil_clean', 'ppd_events', 'oil_windows', 'pairs_df'}.
        • Если передана последовательность/кортеж, порядок ―
          (ppd_clean, oil_clean, ppd_events, oil_windows, pairs_df).
        • Если None, входные CSV будут прочитаны из *csv_path*.
    csv_path : str | Path | None, optional
        Путь к директории, содержащей пять CSV-файлов с именами:
            ppd_clean.csv,  oil_clean.csv,
            ppd_events.csv, oil_windows.csv, pairs_oil_ppd.csv.
        Если не задано, используется директория *clean_data/*.
    save_csv : bool, default=True
        Сохранять ли результат в ``clean_data/corr_results.csv``.

    Returns
    -------
    pd.DataFrame
        Итоговый DataFrame, сформированный функцией :pyfunc:`_calc_corr_df`.
    """
    # ── 1. Загружаем входные данные ────────────────────────────────────── #
    if df is not None:
        # --- вариант: словарь DataFrame ---------------------------------- #
        if isinstance(df, Mapping):
            try:
                ppd_clean = df["ppd_clean"]
                oil_clean = df["oil_clean"]
                ppd_events = df["ppd_events"]
                oil_windows = df["oil_windows"]
                pairs_df = df["pairs_df"]
            except KeyError as exc:
                raise ValueError(
                    "При использовании словаря df необходимы ключи "
                    "{'ppd_clean', 'oil_clean', 'ppd_events', 'oil_windows', 'pairs_df'}."
                ) from exc
        # --- вариант: последовательность / кортеж ------------------------ #
        elif isinstance(df, Sequence) and not isinstance(df, pd.DataFrame):
            if len(df) != 5:
                raise ValueError(
                    "Передайте ровно 5 DataFrame в порядке "
                    "(ppd_clean, oil_clean, ppd_events, oil_windows, pairs_df)."
                )
            (
                ppd_clean,
                oil_clean,
                ppd_events,
                oil_windows,
                pairs_df,
            ) = df
        # --- вариант: единичный DataFrame (не допускается) --------------- #
        else:
            raise TypeError(
                "df должен быть Mapping[str, DataFrame] или "
                "Sequence[DataFrame] (из 5 элементов); "
                "передан одиночный DataFrame."
            )
    else:
        # --- читаем CSV из директории csv_path --------------------------- #
        path = Path(csv_path) if csv_path else BASE_DIR
        ppd_clean = pd.read_csv(path / "ppd_clean.csv")
        oil_clean = pd.read_csv(path / "oil_clean.csv")
        ppd_events = pd.read_csv(path / "ppd_events.csv")
        oil_windows = pd.read_csv(path / "oil_windows.csv")
        pairs_df = pd.read_csv(path / "pairs_oil_ppd.csv")

    # ── 2. Вызываем чистое ядро ────────────────────────────────────────── #
    result_df = _calc_corr_df(
        ppd_clean=ppd_clean,
        oil_clean=oil_clean,
        ppd_events=ppd_events,
        oil_windows=oil_windows,
        pairs=pairs_df,
    )

    # ── 3. Сохраняем CSV (если нужно) ──────────────────────────────────── #
    if save_csv:
        BASE_DIR.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    return result_df


# ───────────────────────────────────────────────────────────────────────
#                                CLI
# ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[CORREL] ▶ расчёт корреляций…")
    df_result = calc_corr(csv_path="clean_data", save_csv=True)
    print(f"[CORREL] ✔ {len(df_result) - 1} пар → {OUT_CSV}")
