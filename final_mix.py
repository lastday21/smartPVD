"""

 Гибридное объединение категорий корреляции и CI:
    Parameters:
        corr_df: DataFrame с колонками ['oil_well','ppd_well','corr_cat']
        ci_df:   DataFrame с колонками ['oil_well','ppd_well','ci_cat']
        pairs_df:DataFrame с ['oil_well','ppd_well','distance']
        allowed_pairs: set кортежей (oil_well, ppd_well)
        dist_limit:   порог дистанции из config.FUSION_DIST_LIMIT
    Returns:
        DataFrame ['oil_well','ppd_well','final_cat']

"""

from __future__ import annotations
from pathlib import Path
import sys
import pandas as pd

from config import FUSION_DIST_LIMIT

# Уровни категорий для сравнения
LEVELS = {"none": 0, "weak": 1, "impact": 2}
INV_LEVELS = {v: k for k, v in LEVELS.items()}


def run_final_mix(
    corr_df: pd.DataFrame,
    ci_df: pd.DataFrame,
    pairs_df: pd.DataFrame,
    allowed_pairs: set[tuple[str, str]] | None = None,
    dist_limit: int = FUSION_DIST_LIMIT,
    filter_by_gt: bool = False,
) -> pd.DataFrame:
    """
    Гибридное объединение категорий корреляции и CI.

    Parameters:
        corr_df (pd.DataFrame): DataFrame с колонками ['oil_well','ppd_well','corr_cat'].
        ci_df (pd.DataFrame):   DataFrame с колонками ['oil_well','ppd_well','ci_cat'].
        pairs_df (pd.DataFrame):DataFrame с колонками ['oil_well','ppd_well','distance'].
        allowed_pairs (set of tuple[str,str] or None): пары для фильтрации (если filter_by_gt=True).
        dist_limit (int):       порог дистанции для корректировки CI.
        filter_by_gt (bool):    если True и allowed_pairs задан, фильтровать пары по GT.

    Returns:
        pd.DataFrame: DataFrame с колонками
            ['oil_well','ppd_well','distance','corr_cat','ci_cat','final_cat'].
    """
    # Приводим ключевые столбцы к str
    for df in (corr_df, ci_df, pairs_df):
        df["oil_well"] = df["oil_well"].astype(str)
        df["ppd_well"] = df["ppd_well"].astype(str)

    # Фильтрация по ground_truth, если включена
    if filter_by_gt and allowed_pairs is not None:
        pairs_df = pairs_df[
            pairs_df.apply(lambda r: (r.oil_well, r.ppd_well) in allowed_pairs, axis=1)
        ]

    # Удаляем служебную строку TOTALS, если она есть
    corr_clean = corr_df[corr_df["oil_well"] != "TOTALS"].copy()
    ci_clean   = ci_df[ci_df["oil_well"]   != "TOTALS"].copy()

    # Собираем базовую таблицу: пары + corr_cat + ci_cat
    base = (
        pairs_df[["oil_well", "ppd_well", "distance"]]
        .merge(
            corr_clean[["oil_well", "ppd_well", "corr_cat"]],
            on=["oil_well", "ppd_well"],
            how="left",
        )
        .merge(
            ci_clean[["oil_well", "ppd_well", "ci_cat"]],
            on=["oil_well", "ppd_well"],
            how="left",
        )
        .fillna({"corr_cat": "none", "ci_cat": "none"})
    )

    # Функция выбора и коррекции финальной категории
    def fuse(row):
        corr_cat = row["corr_cat"]
        ci_cat   = row["ci_cat"]
        dist     = row["distance"]
        # Выбираем максимальную категорию
        chosen = ci_cat if LEVELS[ci_cat] > LEVELS[corr_cat] else corr_cat
        # Приоритет CI, но коррекция на дальнем расстоянии
        if (
            chosen == ci_cat
            and dist > dist_limit
            and LEVELS[ci_cat] >= LEVELS["impact"]
            and LEVELS[corr_cat] <= LEVELS["weak"]
        ):
            chosen = INV_LEVELS[LEVELS[ci_cat] - 1]
        return chosen

    base["final_cat"] = base.apply(fuse, axis=1)
    return base[["oil_well", "ppd_well", "distance", "corr_cat", "ci_cat", "final_cat"]]


if __name__ == "__main__":
    ROOT      = Path(__file__).resolve().parent
    CLEAN_DIR = ROOT / "clean_data"
    START_DIR = ROOT / "start_data"
    CLEAN_DIR.mkdir(exist_ok=True)
    START_DIR.mkdir(exist_ok=True)

    # Читаем входные CSV
    try:
        corr  = pd.read_csv(CLEAN_DIR / "corr_results.csv")
        ci    = pd.read_csv(CLEAN_DIR / "ci_results_agg.csv")
        pairs = pd.read_csv(CLEAN_DIR / "pairs_oil_ppd.csv")
    except FileNotFoundError as e:
        sys.exit(f"[ERROR] Не найден входной файл: {e.filename}")

    # Включаем работу с ground_truth, если нужно
    # Поставьте True, чтобы отфильтровать по GT и добавить expected/acceptable
    filter_by_gt = True

    if filter_by_gt:
        gt_path = START_DIR / "ground_truth.csv"
        if not gt_path.exists():
            sys.exit(f"[ERROR] filter_by_gt=True, но {gt_path} не найден")
        gt_df   = pd.read_csv(gt_path, dtype=str)
        allowed = set(zip(gt_df["well"].astype(str), gt_df["ppd_well"].astype(str)))
    else:
        allowed = None

    # Вычисляем финальный DataFrame
    final = run_final_mix(
        corr_df=corr,
        ci_df=ci,
        pairs_df=pairs,
        allowed_pairs=allowed,
        dist_limit=FUSION_DIST_LIMIT,
        filter_by_gt=filter_by_gt,
    )

    # Если фильтрация GT включена, добавляем expected/acceptable
    if filter_by_gt:
        gt_df = gt_df.rename(columns={"well": "oil_well"})
        gt_df["acceptable"] = (
            gt_df.get("acceptable", "")
            .fillna("")
            .apply(lambda s: [x.strip() for x in s.split(";")] if s else [])
        )
        final = final.merge(
            gt_df[["oil_well", "ppd_well", "expected", "acceptable"]],
            on=["oil_well", "ppd_well"],
            how="left",
        )

    # Сохраняем результат
    out_path = CLEAN_DIR / "final_result.csv"
    final.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[FINAL MIX] Сохранено: {out_path} (пар = {len(final)})")