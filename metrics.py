
"""
metrics.py — расчёт Confidence Index (CI) для SmartPVD-MVP в формате
«чистое ядро / универсальный интерфейс / CLI».

Изменения по шаблону
--------------------
1. **Чистое ядро** — вся логика помещена в функцию
   :pyfunc:`_compute_ci_df`, которая принимает на вход подготовленные
   ``pandas.DataFrame``-ы и **не** выполняет файлового I/O.
2. **Универсальный интерфейс** — функция :pyfunc:`compute_ci`
   загружает/сохраняет CSV при необходимости и делегирует расчёт ядру.
3. **CLI** — блок ``if __name__ == "__main__":`` сохраняет прежнее
   использование скрипта как самостоятельной утилиты.
4. **Документация** — добавлены развёрнутые docstring'и и комментарии
   (технический русский язык).
5. **Стиль** — сохранён PEP 8 и названия переменных; алгоритм не упрощён.

"""
from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple, Union

import importlib
import numpy as np
import pandas as pd

import config  # внешние константы и параметры

warnings.filterwarnings("ignore", category=FutureWarning)

# ────────────────────────────────────────────────────────────────────────
#                            ВСПОМОГАТЕЛЬНЫЕ
# ────────────────────────────────────────────────────────────────────────
def _ci_value(dp: float, dq: float, dp_oil: float) -> float:
    """
    Формула локального *Confidence Index* (CI).

    Параметры
    ----------
    dp : float
        Разница давления закачки ΔP<sub>PPD</sub>.
    dq : float
        Прирост дебита жидкости ΔQ<sub>liq</sub> (фактический или
        скорректированный на бейслайн).
    dp_oil : float
        Прирост давления на приеме ΔP<sub>oil</sub>.

    Returns
    -------
    float
        Значение CI в условных единицах (может быть 0.0).
    """
    if dp == 0:  # физический ноль → нет влияния
        return 0.0

    # нормированные компоненты
    x = dp * (dq / config.divider_q)
    y = dp * (dp_oil / config.divider_p)

    # согласованные / конфликтующие знаки
    if x >= 0 and y >= 0:
        return config.w_q * x + config.w_p * y
    if x >= 0:
        return x
    if y >= 0:
        return y
    return 0.0  # оба отрицательные → влияние не подтверждено


# ────────────────────────────────────────────────────────────────────────
#                               ЧИСТОЕ ЯДРО
# ────────────────────────────────────────────────────────────────────────
def _compute_ci_df(  # pylint: disable=too-many-arguments,too-many-locals
    df_oil_windows: pd.DataFrame,
    df_ppd_events: pd.DataFrame,
    df_oil_clean: pd.DataFrame,
    df_pairs: pd.DataFrame,
    df_ground_truth: pd.DataFrame | None = None,
    *,
    methods: tuple[str, ...] = ("none",),
    filter_by_gt: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    **Ядро**: расчёт детального и сводного отчётов CI без файлового I/O.

    Параметры
    ----------
    df_oil_windows : pd.DataFrame
        Окна отклика добычи: ``['well','ppd_well','q_start','q_end',
        'p_start','p_end','oil_start','oil_end','duration_days_oil',
        'ppd_start']``.
    df_ppd_events : pd.DataFrame
        События ППД с колонками
        ``['well','start_date','baseline_before','baseline_during']``.
    df_oil_clean : pd.DataFrame
        Суточные ряды добычи: ``['well','date','q_oil','p_oil']``.
    df_pairs : pd.DataFrame
        Пары отверстие-закачка и расстояния:
        ``['oil_well','ppd_well','distance']``.
    df_ground_truth : pd.DataFrame | None, optional
        Эталон для сравнения (``['oil_well','ppd_well','expected','acceptable']``).
    methods : tuple[str, ...], default ``("none",)``
        Модификаторы бейслайна: ``"mean"``, ``"regression"``,
        ``"median_ewma"`` либо ``"none"`` (только фактический прирост).
    filter_by_gt : bool, default ``False``
        Оставлять ли только пары, присутствующие в ground truth.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        ``detail_df`` — строка на каждое окно воздействия;
        ``agg_df`` — агрегировано по паре (oil_well, ppd_well) с категорией
        ``ci_cat``.
    """
    # 1. Пред-обработка событий ППД --------------------------------------- #
    df_ppd = df_ppd_events.copy()
    df_ppd["start_date"] = pd.to_datetime(df_ppd["start_date"], dayfirst=True)
    df_ppd["delta_PPD"] = df_ppd["baseline_during"] - df_ppd["baseline_before"]
    df_ppd["deltaQpr"] = df_ppd["delta_PPD"]
    df_ppd_min = (
        df_ppd[["well", "start_date", "delta_PPD", "deltaQpr"]]
        .rename(columns={"well": "ppd_well", "start_date": "ppd_start"})
    )

    # 2. Дата-временные колонки в df_oil_windows -------------------------- #
    df_ow = df_oil_windows.copy()
    df_ow["oil_start"] = pd.to_datetime(df_ow["oil_start"])
    df_ow["oil_end"] = pd.to_datetime(df_ow["oil_end"])
    df_ow["ppd_start"] = pd.to_datetime(df_ow["ppd_start"])

    # 3. Суточные ряды добычи → словарь well → DataFrame ------------------ #
    df_clean = df_oil_clean.copy()
    df_clean["date"] = pd.to_datetime(df_clean["date"], dayfirst=True, errors="coerce")
    clean_by_well: Dict[str, pd.DataFrame] = {
        w: sub.sort_values("date")[["date", "q_oil", "p_oil"]]
        for w, sub in df_clean.groupby("well")
    }

    # 4. Расстояния -------------------------------------------------------- #
    dist: Dict[tuple[str, str], float] = {
        (str(r.oil_well), str(r.ppd_well)): float(r.distance)
        for _, r in df_pairs.iterrows()
    }

    # 5. Объединяем окна и события ППД ------------------------------------ #
    df = pd.merge(df_ow, df_ppd_min, on=["ppd_well", "ppd_start"], how="left")

    # 6. Подключаем baseline-модель при необходимости --------------------- #
    need_baseline = any(m != "none" for m in methods)
    if need_baseline:
        get_baseline = importlib.import_module("metrics_baseline").get_baseline  # noqa: E501

    # 7. Расчёт CI для каждой строки окна --------------------------------- #
    def _process(rec: pd.Series) -> pd.Series:  # pylint: disable=too-many-locals
        well = rec["well"]
        pre_df = clean_by_well.get(well)

        # --- 7.1 baseline ------------------------------------------------ #
        dq_b = {m: 0.0 for m in ("mean", "regression", "median_ewma")}
        dp_b = dq_b.copy()

        if need_baseline and pre_df is not None:
            pre = pre_df[
                (pre_df["date"] >= rec["oil_start"] - pd.Timedelta(days=config.T_pre))
                & (pre_df["date"] < rec["oil_start"])
            ].query("q_oil > 0 & p_oil > 0")

            if len(pre) >= 5:
                for m in dq_b:
                    if m in methods:
                        dq_b[m], dp_b[m] = get_baseline(
                            m,
                            pre,
                            q_start=rec["q_start"],
                            p_start=rec["p_start"],
                            oil_start=rec["oil_start"],
                            oil_end=rec["oil_end"],
                        )

        # --- 7.2 фактические приращения ---------------------------------- #
        dq_act = rec["q_end"] - rec["q_start"]
        dp_act = rec["p_end"] - rec["p_start"]

        # --- 7.3 CI без / с бейслайном ----------------------------------- #
        ci: Dict[str, float] = {"CI_none": _ci_value(rec["delta_PPD"], dq_act, dp_act)}

        if "mean" in methods:
            ci["CI_mean"] = _ci_value(
                rec["delta_PPD"],
                dq_act - dq_b["mean"],
                dp_act - dp_b["mean"],
            )
        if "regression" in methods:
            ci["CI_regression"] = _ci_value(
                rec["delta_PPD"],
                dq_act - dq_b["regression"],
                dp_act - dp_b["regression"],
            )
        if "median_ewma" in methods:
            ci["CI_median_ewma"] = _ci_value(
                rec["delta_PPD"],
                dq_act - dq_b["median_ewma"],
                dp_act - dp_b["median_ewma"],
            )

        # --- 7.4 Затухание по расстоянию --------------------------------- #
        d = dist.get((str(well), str(rec["ppd_well"])))
        if d is not None:
            atten = (
                max(0.0, 1 - d / config.lambda_dist)
                if config.distance_mode == "linear"
                else math.exp(-d / config.lambda_dist)
            )
            for k in ci:
                ci[k] *= atten

        # --- 7.5 итоговая строка ----------------------------------------- #
        out = {k: round(v, 1) for k, v in ci.items()}
        out.update(
            {
                "deltaQpr": rec["deltaQpr"],
                "q_start": rec["q_start"],
                "p_start": rec["p_start"],
                "q_end": rec["q_end"],
                "p_end": rec["p_end"],
                "duration_days_oil": rec.get("duration_days_oil", np.nan),
                "oil_start": rec["oil_start"],
                "oil_end": rec["oil_end"],
                "well": well,
                "ppd_well": rec["ppd_well"],
            }
        )
        return pd.Series(out)

    detail_df = df.assign(**df.apply(_process, axis=1))
    detail_df = detail_df.sort_values(["well", "ppd_well"]).reset_index(drop=True)
    detail_df.insert(0, "№", range(1, len(detail_df) + 1))

    # 8. Агрегация по паре ----------------------------------------------- #
    agg_df = (
        detail_df.groupby(["well", "ppd_well"], as_index=False)["CI_none"]
        .sum()
        .rename(
            columns={
                "well": "oil_well",
                "ppd_well": "ppd_well",
                "CI_none": "CI_value",
            }
        )
    )
    agg_df["CI_value"] = agg_df["CI_value"].round(1)
    agg_df[["oil_well", "ppd_well"]] = agg_df[["oil_well", "ppd_well"]].astype(str)

    # 9. Категория CI ----------------------------------------------------- #
    t1, t2 = config.CI_THRESHOLDS
    agg_df["ci_cat"] = agg_df["CI_value"].apply(
        lambda v: ("none", "weak", "impact")[
            0 if v < t1 else 1 if v < t2 else 2
        ]
    )

    # 10. Ground truth ---------------------------------------------------- #
    if df_ground_truth is not None and not df_ground_truth.empty:
        gt = df_ground_truth.copy()
        if "well" in gt.columns:
            gt = gt.rename(columns={"well": "oil_well"}, errors="ignore")
        gt[["oil_well", "ppd_well"]] = gt[["oil_well", "ppd_well"]].astype(str)
        gt_pairs = set(zip(gt["oil_well"], gt["ppd_well"]))

        # добавляем GT-колонки в agg_df
        agg_df = agg_df.merge(
            gt[["oil_well", "ppd_well", "expected", "acceptable"]],
            on=["oil_well", "ppd_well"],
            how="left",
        )

        # синхронизируем detail_df
        detail_df["oil_well"] = detail_df["well"].astype(str)
        detail_df["ppd_well"] = detail_df["ppd_well"].astype(str)
        detail_df = detail_df.merge(
            agg_df[["oil_well", "ppd_well"]],
            on=["oil_well", "ppd_well"],
            how="inner",
        )

        if filter_by_gt:
            agg_df = agg_df[
                agg_df.apply(
                    lambda r: (r["oil_well"], r["ppd_well"]) in gt_pairs, axis=1
                )
            ]
            detail_df = detail_df[
                detail_df.apply(
                    lambda r: (r["oil_well"], r["ppd_well"]) in gt_pairs, axis=1
                )
            ]
    else:
        # ground_truth отсутствует → пустые датафреймы-заглушки
        agg_df = agg_df.iloc[0:0]
        detail_df = detail_df.iloc[0:0]

    return detail_df, agg_df


# ────────────────────────────────────────────────────────────────────────
#                        УНИВЕРСАЛЬНЫЙ ИНТЕРФЕЙС
# ────────────────────────────────────────────────────────────────────────
def compute_ci(  # pylint: disable=too-many-arguments
    df: Union[
        Mapping[str, pd.DataFrame],
        Sequence[pd.DataFrame],
        None,
    ] = None,
    csv_path: str | Path | None = None,
    *,
    methods: tuple[str, ...] = ("none",),
    save_csv: bool = True,
    filter_by_gt: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Универсальный расчёт Confidence Index.

    Параметры
    ----------
    df : Mapping[str, pd.DataFrame] | Sequence[pd.DataFrame] | None, optional
        * Если передан *словарь*, ожидaются ключи:
          ``{'oil_windows', 'ppd_events', 'oil_clean',
             'pairs', 'ground_truth'}`` (последний опционален).
        * Если передана *последовательность* (tuple/list), порядок —
          ``(oil_windows, ppd_events, oil_clean, pairs, ground_truth?)``.
        * Если ``None`` — CSV считываются из директории *csv_path*.
    csv_path : str | Path | None, optional
        Папка с файлами
        ``oil_windows.csv``, ``ppd_events.csv``, ``oil_clean.csv``,
        ``pairs_oil_ppd.csv``, ``ground_truth.csv`` (последний — опц.).
        По умолчанию ``clean_data/``.
    methods : tuple[str, ...], default ``("none",)``
        Перечень baseline-методов (см. :pyfunc:`_compute_ci_df`).
    save_csv : bool, default ``True``
        Сохранять ли результат в ``clean_data/ci_results*.csv``.
    filter_by_gt : bool, default ``False``
        Сохранять ли только пары, отмеченные в ground_truth.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        ``detail_df`` и ``agg_df`` с результатами CI.
    """
    # ── 1. Подготовка входных DataFrame-ов ─────────────────────────────── #
    if df is not None:
        # --- 1.1 dict[str,DataFrame] ------------------------------------- #
        if isinstance(df, Mapping):
            try:
                df_oil_windows = df["oil_windows"]
                df_ppd_events = df["ppd_events"]
                df_oil_clean = df["oil_clean"]
                df_pairs = df["pairs"]
                df_ground_truth = df.get("ground_truth")
            except KeyError as exc:
                raise ValueError(
                    "Для расчёта CI необходимы ключи "
                    "{'oil_windows','ppd_events','oil_clean','pairs'}."
                ) from exc

        # --- 1.2 sequence[DataFrame] ------------------------------------- #
        elif isinstance(df, Sequence) and not isinstance(df, pd.DataFrame):
            if len(df) not in (4, 5):
                raise ValueError(
                    "При передаче последовательности требуется 4 или 5 "
                    "DataFrame'ов: (oil_windows, ppd_events, oil_clean, "
                    "pairs[, ground_truth])."
                )
            (
                df_oil_windows,
                df_ppd_events,
                df_oil_clean,
                df_pairs,
                *optional,
            ) = df
            df_ground_truth = optional[0] if optional else None
        else:
            raise TypeError(
                "Аргумент df должен быть Mapping[str, DataFrame] "
                "или Sequence[DataFrame]."
            )
    else:
        # --- 1.3 читаем CSV --------------------------------------------- #
        base = Path(csv_path) if csv_path else Path("clean_data")
        df_oil_windows = pd.read_csv(base / "oil_windows.csv")
        df_ppd_events = pd.read_csv(base / "ppd_events.csv")
        df_oil_clean = pd.read_csv(base / "oil_clean.csv")
        df_pairs = pd.read_csv(base / "pairs_oil_ppd.csv")
        gt_path = base / "ground_truth.csv"
        if not gt_path.exists():
            gt_path = Path("start_data") / "ground_truth.csv"
        df_ground_truth = pd.read_csv(gt_path) if gt_path.exists() else None

    # ── 2. Вызываем чистое ядро ────────────────────────────────────────── #
    detail_df, agg_df = _compute_ci_df(
        df_oil_windows=df_oil_windows,
        df_ppd_events=df_ppd_events,
        df_oil_clean=df_oil_clean,
        df_pairs=df_pairs,
        df_ground_truth=df_ground_truth,
        methods=methods,
        filter_by_gt=filter_by_gt,
    )

    # ── 3. Сохраняем CSV при необходимости ────────────────────────────── #
    if save_csv:
        base = Path(csv_path) if csv_path else Path("clean_data")
        base.mkdir(parents=True, exist_ok=True)
        detail_path = base / "ci_results.csv"
        agg_path = base / "ci_results_agg.csv"
        detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")
        agg_df.to_csv(agg_path, index=False, encoding="utf-8-sig")

        # — 3.1 totals-строка для совместимости со старым форматом -------- #
        total = len(agg_df)
        exact = int(
            ((agg_df["expected"] != "") & (agg_df["ci_cat"] == agg_df["expected"])).sum()
            if "expected" in agg_df.columns
            else 0
        )
        acc_lists = (
            agg_df["acceptable"]
            .fillna("")
            .apply(lambda s: [x.strip() for x in s.split(";")] if s else [])
            if "acceptable" in agg_df.columns
            else pd.Series([[]] * len(agg_df))
        )
        nearby = int(
            sum(
                (exp != "") and (cat != exp) and (cat in acc)
                for exp, cat, acc in zip(
                    agg_df.get("expected", ""), agg_df["ci_cat"], acc_lists
                )
            )
        )
        miss = total - exact - nearby
        accuracy = (exact + nearby) / total if total else 0.0
        totals_line = (
            f"TOTALS,exact={exact},nearby={nearby},miss={miss},"
            f"all={total},accuracy={accuracy:.2f}"
        )
        for p in (detail_path, agg_path):
            with open(p, "a", encoding="utf-8-sig") as f:
                f.write(f"{totals_line}")
        print(f"[CI] ✔ сохранено {detail_path} и {agg_path}")

    return detail_df, agg_df


# ────────────────────────────────────────────────────────────────────────
#                                   CLI
# ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[CI] ▶ расчёт CI…")
    compute_ci(
        csv_path="clean_data",
        methods=("none",),
        save_csv=True,
        filter_by_gt=False,
    )
