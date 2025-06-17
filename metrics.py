#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
metrics.py — расчёт Confidence Index (CI) для SmartPVD-MVP.

Модуль читает файлы:

  clean_data/oil_windows.csv    — окна «до/после» ППД (well, ppd_well, q_start, q_end, p_start, p_end, oil_start, oil_end, duration_days_oil)
  clean_data/ppd_events.csv     — события ППД (well, start_date, baseline_before, baseline_during)
  clean_data/oil_clean.csv      — суточные ряды добычи (well, date, q_oil, p_oil)
  clean_data/pairs_oil_ppd.csv  — пары скважин и расстояния (oil_well, ppd_well, distance)
  clean_data/ground_truth.csv   — эталон (oil_well, ppd_well, expected, acceptable)

Добавлен параметр filter_by_gt: если True, в detail_df и agg_df остаются только пары, присутствующие в GT.
Логика расчёта CI не изменена.
"""
from __future__ import annotations
import math, warnings, importlib
from pathlib import Path
from typing import Tuple

import pandas as pd
import numpy as np

import config

warnings.filterwarnings("ignore", category=FutureWarning)
_CSV_CACHE: dict = {}

def _load_csvs(
    clean_data_dir: str,
    oil_windows_fname: str,
    ppd_events_fname: str,
    oil_clean_fname: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame], dict[tuple[str,str], float]]:
    key = (clean_data_dir, oil_windows_fname, ppd_events_fname, oil_clean_fname)
    if key in _CSV_CACHE:
        return _CSV_CACHE[key]
    base = Path(clean_data_dir)
    df_ow = pd.read_csv(base / oil_windows_fname)
    df_ow["oil_start"] = pd.to_datetime(df_ow["oil_start"])
    df_ow["oil_end"]   = pd.to_datetime(df_ow["oil_end"])
    df_ow["ppd_start"] = pd.to_datetime(df_ow["ppd_start"])

    df_ppd = pd.read_csv(base / ppd_events_fname)
    df_ppd["start_date"] = pd.to_datetime(df_ppd["start_date"], dayfirst=True)
    df_ppd["delta_PPD"]  = df_ppd["baseline_during"] - df_ppd["baseline_before"]
    df_ppd["deltaQpr"]   = df_ppd["delta_PPD"]
    df_ppd_min = (
        df_ppd[["well","start_date","delta_PPD","deltaQpr"]]
               .rename(columns={"well":"ppd_well","start_date":"ppd_start"})
    )

    df_clean = pd.read_csv(base / oil_clean_fname)
    df_clean["date"] = pd.to_datetime(df_clean["date"], dayfirst=True, errors="coerce")
    clean_by_well = {w: sub.sort_values("date")[['date','q_oil','p_oil']] for w, sub in df_clean.groupby('well')}

    df_pairs = pd.read_csv(base / "pairs_oil_ppd.csv")
    dist = {(str(r.oil_well), str(r.ppd_well)): float(r.distance) for _, r in df_pairs.iterrows()}

    _CSV_CACHE[key] = (df_ow, df_ppd_min, clean_by_well, dist)
    return _CSV_CACHE[key]

def _ci_value(dp: float, dq: float, dp_oil: float) -> float:
    if dp == 0:
        return 0.0
    x = dp * (dq / config.divider_q)
    y = dp * (dp_oil / config.divider_p)
    if x >= 0 and y >= 0:
        return config.w_q * x + config.w_p * y
    if x >= 0:
        return x
    if y >= 0:
        return y
    return 0.0

def compute_ci_with_pre(
    clean_data_dir: str = "clean_data",
    oil_windows_fname: str = "oil_windows.csv",
    ppd_events_fname: str = "ppd_events.csv",
    oil_clean_fname: str = "oil_clean.csv",
    methods: tuple[str, ...] = ("none",),
    save_csv: bool = False,
    filter_by_gt: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Возвращает детальный и агрегированный отчёты по CI.

    Параметры:
      filter_by_gt: если True — оставляет только пары (oil_well,ppd_well) из ground_truth.
    """
    need_baseline = any(m != "none" for m in methods)
    if need_baseline:
        get_baseline = importlib.import_module("metrics_baseline").get_baseline

    df_ow, df_ppd_min, clean_by_well, dist = _load_csvs(
        clean_data_dir, oil_windows_fname, ppd_events_fname, oil_clean_fname
    )
    df = pd.merge(df_ow, df_ppd_min, on=["ppd_well","ppd_start"], how="left")

    def _process(rec: pd.Series) -> pd.Series:
        well = rec["well"]
        pre_df = clean_by_well.get(well)
        dq_b = {m:0.0 for m in ("mean","regression","median_ewma")}
        dp_b = dq_b.copy()
        if need_baseline and pre_df is not None:
            pre = pre_df[(pre_df["date"] >= rec["oil_start"] - pd.Timedelta(days=config.T_pre)) &
                         (pre_df["date"] < rec["oil_start"])].query("q_oil>0 & p_oil>0")
            if len(pre) >= 5:
                for m in dq_b:
                    if m in methods:
                        dq_b[m], dp_b[m] = get_baseline(
                            m, pre,
                            q_start=rec["q_start"], p_start=rec["p_start"],
                            oil_start=rec["oil_start"], oil_end=rec["oil_end"]
                        )
        dq_act = rec["q_end"] - rec["q_start"]
        dp_act = rec["p_end"] - rec["p_start"]
        ci = {"CI_none": _ci_value(rec["delta_PPD"], dq_act, dp_act)}
        if "mean" in methods:
            ci["CI_mean"] = _ci_value(rec["delta_PPD"], dq_act - dq_b["mean"], dp_act - dp_b["mean"])
        if "regression" in methods:
            ci["CI_regression"] = _ci_value(rec["delta_PPD"], dq_act - dq_b["regression"], dp_act - dp_b["regression"])
        if "median_ewma" in methods:
            ci["CI_median_ewma"] = _ci_value(rec["delta_PPD"], dq_act - dq_b["median_ewma"], dp_act - dp_b["median_ewma"])
        d = dist.get((str(well), str(rec["ppd_well"])) )
        if d is not None:
            atten = (max(0.0, 1 - d/config.lambda_dist) if config.distance_mode == "linear" else math.exp(-d/config.lambda_dist))
            for k in ci:
                ci[k] *= atten
        out = {k: round(v,1) for k,v in ci.items()}
        out.update({
            "deltaQpr": rec["deltaQpr"], "q_start": rec["q_start"], "p_start": rec["p_start"],
            "q_end": rec["q_end"], "p_end": rec["p_end"],
            "duration_days_oil": rec.get("duration_days_oil", np.nan),
            "oil_start": rec["oil_start"], "oil_end": rec["oil_end"],
            "well": well, "ppd_well": rec["ppd_well"]
        })
        return pd.Series(out)

    detail_df = df.assign(**df.apply(_process, axis=1))
    detail_df = detail_df.sort_values(["well","ppd_well"]).reset_index(drop=True)
    detail_df.insert(0, "№", range(1, len(detail_df)+1))

    agg_df = (
        detail_df.groupby(["well","ppd_well"], as_index=False)["CI_none"]
                 .sum()
                 .rename(columns={"well":"oil_well","ppd_well":"ppd_well","CI_none":"CI_value"})
    )
    agg_df["CI_value"] = agg_df["CI_value"].round(1)
    agg_df["oil_well"] = agg_df["oil_well"].astype(str)
    agg_df["ppd_well"] = agg_df["ppd_well"].astype(str)

    t1, t2, t3 = config.CI_THRESHOLDS
    agg_df["ci_cat"] = agg_df["CI_value"].apply(
        lambda v: ("none","weak","medium","strong")[0 if v<t1 else 1 if v<t2 else 2 if v<t3 else 3]
    )

    # объединение с эталоном и фильтрация
    gt_path = Path(clean_data_dir)/"ground_truth.csv"
    if not gt_path.exists():
        gt_path = Path("start_data")/"ground_truth.csv"
    if gt_path.exists():
        gt = pd.read_csv(gt_path, dtype=str)
        if "well" in gt.columns:
            gt = gt.rename(columns={"well":"oil_well"}, errors="ignore")
        for c in ("oil_well","ppd_well"): gt[c] = gt[c].astype(str)
        gt_pairs = set(zip(gt["oil_well"], gt["ppd_well"]))
        # добавляем GT-колонки
        agg_df = agg_df.merge(gt[["oil_well","ppd_well","expected","acceptable"]],
                              on=["oil_well","ppd_well"], how="left")
        # подготовка detail_df: строковые ключи
        detail_df["oil_well"] = detail_df["well"].astype(str)
        detail_df["ppd_well"] = detail_df["ppd_well"].astype(str)
        detail_df = detail_df.merge(
            agg_df[["oil_well","ppd_well"]],
            on=["oil_well","ppd_well"], how="inner"
        )
        if filter_by_gt:
            agg_df = agg_df[agg_df.apply(lambda r: (r["oil_well"],r["ppd_well"]) in gt_pairs, axis=1)]
            detail_df = detail_df[detail_df.apply(lambda r: (r["oil_well"],r["ppd_well"]) in gt_pairs, axis=1)]
    else:
        agg_df = agg_df.iloc[0:0]
        detail_df = detail_df.iloc[0:0]

    if save_csv:
        detail_path = Path(clean_data_dir)/"ci_results.csv"
        agg_path    = Path(clean_data_dir)/"ci_results_agg.csv"
        detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")
        agg_df.to_csv(agg_path,    index=False, encoding="utf-8-sig")

        total = len(agg_df)
        # точные совпадения
        exact = int(((agg_df["expected"]!="") & (agg_df["ci_cat"]==agg_df["expected"])).sum())
        # «nearby» – допустимые (acceptable), но не точные
        acc_lists = agg_df["acceptable"].fillna("").apply(lambda s: [x.strip() for x in s.split(";")] if s else [])
        nearby = int(sum((exp!="") and (cat!=exp) and (cat in acc) for exp,cat,acc in zip(
            agg_df["expected"], agg_df["ci_cat"], acc_lists)))
        # промахи (категории ни в expected, ни в acceptable)
        miss = total - exact - nearby
        # точность учитывает точные и «nearby»
        accuracy = (exact + nearby)/total if total else 0.0

        totals_line = (f"TOTALS,exact={exact},nearby={nearby},miss={miss},"
                       f"all={total},accuracy={accuracy:.2f}")

        for p in (detail_path, agg_path):
            with open(p, "a", encoding="utf-8-sig") as f:
                f.write(f"{totals_line}")
        print(f"[CI] ✔ сохранено {detail_path} и {agg_path}")

    return detail_df, agg_df

if __name__ == "__main__":
    print("[CI] ▶ расчёт…")
    compute_ci_with_pre(
        clean_data_dir="clean_data",
        oil_windows_fname="oil_windows.csv",
        ppd_events_fname="ppd_events.csv",
        oil_clean_fname="oil_clean.csv",
        methods=("none",),
        save_csv=True,
        filter_by_gt=True,
    )
