"""
Этот модуль выполняет расчёт Confidence Index (CI) для каждого окна «до/после»
ППД-мероприятия и сохраняет результат в clean_data/ci_results_pro.csv.

Что читается:
-------------
  1) clean_data/oil_windows.csv
     - Окна наблюдений «до/после» ППД
     - Столбцы: well, ppd_well, q_start, q_end, p_start, p_end, oil_start, oil_end, duration_days_oil

  2) clean_data/ppd_events.csv
     - Метаданные ППД-мероприятий
     - Столбцы: well (P-скважина), start_date, baseline_before, baseline_during

  3) clean_data/oil_clean.csv
     - Исторические суточные данные по дебиту и давлению
     - Столбцы: well, date, q_oil, p_oil

Как устроено:
-------------
  • Все три CSV читаются и кешируются при первом вызове (каждый файл – однажды).
  • oil_clean.csv дополнительно группируется по скважинам в словарь для быстрого доступа.
  • Для каждой строки из oil_windows объединяется запись ППД, затем по “пред-окну” берутся
    данные из oil_clean, и, при запросе, вызываются базовые методы из metrics_baseline.py.

Основная функция:
-----------------
  compute_ci_with_pre(
      clean_data_dir: str = "clean_data",
      oil_windows_fname: str = "oil_windows.csv",
      ppd_events_fname: str = "ppd_events.csv",
      oil_clean_fname: str = "oil_clean.csv",
      methods: tuple[str, ...] = ("none",),
  ) -> pd.DataFrame

  Параметр methods:
    - "none"        – без базлайна (CI только по разнице Q/P);
    - "mean"        – коррекция по среднему пред-окна;
    - "regression"  – коррекция линейной регрессией;
    - "median_ewma" – медиана + экспоненциальное сглаживание.

  Возвращает DataFrame со строками:
    №, well, ppd_well, deltaQpr, q_start, p_start, q_end, p_end,
    duration_days_oil, oil_start, oil_end,
    deltaQbaseCI_mean, deltaPbaseCI_mean,
    deltaQbaseCI_regression, deltaPbaseCI_regression,
    deltaQbaseCI_median_ewma, deltaPbaseCI_median_ewma,
    CI_none, CI_mean, CI_regression, CI_median_ewma

  Параллельных вычислений в этом модуле нет — он рассчитан на одиночный быстрый прогон.
  Для полного перебора гиперпараметров используется отдельный скрипт hyperparam_search.py.

Пример использования:
---------------------
  import metrics

  # только «грубый» CI
  df = metrics.compute_ci_with_pre()

  # с простым baseline-средним
  df = metrics.compute_ci_with_pre(methods=("none","mean"))

  # со всеми методами
  df = metrics.compute_ci_with_pre(
      methods=("none","mean","regression","median_ewma")
  )
"""

from __future__ import annotations
import os, math, warnings, importlib
import pandas as pd, numpy as np
import config

warnings.filterwarnings("ignore", category=FutureWarning)

# ──────────────────────── кеш CSV ────────────────────────────────
_cached = {}
def _load_csvs(dir_: str, ow: str, ppd: str, oc: str):
    """Чтение и кеширование исходных файлов + расстояния."""
    key = (dir_, ow, ppd, oc)
    if key in _cached:
        return _cached[key]

    # 1) окна + ппд (как было)
    df_ow = pd.read_csv(os.path.join(dir_, ow))
    df_ow["oil_start"] = pd.to_datetime(df_ow["oil_start"])
    df_ow["oil_end"]   = pd.to_datetime(df_ow["oil_end"])
    df_ow["ppd_start"] = pd.to_datetime(df_ow["ppd_start"])

    df_ppd = pd.read_csv(os.path.join(dir_, ppd))
    df_ppd["start_date"] = pd.to_datetime(df_ppd["start_date"], dayfirst=True)
    df_ppd["delta_PPD"]  = (
        df_ppd["baseline_during"] - df_ppd["baseline_before"]
    ).apply(lambda d: 1 if d > 0 else (-1 if d < 0 else 0))
    df_ppd["deltaQpr"]   = df_ppd["baseline_during"] - df_ppd["baseline_before"]
    df_ppd_min = (
        df_ppd[["well","start_date","delta_PPD","deltaQpr"]]
          .rename(columns={"well":"ppd_well","start_date":"ppd_start"})
    )

    # 2) oil_clean
    df_clean = pd.read_csv(os.path.join(dir_, oc))
    df_clean["date"] = pd.to_datetime(df_clean["date"], dayfirst=True, errors="coerce")
    by_well = {
        w: sub.sort_values("date")[["date","q_oil","p_oil"]]
        for w, sub in df_clean.groupby("well")
    }

    # 3) пары расстояний
    dist_path = os.path.join(dir_, "pairs_oil_ppd.csv")
    df_dist = pd.read_csv(dist_path)
    dist = {
        (str(r.oil_well), str(r.ppd_well)): float(r.distance)
        for _, r in df_dist.iterrows()
    }

    _cached[key] = (df_ow, df_ppd_min, by_well, dist)
    return _cached[key]

# ──────────────────────── формула CI ─────────────────────────────
def _ci_value(dp, eps_q, eps_p):
    if dp == 0:
        return 0.0
    x = dp * (eps_q / config.divider_q)
    y = dp * (eps_p / config.divider_p)
    if x >= 0 and y >= 0:
        return config.w_q * x + config.w_p * y
    if x >= 0 and y < 0:
        return x
    if x < 0 and y >= 0:
        return y
    return 0.0

# ──────────────────── публичная функция ─────────────────────────
def compute_ci_with_pre(
        clean_data_dir="clean_data",
        oil_windows_fname="oil_windows.csv",
        ppd_events_fname="ppd_events.csv",
        oil_clean_fname="oil_clean.csv",
        methods=("none",),
):
    need_bl = any(m != "none" for m in methods)
    if need_bl:
        get_baseline = importlib.import_module("metrics_baseline").get_baseline

    df_ow, df_ppd_min, clean_by_well, dist = _load_csvs(
        clean_data_dir, oil_windows_fname, ppd_events_fname, oil_clean_fname
    )
    df = pd.merge(df_ow, df_ppd_min, on=["ppd_well","ppd_start"], how="left")

    def process(row):
        well, rec = row["well"], row
        pre_df = clean_by_well.get(well)
        dq_b = {"mean":0, "regression":0, "median_ewma":0}
        dp_b = dq_b.copy()

        # baseline (как раньше)
        if need_bl and pre_df is not None:
            pre = pre_df[
                (pre_df["date"] >= rec["oil_start"] - pd.Timedelta(days=config.T_pre)) &
                (pre_df["date"] <  rec["oil_start"])
            ].query("q_oil>0 & p_oil>0")
            if len(pre) >= 5:
                for m in dq_b:
                    if m in methods:
                        dq_b[m], dp_b[m] = get_baseline(
                            m, pre, q_start=rec["q_start"], p_start=rec["p_start"],
                            oil_start=rec["oil_start"], oil_end=rec["oil_end"]
                        )

        dq_act = rec["q_end"] - rec["q_start"]
        dp_act = rec["p_end"] - rec["p_start"]
        ci = {"CI_none": _ci_value(rec["delta_PPD"], dq_act, dp_act)}
        if "mean" in methods:
            ci["CI_mean"] = _ci_value(rec["delta_PPD"],
                                      dq_act-dq_b["mean"], dp_act-dp_b["mean"])
        if "regression" in methods:
            ci["CI_regression"] = _ci_value(rec["delta_PPD"],
                                            dq_act-dq_b["regression"], dp_act-dp_b["regression"])
        if "median_ewma" in methods:
            ci["CI_median_ewma"] = _ci_value(rec["delta_PPD"],
                                             dq_act-dq_b["median_ewma"], dp_act-dp_b["median_ewma"])

        # ─── модификатор расстояния ───
        d = dist.get((str(well), str(rec["ppd_well"])))
        if d is not None:
            if config.distance_mode == "linear":
                atten = max(0.0, 1 - d / config.lambda_dist)
            else:
                atten = math.exp(-d / config.lambda_dist)
            for k in ci:
                ci[k] *= atten

        # round & return
        out = {
            "deltaQpr": rec["deltaQpr"],
            "q_start": rec["q_start"], "p_start": rec["p_start"],
            "q_end": rec["q_end"],     "p_end": rec["p_end"],
            "duration_days_oil": rec.get("duration_days_oil", np.nan),
            "oil_start": rec["oil_start"], "oil_end": rec["oil_end"],
            "deltaQbaseCI_mean": round(dq_b["mean"],1),
            "deltaPbaseCI_mean": round(dp_b["mean"],1),
            "deltaQbaseCI_regression": round(dq_b["regression"],1),
            "deltaPbaseCI_regression": round(dp_b["regression"],1),
            "deltaQbaseCI_median_ewma": round(dq_b["median_ewma"],1),
            "deltaPbaseCI_median_ewma": round(dp_b["median_ewma"],1),
        }
        out.update({k: round(v,1) for k,v in ci.items()})
        return pd.Series(out)

    res = df.assign(**df.apply(process, axis=1))
    res = res.sort_values(["well","ppd_well"]).reset_index(drop=True)
    res.insert(0, "№", range(1, len(res)+1))
    res.to_csv(os.path.join(clean_data_dir, "ci_results.csv"),
               index=False, float_format="%.1f")
    return res

if __name__ == "__main__":
    compute_ci_with_pre(methods=("none",))
