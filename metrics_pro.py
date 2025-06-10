# metrics_pro.py

import os
import warnings
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import config   # вместо `from config import ...`

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but .* was fitted with feature names"
)

# ─── Модульный кеш: читаем CSV и разбиваем once per process ─────────────────────
_df_ow = None
_df_ppd_min = None
_df_clean = None
_clean_by_well = None

def _load_and_cache(clean_data_dir, oil_windows_fname, ppd_events_fname, oil_clean_fname):
    global _df_ow, _df_ppd_min, _df_clean, _clean_by_well
    if _df_ow is None:
        # 1) oil_windows
        path_ow = os.path.join(clean_data_dir, oil_windows_fname)
        df_ow = pd.read_csv(path_ow)
        df_ow["oil_start"] = pd.to_datetime(df_ow["oil_start"], errors="coerce")
        df_ow["oil_end"]   = pd.to_datetime(df_ow["oil_end"], errors="coerce")
        df_ow["ppd_start"] = pd.to_datetime(df_ow["ppd_start"], errors="coerce")
        _df_ow = df_ow

        # 2) ppd_events -> minimal
        path_ppd = os.path.join(clean_data_dir, ppd_events_fname)
        df_ppd = pd.read_csv(path_ppd)
        df_ppd["start_date"] = pd.to_datetime(df_ppd["start_date"], dayfirst=True, errors="coerce")
        df_ppd["delta_PPD"]  = (df_ppd["baseline_during"] - df_ppd["baseline_before"])\
                                .apply(lambda d: 1 if d>0 else (-1 if d<0 else 0))
        df_ppd["deltaQpr"]   = df_ppd["baseline_during"] - df_ppd["baseline_before"]
        _df_ppd_min = (
            df_ppd[["well","start_date","delta_PPD","deltaQpr"]]
            .rename(columns={"well":"ppd_well","start_date":"ppd_start"})
        )

        # 3) oil_clean
        path_oc = os.path.join(clean_data_dir, oil_clean_fname)
        df_clean = pd.read_csv(path_oc)
        df_clean["date"]   = pd.to_datetime(df_clean["date"], errors="coerce")
        _df_clean = df_clean

        # 4) предварительное группирование по well
        _clean_by_well = {
            well: sub_df.sort_values("date")[["date","q_oil","p_oil"]]
            for well, sub_df in df_clean.groupby("well")
        }

    return _df_ow, _df_ppd_min, _df_clean, _clean_by_well

# ─── Базовая функция CI (читает динамические config) ─────────────────────────────
def compute_ci(dp: float, eps_q: float, eps_p: float) -> float:
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

# ─── Основная функция: возвращает DataFrame, как раньше ────────────────────────
def compute_ci_with_pre(
    clean_data_dir: str = "clean_data",
    oil_windows_fname: str = "oil_windows.csv",
    ppd_events_fname: str = "ppd_events.csv",
    oil_clean_fname: str = "oil_clean.csv",
) -> pd.DataFrame:

    # Загрузить (одноразово) и получить кешированные DF + группы
    df_ow, df_ppd_min, df_clean, clean_by_well = _load_and_cache(
        clean_data_dir, oil_windows_fname, ppd_events_fname, oil_clean_fname
    )

    # Объединяем события
    df = pd.merge(df_ow, df_ppd_min, on=["ppd_well","ppd_start"], how="left").copy()

    # Процессинг одной строки
    def process(row):
        well       = row["well"]
        oil_start  = row["oil_start"]
        oil_end    = row["oil_end"]
        dp         = row["delta_PPD"]
        dq_act     = row["q_end"] - row["q_start"]
        dp_act     = (row["p_end"] - row["p_start"]) if pd.notnull(row["p_start"]) else 0.0

        # Взять пред-окно из кеша, затем фильтровать по дате
        pre_df = clean_by_well.get(well)
        dq_b_mean = dp_b_mean = 0.0
        dq_b_reg  = dp_b_reg  = 0.0
        dq_b_med  = dp_b_med  = 0.0

        if pre_df is not None:
            # mask только по дате
            pre = pre_df[
                (pre_df["date"] >= oil_start - pd.Timedelta(days=config.pre_len)) &
                (pre_df["date"] <  oil_start)
            ].query("q_oil > 0 & p_oil > 0")

            if len(pre) >= 5:
                # mean
                dq_b_mean = pre["q_oil"].mean() - row["q_start"]
                dp_b_mean = pre["p_oil"].mean() - row["p_start"]

                # regression
                pr = pre
                t0  = pr["date"].iat[0]
                days = (pr["date"] - t0).dt.days.values.reshape(-1,1)
                lrq = LinearRegression().fit(days, pr["q_oil"].values)
                lrp = LinearRegression().fit(days, pr["p_oil"].values)
                d0  = (oil_start - t0).days
                d1  = (oil_end   - t0).days
                dq_b_reg = lrq.predict([[d1]])[0] - lrq.predict([[d0]])[0]
                dp_b_reg = lrp.predict([[d1]])[0] - lrp.predict([[d0]])[0]

                # median + EWMA
                med_q = pr["q_oil"].median()
                med_p = pr["p_oil"].median()
                ewq   = pr["q_oil"].ewm(alpha=0.2).mean()
                ewp   = pr["p_oil"].ewm(alpha=0.2).mean()
                dq_b_med = ((med_q - row["q_start"]) + (ewq.iat[-1] - ewq.iat[0]))/2
                dp_b_med = ((med_p - row["p_start"]) + (ewp.iat[-1] - ewp.iat[0]))/2

        # округляем базовые дельты
        dq_b_mean, dp_b_mean = round(dq_b_mean,1), round(dp_b_mean,1)
        dq_b_reg,  dp_b_reg  = round(dq_b_reg,1),  round(dp_b_reg,1)
        dq_b_med,  dp_b_med  = round(dq_b_med,1),  round(dp_b_med,1)

        # вычисляем все 4 CI
        ci_none = round(compute_ci(dp, dq_act,      dp_act),1)
        ci_mean = round(compute_ci(dp, dq_act-dq_b_mean, dp_act-dp_b_mean),1)
        ci_reg  = round(compute_ci(dp, dq_act-dq_b_reg,  dp_act-dp_b_reg),1)
        ci_med  = round(compute_ci(dp, dq_act-dq_b_med,  dp_act-dp_b_med),1)

        return pd.Series({
            "deltaQpr": row["deltaQpr"],
            "q_start": row["q_start"],   "p_start": row["p_start"],
            "q_end": row["q_end"],       "p_end": row["p_end"],
            "duration_days_oil": row.get("duration_days_oil", np.nan),
            "oil_start": oil_start,      "oil_end": oil_end,
            "deltaQbaseCI_mean":       dq_b_mean,
            "deltaPbaseCI_mean":       dp_b_mean,
            "deltaQbaseCI_regression": dq_b_reg,
            "deltaPbaseCI_regression": dp_b_reg,
            "deltaQbaseCI_median_ewma":dq_b_med,
            "deltaPbaseCI_median_ewma":dp_b_med,
            "CI_none":      ci_none,
            "CI_mean":      ci_mean,
            "CI_regression":ci_reg,
            "CI_median_ewma":ci_med
        })

    # Применяем и сохраняем
    res = df.assign(**df.apply(process, axis=1))
    res = res.sort_values(["well","ppd_well"]).reset_index(drop=True)
    res.insert(0, "№", range(1, len(res)+1))

    # сохраняем
    out = os.path.join(clean_data_dir, "ci_results_pro.csv")
    res.to_csv(out, index=False)
    print(f"Сохранено: {out} ({len(res)} строк)")

    return res

# для standalone вызова
if __name__ == "__main__":
    compute_ci_with_pre()
