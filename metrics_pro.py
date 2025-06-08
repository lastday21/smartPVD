"""
metrics.py

Модуль вычисляет несколько вариантов CI (Composite Influence) для каждой пары скважина–ППД:
1) CI без baseline-коррекции
2) CI с mean-baseline
3) CI с regression-baseline (существующий метод)
4) CI с median+EWMA baseline

Результат сохраняется в clean_data/ci_results_pro.csv с колонками:
  №, well_oil, ppd_well, deltaQpr,
  q_start, p_start, q_end, p_end,
  duration_days_oil, oil_start, oil_end,
  deltaQbaseCI_mean, deltaPbaseCI_mean,
  deltaQbaseCI_regression, deltaPbaseCI_regression,
  deltaQbaseCI_median_ewma, deltaPbaseCI_median_ewma,
  CI_none, CI_mean, CI_regression, CI_median_ewma

Все базовые дельты и CI округляются до одной десятой.
Строки сортируются по возрастанию номера скважины (well).
"""
import os
import warnings
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from config import T_pre, divider_q, divider_p, w_q, w_p

# Отключаем предупреждения sklearn
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but .* was fitted with feature names"
)


def compute_ci(dp: float, eps_q: float, eps_p: float) -> float:
    """Вычисление CI по базовой формуле из x, y"""
    if dp == 0:
        return 0.0
    x = dp * (eps_q / divider_q)
    y = dp * (eps_p / divider_p)
    if x >= 0 and y >= 0:
        return w_q * x + w_p * y
    if x >= 0 and y < 0:
        return x
    if x < 0 and y >= 0:
        return y
    return 0.0


def compute_ci_with_pre(
    clean_data_dir: str = "clean_data",
    oil_windows_fname: str = "oil_windows.csv",
    ppd_events_fname: str = "ppd_events.csv",
    oil_clean_fname: str = "oil_clean.csv"
) -> pd.DataFrame:
    # Пути к файлам
    path_ow = os.path.join(clean_data_dir, oil_windows_fname)
    path_ppd = os.path.join(clean_data_dir, ppd_events_fname)
    path_oc = os.path.join(clean_data_dir, oil_clean_fname)
    output_path = os.path.join(clean_data_dir, "ci_results_pro.csv")

    # Загрузка данных
    df_ow = pd.read_csv(path_ow)
    df_ow["oil_start"] = pd.to_datetime(df_ow["oil_start"], errors="coerce")
    df_ow["oil_end"] = pd.to_datetime(df_ow["oil_end"], errors="coerce")
    df_ow["ppd_start"] = pd.to_datetime(df_ow["ppd_start"], errors="coerce")

    df_ppd = pd.read_csv(path_ppd)
    df_ppd["start_date"] = pd.to_datetime(df_ppd["start_date"], dayfirst=True, errors="coerce")
    df_ppd["delta_PPD"] = (df_ppd["baseline_during"] - df_ppd["baseline_before"]).apply(
        lambda d: 1 if d > 0 else (-1 if d < 0 else 0)
    )
    df_ppd["deltaQpr"] = df_ppd["baseline_during"] - df_ppd["baseline_before"]
    df_ppd_min = df_ppd[["well","start_date","delta_PPD","deltaQpr"]].rename(
        columns={"well":"ppd_well","start_date":"ppd_start"}
    )

    df_clean = pd.read_csv(path_oc)
    df_clean["date"] = pd.to_datetime(df_clean["date"], errors="coerce")

    # Объединение с oil_windows
    df = pd.merge(df_ow, df_ppd_min, on=["ppd_well","ppd_start"], how="left").copy()

    def process(row):
        well = row["well"]
        oil_start, oil_end = row["oil_start"], row["oil_end"]
        dp = row["delta_PPD"]
        dq_act = row["q_end"] - row["q_start"]
        dp_act = (row["p_end"] - row["p_start"]) if pd.notnull(row["p_start"]) and pd.notnull(row["p_end"]) else 0.0

        # Пред-окно
        mask = (
            (df_clean["well"] == well) &
            (df_clean["date"] >= oil_start - pd.Timedelta(days=T_pre)) &
            (df_clean["date"] < oil_start)
        )
        pre = (
            df_clean.loc[mask, ["date","q_oil","p_oil"]]
            .query("q_oil > 0 & p_oil > 0").sort_values("date")
        )

        # Инициализация базовых смещений
        dq_b_mean = dp_b_mean = 0.0
        dq_b_reg = dp_b_reg = 0.0
        dq_b_med = dp_b_med = 0.0

        if len(pre) >= 5:
            # mean-baseline
            dq_b_mean = pre["q_oil"].mean() - row["q_start"]
            dp_b_mean = pre["p_oil"].mean() - row["p_start"]
            # regression-baseline
            pr = pre.reset_index(drop=True)
            t0 = pr.loc[0, "date"]
            days = (pr["date"] - t0).dt.days.values.reshape(-1,1)
            lrq = LinearRegression().fit(days, pr["q_oil"].values)
            lrp = LinearRegression().fit(days, pr["p_oil"].values)
            d0 = (oil_start - t0).days
            d1 = (oil_end - t0).days
            dq_b_reg = lrq.predict([[d1]])[0] - lrq.predict([[d0]])[0]
            dp_b_reg = lrp.predict([[d1]])[0] - lrp.predict([[d0]])[0]
            # median+EWMA
            med_q = pr["q_oil"].median()
            med_p = pr["p_oil"].median()
            ewq = pr["q_oil"].ewm(alpha=0.2).mean()
            ewp = pr["p_oil"].ewm(alpha=0.2).mean()
            dq_b_med = ((med_q - row["q_start"]) + (ewq.iloc[-1] - ewq.iloc[0])) / 2
            dp_b_med = ((med_p - row["p_start"]) + (ewp.iloc[-1] - ewp.iloc[0])) / 2

        # Округление до десятых
        dq_b_mean, dp_b_mean = round(dq_b_mean,1), round(dp_b_mean,1)
        dq_b_reg, dp_b_reg = round(dq_b_reg,1), round(dp_b_reg,1)
        dq_b_med, dp_b_med = round(dq_b_med,1), round(dp_b_med,1)

        # CI
        ci_none = round(compute_ci(dp, dq_act, dp_act), 1)
        ci_mean = round(compute_ci(dp, dq_act - dq_b_mean, dp_act - dp_b_mean), 1)
        ci_reg = round(compute_ci(dp, dq_act - dq_b_reg, dp_act - dp_b_reg), 1)
        ci_med = round(compute_ci(dp, dq_act - dq_b_med, dp_act - dp_b_med), 1)

        return pd.Series({
            "deltaQpr": row["deltaQpr"],
            "q_start": row["q_start"], "p_start": row["p_start"],
            "q_end": row["q_end"], "p_end": row["p_end"],
            "duration_days_oil": row.get("duration_days_oil", np.nan),
            "oil_start": oil_start, "oil_end": oil_end,
            "deltaQbaseCI_mean": dq_b_mean,
            "deltaPbaseCI_mean": dp_b_mean,
            "deltaQbaseCI_regression": dq_b_reg,
            "deltaPbaseCI_regression": dp_b_reg,
            "deltaQbaseCI_median_ewma": dq_b_med,
            "deltaPbaseCI_median_ewma": dp_b_med,
            "CI_none": ci_none,
            "CI_mean": ci_mean,
            "CI_regression": ci_reg,
            "CI_median_ewma": ci_med
        })

    res = df.apply(process, axis=1)
    # сортировка по номеру скважины
    res.insert(0, "well", df["well"])
    res.insert(1, "ppd_well", df["ppd_well"])
    res = res.sort_values(by=["well", "ppd_well"]).reset_index(drop=True)
    # нумерация по порядку после сортировки
    res.insert(0, "№", range(1, len(res) + 1))

    # сохранение
    res.to_csv(output_path, index=False)
    print(f"Сохранено: {output_path}, строк: {len(res)}")
    return res


if __name__ == "__main__":
    compute_ci_with_pre()
