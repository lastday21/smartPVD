"""
metrics_baseline.py
───────────────────────────────────────────────────────────────────────────────
Содержит ВСЮ «дорогую» логику базлайна, которую мы больше не хотим держать
в основном прод-коде.  Модуль подгружается динамически ТОЛЬКО тогда, когда
в  metrics_pro.compute_ci_with_pre(methods=…)  встречаются строки:
    "mean", "regression", "median_ewma"

Функция
--------
get_baseline(method, pre_df, q_start, p_start, oil_start, oil_end) -> (dq, dp)

    method : str
        "mean", "regression" или "median_ewma".
    pre_df : DataFrame
        Отфильтрованные данные «пред-окна» (как минимум 5 строк),
        колонки: date, q_oil, p_oil.
    q_start, p_start : float
        Q и P ровно в момент начала события.
    oil_start, oil_end : Timestamp
        Даты начала и конца окна «до / после».

Возвращает
----------
tuple(float, float)
    deltaQbase, deltaPbase – ОКРУГЛЁННЫЕ до 0.1.

Для regression используется sklearn.linear_model.LinearRegression.
"""

import pandas as pd
from sklearn.linear_model import LinearRegression


# ─────────────────────────────────────────  helpers
def _mean_baseline(pre: pd.DataFrame, q0: float, p0: float):
    dq = pre["q_oil"].mean() - q0
    dp = pre["p_oil"].mean() - p0
    return round(dq, 1), round(dp, 1)


def _regression_baseline(pre: pd.DataFrame,
                         q0: float, p0: float,
                         t_start, t_end):
    t0 = pre["date"].iat[0]
    days = (pre["date"] - t0).dt.days.values.reshape(-1, 1)

    lr_q = LinearRegression().fit(days, pre["q_oil"].values)
    lr_p = LinearRegression().fit(days, pre["p_oil"].values)

    d0, d1 = (t_start - t0).days, (t_end - t0).days
    dq = lr_q.predict([[d1]])[0] - lr_q.predict([[d0]])[0]
    dp = lr_p.predict([[d1]])[0] - lr_p.predict([[d0]])[0]
    return round(dq, 1), round(dp, 1)


def _median_ewma_baseline(pre: pd.DataFrame, q0: float, p0: float):
    med_q = pre["q_oil"].median()
    med_p = pre["p_oil"].median()
    ewq = pre["q_oil"].ewm(alpha=0.2).mean()
    ewp = pre["p_oil"].ewm(alpha=0.2).mean()
    dq = ((med_q - q0) + (ewq.iat[-1] - ewq.iat[0])) / 2
    dp = ((med_p - p0) + (ewp.iat[-1] - ewp.iat[0])) / 2
    return round(dq, 1), round(dp, 1)


# ─────────────────────────────────────────  public
def get_baseline(method: str,
                 pre_df: pd.DataFrame,
                 q_start: float,
                 p_start: float,
                 oil_start=None,
                 oil_end=None):
    """Вернуть (deltaQbase, deltaPbase) для выбранного метода."""
    if method == "mean":
        return _mean_baseline(pre_df, q_start, p_start)
    if method == "regression":
        return _regression_baseline(
            pre_df, q_start, p_start, oil_start, oil_end
        )
    if method == "median_ewma":
        return _median_ewma_baseline(pre_df, q_start, p_start)
    # method == "none" или неизвестный
    return 0.0, 0.0
