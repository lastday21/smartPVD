"""
metrics.py

Этот модуль вычисляет показатель CI (Composite Influence) для каждой пары скважина–ППД → нефтянка.
CI отражает силу влияния изменения закачки ППД на дебит (Q) и давление (P) нефтяной скважины,
с учётом предварительного тренда (baseline) за T_pre дней.

Фильтрует все нулевые замеры q_oil и p_oil в предокне.
Если в предокне после фильтрации <5 точек, использует прямую дельту без baseline.

Использует файлы в clean_data:
  - oil_windows.csv
  - ppd_events.csv
  - oil_clean.csv

Глобальные параметры в config.py:
  T_pre, divider_q, divider_p, w_q, w_p

Результат сохраняется в clean_data/ci_results.csv
с колонками:
  №, well_oil, ppd_well, deltaQpr,
  q_start, p_start, q_end, p_end,
  duration_days_oil, start_date, end_date,
  deltaQbase, deltaPbase, CI
"""

import os
import warnings
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from config import T_pre, divider_q, divider_p, w_q, w_p

# Отключаем предупреждения sklearn о feature names
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but .* was fitted with feature names"
)

def compute_ci_with_pre(
    clean_data_dir: str = "clean_data",
    oil_windows_fname: str = "oil_windows.csv",
    ppd_events_fname: str = "ppd_events.csv",
    oil_clean_fname: str = "oil_clean.csv"
) -> pd.DataFrame:
    # Пути к файлам
    path_oil_windows = os.path.join(clean_data_dir, oil_windows_fname)
    path_ppd_events  = os.path.join(clean_data_dir, ppd_events_fname)
    path_oil_clean   = os.path.join(clean_data_dir, oil_clean_fname)
    output_path      = os.path.join(clean_data_dir, "ci_results.csv")

    # 1) Загрузка и конвертация дат
    df_oil_windows = pd.read_csv(path_oil_windows)
    df_oil_windows["oil_start"] = pd.to_datetime(df_oil_windows["oil_start"], errors="coerce")
    df_oil_windows["oil_end"]   = pd.to_datetime(df_oil_windows["oil_end"],   errors="coerce")
    df_oil_windows["ppd_start"] = pd.to_datetime(df_oil_windows["ppd_start"], errors="coerce")

    df_ppd_events = pd.read_csv(path_ppd_events)
    df_ppd_events["start_date"] = pd.to_datetime(df_ppd_events["start_date"], dayfirst=True, errors="coerce")
    df_ppd_events["end_date"]   = pd.to_datetime(df_ppd_events["end_date"],   dayfirst=True, errors="coerce")

    df_oil_clean = pd.read_csv(path_oil_clean)
    df_oil_clean["date"] = pd.to_datetime(df_oil_clean["date"], errors="coerce")

    # 2) Вычисление Δ_PPD и deltaQpr
    def sign_delta(b, d):
        if pd.isna(b) or pd.isna(d):
            return 0
        return 1 if d > b else (-1 if d < b else 0)

    df_ppd_events["delta_PPD"] = df_ppd_events.apply(
        lambda r: sign_delta(r["baseline_before"], r["baseline_during"]), axis=1
    )
    df_ppd_events["deltaQpr"] = df_ppd_events["baseline_during"] - df_ppd_events["baseline_before"]

    df_ppd_min = df_ppd_events[["well","start_date","delta_PPD","deltaQpr"]].copy()
    df_ppd_min.rename(columns={"well":"ppd_well","start_date":"ppd_start"}, inplace=True)

    # 3) Объединение окон и событий
    df_ow = pd.merge(df_oil_windows, df_ppd_min, how="left", on=["ppd_well","ppd_start"])

    # 4) Обработка каждого окна
    def process_row(row):
        well = row["well"]
        oil_start = row["oil_start"]
        oil_end   = row["oil_end"]
        dp = row["delta_PPD"]

        # Фактические дельты
        dq_act = row["q_end"] - row["q_start"]
        p0, p1 = row["p_start"], row["p_end"]
        dp_act = 0.0 if p0==0 or p1==0 or pd.isna(p0) or pd.isna(p1) else (p1 - p0)

        # Предокно + фильтрация нулевых замеров
        mask = (
            (df_oil_clean["well"] == well) &
            (df_oil_clean["date"] >= oil_start - pd.Timedelta(days=T_pre)) &
            (df_oil_clean["date"] < oil_start)
        )
        df_pre = df_oil_clean.loc[mask, ["date","q_oil","p_oil"]].copy()
        df_pre = df_pre[(df_pre["q_oil"] > 0) & (df_pre["p_oil"] > 0)]
        df_pre.sort_values("date", inplace=True)

        # Если недостаточно точек, используем прямую дельту без baseline
        if len(df_pre) < 5:
            dq_base = 0.0
            dp_base = 0.0
            eps_q = dq_act
            eps_p = dp_act
        else:
            # Подготовка для регрессии
            df_pre = df_pre.reset_index(drop=True)
            t0 = df_pre.loc[0, "date"]
            df_pre["day_ind"] = (df_pre["date"] - t0).dt.days.astype(float)

            # Линейная регрессия для дебита
            X_q = df_pre["day_ind"].to_numpy().reshape(-1, 1)
            y_q = df_pre["q_oil"].to_numpy()
            lr_q = LinearRegression().fit(X_q, y_q)

            # Линейная регрессия для давления
            X_p = df_pre["day_ind"].to_numpy().reshape(-1, 1)
            y_p = df_pre["p_oil"].to_numpy()
            lr_p = LinearRegression().fit(X_p, y_p)

            # Расчёт прогнозов на моменты начала и конца
            d0 = float((oil_start - t0).days)
            d1 = float((oil_end   - t0).days)
            Q0 = lr_q.predict(np.array([[d0]]))[0]
            Q1 = lr_q.predict(np.array([[d1]]))[0]
            P0 = lr_p.predict(np.array([[d0]]))[0]
            P1 = lr_p.predict(np.array([[d1]]))[0]

            dq_base = Q1 - Q0
            dp_base = P1 - P0
            eps_q = dq_act - dq_base
            eps_p = dp_act - dp_base

        # Расчёт x, y и CI
        if dp == 0:
            x = y = 0.0
        else:
            x = dp * (eps_q / divider_q)
            y = dp * (eps_p / divider_p)

        if x >= 0 and y >= 0:
            CI = w_q * x + w_p * y
        elif x >= 0 and y < 0:
            CI = x
        elif x < 0 and y >= 0:
            CI = y
        else:
            CI = 0.0

        return pd.Series({
            "x": x,
            "y": y,
            "CI": CI,
            "deltaQbase": int(round(dq_base)),
            "deltaPbase": round(dp_base, 1)
        })

    df_ow[["x","y","CI","deltaQbase","deltaPbase"]] = df_ow.apply(process_row, axis=1)

    # 5) Формирование итогового DataFrame
    df = df_ow[[
        "well","ppd_well","deltaQpr",
        "q_start","p_start","q_end","p_end",
        "duration_days_oil","oil_start","oil_end",
        "deltaQbase","deltaPbase","CI"
    ]].copy()
    df.insert(0, "№", range(1, len(df) + 1))
    df.rename(columns={
        "well": "well_oil",
        "oil_start": "start_date",
        "oil_end":   "end_date"
    }, inplace=True)
    df["CI"] = df["CI"].round(1)
    df.sort_values("well_oil", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["№"] = range(1, len(df) + 1)

    # Сохранение результата
    df.to_csv(output_path, index=False)
    print(f"Сохранено: {output_path}, строк: {len(df)}")
    return df

if __name__ == "__main__":
    compute_ci_with_pre()
