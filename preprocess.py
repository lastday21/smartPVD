# preprocess.py

"""
Модуль предобработки данных для SmartPVD.

Содержит:
  – clean_ppd(df): очистка и заполнение рядов нагнетательных скважин (ППД)
  – clean_oil(df): очистка и заполнение рядов нефтяных скважин
  – resample_and_fill(series, kind): ресемплинг и заполнение пропусков на ежедневной частоте

Все функции обрабатывают записи в исходном порядке, не смешивая данные разных скважин.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from config import GAP_LIMIT, FREQ_THRESH, MIN_WORK_PPD, NO_PRESS_WITH_Q_LIMIT


def _num(col: pd.Series) -> pd.Series:
    """
    Преобразует строковые числовые значения вида "1 234,56" в float 1234.56.
    Удаляет неразрывный пробел и пробелы, заменяет запятую на точку.
    Нечисловые → NaN.
    """
    return pd.to_numeric(
        col.astype(str)
           .str.replace("\u00A0", "", regex=False)
           .str.replace(" ", "", regex=False)
           .str.replace(",", ".", regex=False),
        errors="coerce"
    )


def _interp_bf_ff(s: pd.Series) -> pd.Series:
    """
    Интерполяция до GAP_LIMIT дней, затем back-fill и forward-fill.
    Используется для q_ppd, q_oil, p_cust, p_oil.
    """
    return (
        s.interpolate(limit=GAP_LIMIT, limit_direction="both")
         .bfill()
         .ffill()
    )


def _bf_ff(s: pd.Series) -> pd.Series:
    """
    Сначала back-fill, потом forward-fill.
    Используется для water_cut, freq, t_work.
    """
    return s.bfill().ffill()


def clean_ppd(df: pd.DataFrame) -> pd.DataFrame:
    """
    Очистка и заполнение параметров нагнетательных скважин (ППД).

    1. Нормализация колонок q_ppd, p_cust, d_choke.
    2. Шум: q_ppd < MIN_WORK_PPD → NaN.
    3. Определение рабочих периодов:
       – старт при p_cust > 0,
       – тех. провал <5 дней подряд (нет давления и нет q) → считаем работой,
       – отдых без давления, но с q ≥ MIN_WORK_PPD до NO_PRESS_WITH_Q_LIMIT дней → считаем работой,
       – при превышении лимитов → ретроактивный стоп всего блока.
    4. Заполнение внутри работы:
       – p_cust → back-fill,
       – q_ppd  → внутри блока bfill→ffill,
       – d_choke → ffill→bfill.
    5. Окончательное округление в целые.
    """
    wells = []
    # 1) нормализация
    for c in ("q_ppd", "p_cust", "d_choke"):
        df[c] = _num(df.get(c, pd.NA))

    # 2–5) по каждой скважине
    for well, sub in df.groupby("well", sort=False):
        sub = sub.reset_index(drop=True)

        # d_choke: всегда тянем прошлое/будущее
        sub["d_choke"] = sub["d_choke"].ffill().bfill().fillna(0)
        # шум в дебите
        sub["q_ppd"] = sub["q_ppd"].where(sub["q_ppd"] >= MIN_WORK_PPD, np.nan)

        work_flag   = []
        in_work     = False
        no_p_with_q = 0
        no_both     = 0

        # 3) вычисляем флаг работы
        for i, row in sub.iterrows():
            has_p = row["p_cust"] > 0
            has_q = row["q_ppd"] >= MIN_WORK_PPD

            if not in_work and has_p:
                in_work = True

            if in_work:
                if not has_p and has_q:
                    no_p_with_q += 1
                    if no_p_with_q >= NO_PRESS_WITH_Q_LIMIT:
                        start = len(work_flag) - (no_p_with_q - 1)
                        for k in range(start, len(work_flag)):
                            work_flag[k] = False
                        in_work = False
                        no_p_with_q = no_both = 0
                elif not has_p and not has_q:
                    no_both += 1
                    if no_both >= 5:
                        start = len(work_flag) - (no_both - 1)
                        for k in range(start, len(work_flag)):
                            work_flag[k] = False
                        in_work = False
                        no_p_with_q = no_both = 0
                else:
                    no_p_with_q = no_both = 0

            work_flag.append(in_work)

        sub["work"] = work_flag

        # 4a) p_cust: back-fill внутри работы, иначе 0
        last_p = 0.0
        out_p  = []
        for i, row in sub.iterrows():
            if sub.at[i, "work"]:
                if row["p_cust"] > 0:
                    last_p = row["p_cust"]
                out_p.append(last_p)
            else:
                last_p = 0.0
                out_p.append(0.0)
        sub["p_cust"] = pd.Series(out_p, index=sub.index)

        # 4b) q_ppd: bfill→ffill внутри рабочего блока
        out_q = pd.Series(0.0, index=sub.index)
        grp2  = sub["work"].ne(sub["work"].shift()).cumsum()
        for _, idxs in sub.groupby(grp2, sort=False).groups.items():
            if sub.loc[idxs, "work"].iat[0]:
                seg = sub.loc[idxs, "q_ppd"]
                filled = seg.bfill().ffill().fillna(0)
                out_q.loc[idxs] = filled
        sub["q_ppd"] = out_q

        # 5) округление
        sub["d_choke"] = sub["d_choke"].astype(int)
        sub["p_cust"]  = sub["p_cust"].round(0).astype(int)
        sub["q_ppd"]   = sub["q_ppd"].round(0).astype(int)

        wells.append(sub.drop(columns="work"))

    return pd.concat(wells, ignore_index=True)


def clean_oil(df: pd.DataFrame) -> pd.DataFrame:
    """
    Очистка и заполнение параметров нефтяных скважин.

    1. Нормализация колонок q_oil, water_cut, p_oil, freq, t_work.
    2. raw_work = (freq > FREQ_THRESH) или (q_oil > 0 и t_work > 0).
    3. Старт работы сразу при raw=True; стоп при ≥5 подряд raw=False
       (ретроактивное обнуление всего блока).
    4. Заполнение внутри work:
       – q_oil, p_oil → interpolate→bfill→ffill,
       – water_cut, freq, t_work → bfill→ffill;
       вне работы все поля = 0.
    5. Окончательное округление:
       water_cut, freq → int,
       t_work, p_oil → одна десятая.
    """
    out = []

    # 1) нормализация
    for c in ("q_oil", "water_cut", "p_oil", "freq", "t_work"):
        df[c] = _num(df.get(c, pd.NA)) if c in df.columns else pd.Series(pd.NA, index=df.index)

    # 2–5) по каждой скважине
    for well, sub in df.groupby("well", sort=False):
        sub = sub.reset_index(drop=True)

        raw = (sub["freq"] > FREQ_THRESH) | ((sub["q_oil"] > 0) & (sub["t_work"] > 0))

        in_work     = False
        miss_streak = 0
        flags       = []

        # 3) формируем флаг работы
        for is_raw in raw:
            if not in_work:
                if is_raw:
                    in_work = True
                    miss_streak = 0
            else:
                if not is_raw:
                    miss_streak += 1
                    if miss_streak >= 5:
                        flags.append(False)
                        start = len(flags) - miss_streak
                        for k in range(start, len(flags)):
                            flags[k] = False
                        in_work = False
                        miss_streak = 0
                        continue
                else:
                    miss_streak = 0
            flags.append(in_work)

        sub["work"] = flags

        # 4a) water_cut, freq, t_work: bfill→ffill внутри работы, иначе 0
        for c in ("water_cut", "freq", "t_work"):
            tmp = sub[c].where(sub["work"]).pipe(_bf_ff)
            sub[c] = tmp.where(sub["work"], 0)

        # 4b) q_oil, p_oil: interpolate→bfill→ffill внутри работы, иначе 0
        for c in ("q_oil", "p_oil"):
            tmp = sub[c].where(sub["work"]).pipe(_interp_bf_ff)
            sub[c] = tmp.where(sub["work"], 0)

        # 5) окончательное округление
        sub["water_cut"] = sub["water_cut"].round(0).astype(int)
        sub["freq"]      = sub["freq"].round(0).astype(int)
        sub["t_work"]    = sub["t_work"].round(1)
        sub["p_oil"]     = sub["p_oil"].round(1)

        out.append(sub.drop(columns="work"))

    return pd.concat(out, ignore_index=True)


def resample_and_fill(series: pd.Series, *, kind: str) -> pd.Series:
    """
    Суточный ресемплинг:
      – water_cut, freq, t_work → bfill→ffill,
      – d_choke → ffill→bfill,
      – остальные → interpolate→bfill→ffill,
      – если серия пустая → заполняем нулями.
    """
    daily = series.resample("D").asfreq()
    name  = series.name

    if name in ("water_cut", "freq", "t_work"):
        return _bf_ff(daily)
    elif name == "d_choke":
        return daily.ffill().bfill()
    else:
        if daily.isna().all():
            return pd.Series(0, index=daily.index, name=name)
        return _interp_bf_ff(daily)
