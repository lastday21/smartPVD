"""
Этот модуль:
1. Читает исходные Excel-файлы ППД, добычи и координат.
2. Очищает и фильтрует данные по заданным правилам.
3. Создаёт суточные ряды с интерполяцией и заполнением пропусков.
4. По запросу сохраняет итоговые CSV для отладки.

Ключевые функции:
- load_ppd, load_oil, load_coords — чтение и приведение колонок к стандартным именам.
- clean_ppd      — очистка ППД-рядов.
- clean_oil      — очистка добычи нефти.
- resample_and_fill — суточный ресемплинг с заполнением пропусков.
- build_clean_data — полный конвейер, возвращает DataFrame-ы и опционально пишет CSV.
"""

from __future__ import annotations
from pathlib import Path
import re
import pandas as pd
import numpy as np

# --------------------------- конфиг -------------------------------------
from config import (
    PPD_FILE, OIL_FILE, COORD_FILE,
    PPD_SHEET_NAME, OIL_SHEET_NAME,
    GAP_LIMIT, FREQ_THRESH, MIN_WORK_PPD, NO_PRESS_WITH_Q_LIMIT,
)

# Папка для отладочных CSV
CLEAN_DIR = Path("clean_data")
CLEAN_DIR.mkdir(exist_ok=True)

# --------------------------------------------------------------------
# Вспомогательные функции
# --------------------------------------------------------------------

def _num(col: pd.Series) -> pd.Series:
    """
    Преобразует строковый формат чисел '1 234,56' → 1234.56 (float).
    Удаляет пробелы и заменяет ',' на '.'.
    """
    return pd.to_numeric(
        col.astype(str)
           .str.replace("\u00A0", "", regex=False)
           .str.replace(" ",     "", regex=False)
           .str.replace(",",     ".", regex=False),
        errors="coerce",
    )


def _interp_bf_ff(s: pd.Series) -> pd.Series:
    """Интерполяция пропусков до GAP_LIMIT дней, затем backfill + forwardfill."""
    return s.interpolate(limit=GAP_LIMIT, limit_direction="both").bfill().ffill()


def _bf_ff(s: pd.Series) -> pd.Series:
    """Заполнение пропусков: сначала назад (bfill), затем вперёд (ffill)."""
    return s.bfill().ffill()

# --------------------------------------------------------------------
# Очистка данных ППД
# --------------------------------------------------------------------

def clean_ppd(df: pd.DataFrame) -> pd.DataFrame:
    """
    Очищает данные приема жидкости:
    - Приводит колонки q_ppd, p_cust, d_choke к числовому виду.
    - Фильтрует нерелевантные значения по MIN_WORK_PPD и уровню давления.
    - Определяет рабочие интервалы, заполняет пропуски значения давления и расхода.
    """
    wells = []

    # Приведение к числовому
    for c in ("q_ppd", "p_cust", "d_choke"):
        df[c] = _num(df.get(c, pd.NA))

    # По каждой скважине
    for _, sub in df.groupby("well", sort=False):
        sub = sub.reset_index(drop=True)

        # Начальное заполнение d_choke и фильтрация q_ppd
        sub["d_choke"] = sub["d_choke"].ffill().bfill().fillna(0)
        sub["q_ppd"]   = sub["q_ppd"].where(sub["q_ppd"] >= MIN_WORK_PPD, np.nan)

        # Определение флага "в работе"
        work_flag, in_work, no_pq, no_both = [], False, 0, 0

        for _, row in sub.iterrows():
            has_p = row["p_cust"] > 0
            has_q = row["q_ppd"] >= MIN_WORK_PPD

            if not in_work and has_p:
                in_work = True

            if in_work:
                if not has_p and has_q:
                    no_pq += 1
                    if no_pq >= NO_PRESS_WITH_Q_LIMIT:
                        start = len(work_flag) - (no_pq - 1)
                        for k in range(start, len(work_flag)):
                            work_flag[k] = False
                        in_work = False; no_pq = no_both = 0
                elif not has_p and not has_q:
                    no_both += 1
                    if no_both >= 5:
                        start = len(work_flag) - (no_both - 1)
                        for k in range(start, len(work_flag)):
                            work_flag[k] = False
                        in_work = False; no_pq = no_both = 0
                else:
                    no_pq = no_both = 0
            work_flag.append(in_work)

        sub["work"] = work_flag

        # Заполнение p_cust внутри рабочего интервала
        last_p, p_out = 0.0, []
        for i, row in sub.iterrows():
            if sub.at[i, "work"]:
                if row["p_cust"] > 0:
                    last_p = row["p_cust"]
                p_out.append(last_p)
            else:
                last_p = 0.0; p_out.append(0.0)
        sub["p_cust"] = pd.Series(p_out, index=sub.index)

        # Заполнение q_ppd внутри рабочего интервала
        out_q = pd.Series(0.0, index=sub.index)
        grp = sub["work"].ne(sub["work"].shift()).cumsum()
        for _, idx in sub.groupby(grp, sort=False).groups.items():
            if sub.loc[idx, "work"].iat[0]:
                seg = sub.loc[idx, "q_ppd"].bfill().ffill().fillna(0)
                out_q.loc[idx] = seg
        sub["q_ppd"] = out_q

        # 5) округление
        sub["d_choke"] = sub["d_choke"].astype(int)
        sub["p_cust"]  = sub["p_cust"].round(0).astype(int)
        sub["q_ppd"]   = sub["q_ppd"].round(0).astype(int)

        wells.append(sub.drop(columns="work"))

    return pd.concat(wells, ignore_index=True)

# --------------------------------------------------------------------
# Очистка данных добычи нефти
# --------------------------------------------------------------------

def clean_oil(df: pd.DataFrame) -> pd.DataFrame:
    """
        Очищает данные добычи нефти:
        - Приводит колонки к числовому виду.
        - Определяет интервалы работы по частоте (FREQ_THRESH) и объему.
        - Заполняет пропуски разными методами: backfill, forwardfill, интерполяция.
    """
    out = []
    # Нормализация колонок
    for col in ("q_oil", "water_cut", "p_oil", "freq", "t_work"):
        df[col] = _num(df.get(col, pd.NA))
    for _, group in df.groupby("well", sort=False):
        sub = group.reset_index(drop=True)
        raw = (sub["freq"] > FREQ_THRESH) | ((sub["q_oil"] > 0) & (sub["t_work"] > 0))
        work = []
        miss = 0
        in_work = False
        for is_raw in raw:
            if not in_work and is_raw:
                in_work = True
                miss = 0
            elif in_work and not is_raw:
                miss += 1
                if miss >= 5:
                    start = len(work) - miss + 1
                    for i in range(start, len(work)):
                        work[i] = False
                    in_work = False
                    miss = 0
                    continue
            else:
                miss = 0
            work.append(in_work)
        # Дополняем флаг False до конца
        work.extend([False] * (len(sub) - len(work)))
        sub["work"] = work

        # Заполнение пропусков
        for col in ("water_cut", "freq", "t_work"):
            filled = sub[col].where(sub["work"]).pipe(_bf_ff)
            sub[col] = filled.where(sub["work"], 0)
        for col in ("q_oil", "p_oil"):
            filled = sub[col].where(sub["work"]).pipe(_interp_bf_ff)
            sub[col] = filled.where(sub["work"], 0)

        # Округление
        sub["water_cut"] = sub["water_cut"].round(0).astype(int)
        sub["freq"] = sub["freq"].round(0).astype(int)
        sub["t_work"] = sub["t_work"].round(1)
        sub["p_oil"] = sub["p_oil"].round(1)
        out.append(sub.drop(columns="work"))
    return pd.concat(out, ignore_index=True)

# --------------------------------------------------------------------
# Суточный ресемплинг
# --------------------------------------------------------------------

def resample_and_fill(series: pd.Series, *, kind: str) -> pd.Series:
    """
        Выполняет суточный ресемплинг и заполняет пропуски:
          - "water_cut", "freq", "t_work": backfill + forwardfill
          - "d_choke": forwardfill + backfill
          - прочие: интерполяция + fill
    """
    daily = series.resample("D").asfreq()
    name  = series.name
    if name in ("water_cut", "freq", "t_work"):
        return _bf_ff(daily)
    if name == "d_choke":
        return daily.ffill().bfill()
    if daily.isna().all():
        return pd.Series(0, index=daily.index, name=name)
    return _interp_bf_ff(daily)

# --------------------------------------------------------------------
# Чтение исходных Excel
# --------------------------------------------------------------------

def _find_header(path: Path, probe: int = 20) -> int:
    """
    Определяет номер строки, в которой начинается заголовок (по слову 'дата').
    """
    sample = pd.read_excel(path, header=None, nrows=probe, dtype=str)
    for i in range(probe):
        if sample.iloc[i].astype(str).str.contains("дата", case=False, na=False).any():
            return i
    return 0


def _read_raw(path: Path, sheet_name: str) -> pd.DataFrame:
    """
    Читает лист Excel с автопоиском заголовка, чистит названия колонок, приводит date к datetime.
    """
    header = _find_header(path)
    df = pd.read_excel(path, sheet_name=sheet_name, header=header, dtype=str)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df.columns = df.columns.str.replace("\u00A0", " ").str.strip()
    for col in df.columns:
        if re.search(r"(?i)дата", col):
            df = df.rename(columns={col: "date"})
            break
    df["date"] = pd.to_datetime(df["date"], dayfirst=False, errors="coerce")
    return df

def _ru(s: str) -> str:  # helper
    return s.lower().replace("\u00a0", " ").replace(" ", "")

# Маппинг названий колонок из русского в английский
RU2ENG_OIL = {
    "№п/п": "idx", "мест.": "field", "№скважины": "well", "куст": "cluster",
    "дата": "date", "qж": "q_oil", "обв": "water_cut",
    "рприем": "p_oil", "fвращтм": "freq", "траб(тм)": "t_work",
    "trab(тм)": "t_work",
}
RU2ENG_PPD = {
    "№п/п": "idx", "мест.": "field", "№скважины": "well", "куст": "cluster",
    "дата": "date", "dшт": "d_choke",
    "pкуст": "p_cust", "ркуст": "p_cust", "ркустовое": "p_cust",
    "qприем.тех": "q_ppd",
}


def load_ppd(path: Path | str = PPD_FILE) -> pd.DataFrame:
    """Читает и переименовывает колонки ППД."""
    df = _read_raw(Path(path), PPD_SHEET_NAME)
    df = df.rename(columns={c: RU2ENG_PPD[_ru(c)] for c in df.columns if _ru(c) in RU2ENG_PPD})
    return df


def load_oil(path: Path | str = OIL_FILE) -> pd.DataFrame:
    """Читает и переименовывает колонки добычи нефти."""
    df = _read_raw(Path(path), OIL_SHEET_NAME)
    df = df.rename(columns={c: RU2ENG_OIL[_ru(c)] for c in df.columns if _ru(c) in RU2ENG_OIL})
    for col in df.columns:
        if re.search(r"(?i)\(тм\)|раб", col):
            df = df.rename(columns={col: "t_work"}); break
    return df


def load_coords(path: Path | str = COORD_FILE) -> pd.DataFrame:
    """Читает таблицу координат и переименовывает колонки."""
    return (
        pd.read_excel(path)
          .rename(columns={"Скважина": "well", "X": "x", "Y": "y"})
          [["well", "x", "y"]]
    )

# — ресемпл колонки

def _daily(df: pd.DataFrame, col: str, *, kind: str) -> pd.DataFrame:
    return (
        df.set_index("date")
          .groupby("well", sort=False)[col]
          .apply(lambda s: resample_and_fill(s, kind=kind))
          .reset_index()
    )

# --------------------------------------------------------------------
# Основная функция конвейера
# --------------------------------------------------------------------

def build_clean_data(*, save_csv: bool = False):
    """
    Полный конвейер обработки:
    1. load_ppd, load_oil, load_coords
    2. clean_ppd, clean_oil
    3. Суточный ресемплинг q, p, d_choke для PPD и q, p, water_cut, freq, t_work для нефти
    4. Приведение типов
    5. Если save_csv=True, сохраняет CSV в clean_data/ в исходном формате

    Возвращает:
        ppd_daily, oil_daily, coords_df
    """
    # 1. читаем исходники
    ppd_raw = load_ppd(); oil_raw = load_oil(); coords = load_coords()

    # 2. чистка
    ppd_cln = clean_ppd(ppd_raw)
    oil_cln = clean_oil(oil_raw)

    # 3. суточные ряды PPD
    ppd_q  = _daily(ppd_cln, "q_ppd",  kind="ppd")
    ppd_p  = _daily(ppd_cln, "p_cust", kind="ppd")
    ppd_d  = _daily(ppd_cln, "d_choke", kind="ppd")
    ppd_daily = ppd_q.merge(ppd_p, on=("well","date")).merge(ppd_d, on=("well","date"))

    # 4. суточные ряды OIL
    oil_q  = _daily(oil_cln, "q_oil",     kind="oil")
    oil_p  = _daily(oil_cln, "p_oil",     kind="oil")
    oil_wc = _daily(oil_cln, "water_cut", kind="oil")
    oil_f  = _daily(oil_cln, "freq",      kind="oil")
    oil_tw = _daily(oil_cln, "t_work",    kind="oil")
    oil_daily = (oil_q.merge(oil_p, on=("well","date"))
                       .merge(oil_wc, on=("well","date"))
                       .merge(oil_f,  on=("well","date"))
                       .merge(oil_tw, on=("well","date")))

    # 5. кастинг типов
    ppd_daily[["q_ppd","p_cust","d_choke"]] = ppd_daily[["q_ppd","p_cust","d_choke"]].astype(int)
    oil_daily["q_oil"]     = oil_daily["q_oil"].astype(int)
    oil_daily["water_cut"] = oil_daily["water_cut"].astype(int)
    oil_daily["freq"]      = oil_daily["freq"].astype(int)
    oil_daily["t_work"]    = oil_daily["t_work"].round(1)

    if save_csv:
        # 6. финальные CSV (дословно)
        meta_ppd = ppd_cln.drop_duplicates("well")[["field","well","cluster"]]
        meta_oil = oil_cln.drop_duplicates("well")[["field","well","cluster"]]

        out_ppd = (ppd_daily.merge(meta_ppd, on="well", how="left")
                            [["field","well","cluster","date","d_choke","p_cust","q_ppd"]])
        out_ppd.insert(0, "№ п/п", range(1, len(out_ppd)+1))
        out_ppd = out_ppd.rename(columns={"cluster":"Куст"})
        out_ppd["date"] = pd.to_datetime(out_ppd["date"], dayfirst=False).dt.strftime("%d.%m.%Y")
        out_ppd.to_csv(CLEAN_DIR/"ppd_clean1.csv", index=False, encoding="utf-8-sig")

        out_oil = (oil_daily.merge(meta_oil, on="well", how="left")
                             [["field","well","cluster","date","q_oil","water_cut","p_oil","freq","t_work"]])
        out_oil.insert(0, "№ п/п", range(1, len(out_oil)+1))
        out_oil = out_oil.rename(columns={"cluster":"Куст","water_cut":"watercut","t_work":"Tраб(ТМ)"})
        out_oil["date"] = pd.to_datetime(out_oil["date"], dayfirst=False).dt.strftime("%d.%m.%Y")
        out_oil.to_csv(CLEAN_DIR/"oil_clean1.csv", index=False, encoding="utf-8-sig")

        coords.to_csv(CLEAN_DIR/"coords_clean.csv", index=False, encoding="utf-8-sig")
        print("✓ CSV‑файлы сохранены →", CLEAN_DIR.resolve())

    return ppd_daily, oil_daily, coords

# -------------------------------------------------------------------------
if __name__ == "__main__":
    build_clean_data(save_csv=True)
