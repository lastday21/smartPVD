from __future__ import annotations
"""
oil_windows.py
==============

Строит «окна отклика» нефтяных скважин на события ППД и формирует
таблицу *oil_windows.csv* (или возвращает DataFrame в виде функции).

Входные файлы  ──────────────────────────────────────────────────────
clean_data/oil_clean.csv      – суточные Q/P по oil-скважинам  
clean_data/ppd_events.csv     – события на PPD-скважинах  
clean_data/pairs_oil_ppd.csv  – пары вида «ppd_well, oil_well»

Результирующий CSV содержит ровно такие колонки:

    well,
    oil_start, q_start, p_start,
    oil_end,  q_end,  p_end,
    duration_days_oil,
    ppd_well, ppd_start, ppd_end,
    duration_days_ppd

При импорте в пайплайн используем функцию
    build_oil_windows(ppd_events_df, oil_df, mapping_dict) → DataFrame
а при запуске `python oil_windows.py` скрипт автоматически
читает/пишет необходимые CSV.
"""

from pathlib import Path
from typing import Dict, List

import pandas as pd

from window_selector import select_response_window, make_window_passport

# ---------------------------------------------------------------------------
# Пути к входным / выходным файлам
# ---------------------------------------------------------------------------
DATA_DIR          = Path("clean_data")
OIL_CLEAN_PATH    = DATA_DIR / "oil_clean.csv"
PPD_EVENTS_PATH   = DATA_DIR / "ppd_events.csv"
PAIRS_PATH        = DATA_DIR / "pairs_oil_ppd.csv"
WINDOWS_OUT_PATH  = DATA_DIR / "oil_windows.csv"

# ---------------------------------------------------------------------------
# Вспомогательные ридеры
# ---------------------------------------------------------------------------
def _read_csv_dates(
        path: Path,
        *,
        date_cols: Dict[str, str]
) -> pd.DataFrame:
    """
    Читает CSV и конвертирует перечисленные колонки в datetime (day-first).

    date_cols – словарь {старое_имя_колонки : новое_имя_колонки}
                если имя уже нужное, можно указать {'date':'date'}
    """
    df = pd.read_csv(path)
    for old, new in date_cols.items():
        if old in df.columns:
            df = df.rename(columns={old: new})
        df[new] = pd.to_datetime(df[new], dayfirst=True, errors="coerce")
    return df.dropna(subset=date_cols.values())


def _event_dates(ev):
    """Возвращает (start, end) вне зависимости от имён полей в ev."""
    _get = lambda *names: next((getattr(ev, n) for n in names if hasattr(ev, n)), None)
    return (
        pd.to_datetime(_get("start", "start_date")),
        pd.to_datetime(_get("end",   "end_date"))
    )


def _load_oil_df(path: Path) -> pd.DataFrame:
    df = _read_csv_dates(path, date_cols={"date": "date"})
    df["well"] = df["well"].astype(str)
    return df


def _load_ppd_events(path: Path) -> pd.DataFrame:
    df = _read_csv_dates(path, date_cols={"start_date": "start", "end_date": "end"})
    df = df.rename(columns={"well": "ppd_well"})
    df["ppd_well"] = df["ppd_well"].astype(str)
    return df


def _load_pairs(path: Path) -> Dict[str, List[str]]:
    pairs = pd.read_csv(path, dtype={"ppd_well": str, "oil_well": str})
    mapping: Dict[str, List[str]] = {}
    for ppd, oil in pairs[["ppd_well", "oil_well"]].itertuples(index=False):
        mapping.setdefault(ppd, []).append(oil)
    return mapping

# ---------------------------------------------------------------------------
# Основная функция (используется в пайплайне)
# ---------------------------------------------------------------------------
def build_oil_windows(
        ppd_events: pd.DataFrame,
        oil_df: pd.DataFrame,
        mapping: Dict[str, List[str]]
) -> pd.DataFrame:
    """
    Перебирает все события ППД и связанные с ними нефтяные скважины,
    определяет окно отклика и формирует «паспорт» окна.

    ppd_events : DataFrame  – колонки минимум ['ppd_well','start','end', …]
    oil_df     : DataFrame  – суточные ряды, колонки минимум ['well','date',…]
    mapping    : dict       – {ppd_well: [oil_well1, oil_well2, …]}

    Возвращает DataFrame нужного формата (см. cols_order).
    """
    # 1. Готовим oil-DataFrame: datetime + MultiIndex (well, date)
    oil_df["date"] = pd.to_datetime(oil_df["date"], dayfirst=True)
    oil_idx = oil_df.set_index(["well", "date"]).sort_index()

    records: List[dict] = []

    # 2. Обход всех PPD-событий
    for ev in ppd_events.itertuples():
        ppd_start, ppd_end = _event_dates(ev)

        # пробегаемся по всем oil-скважинам, связанным с текущей ppd_well
        for oil_well in mapping.get(str(ev.ppd_well), []):
            try:
                series = oil_idx.loc[oil_well]            # DataFrame индекc=date
            except KeyError:
                continue  # такой скважины нет в oil-датасете

            win = select_response_window(ev, series)
            if win is None:
                continue
            oil_start, oil_end = win

            passport = make_window_passport(series, oil_start, oil_end)
            if passport is None:
                continue

            # дополняем паспорт «служебными» полями
            passport |= {
                "well": oil_well,
                "oil_start": oil_start,
                "oil_end":   oil_end,
                "duration_days_oil": (oil_end - oil_start).days + 1,
                #
                "ppd_well": ev.ppd_well,
                "ppd_start": ppd_start,
                "ppd_end":   ppd_end,
                "duration_days_ppd": (ppd_end - ppd_start).days + 1,
            }
            records.append(passport)

    # 3. Сбор итогового DataFrame – порядок колонок фиксированный
    cols_order = [
        "well",
        "oil_start", "q_start", "p_start",
        "oil_end",   "q_end",   "p_end",
        "duration_days_oil",
        "ppd_well",  "ppd_start", "ppd_end",
        "duration_days_ppd",
    ]
    return pd.DataFrame(records)[cols_order]

# ---------------------------------------------------------------------------
# CLI-запуск: «python oil_windows.py»
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    oil_df      = _load_oil_df(OIL_CLEAN_PATH)
    ppd_events  = _load_ppd_events(PPD_EVENTS_PATH)
    mapping     = _load_pairs(PAIRS_PATH)

    windows_df = build_oil_windows(ppd_events, oil_df, mapping)

    if windows_df.empty:
        print("Не найдено ни одного окна — проверьте входные данные/пары")
    else:
        windows_df.to_csv(WINDOWS_OUT_PATH, index=False, encoding="utf-8-sig")
        print(f"✔ Сформировано окон: {len(windows_df)}  →  {WINDOWS_OUT_PATH}")
