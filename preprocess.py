"""
preprocess_refactored.py — модуль пред‑обработки исходных данных ППД и добычи нефти
для проекта *smartPVD*.

🛠 **Ключевая идея рефакторинга**
---------------------------------
Модуль разведен на две независимые части:

1. `_build_clean_data_df`— *чистое ядро* (pure‑function), которое **не выполняет
   никакого ввода‑вывода**, а работает исключительно с уже загруженными
   `pandas.DataFrame`. Оно проводит всю логику очистки, ресемплинга и
   агрегации, возвращая необходимые DataFrame‑ы для дальнейшего пайплайна.
2. `build_clean_data`— *универсальный интерфейс* (обёртка), который в режиме
   **in‑memory** принимает входные DataFrame‑ы, а в режиме **CLI**/отладки сам
   загружает исходные Excel/CSV‑файлы, опционально сохраняет промежуточные CSV и
   полностью повторяет прежнее поведение старой функции (совместимость 100%).

Таким образом:
• Вконвейере SmartPVD вы вызываете `build_clean_data(ppd_df=..., oil_df=...,
  coords_df=..., save_csv=False)` ➜ никакого I/Oмежду модулями.
• Для ручной проверки достаточен `python preprocess_refactored.py` — будут
  прочитаны исходники и сохранены готовые CSV в`clean_data/`.
"""
from __future__ import annotations

from pathlib import Path
import re
import pandas as pd
import numpy as np

# --------------------------- конфигурация -----------------------------
from config import (
    PPD_FILE, OIL_FILE, COORD_FILE,
    PPD_SHEET_NAME, OIL_SHEET_NAME,
    GAP_LIMIT, FREQ_THRESH, MIN_WORK_PPD, NO_PRESS_WITH_Q_LIMIT,
)

# Папка для CSV‑файлов отладки
CLEAN_DIR = Path("clean_data")
CLEAN_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------
# ⬇️  Вспомогательные функции (без изменений логики)
# ---------------------------------------------------------------------

def _num(col: pd.Series) -> pd.Series:
    """Преобразует строковый формат чисел «1234,56» → 1234.56 (float)."""
    return pd.to_numeric(
        col.astype(str)
           .str.replace("\u00A0", "", regex=False)  # неразрывный пробел
           .str.replace(" ",     "", regex=False)
           .str.replace(",",     ".", regex=False),
        errors="coerce",
    )


def _interp_bf_ff(s: pd.Series) -> pd.Series:
    """Интерп. пропусков ≤ *GAP_LIMIT* дней, затем backfill+forwardfill."""
    return s.interpolate(limit=GAP_LIMIT, limit_direction="both").bfill().ffill()


def _bf_ff(s: pd.Series) -> pd.Series:
    """Простое заполнение: `bfill` → `ffill`."""
    return s.bfill().ffill()

# ---------------------------------------------------------------------
# ⬇️  Очистка данных ППД
# ---------------------------------------------------------------------

def clean_ppd(df: pd.DataFrame) -> pd.DataFrame:
    """Очистка и нормализация сырых рядов приёма жидкости (ППД).

    Параметры
    ----------
    df : pd.DataFrame
        Исходная таблица с колонками («дата», «q_ppd», «p_cust», «d_choke», …).

    Возвращает
    ----------
    pd.DataFrame
        Очищенный датафрейм, где пропуски заполнены, значения отфильтрованы, а
        типы приведены к *int*.
    """
    wells: list[pd.DataFrame] = []

    # 1) Приведение ключевых колонок к float
    for col in ("q_ppd", "p_cust", "d_choke"):
        df[col] = _num(df.get(col, pd.NA))

    # 2) Обработка по каждой скважине отдельно —логика прежняя
    for _, sub in df.groupby("well", sort=False):
        sub = sub.reset_index(drop=True)

        # 2.1 Диаметр штуцера и фильтрация расхода
        sub["d_choke"] = sub["d_choke"].ffill().bfill().fillna(0)
        sub["q_ppd"]   = sub["q_ppd"].where(sub["q_ppd"] >= MIN_WORK_PPD, np.nan)

        # 2.2 Построение флага «скважина в работе»
        work_flag: list[bool] = []
        in_work, no_pq, no_both = False, 0, 0
        for _, row in sub.iterrows():
            has_p = row["p_cust"] > 0
            has_q = row["q_ppd"] >= MIN_WORK_PPD

            # запуск рабочего режима
            if not in_work and has_p:
                in_work = True
            # внутри рабочего режима —отслеживаем пропуски давления/расхода
            if in_work:
                if not has_p and has_q:
                    no_pq += 1
                    if no_pq >= NO_PRESS_WITH_Q_LIMIT:
                        # откатываем предыдущие флаги
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

        # 2.3 Заполнение давления внутри рабочих интервалов
        last_p, p_out = 0.0, []
        for i, row in sub.iterrows():
            if sub.at[i, "work"]:
                if row["p_cust"] > 0:
                    last_p = row["p_cust"]
                p_out.append(last_p)
            else:
                last_p = 0.0; p_out.append(0.0)
        sub["p_cust"] = pd.Series(p_out, index=sub.index)

        # 2.4 Заполнение расхода внутри рабочего интервала
        out_q = pd.Series(0.0, index=sub.index)
        grp = sub["work"].ne(sub["work"].shift()).cumsum()
        for _, idx in sub.groupby(grp, sort=False).groups.items():
            if sub.loc[idx, "work"].iat[0]:
                seg = sub.loc[idx, "q_ppd"].bfill().ffill().fillna(0)
                out_q.loc[idx] = seg
        sub["q_ppd"] = out_q

        # 2.5 Финальное округление и сохранение
        sub["d_choke"] = sub["d_choke"].astype(int)
        sub["p_cust"]  = sub["p_cust"].round(0).astype(int)
        sub["q_ppd"]   = sub["q_ppd"].round(0).astype(int)
        wells.append(sub.drop(columns="work"))

    return pd.concat(wells, ignore_index=True)

# ---------------------------------------------------------------------
# ⬇  Очистка данных добычи нефти
# ---------------------------------------------------------------------

def clean_oil(df: pd.DataFrame) -> pd.DataFrame:
    """Очистка суточных данных добычи нефти.

    Алгоритм в точности повторяет старую логику: фильтруем рабочие интервалы по
    частоте/объёму, заполняем пропуски, округляем значения.
    """
    out: list[pd.DataFrame] = []

    # 1) Нормализация форматов чисел
    for col in ("q_oil", "water_cut", "p_oil", "freq", "t_work"):
        df[col] = _num(df.get(col, pd.NA))

    # 2) По каждой скважине
    for _, group in df.groupby("well", sort=False):
        sub = group.reset_index(drop=True)

        # 2.1 Флаг «показания есть»
        raw_flag = (sub["freq"] > FREQ_THRESH) | ((sub["q_oil"] > 0) & (sub["t_work"] > 0))
        work: list[bool] = []
        miss, in_work = 0, False
        for is_raw in raw_flag:
            if not in_work and is_raw:
                in_work = True; miss = 0
            elif in_work and not is_raw:
                miss += 1
                if miss >= 5:
                    # откат флага
                    start = len(work) - miss + 1
                    for i in range(start, len(work)):
                        work[i] = False
                    in_work = False; miss = 0; continue
            else:
                miss = 0
            work.append(in_work)
        work.extend([False] * (len(sub) - len(work)))
        sub["work"] = work

        # 2.2 Заполнение пропусков по группам колонок
        for col in ("water_cut", "freq", "t_work"):
            filled = sub[col].where(sub["work"]).pipe(_bf_ff)
            sub[col] = filled.where(sub["work"], 0)
        for col in ("q_oil", "p_oil"):
            filled = sub[col].where(sub["work"]).pipe(_interp_bf_ff)
            sub[col] = filled.where(sub["work"], 0)

        # 2.3 Округление и финал
        sub["water_cut"] = sub["water_cut"].round(0).astype(int)
        sub["freq"]       = sub["freq"].round(0).astype(int)
        sub["t_work"]     = sub["t_work"].round(1)
        sub["p_oil"]      = sub["p_oil"].round(1)
        out.append(sub.drop(columns="work"))

    return pd.concat(out, ignore_index=True)

# ---------------------------------------------------------------------
# ⬇  Суточный ресемплинг (без изменений)
# ---------------------------------------------------------------------

def resample_and_fill(series: pd.Series, *, kind: str) -> pd.Series:
    """Суточный ресемплинг + заполнение пропусков (логика прежняя)."""
    daily = series.resample("D").asfreq()
    name  = series.name
    if name in ("water_cut", "freq", "t_work"):
        return _bf_ff(daily)
    if name == "d_choke":
        return daily.ffill().bfill()
    if daily.isna().all():
        return pd.Series(0, index=daily.index, name=name)
    return _interp_bf_ff(daily)

# ---------------------------------------------------------------------
# ⬇  Чтение «сырого» Excel/CSV —остаются нетронутыми и используются только
#     во внешнем интерфейсе при отсутствии уже загруженных DataFrame‑ов.
# ---------------------------------------------------------------------

def _find_header(path: Path, probe: int = 20) -> int:
    """Ищет строку заголовка по наличию слова «дата»."""
    sample = pd.read_excel(path, header=None, nrows=probe, dtype=str)
    for i in range(probe):
        if sample.iloc[i].astype(str).str.contains("дата", case=False, na=False).any():
            return i
    return 0


def _read_raw(path: Path, sheet_name: str) -> pd.DataFrame:
    """Чтение листа Excel с автопоиском шапки + очистка имён колонок."""
    header = _find_header(path)
    df = pd.read_excel(path, sheet_name=sheet_name, header=header, dtype=str)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df.columns = df.columns.str.replace("\u00A0", " ").str.strip()
    for col in df.columns:
        if re.search(r"(?i)дата", col):
            df = df.rename(columns={col: "date"}); break
    df["date"] = pd.to_datetime(df["date"], dayfirst=False, errors="coerce")
    return df


def _ru(s: str) -> str:
    """Утилита: русские имена → нормализованная форма (lower, без пробелов)."""
    return s.lower().replace("\u00a0", " ").replace(" ", "")

# Маппинг колонок
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
    """Читает Excel ППД и переименовывает колонки под стандарт."""
    df = _read_raw(Path(path), PPD_SHEET_NAME)
    df = df.rename({c: RU2ENG_PPD[_ru(c)] for c in df.columns if _ru(c) in RU2ENG_PPD}, axis=1)
    return df


def load_oil(path: Path | str = OIL_FILE) -> pd.DataFrame:
    """Читает Excel добычи нефти и переименовывает колонки."""
    df = _read_raw(Path(path), OIL_SHEET_NAME)
    df = df.rename({c: RU2ENG_OIL[_ru(c)] for c in df.columns if _ru(c) in RU2ENG_OIL}, axis=1)
    for col in df.columns:
        if re.search(r"(?i)\(тм\)|раб", col):
            df = df.rename(columns={col: "t_work"}); break
    return df


def load_coords(path: Path | str = COORD_FILE) -> pd.DataFrame:
    """Читает таблицу координат (X, Y)."""
    return (
        pd.read_excel(path)
          .rename(columns={"Скважина": "well", "X": "x", "Y": "y"})
          [["well", "x", "y"]]
    )

# ---------------------------------------------------------------------
# ⬇️  Вспом. функция для суточного ресемплинга колонки
# ---------------------------------------------------------------------

def _daily(df: pd.DataFrame, col: str, *, kind: str) -> pd.DataFrame:
    """Группирует по скважине и ресемплит указанную колонку на сутки."""
    return (
        df.set_index("date")
          .groupby("well", sort=False)[col]
          .apply(lambda s: resample_and_fill(s, kind=kind))
          .reset_index()
    )

# =====================================================================
#                         Ч И С Т О Е   Я Д Р О
# =====================================================================

def _build_clean_data_df(
    ppd_raw: pd.DataFrame,
    oil_raw: pd.DataFrame,
    coords: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Собирает полный чистый датасет *без* файловых операций.

    Параметры
    ----------
    ppd_raw : pd.DataFrame
        Сырые данные ППД после `load_ppd` (русские колонки уже переименованы).
    oil_raw : pd.DataFrame
        Сырые данные добычи нефти после `load_oil`.
    coords : pd.DataFrame
        Координаты скважин (`well`, `x`, `y`).

    Возвращает
    ----------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        `ppd_daily`, `oil_daily`, `coords` —готовые суточные ряды.
    """

    # 1) Очистка (без I/O)
    ppd_cln = clean_ppd(ppd_raw)
    oil_cln = clean_oil(oil_raw)

    # 2) Суточные ряды ППД
    ppd_q  = _daily(ppd_cln, "q_ppd",  kind="ppd")
    ppd_p  = _daily(ppd_cln, "p_cust", kind="ppd")
    ppd_d  = _daily(ppd_cln, "d_choke", kind="ppd")
    ppd_daily = ppd_q.merge(ppd_p, on=("well", "date")).merge(ppd_d, on=("well", "date"))

    # 3) Суточные ряды нефти
    oil_q  = _daily(oil_cln, "q_oil",     kind="oil")
    oil_p  = _daily(oil_cln, "p_oil",     kind="oil")
    oil_wc = _daily(oil_cln, "water_cut", kind="oil")
    oil_f  = _daily(oil_cln, "freq",      kind="oil")
    oil_tw = _daily(oil_cln, "t_work",    kind="oil")
    oil_daily = (
        oil_q.merge(oil_p, on=("well", "date"))
              .merge(oil_wc, on=("well", "date"))
              .merge(oil_f,  on=("well", "date"))
              .merge(oil_tw, on=("well", "date"))
    )

    # 4) Приведение типов/округление (как раньше)
    ppd_daily[["q_ppd", "p_cust", "d_choke"]] = ppd_daily[["q_ppd", "p_cust", "d_choke"]].astype(int)
    oil_daily["q_oil"]     = oil_daily["q_oil"].astype(int)
    oil_daily["water_cut"] = oil_daily["water_cut"].astype(int)
    oil_daily["freq"]      = oil_daily["freq"].astype(int)
    oil_daily["t_work"]    = oil_daily["t_work"].round(1)

    return ppd_daily, oil_daily, coords

# =====================================================================
#                    У Н И В Е Р С А Л Ь Н Ы Й   И Н Т Е Р Ф Е Й С
# =====================================================================

def build_clean_data(
    *,
    ppd_df: pd.DataFrame | None = None,
    oil_df: pd.DataFrame | None = None,
    coords_df: pd.DataFrame | None = None,
    ppd_path: str | Path | None = None,
    oil_path: str | Path | None = None,
    coords_path: str | Path | None = None,
    save_csv: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Полный конвейер пред‑обработки.

    Параметры
    ----------
    ppd_df, oil_df, coords_df : pd.DataFrame | None
        Уже загруженные DataFrame‑ы. Если переданы, **файлы не читаем**.
    ppd_path, oil_path, coords_path : str | Path | None
        Пути к исходным Excel/CSV. Используются, когда соответствующий
        `*_df` не передан. По умолчанию берутся из глобального конфига.
    save_csv : bool, default **True**
        Если *True* — сохраняет `ppd_clean.csv`, `oil_clean.csv`, `coords_clean.csv`
        в `clean_data/` (поведение старой версии для отладки).

    Возвращает
    ----------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Готовые суточные ряды: `ppd_daily`, `oil_daily` и `coords_df`.
    """

    # 1) Получаем исходные данные (файлы либо уже загруженные DF)
    if ppd_df is None:
        ppd_df = load_ppd(ppd_path or PPD_FILE)
    if oil_df is None:
        oil_df = load_oil(oil_path or OIL_FILE)
    if coords_df is None:
        coords_df = load_coords(coords_path or COORD_FILE)

    # 2) Чистое ядро — никаких файловых операций внутри
    ppd_daily, oil_daily, coords_df = _build_clean_data_df(ppd_df, oil_df, coords_df)

    # 3) Сохраняем CSV для отладки (при необходимости)
    if save_csv:
        # метаданные (поле, куст) берём из *очищенных* исходников
        meta_ppd = clean_ppd(ppd_df).drop_duplicates("well")[["field", "well", "cluster"]]
        meta_oil = clean_oil(oil_df).drop_duplicates("well")[["field", "well", "cluster"]]

        # ----- ППД -----------------------------------------------------
        out_ppd = (
            ppd_daily.merge(meta_ppd, on="well", how="left")
                     [["field", "well", "cluster", "date", "d_choke", "p_cust", "q_ppd"]]
        )
        out_ppd.insert(0, "№ п/п", range(1, len(out_ppd) + 1))
        out_ppd = out_ppd.rename(columns={"cluster": "Куст"})
        out_ppd["date"] = pd.to_datetime(out_ppd["date"]).dt.strftime("%d.%m.%Y")
        out_ppd.to_csv(CLEAN_DIR / "ppd_clean.csv", index=False, encoding="utf-8-sig")

        # ----- Нефть ---------------------------------------------------
        out_oil = (
            oil_daily.merge(meta_oil, on="well", how="left")
                     [["field", "well", "cluster", "date", "q_oil", "water_cut", "p_oil", "freq", "t_work"]]
        )
        out_oil.insert(0, "№ п/п", range(1, len(out_oil) + 1))
        out_oil = out_oil.rename(columns={
            "cluster": "Куст",
            "water_cut": "watercut",
            "t_work": "Tраб(ТМ)",
        })
        out_oil["date"] = pd.to_datetime(out_oil["date"]).dt.strftime("%d.%m.%Y")
        out_oil.to_csv(CLEAN_DIR / "oil_clean.csv", index=False, encoding="utf-8-sig")

        # ----- Координаты ---------------------------------------------
        coords_df.to_csv(CLEAN_DIR / "coords_clean.csv", index=False, encoding="utf-8-sig")
        print(f"✓ CSV‑файлы сохранены → {CLEAN_DIR.resolve()}")

    return ppd_daily, oil_daily, coords_df

# ---------------------------------------------------------------------
# ⬇  CLI‑режим (запуск модуля напрямую) — полностью эквивалентен прежнему.
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # При ручном запуске читаем Excel и сохраняем промежуточные CSV
    build_clean_data(save_csv=True)
