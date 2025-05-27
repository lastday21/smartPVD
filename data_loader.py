"""
SmartPVD • DATA LOADER
──────────────────────────────────────────────────────────────────────────────
Читает три исходных Excel-файла (координаты, нефтяные скважины, ППД),
очищает данные через функции из `preprocess.py` и формирует итоговые CSV:
    clean_data/ppd_clean.csv   – № п/п,field,well,Куст,date,d_choke,p_cust,q_ppd
    clean_data/oil_clean.csv   – № п/п,field,well,Куст,date,q_oil,watercut,
                                 p_oil,freq,Tраб(ТМ)
    clean_data/coords_clean.csv – well,x,y
CSV-файлы сохраняются в UTF-8-BOM (Excel friendly).
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from config import (
    PPD_FILE,            # путь к «Параметры ППД …xlsx»
    OIL_FILE,            # путь к «Параметры нефтяной …xlsx»
    COORD_FILE,          # путь к «Координаты…xlsx»
    GAP_LIMIT,           # лимит интерполяции (дней) – нужен в preprocess
)
from preprocess import clean_ppd, clean_oil, resample_and_fill

# Папка назначения для «чистых» CSV
CLEAN_DIR = Path("clean_data")

# ───────────────────────── internal helpers ─────────────────────────

def _find_header(path: Path, probe: int = 20) -> int:
    """Автоматически находим строку-шапку (содержит слово «дата»)."""
    sample = pd.read_excel(path, header=None, nrows=probe, dtype=str)
    for i in range(probe):
        if sample.iloc[i].astype(str).str.contains("дата", case=False, na=False).any():
            return i
    return 0


def _read_raw(path: Path) -> pd.DataFrame:
    """Читаем Excel, нормализуем имена колонок."""
    header = _find_header(path)
    df = pd.read_excel(path, header=header, dtype=str)
    df.columns = df.columns.str.replace("\u00A0", " ").str.strip()
    return df


def _to_float(col: pd.Series) -> pd.Series:
    """Меняем запятую на точку и переводим в float."""
    return pd.to_numeric(col.str.replace(",", ".", regex=False), errors="coerce")


def _daily(df: pd.DataFrame, value: str, *, kind: str) -> pd.DataFrame:
    """Суточный ресемпл столбца `value` с ограниченной интерполяцией."""
    return (
        df.set_index("date")
          .groupby("well")[value]
          .apply(lambda s: resample_and_fill(s, kind=kind))
          .reset_index(name=value)
    )

# ───────────────────────── loaders per sheet ─────────────────────────

def load_ppd(path: Path | str = PPD_FILE) -> pd.DataFrame:
    df = _read_raw(Path(path))
    rename = {
        "№ п/п": "idx",
        "№ скважины": "well",
        "Куст": "cluster",
        "Дата": "date",
        "Dшт": "d_choke",
        "Pкуст": "p_cust",
        "Qприем.Тех": "q_ppd",
        "Мест.": "field",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    for col in ("d_choke", "p_cust", "q_ppd"):
        if col in df.columns:
            df[col] = _to_float(df[col])
    return df.dropna(subset=["well", "date"]).reset_index(drop=True)


def load_oil(path: Path | str = OIL_FILE) -> pd.DataFrame:
    df = _read_raw(Path(path))
    rename = {
        "№ п/п": "idx",
        "№ скважины": "well",
        "Куст": "cluster",
        "Дата": "date",
        "Qж": "q_oil",
        "Обв": "water_cut",
        "Рприем": "p_oil",
        "F вращ ТМ": "freq",
        # Разные варианты написания «Tраб(ТМ)»
        "Траб (ТМ)": "t_work",
        "Траб(ТМ)": "t_work",
        "Tраб (ТМ)": "t_work",
        "Tраб(ТМ)": "t_work",
        "Мест.": "field",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    for col in ("q_oil", "water_cut", "p_oil", "freq", "t_work"):
        if col in df.columns:
            df[col] = _to_float(df[col])
    return df.dropna(subset=["well", "date"]).reset_index(drop=True)


def load_coords(path: Path | str = COORD_FILE) -> pd.DataFrame:
    df = _read_raw(Path(path))
    df = df.rename(columns={
        "Скважина": "well",
        "X": "x",
        "Y": "y",
        "Пласт": "layer",
        "Месторождение": "field",
    })
    keep = [c for c in ("well", "x", "y", "layer", "field") if c in df.columns]
    return df[keep].reset_index(drop=True)

# ───────────────────────── main pipeline ─────────────────────────

def build_clean_files() -> None:
    """Генерирует clean_data/*.csv согласно ТЗ."""

    CLEAN_DIR.mkdir(exist_ok=True)

    # 1️⃣ Чтение исходных файлов
    ppd_raw = load_ppd()
    oil_raw = load_oil()
    coords   = load_coords()

    # 2️⃣ Очистка данных
    ppd_cln = clean_ppd(ppd_raw)
    oil_cln = clean_oil(oil_raw)

    # 3️⃣ Суточные ряды
    ppd_daily   = _daily(ppd_cln, "q_ppd", kind="ppd")
    oil_q_daily = _daily(oil_cln, "q_oil", kind="oil")
    oil_p_daily = _daily(oil_cln, "p_oil", kind="oil")

    # 4️⃣ Паспортные таблицы
    meta_ppd = ppd_cln.drop_duplicates("well")[[
        c for c in ("idx", "well", "field", "cluster", "d_choke", "p_cust") if c in ppd_cln.columns
    ]]
    meta_oil = oil_cln.drop_duplicates("well")[[
        c for c in ("idx", "well", "field", "cluster", "water_cut", "freq", "t_work") if c in oil_cln.columns
    ]]

    # 5️⃣ PPD CSV
    out_ppd = ppd_daily.merge(meta_ppd, on="well", how="left")
    out_ppd = out_ppd[[c for c in (
        "idx", "field", "well", "cluster", "date", "d_choke", "p_cust", "q_ppd"
    ) if c in out_ppd.columns]]
    out_ppd.rename(columns={"idx": "№ п/п", "cluster": "Куст"}, inplace=True)
    out_ppd.to_csv(CLEAN_DIR / "ppd_clean.csv", index=False, encoding="utf-8-sig")

    # 6️⃣ OIL CSV
    out_oil = pd.merge(oil_q_daily, oil_p_daily, on=("well", "date"))
    out_oil = out_oil.merge(meta_oil, on="well", how="left")
    out_oil = out_oil[[c for c in (
        "idx", "field", "well", "cluster", "date",
        "q_oil", "water_cut", "p_oil", "freq", "t_work"
    ) if c in out_oil.columns]]
    out_oil.rename(columns={
        "idx": "№ п/п",
        "cluster": "Куст",
        "water_cut": "watercut",
        "t_work": "Tраб(ТМ)",
    }, inplace=True)
    out_oil.to_csv(CLEAN_DIR / "oil_clean.csv", index=False, encoding="utf-8-sig")

    # 7️⃣ COORDS CSV
    coords[["well", "x", "y"]].to_csv(CLEAN_DIR / "coords_clean.csv", index=False, encoding="utf-8-sig")

    print("✓ CSV-файлы сохранены →", CLEAN_DIR.resolve())


if __name__ == "__main__":
    build_clean_files()  # запускаем, если вызван как скрипт