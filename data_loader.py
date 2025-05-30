"""
Скрипт чтения исходных Excel’ев, очистки рядов и сохранения чистых CSV:
– load_ppd, load_oil, load_coords — парсинг и первичное переименование,
– build_clean_files — основная последовательность сборки clean_data/*.csv.
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import re

from config import (
    PPD_FILE, OIL_FILE, COORD_FILE,
    PPD_SHEET_NAME, OIL_SHEET_NAME
)
from preprocess import clean_ppd, clean_oil, resample_and_fill

CLEAN_DIR = Path("clean_data")
CLEAN_DIR.mkdir(exist_ok=True)


def _find_header(path: Path, probe: int = 20) -> int:
    """
    Ищем строку, где впервые встречается слово «дата»,
    чтобы передать её в качестве header при read_excel.
    """
    sample = pd.read_excel(path, header=None, nrows=probe, dtype=str)
    for i in range(probe):
        if sample.iloc[i].astype(str).str.contains("дата", case=False, na=False).any():
            return i
    return 0


def _read_raw(path: Path, sheet_name) -> pd.DataFrame:
    """
    Читает лист Excel, отбрасывает колонки Unnamed, нормализует заголовок date,
    и конвертирует его в datetime.
    """
    header = _find_header(path)
    df = pd.read_excel(path, sheet_name=sheet_name, header=header, dtype=str)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df.columns = df.columns.str.replace("\u00A0", " ").str.strip()
    for col in df.columns:
        if col.lower().startswith("дата"):
            df = df.rename(columns={col: "date"})
            break
    df["date"] = pd.to_datetime(df["date"], dayfirst=False, errors="coerce")
    return df


def _ru(s: str) -> str:
    """Убираем пробелы и приводим к нижнему регистру для key-lookup."""
    return s.lower().replace("\u00a0", " ").replace(" ", "")


# Расширенный маппинг для кириллицы и латиницы
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
    "qприем.тех": "q_ppd"
}


def load_ppd(path: Path | str = PPD_FILE) -> pd.DataFrame:
    df = _read_raw(Path(path), PPD_SHEET_NAME)
    df = df.rename(columns={
        c: RU2ENG_PPD[_ru(c)]
        for c in df.columns
        if _ru(c) in RU2ENG_PPD
    })
    return df


def load_oil(path: Path | str = OIL_FILE) -> pd.DataFrame:
    df = _read_raw(Path(path), OIL_SHEET_NAME)
    df = df.rename(columns={
        c: RU2ENG_OIL[_ru(c)]
        for c in df.columns
        if _ru(c) in RU2ENG_OIL
    })

    #  дополнительный «ловец» t_work по regex
    for col in df.columns:
        if re.search(r'(?i)\(тм\)|раб', col):
            df = df.rename(columns={col: "t_work"})
            break

    return df


def load_coords(path: Path | str = COORD_FILE) -> pd.DataFrame:
    """
    Чтение координат: ожидаем колонки «Скважина», «X», «Y» → well, x, y.
    """
    df = pd.read_excel(path).rename(columns={"Скважина": "well", "X": "x", "Y": "y"})
    return df[["well", "x", "y"]]


def _daily(df: pd.DataFrame, col: str, *, kind: str) -> pd.DataFrame:
    """
    Для каждой well ресемплим колонку col на ежедневную частоту
    с помощью resample_and_fill.
    Сохраняется порядок well.
    """
    return (
        df.set_index("date")
          .groupby("well", sort=False)[col]
          .apply(lambda s: resample_and_fill(s, kind=kind))
          .reset_index()
    )


def build_clean_files() -> None:
    """
    Основной конвейер:
    1) load_ppd/load_oil/load_coords
    2) clean_ppd, clean_oil
    3) суточный ресемпл _daily для PPD и OIL
    4) кастинг всех числовых полей в нужные типы
    5) сборка финальных таблиц + добавление порядкового № п/п
    6) сохранение CSV (utf-8-sig, без индекса)
    """
    # 1. Читаем исходники
    ppd_raw = load_ppd()
    oil_raw = load_oil()
    coords  = load_coords()

    # 2. Грубая очистка
    ppd_cln = clean_ppd(ppd_raw)
    oil_cln = clean_oil(oil_raw)

    # 3. Суточные ряды PPD
    ppd_q  = _daily(ppd_cln, "q_ppd",  kind="ppd")
    ppd_p  = _daily(ppd_cln, "p_cust", kind="ppd")
    ppd_d  = _daily(ppd_cln, "d_choke", kind="ppd")
    ppd_daily = (
        ppd_q.merge(ppd_p, on=("well", "date"))
             .merge(ppd_d, on=("well", "date"))
    )

    # 4. Суточные ряды OIL
    oil_q  = _daily(oil_cln, "q_oil",     kind="oil")
    oil_p  = _daily(oil_cln, "p_oil",     kind="oil")
    oil_wc = _daily(oil_cln, "water_cut", kind="oil")
    oil_f  = _daily(oil_cln, "freq",      kind="oil")
    oil_tw = _daily(oil_cln, "t_work",    kind="oil")
    oil_daily = (
        oil_q.merge(oil_p,  on=("well", "date"))
             .merge(oil_wc, on=("well", "date"))
             .merge(oil_f,  on=("well", "date"))
             .merge(oil_tw, on=("well", "date"))
    )

    # 5. Кастинг типов сразу после ресемпла
    ppd_daily[["q_ppd","p_cust","d_choke"]] = ppd_daily[["q_ppd","p_cust","d_choke"]].astype(int)
    oil_daily["q_oil"]     = oil_daily["q_oil"].astype(int)
    oil_daily["water_cut"] = oil_daily["water_cut"].astype(int)
    oil_daily["freq"]      = oil_daily["freq"].astype(int)
    oil_daily["t_work"]    = oil_daily["t_work"].round(1)

    # 6. «Паспорта» для PPD и OIL (field, well, cluster)
    meta_ppd = ppd_cln.drop_duplicates("well")[["field","well","cluster"]]
    meta_oil = oil_cln.drop_duplicates("well")[["field","well","cluster"]]

    # 7. Финальный PPD CSV
    out_ppd = ppd_daily.merge(meta_ppd, on="well", how="left")
    out_ppd = out_ppd[["field","well","cluster","date","d_choke","p_cust","q_ppd"]]
    out_ppd.insert(0, "№ п/п", range(1, len(out_ppd)+1))
    out_ppd = out_ppd.rename(columns={"cluster":"Куст"})
    out_ppd["date"] = pd.to_datetime(out_ppd["date"], dayfirst=False).dt.strftime("%d.%m.%Y")
    out_ppd.to_csv(CLEAN_DIR/"ppd_clean.csv", index=False, encoding="utf-8-sig")

    # 8. Финальный OIL CSV
    out_oil = oil_daily.merge(meta_oil, on="well", how="left")
    out_oil = out_oil[["field","well","cluster","date","q_oil","water_cut","p_oil","freq","t_work"]]
    out_oil.insert(0, "№ п/п", range(1, len(out_oil)+1))
    out_oil = out_oil.rename(columns={"cluster":"Куст","water_cut":"watercut","t_work":"Tраб(ТМ)"})
    out_oil["date"] = pd.to_datetime(out_oil["date"], dayfirst=False).dt.strftime("%d.%m.%Y")
    out_oil.to_csv(CLEAN_DIR/"oil_clean.csv", index=False, encoding="utf-8-sig")

    # 9. Координаты
    coords.to_csv(CLEAN_DIR/"coords_clean.csv", index=False, encoding="utf-8-sig")

    print("✓ CSV-файлы сохранены →", CLEAN_DIR.resolve())


if __name__ == "__main__":
    build_clean_files()
    oil_raw = load_oil()
    print(">>> Oil raw columns:", oil_raw.columns.tolist())
