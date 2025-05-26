"""
Загрузка суточных выгрузок SmartPVD → очистка → сохранение CSV.

Создаёт 3 файла в каталоге, заданном ключом ``--outdir``
(по-умолчанию ``clean_data``):

    clean_data/ppd_clean.csv
    clean_data/oil_clean.csv
    clean_data/coords_clean.csv
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Iterable, Mapping, Optional
import config

# ──────────────────────────────────────────────────────────────────────────────
# Пути и константы
# ──────────────────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────────────────────
# Вспомогательные утилиты
# ──────────────────────────────────────────────────────────────────────────────
def _detect_header_row(path: Path, keyword: str = "Дата", search_rows: int = 15) -> int:
    sample = pd.read_excel(path, header=None, nrows=search_rows)
    for i in range(search_rows):
        if sample.iloc[i].astype(str).str.contains(keyword, case=False, na=False).any():
            return i
    return 0


def _read_excel(path: Path, rename_map: Mapping[str, str]) -> pd.DataFrame:
    header = _detect_header_row(path)
    df = pd.read_excel(path, header=header)
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")

    for col in df.columns.difference(["date", "field"]):
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", ".", regex=False)
            .replace({"": np.nan, "nan": np.nan})
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _interpolate_short_gaps(s: pd.Series, limit: int = config.GAP_LIMIT) -> pd.Series:
    return (
        s.astype(float)
        .interpolate(limit=limit, limit_area="inside")
        .ffill(limit=limit)
    )

# ──────────────────────────────────────────────────────────────────────────────
# PPD
# ──────────────────────────────────────────────────────────────────────────────
def load_ppd(path: str | Path | None = None,
             wells: Optional[Iterable[str]] = None) -> pd.DataFrame:
    path = Path(path or config.PPD_FILE)
    df = _read_excel(path, {
        "Дата": "date", "№ скважины": "well",
        "Qприем.Тех": "q_ppd", "Pкуст": "p_cust",
        "Dшт": "d_choke", "Мест.": "field",
    })

    if wells:
        df = df[df["well"].isin(wells)]
    df = df.sort_values(["well", "date"]).reset_index(drop=True)

    def _clean(grp: pd.DataFrame) -> pd.DataFrame:
        grp = grp.copy()
        grp[["q_ppd", "p_cust"]] = grp[["q_ppd", "p_cust"]].astype(float)
        grp["field"] = grp["field"].bfill().ffill()

        glitch = (grp["q_ppd"].eq(0)) & (grp["p_cust"] > 0)
        roll   = grp["q_ppd"].replace(0, np.nan).rolling(config.WINDOW_DAYS, 1).mean()
        grp.loc[glitch, "q_ppd"] = roll[glitch]

        no_press = grp["p_cust"].fillna(0).eq(0) & grp["q_ppd"].lt(config.MIN_WORK_PPD)
        grp.loc[no_press, "q_ppd"] = 0.0

        grp["q_ppd"] = _interpolate_short_gaps(grp["q_ppd"]).fillna(0)
        grp["p_cust"] = _interpolate_short_gaps(grp["p_cust"])
        grp["d_choke"] = grp["d_choke"].ffill()
        return grp

    return df.groupby("well", group_keys=False).apply(_clean)


# ──────────────────────────────────────────────────────────────────────────────
# OIL
# ──────────────────────────────────────────────────────────────────────────────
def load_oil(path: str | Path | None = None,
             wells: Optional[Iterable[str]] = None) -> pd.DataFrame:
    path = Path(path or config.OIL_FILE)
    df = _read_excel(path, {
        "Дата": "date", "№ скважины": "well",
        "Qж": "q_oil", "Рприем": "p_oil",
        "F вращ ТМ": "freq", "Траб(ТМ)": "t_work",
        "Обв": "watercut", "Мест.": "field",
        "Обв ХАЛ": "_drop",
    })
    df = df[[c for c in df.columns if not c.startswith("_drop")]]

    if wells:
        df = df[df["well"].isin(wells)]
    df = df.sort_values(["well", "date"]).reset_index(drop=True)

    def _clean(grp: pd.DataFrame) -> pd.DataFrame:
        grp = grp.copy()
        grp[["q_oil", "p_oil", "freq"]] = grp[["q_oil", "p_oil", "freq"]].astype(float)
        grp["field"] = grp["field"].bfill().ffill()

        glitch = grp["q_oil"].eq(0) & grp["freq"].ge(config.FREQ_THRESH)
        roll   = grp["q_oil"].replace(0, np.nan).rolling(config.WINDOW_DAYS, 1).mean()
        grp.loc[glitch, "q_oil"] = roll[glitch]

        grp["p_oil"] = _interpolate_short_gaps(grp["p_oil"]).fillna(0)
        if "watercut" in grp.columns:
            grp["watercut"] = grp["watercut"].bfill().ffill()
        return grp

    return df.groupby("well", group_keys=False).apply(_clean)


# ──────────────────────────────────────────────────────────────────────────────
# COORDS
# ──────────────────────────────────────────────────────────────────────────────
def load_coords(path: str | Path | None = None) -> pd.DataFrame:
    path = Path(path or config.COORD_FILE)
    df = pd.read_excel(path).rename(columns={"Скважина": "well", "X": "x", "Y": "y"})
    return df[["well", "x", "y"]]


# ──────────────────────────────────────────────────────────────────────────────
# CLI → сохранить CSV
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Clean SmartPVD data → CSV")
    ap.add_argument("--outdir", default="clean_data",
                    help="Каталог, куда положить ppd/oil/coords CSV")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    load_ppd().to_csv(outdir / "ppd_clean.csv",  index=False, encoding="utf-8")
    load_oil().to_csv(outdir / "oil_clean.csv",  index=False, encoding="utf-8")
    load_coords().to_csv(outdir / "coords_clean.csv", index=False, encoding="utf-8")

    print(f"✓ CSV-файлы сохранены в «{outdir.resolve()}»")
