# ────────────────────────────── preprocess.py ──────────────────────────────
"""
Pre‑processing utilities for SmartPVD.

• clean_ppd(df) – очистка рядов ППД (глюки расхода, остановы, интерполяция).
• clean_oil(df) – базовая очистка добывающих (шумовые нули, t_work, …).
• resample_and_fill(series, kind) – суточный ресемпл + интерполяция ≤ GAP_LIMIT.

Все функции независимы от путей к файлам – только DataFrame‑входы.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Literal

from config import GAP_LIMIT, FREQ_THRESH, MIN_WORK_PPD

__all__ = [
    "clean_ppd",
    "clean_oil",
    "resample_and_fill",
]

# ───────────────────────────── helpers ─────────────────────────────

def resample_and_fill(series: pd.Series, *, kind: Literal["ppd", "oil"]) -> pd.Series:
    """Daily resample → limited linear interp (≤GAP_LIMIT) → fill NaN=0."""
    daily = series.resample("D").mean()
    filled = daily.interpolate(limit=GAP_LIMIT, limit_direction="both")
    return filled.fillna(0)

# ───────────────────────────── cleaners ─────────────────────────────

def clean_ppd(df: pd.DataFrame) -> pd.DataFrame:
    """Убираем шумы расходомера ППД, оставляя реальные остановы нулём."""
    df = df.copy()
    if "q_ppd" not in df.columns:
        return df  # нечего чистить

    # 1) Глюк: расход < MIN_WORK_PPD, а давление есть → NaN (потом ffill)
    glitch = (df["q_ppd"] < MIN_WORK_PPD) & (df.get("p_cust", 0) > 0)
    df.loc[glitch, "q_ppd"] = np.nan

    # 2) Реальный останов: расход низкий и давления нет → 0
    stop = (df["q_ppd"] < MIN_WORK_PPD) & (df.get("p_cust", 0) <= 0)
    df.loc[stop, "q_ppd"] = 0.0

    # Заполняем мелкие разрывы
    df["q_ppd"] = df.groupby("well")["q_ppd"].ffill().bfill()
    return df


def clean_oil(df: pd.DataFrame) -> pd.DataFrame:
    """Базовая очистка нефте‑скважин; безопасна, если каких‑то колонок нет."""
    df = df.copy()

    # — Q_oil: шумовой ноль, когда частота > порога
    if {"q_oil", "freq"}.issubset(df.columns):
        noise = (df["q_oil"] == 0) & (df["freq"] > FREQ_THRESH)
        df.loc[noise, "q_oil"] = np.nan
        df.loc[df["freq"] <= FREQ_THRESH, "q_oil"] = 0.0

    # — t_work: подставляем последний ненулевой, если насос реально работал
    if {"t_work", "freq"}.issubset(df.columns):
        mask = (df["freq"] > FREQ_THRESH) & (df["t_work"].fillna(0) == 0)
        df.loc[mask, "t_work"] = np.nan
        df["t_work"] = df.groupby("well")["t_work"].ffill().fillna(0)

    # p_oil не трогаем здесь – дневная интерполяция идёт позже в data_loader
    return df
