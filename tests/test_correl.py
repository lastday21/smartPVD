"""
Цель — убедиться, что корреляционный модуль строит связи правильно:

* `_best_ccf`            — ищет максимум |ρ| кросс-корреляции ∆-рядов;
* `_calc_corr_df`        — ядро: собирает всё воедино и проставляет
                           категорию ('none' / 'weak' / 'impact');
* `calc_corr` обёртка не проверяется отдельно: она только проксирует вызов.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import pytest

import correl as cr

# ------------------------------------------------------------------ #
#  «ужимаем» требование по точкам, чтобы игрушечные ряды проходили
# ------------------------------------------------------------------ #
@pytest.fixture(autouse=True)
def _patch_min_points(monkeypatch):
    monkeypatch.setattr(cr, "MIN_POINTS_CCF", 5, raising=False)


# ------------------------------------------------------------------ #
#  helpers
# ------------------------------------------------------------------ #
def _series(vals: List[float], start="2025-01-01"):
    idx = pd.date_range(start, periods=len(vals), freq="D")
    return pd.Series(vals, index=idx)


def _make_frames(neg: bool = False):
    """Минимальный набор табличек для пары O1–P1."""
    dates = pd.date_range("2025-01-01", periods=10, freq="D")
    ppd_clean = pd.DataFrame({"well": "P1", "date": dates, "q_ppd": np.arange(10, 20)})
    oil_clean = pd.DataFrame({"well": "O1", "date": dates, "q_oil": np.arange(0, 20, 2)})

    ev_dates = [pd.Timestamp("2025-01-05"), pd.Timestamp("2025-01-08")]
    inj_delta = [20.0, 40.0]
    liq_delta = [-5.0, -10.0] if neg else [5.0, 10.0]

    ppd_events = pd.DataFrame(
        {
            "ppd_well": "P1",
            "start_date": ev_dates,
            "baseline_before": [100.0, 100.0],
            "baseline_during": [b + d for b, d in zip([100.0, 100.0], inj_delta)],
        }
    )

    oil_windows = pd.DataFrame(
        {
            "oil_well": "O1",
            "ppd_well": "P1",
            "ppd_start": ev_dates,
            "q_start": [30.0, 30.0],
            "q_end": [30.0 + d for d in liq_delta],
        }
    )

    pairs = pd.DataFrame({"oil_well": ["O1"], "ppd_well": ["P1"]})

    return ppd_clean, oil_clean, ppd_events, oil_windows, pairs


# ------------------------------------------------------------------ #
# 1. _best_ccf
# ------------------------------------------------------------------ #
def test_best_ccf_perfect_positive():
    inj = _series([0, 1, 3, 6, 10])
    liq = _series([0, 2, 6, 12, 20])
    assert cr._best_ccf(inj, liq) == 1.0


def test_best_ccf_insufficient_points(monkeypatch):
    monkeypatch.setattr(cr, "MIN_POINTS_CCF", 6, raising=False)
    ser = _series([0, 1, 2, 3, 4])
    assert cr._best_ccf(ser, ser) == 0.0


# ------------------------------------------------------------------ #
# 2. _calc_corr_df
# ------------------------------------------------------------------ #
def test_calc_corr_positive(monkeypatch):
    """
    Патчим _best_ccf → +0.8, чтобы гарантировать сильную положительную связь.
    """
    monkeypatch.setattr(cr, "_best_ccf", lambda *_: 0.8, raising=False)

    frames = _make_frames(neg=False)
    df = cr._calc_corr_df(*frames)
    row = df.iloc[0]

    assert row["event_corr"] == pytest.approx(1.0)
    assert row["ccf_corr"] == pytest.approx(0.8)
    assert row["corr_cat"] == "impact"        # положительная, без штрафа


def test_calc_corr_negative(monkeypatch):
    """
    Отрицательная event_corr и нулевая ccf → итоговая категория «weak».
    """
    monkeypatch.setattr(cr, "_best_ccf", lambda *_: 0.0, raising=False)

    frames = _make_frames(neg=True)
    df = cr._calc_corr_df(*frames)
    row = df.iloc[0]

    assert row["event_corr"] == pytest.approx(-1.0)
    assert row["corr_cat"] == "weak"          # impact (2) – penalty (1) = weak