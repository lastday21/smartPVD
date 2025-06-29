"""
1.  **_ci_value** – формула. Прогоняем все комбинации знаков ΔQ и ΔPприём.
2.  **_compute_ci_df** – «игрушечный» датасет: одна пара «нефть – ППД»,
    чтобы CI можно было посчитать вручную.
3.  **compute_ci** – фасад должен вернуть те же data-frames, что и ядро.
4.  Экспоненциальное затухание: убеждаемся, что при distance_mode="exp"
    CI реально уменьшается по e^{-d/λ}.

"""
from __future__ import annotations

import math

import pandas as pd
import pytest

import metrics as mt
import config as cfg


# ----------------------------------------------------------------------
# Вспомогалки
# ----------------------------------------------------------------------
def _ci_manual(dp: float, dq: float, dp_o: float) -> float:
    """
    То же, что делает `_ci_value`, только расписано «на бумажке».
    """
    x = dp * (dq / cfg.divider_q)
    y = dp * (dp_o / cfg.divider_p)

    if x >= 0 and y >= 0:
        return cfg.w_q * x + cfg.w_p * y
    if x >= 0:
        return x
    if y >= 0:
        return y
    return 0.0


def _toy_frames(distance: float = 100.0):
    """
    Мини-датасет с одной парой (O1, P1). Данных достаточно, чтобы ядро
    `_compute_ci_df` отработало без единой ветки `if len(df)==0`.
    """
    df_ow = pd.DataFrame(
        {
            "well": ["O1"],
            "ppd_well": ["P1"],
            "q_start": [30.0],
            "q_end": [35.0],          # ΔQ = +5
            "p_start": [100.0],
            "p_end": [105.0],         # ΔP_oil = +5
            "oil_start": [pd.Timestamp("2025-01-02")],
            "oil_end": [pd.Timestamp("2025-01-05")],
            "duration_days_oil": [3],
            "ppd_start": [pd.Timestamp("2025-01-01")],
        }
    )

    df_ppd = pd.DataFrame(
        {
            "well": ["P1"],
            "start_date": ["2025-01-01"],
            "baseline_before": [100.0],
            "baseline_during": [120.0],   # ΔP_PPD = +20
        }
    )

    # чистая нефть — ядро пытается парсить колонку date, поэтому она обязана быть
    df_clean = pd.DataFrame(
        {
            "well": ["O1"],
            "date": [pd.Timestamp("2024-12-20")],
            "q_oil": [30.0],
            "p_oil": [100.0],
        }
    )

    df_pairs = pd.DataFrame(
        {"oil_well": ["O1"], "ppd_well": ["P1"], "distance": [distance]}
    )

    # ground truth: ядро просто проверяет, что файл «не пустой»
    df_gt = pd.DataFrame(
        {"oil_well": ["O1"], "ppd_well": ["P1"], "expected": ["impact"], "acceptable": [""]}
    )

    return df_ow, df_ppd, df_clean, df_pairs, df_gt


# ----------------------------------------------------------------------
# 1. Локальная функция _ci_value
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "dp,dq,dp_o",
    [
        (10,  5,  2),   # все приросты +
        (10,  3, -1),   # только ΔQ +
        (10, -3,  1),   # только ΔP_oil +
        (10, -3, -1),   # оба −
        (0,   5,  2),   # ΔP_PPD = 0
    ],
)
def test_ci_value(dp, dq, dp_o):
    """Формула должна вести себя ровно по описанным правилам."""
    assert mt._ci_value(dp, dq, dp_o) == pytest.approx(_ci_manual(dp, dq, dp_o))


# ----------------------------------------------------------------------
# 2. Ядро _compute_ci_df
# ----------------------------------------------------------------------
def _expected_linear(distance: float) -> float:
    """Ручной расчёт CI с линейным затуханием."""
    dp, dq, dp_o = 20.0, 5.0, 5.0
    raw = _ci_manual(dp, dq, dp_o)
    atten = max(0.0, 1 - distance / cfg.lambda_dist)
    return round(raw * atten, 1)


def test_compute_ci_df_linear():
    frames = _toy_frames(distance=100.0)
    detail, agg = mt._compute_ci_df(*frames, methods=("none",))

    expected = _expected_linear(100.0)

    assert len(detail) == 1
    assert detail.at[0, "CI_none"] == expected
    assert agg.at[0, "CI_value"] == expected


def test_distance_mode_exp(monkeypatch):
    """Проверяем e^{-d/λ} при distance_mode='exp'."""
    monkeypatch.setattr(cfg, "distance_mode", "exp", raising=False)
    monkeypatch.setattr(cfg, "lambda_dist", 1_000.0, raising=False)

    frames = _toy_frames(distance=1_000.0)  # d = λ
    detail, _ = mt._compute_ci_df(*frames, methods=("none",))

    dp, dq, dp_o = 20.0, 5.0, 5.0
    raw = _ci_manual(dp, dq, dp_o)
    expected = round(raw * math.e**-1, 1)

    assert detail.at[0, "CI_none"] == expected


def test_ci_category_assignment():
    """Категория должна соответствовать порогам из config.CI_THRESHOLDS."""
    detail, agg = mt._compute_ci_df(*_toy_frames(), methods=("none",))

    ci_val = agg.at[0, "CI_value"]
    low, high = cfg.CI_THRESHOLDS
    cat = "none" if ci_val < low else "weak" if ci_val < high else "impact"

    assert agg.at[0, "ci_cat"] == cat


# ----------------------------------------------------------------------
# 3. compute_ci (обёртка)
# ----------------------------------------------------------------------
def test_compute_ci_wrapper_smoke():
    """Фасад compute_ci отдаёт те же данные, что и ядро."""
    ow, ppd, clean, pairs, gt = _toy_frames()

    d_core, a_core = mt._compute_ci_df(ow, ppd, clean, pairs, gt, methods=("none",))

    mapping = {
        "oil_windows": ow,
        "ppd_events": ppd,
        "oil_clean": clean,
        "pairs": pairs,
        "ground_truth": gt,
    }
    d_wrap, a_wrap = mt.compute_ci(mapping, methods=("none",), save_csv=False)

    pd.testing.assert_frame_equal(d_core, d_wrap)
    pd.testing.assert_frame_equal(a_core, a_wrap)