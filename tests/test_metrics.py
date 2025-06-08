"""
Comprehensive unit tests for metrics.compute_ci_with_pre

Покрывают все ветви расчёта CI, оба варианта baseline (<5 и ≥5 точек) и все возможные
сочетания знаков Δ_PPD, ε_Q и ε_P.

• dp == 0  → CI = 0
• x ≥ 0, y ≥ 0 → CI = w_q·x + w_p·y
• x ≥ 0, y < 0  → CI = x
• x < 0, y ≥ 0  → CI = y
• x < 0, y < 0  → CI = 0
• baseline‑ветка, когда пред‑окно содержит ≥ 5 замеров

Тонкость: production‑код округляет CI до одного знака после запятой
(df["CI"] = df["CI"].round(1)).  В тестах мы теперь делаем то же, иначе
из‑за округления возможны расхождения на ±0.05.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import config
from metrics import compute_ci_with_pre


# ---------------------------------------------------------------------------
#                       Хелпер для генерации синтетических файлов
# ---------------------------------------------------------------------------

def _prepare_case(
    tmp_path: Path,
    *,
    dp_sign: int,
    eps_q_sign: int,
    eps_p_sign: int,
    pre_len: int,
):
    """Создаёт минимальный набор CSV‑файлов для одного синтетического окна.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Фикстура pytest для временной директории.
    dp_sign : {1, –1, 0}
        Желаемый знак delta_PPD (1 — рост, –1 — падение, 0 — без изменения).
    eps_q_sign, eps_p_sign : {1, –1, 0}
        Знаки ε_Q и εЙ_P (дельты с учётом baseline).  Комбинируя их с dp_sign
        получаем все варианты x и y в формуле CI.
    pre_len : int
        Кол‑во точек в пред‑окне (<5 → baseline пропускается).
    """
    clean_dir = tmp_path / "clean_data"
    clean_dir.mkdir()

    # --- Геометрия окна ---
    well = "W1"
    ppd = "P1"
    oil_start = pd.Timestamp("2023-01-10")
    oil_end = pd.Timestamp("2023-01-20")
    ppd_start = pd.Timestamp("2023-01-10")

    # --- Фактические дельты скважины ---
    dq_act = 20 * eps_q_sign  # +20, –20 или 0
    dp_act = 10 * eps_p_sign  # +10, –10 или 0

    q_start = 100.0
    q_end = q_start + dq_act

    p_start = 150.0
    p_end = p_start + dp_act

    # ------------------------- oil_windows.csv -----------------------------
    df_ow = pd.DataFrame({
        "well": [well],
        "ppd_well": [ppd],
        "q_start": [q_start],
        "p_start": [p_start],
        "q_end": [q_end],
        "p_end": [p_end],
        "duration_days_oil": [(oil_end - oil_start).days],
        "oil_start": [oil_start],
        "oil_end": [oil_end],
        "ppd_start": [ppd_start],
    })
    df_ow.to_csv(clean_dir / "oil_windows.csv", index=False)

    # ------------------------- ppd_events.csv -----------------------------
    if dp_sign == 1:
        before, during = 10.0, 20.0
    elif dp_sign == -1:
        before, during = 20.0, 10.0
    else:  # dp_sign == 0
        before = during = 10.0

    df_pe = pd.DataFrame({
        "well": [ppd],
        "start_date": [ppd_start.strftime("%d.%m.%Y")],  # day‑first
        "end_date": [(ppd_start + pd.Timedelta(days=1)).strftime("%d.%m.%Y")],
        "baseline_before": [before],
        "baseline_during": [during],
    })
    df_pe.to_csv(clean_dir / "ppd_events.csv", index=False)

    # -------------------------- oil_clean.csv -----------------------------
    pre_dates = pd.date_range(
        oil_start - pd.Timedelta(days=pre_len), periods=pre_len, freq="D"
    )
    df_clean = pd.DataFrame({
        "well": [well] * pre_len,
        "date": pre_dates,
        "q_oil": [q_start] * pre_len,  # постоянное значение → тренд 0
        "p_oil": [p_start] * pre_len,
    })
    df_clean.to_csv(clean_dir / "oil_clean.csv", index=False)

    return clean_dir, dq_act, dp_act, dp_sign


# ---------------------------------------------------------------------------
#            Локальная реализация формулы CI (без округления!)
# ---------------------------------------------------------------------------

def _expected_ci(dp: int, eps_q: float, eps_p: float) -> float:
    """Возвращает *неокруглённое* значение CI по вариантам из metrics.py."""

    if dp == 0:
        return 0.0

    x = dp * (eps_q / config.divider_q)
    y = dp * (eps_p / config.divider_p)

    if x >= 0 and y >= 0:
        return config.w_q * x + config.w_p * y
    if x >= 0 and y < 0:
        return x
    if x < 0 and y >= 0:
        return y
    return 0.0


# ---------------------------------------------------------------------------
#                                ТЕСТЫ
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "dp_sign,eps_q_sign,eps_p_sign,pre_len",
    [
        # --- Δ_PPD = 0 -----------------------------------------------------
        (0,  1,  1, 3),  # CI == 0

        # --- Δ_PPD > 0 -----------------------------------------------------
        (1,  1,  1, 3),  # x ≥ 0, y ≥ 0
        (1,  1, -1, 3),  # x ≥ 0, y < 0
        (1, -1,  1, 3),  # x < 0, y ≥ 0
        (1, -1, -1, 3),  # x < 0, y < 0 → CI = 0
        (1,  1,  1, 6),  # baseline‑ветка (≥5 точек, тренд 0)

        # --- Δ_PPD < 0 -----------------------------------------------------
        (-1,  1,  1, 3),  # x ≤ 0, y ≤ 0 (оба ≥0 после умножения?)
        (-1,  1, -1, 3),  # x ≤ 0, y > 0
        (-1, -1,  1, 3),  # x > 0, y ≤ 0
        (-1, -1, -1, 3),  # x > 0, y > 0 (оба ≥0)
        (-1,  1,  1, 6),  # baseline, Δ_PPD < 0
    ],
)

def test_ci_branches(tmp_path: Path, dp_sign, eps_q_sign, eps_p_sign, pre_len):
    """Проверяет корректность расчёта CI во всех ветках кода."""

    clean_dir, dq_act, dp_act, dp = _prepare_case(
        tmp_path,
        dp_sign=dp_sign,
        eps_q_sign=eps_q_sign,
        eps_p_sign=eps_p_sign,
        pre_len=pre_len,
    )

    df = compute_ci_with_pre(clean_data_dir=str(clean_dir))
    assert len(df) == 1, "Ожидалась ровно одна строка результата"

    row = df.iloc[0]

    # --- baseline проверки -------------------------------------------------
    if pre_len < 5:
        # baseline не должен участвовать
        assert row["deltaQbase"] == 0
        assert row["deltaPbase"] == 0.0
        eps_q = dq_act
        eps_p = dp_act
    else:
        # baseline рассчитан, но с нулевым трендом он занулит eps
        assert row["deltaQbase"] == 0
        assert row["deltaPbase"] == 0.0
        eps_q = dq_act
        eps_p = dp_act

    # --- CI ----------------------------------------------------------------
    expected_ci_raw = _expected_ci(dp, eps_q, eps_p)
    expected_ci = round(expected_ci_raw, 1)  # production‑округление

    assert row["CI"] == expected_ci, (
        f"Ожидали CI={expected_ci} (неокр. {expected_ci_raw:.3f}), "
        f"получили {row['CI']}"
    )
