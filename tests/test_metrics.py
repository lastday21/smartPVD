"""
Набор unit-тестов для функции compute_ci_with_pre из metrics.py.

Основные цели:
  1. Проверить базовый расчёт CI_none (без использования baseline) в ситуациях:
     - delta_PPD == 0 → CI_none == 0
     - delta_PPD != 0 с разными знаками изменений Q/P
  2. Убедиться, что при передаче разных кортежей methods:
     ("none",), ("none","mean"), ("none","regression"), ("none","median_ewma"),
     ("none","mean","regression","median_ewma")
     в выходном DataFrame появляются соответствующие колонки CI_<method>.
  3. Проверить корректность линейного затухания:
     attenuation = max(0, 1 - distance / lambda_dist)
  4. Проверить корректность экспоненциального затухания:
     attenuation = exp(-distance / lambda_dist)

Структура тестов:
  • _prepare_case(...) — создаёт во временной папке tmp_path/clean_data необходимые CSV:
      - oil_windows.csv
      - ppd_events.csv
      - oil_clean.csv
      - pairs_oil_ppd.csv
    и возвращает фактические дельты dq_act, dp_act для сравнения.

  • _expected_ci(...) — дублирует логику compute_ci_with_pre (без учёта округления),
    чтобы получить _raw_ значение CI_none и сравнить его с выходным значением,
    округлённым до 1-го знака.

  • test_ci_none_basic(...) — параметризованный тест для базового метода ("none"),
    проверяет CI_none для разных комбинаций dp_sign, eps_q_sign, eps_p_sign.

  • test_ci_multiple_methods(...) — проверяет, что при разных списках methods
    в DataFrame появляются все требуемые колонки CI_<method>.

  • test_distance_attenuation_linear(...) и test_distance_attenuation_exp(...)
    — проверяют правильность применения коэффициента затухания при
    distance_mode = "linear" или "exp".


"""

import os
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import config
from metrics import compute_ci_with_pre




#                   Хелпер для генерации синтетических входных CSV
def _prepare_case(tmp_path: Path, *, dp_sign: int, eps_q_sign: int, eps_p_sign: int):
    """
    Создаёт в tmp_path/clean_data файлы:
      • oil_windows.csv
      • ppd_events.csv
      • oil_clean.csv
      • pairs_oil_ppd.csv
    Возвращает (dq_act, dp_act), где
      dq_act = q_end - q_start,
      dp_act = p_end - p_start.
    """
    clean_dir = tmp_path / "clean_data"
    clean_dir.mkdir()
    # параметры окна «до/после» нефти
    well = "W1"
    ppd  = "P1"
    oil_start = pd.Timestamp("2023-01-10")
    oil_end   = pd.Timestamp("2023-01-20")
    ppd_start = oil_start
    # искусственные изменения Q/P
    dq_act = 20 * eps_q_sign
    dp_act = 10 * eps_p_sign
    # 1) oil_windows.csv
    df_ow = pd.DataFrame({
        "well": [well],
        "ppd_well": [ppd],
        "q_start": [100.0],
        "p_start": [150.0],
        "q_end":   [100.0 + dq_act],
        "p_end":   [150.0 + dp_act],
        "duration_days_oil": [(oil_end - oil_start).days],
        "oil_start": [oil_start],
        "oil_end":   [oil_end],
        "ppd_start": [ppd_start],
    })
    df_ow.to_csv(clean_dir / "oil_windows.csv", index=False)
    # 2) ppd_events.csv (baseline_before / baseline_during)
    before = during = 10.0
    if dp_sign == 1:
        before, during = 10.0, 20.0
    elif dp_sign == -1:
        before, during = 20.0, 10.0
    df_pe = pd.DataFrame({
        "well": [ppd],
        "start_date": [ppd_start.strftime("%d.%m.%Y")],
        "end_date":   [(ppd_start + pd.Timedelta(days=1)).strftime("%d.%m.%Y")],
        "baseline_before": [before],
        "baseline_during": [during],
    })
    df_pe.to_csv(clean_dir / "ppd_events.csv", index=False)
    # 3) oil_clean.csv (достаточно T_pre−1 записей, baseline не попадёт в расчёт)
    T_pre = config.T_pre - 1
    dates = pd.date_range(oil_start - pd.Timedelta(days=T_pre), periods=T_pre)
    df_clean = pd.DataFrame({
        "well": [well] * T_pre,
        "date": dates,
        "q_oil": [100.0] * T_pre,
        "p_oil": [150.0] * T_pre,
    })
    df_clean.to_csv(clean_dir / "oil_clean.csv", index=False)
    # 4) pairs_oil_ppd.csv с нулевым расстоянием
    df_pairs = pd.DataFrame({
        "oil_well": [well],
        "ppd_well": [ppd],
        "distance": [0.0],
    })
    df_pairs.to_csv(clean_dir / "pairs_oil_ppd.csv", index=False)
    return dq_act, dp_act


#                       Локальная функция для проверки CI-значения

def _expected_ci(dp_sign: int, eps_q: float, eps_p: float) -> float:
    """
    Дублирует логику _ci_value из metrics.py:
      dp_sign — rec['delta_PPD'] (−1,0,+1),
      eps_q — delta Q (q_end − q_start),
      eps_p — delta P (p_end − p_start).
    """
    # dp_sign уже кодирован как ±1 или 0
    if dp_sign == 0:
        return 0.0
    x = dp_sign * (eps_q / config.divider_q)
    y = dp_sign * (eps_p / config.divider_p)
    if x >= 0 and y >= 0:
        return config.w_q * x + config.w_p * y
    if x >= 0 and y < 0:
        return x
    if x < 0 and y >= 0:
        return y
    return 0.0


#                    Тест базового CI_none (без baseline)

@pytest.mark.parametrize("dp_sign,eps_q_sign,eps_p_sign", [
    (0,  1,  1),
    (1,  1,  1),
    (1,  1, -1),
    (1, -1,  1),
    (1, -1, -1),
    (-1, 1,  1),
    (-1, 1, -1),
    (-1,-1,  1),
    (-1,-1, -1),
])
def test_ci_none_basic(tmp_path, dp_sign, eps_q_sign, eps_p_sign):
    dq_act, dp_act = _prepare_case(
        tmp_path, dp_sign=dp_sign, eps_q_sign=eps_q_sign, eps_p_sign=eps_p_sign
    )
    df = compute_ci_with_pre(clean_data_dir=str(tmp_path / "clean_data"), methods=("none",))
    assert len(df) == 1
    row = df.iloc[0]
    # проверяем, что rec["delta_PPD"] попало в вывод
    assert "delta_PPD" in row.index
    # ожидаем Raw CI по sign-коду δP
    raw = _expected_ci(int(row["delta_PPD"]), dq_act, dp_act)
    expected = round(raw, 1)
    assert "CI_none" in row.index
    assert pytest.approx(row["CI_none"], rel=1e-4) == expected


#                 Тесты на различные методы ("mean", "regression", ...)
@pytest.mark.parametrize("methods", [
    ("none",),
    ("none", "mean"),
    ("none", "regression"),
    ("none", "median_ewma"),
    ("none", "mean", "regression", "median_ewma"),
])
def test_ci_multiple_methods(tmp_path, methods):
    # создаём кейс с позитивным dp_sign, чтобы метод mean/regression хотя бы 1 строку вернул
    dq_act, dp_act = _prepare_case(tmp_path, dp_sign=1, eps_q_sign=1, eps_p_sign=1)
    # базовые методы подключают baseline, но config.T_pre задан выше
    df = compute_ci_with_pre(clean_data_dir=str(tmp_path / "clean_data"), methods=methods)
    # проверяем наличие колонок
    expected = {f"CI_{m}" for m in methods} | {"CI_none"}
    missing = expected - set(df.columns)
    assert not missing, f"Отсутствуют колонки: {missing}"
    assert len(df) >= 1


#                     Тест линейного затухания по distance
def test_distance_attenuation_linear(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "distance_mode", "linear")
    monkeypatch.setattr(config, "lambda_dist", 10.0)
    dq_act, dp_act = _prepare_case(tmp_path, dp_sign=1, eps_q_sign=1, eps_p_sign=1)
    # переопределяем пары с distance=5
    pd.DataFrame({
        "oil_well": ["W1"],
        "ppd_well": ["P1"],
        "distance": [5.0],
    }).to_csv(tmp_path / "clean_data" / "pairs_oil_ppd.csv", index=False)
    df = compute_ci_with_pre(clean_data_dir=str(tmp_path / "clean_data"), methods=("none",))
    row = df.iloc[0]
    raw = _expected_ci(int(row["delta_PPD"]), dq_act, dp_act)
    atten = max(0.0, 1 - 5.0 / 10.0)
    expected = round(raw * atten, 1)
    assert pytest.approx(row["CI_none"], rel=1e-4) == expected


#                    Тест экспоненциального затухания
def test_distance_attenuation_exp(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "distance_mode", "exp")
    monkeypatch.setattr(config, "lambda_dist", 2.0)
    dq_act, dp_act = _prepare_case(tmp_path, dp_sign=1, eps_q_sign=1, eps_p_sign=1)
    pd.DataFrame({
        "oil_well": ["W1"],
        "ppd_well": ["P1"],
        "distance": [2.0],
    }).to_csv(tmp_path / "clean_data" / "pairs_oil_ppd.csv", index=False)
    df = compute_ci_with_pre(clean_data_dir=str(tmp_path / "clean_data"), methods=("none",))
    row = df.iloc[0]
    raw = _expected_ci(int(row["delta_PPD"]), dq_act, dp_act)
    atten = math.exp(-2.0 / 2.0)
    expected = round(raw * atten, 1)
    assert pytest.approx(row["CI_none"], rel=1e-4) == expected
