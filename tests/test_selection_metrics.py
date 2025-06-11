"""
Unit-тесты для ключевых функций модуля selection_metrics.py

Проверяют:

1. categorize_by(ci_val, t1, t2, t3)
   • CI < t1 → "none"
   • t1 ≤ CI < t2 → "weak"
   • t2 ≤ CI < t3 → "medium"
   • CI ≥ t3 → "strong"

2. prepare_rep(sums_df)
   При наличии:
     sums_df = DataFrame с колонками ["well","ppd_well","CI_none"]
     selection_metrics.gt = DataFrame с MultiIndex ["well","ppd_well"] и полями ["expected","acceptable"]
   Ожидаем:
     DataFrame с объединёнными колонками ["well","ppd_well","CI_none","expected","acceptable"]

3. one_hyper_run(params)
   Для заданных:
     • params = (dq, dp, wq, lambda_dist, mode)
     • stub-метод metrics.compute_ci_with_pre возвращает фиксированный DataFrame CI по окнам
     • stub gt содержит разметку для тех же пар
     • THRESHOLDS содержит хотя бы 2 набора
   Проверяем:
     – Возвращается список словарей длиной = number of THRESHOLDS
     – Для каждого набора порогов правильно считается exact/off_by_1/miss

Структура файла:
  • pytest fixtures:
      – tmp_path — для генерации временных CSV
      – monkeypatch — для подмены config и selection_metrics.gt/THRESHOLDS/metrics.compute_ci_with_pre
  • Три теста:
      1. test_categorize_by
      2. test_prepare_rep
      3. test_one_hyper_run

"""

import sys
# Чтобы argparse внутри selection_metrics не "съел" pytest-аргументы:
sys.argv = ["selection_metrics.py"]

import pytest
import pandas as pd

import selection_metrics


def test_categorize_by():
    # при порогах (1,2,3):
    # CI <1 → none; 1≤CI<2 → weak; 2≤CI<3 → medium; CI≥3 → strong
    cat = selection_metrics.categorize_by
    assert cat(0.9, 1, 2, 3) == "none"
    assert cat(1.0, 1, 2, 3) == "weak"
    assert cat(1.9, 1, 2, 3) == "weak"
    assert cat(2.0, 1, 2, 3) == "medium"
    assert cat(2.9, 1, 2, 3) == "medium"
    assert cat(3.0, 1, 2, 3) == "strong"
    assert cat(10.0, 1, 2, 3) == "strong"


def test_prepare_rep(tmp_path, monkeypatch):
    # создаём DataFrame сумм CI
    sums = pd.DataFrame([
        {"well": "w1", "ppd_well": "p1", "CI_none": 1.2},
        {"well": "w2", "ppd_well": "p2", "CI_none": 3.4},
    ])
    # stub ground_truth — две записи
    gt = pd.DataFrame(
        [
            {"expected": "weak",   "acceptable": ["none"]},
            {"expected": "strong", "acceptable": ["medium"]},
        ],
        index=pd.MultiIndex.from_tuples(
            [("w1", "p1"), ("w2", "p2")],
            names=["well", "ppd_well"]
        )
    )
    monkeypatch.setattr(selection_metrics, "gt", gt)
    # вызов
    rep = selection_metrics.prepare_rep(sums)
    # проверяем строки
    assert len(rep) == 2
    r1 = rep.set_index(["well", "ppd_well"]).loc[("w1", "p1")]
    assert r1["CI_none"] == 1.2
    assert r1["expected"] == "weak"
    assert r1["acceptable"] == ["none"]
    r2 = rep.set_index(["well", "ppd_well"]).loc[("w2", "p2")]
    assert r2["CI_none"] == 3.4
    assert r2["expected"] == "strong"
    assert r2["acceptable"] == ["medium"]


def test_one_hyper_run(monkeypatch):
    # stub вычисления CI: вернём два окна для пары (w1,p1)
    dummy_ci = pd.DataFrame([
        {"well": "w1", "ppd_well": "p1", "CI_none": 1.5},
        {"well": "w1", "ppd_well": "p1", "CI_none": 2.5},
    ])
    monkeypatch.setattr(
        selection_metrics.metrics,
        "compute_ci_with_pre",
        lambda **kwargs: dummy_ci
    )
    # stub ground_truth: одна запись для (w1,p1)
    gt = pd.DataFrame(
        [{"expected": "medium", "acceptable": ["weak", "strong"]}],
        index=pd.MultiIndex.from_tuples([("w1", "p1")], names=["well", "ppd_well"])
    )
    monkeypatch.setattr(selection_metrics, "gt", gt)
    # пороги: два набора
    monkeypatch.setattr(selection_metrics, "THRESHOLDS", [(2.0, 4.0, 6.0), (1.0, 2.0, 3.0)])
    # собираем одну hyper-параметр-комбинацию
    params = (1.0, 2.0, 0.5, 400, "linear")
    rows = selection_metrics.one_hyper_run(params)
    # должно быть по два порога → две строки
    assert isinstance(rows, list) and len(rows) == 2

    # для порога (2.0,4.0,6.0): CI_sum = 1.5+2.5=4.0 → category medium, expected medium → exact=1
    row1 = next(r for r in rows if r["T1"] == 2.0)
    assert row1["exact"] == 1
    assert row1["off_by_1"] == 0
    assert row1["miss"] == 0

    # для порога (1.0,2.0,3.0): CI_sum=4.0 → 4.0≥3.0 → strong, expected medium → off_by_1=1 (medium in acceptable)
    row2 = next(r for r in rows if r["T1"] == 1.0)
    assert row2["exact"] == 0
    assert row2["off_by_1"] == 1
    assert row2["miss"] == 0
