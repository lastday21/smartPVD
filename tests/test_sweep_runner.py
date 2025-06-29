"""

Лёгкие unit-тесты для sweep_runner.py.  Без Optuna и тяжёлых вызовов —
только вспомогательные функции, но именно они держат логику
«перебора + оценка».

* patch_cfg   — контекстный менеджер для временной правки config/correl
* gen_early   — сборка «ранних» конфигов
* evaluate    — сравнение финального DataFrame с ground_truth.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd
import pytest

import sweep_runner as sw


# ----------------------------------------------------------------------
# 1. patch_cfg
# ----------------------------------------------------------------------
def test_patch_cfg_temporary_change(monkeypatch):
    """
    patch_cfg должен:
    1. Подменить значение в config **на время контекста**.
    2. Вернуть исходное после выхода из with-блока.
    """
    import config, correl

    # добавляем тестовый атрибут в оба модуля
    monkeypatch.setattr(config, "TEST_PARAM", 0, raising=False)
    monkeypatch.setattr(correl, "TEST_PARAM", 0, raising=False)

    with sw.patch_cfg(TEST_PARAM=42):
        assert config.TEST_PARAM == 42
        assert correl.TEST_PARAM == 42

    # после выхода всё откатилось
    assert config.TEST_PARAM == 0
    assert correl.TEST_PARAM == 0


# ----------------------------------------------------------------------
# 2. gen_early
# ----------------------------------------------------------------------
def test_gen_early_quick_vs_full(monkeypatch):
    """
    quick=True → берём ровно QUICK_EARLY конфигов;
    quick=False → полный декартов продукт EARLY_GRID.
    """
    # вычисляем полный размер сетки
    prod = 1
    for v in sw.EARLY_GRID.values():
        prod *= len(v)

    quick = sw.gen_early(True)
    full  = sw.gen_early(False)

    assert len(quick) == min(sw.QUICK_EARLY, prod)
    assert len(full)  == prod
    # quick — это префикс полного списка
    assert quick == full[:len(quick)]


# ----------------------------------------------------------------------
# helpers для evaluate
# ----------------------------------------------------------------------
def _fake_final(rows: List[Tuple[str, str, str]]) -> pd.DataFrame:
    """
    rows: (oil, ppd, final_cat)
    """
    return pd.DataFrame(
        [{"oil_well": o, "ppd_well": p, "final_cat": c} for o, p, c in rows]
    )


def _fake_gt(rows: List[Tuple[str, str, str, str]]) -> pd.DataFrame:
    """
    rows: (oil, ppd, expected, acceptable)
    acceptable — строка вида "weak;none" или ""
    """
    df = pd.DataFrame(
        [
            {"oil_well": o, "ppd_well": p,
             "expected": e, "acceptable": acc}
            for o, p, e, acc in rows
        ]
    )
    return df


# ----------------------------------------------------------------------
# 3-A. evaluate: gt-файл отсутствует
# ----------------------------------------------------------------------
def test_evaluate_no_gt(monkeypatch):
    """
    Если start_data/ground_truth.csv нет, функция должна вернуть
    (0, 0, total, 0.0).
    """
    monkeypatch.setattr(Path, "exists", lambda self: False, raising=False)

    final = _fake_final([("O1", "P1", "impact"), ("O2", "P2", "weak")])
    exact, nearby, miss, acc = sw.evaluate(final)

    assert (exact, nearby, miss, acc) == (0, 0, 2, 0.0)


# ----------------------------------------------------------------------
# 3-B. evaluate: gt-файл присутствует
# ----------------------------------------------------------------------
def test_evaluate_with_gt(monkeypatch):
    """
    Проверяем точный/допустимый/ошибочный подсчёт.
    Scenario:
        O1–P1 → impact (exact)
        O2–P2 → weak   (nearby, т.к. acceptable включает weak)
        O3–P3 → weak   (miss, ожидают impact, acceptable пустое)
    """
    # 1. Подмена Path.exists → True для ground_truth.csv
    # 1. Path.exists должен вернуть True СТРОГО для ground_truth.csv
    def fake_exists(self):
        return self.name == "ground_truth.csv"

    monkeypatch.setattr(Path, "exists", fake_exists, raising=False)

    # 2. Подмена pd.read_csv → возвращает наш GT-DataFrame
    gt_df = _fake_gt([
        ("O1", "P1", "impact",  ""),
        ("O2", "P2", "impact",  "weak"),
        ("O3", "P3", "impact",  ""),
    ])
    real_read_csv = pd.read_csv

    def fake_read_csv(path, *a, **kw):
        if str(path).endswith("ground_truth.csv"):
            return gt_df.copy()
        return real_read_csv(path, *a, **kw)

    monkeypatch.setattr(pd, "read_csv", fake_read_csv, raising=True)

    # 3. Финальный DataFrame
    final = _fake_final([
        ("O1", "P1", "impact"),   # exact
        ("O2", "P2", "weak"),     # nearby
        ("O3", "P3", "weak"),     # miss
    ])

    exact, nearby, miss, acc = sw.evaluate(final)
    assert (exact, nearby, miss) == (1, 1, 1)
    assert acc == pytest.approx(2 / 3, rel=1e-6)
