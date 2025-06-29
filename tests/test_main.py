"""
Проверяем:
1. Собирает пайплайн без ошибок, когда все внутренние шаги
   «заглушены» минимальным набором DataFrame-ов.
2. Корректно создаёт CSV-файл *final_result.csv*.
3. При наличии ground_truth — мёрджит его и печатает строку TOTALS.

"""

from __future__ import annotations

import importlib

import pandas as pd
import pytest


# ----------------------------------------------------------------------
# helpers: микроскопические DataFrame-ы для замены «боевых» данных
# ----------------------------------------------------------------------
def _stub_build_clean_data(*_a, **_kw):
    ppd = pd.DataFrame({"well": ["P1"], "date": ["2025-01-01"], "q_ppd": [10]})
    oil = pd.DataFrame({"well": ["O1"], "date": ["2025-01-01"], "q_oil": [20]})
    coords = pd.DataFrame({"well": ["P1", "O1"], "x": [0, 0], "y": [0, 1]})
    return ppd, oil, coords


def _stub_detect_ppd_events(*_a, **_kw):
    return pd.DataFrame(
        {"ppd_well": ["P1"], "start": ["2025-01-01"], "end": ["2025-01-02"]}
    )


def _stub_build_pairs(*_a, **_kw):
    return pd.DataFrame({"oil_well": ["O1"], "ppd_well": ["P1"], "distance": [50]})


def _stub_build_oil_windows(*_a, **_kw):
    return pd.DataFrame(
        {
            "oil_well": ["O1"],
            "ppd_well": ["P1"],
            "oil_start": ["2025-01-02"],
            "oil_end": ["2025-01-05"],
            "q_start": [30],
            "q_end": [35],
        }
    )


def _stub_compute_ci(*_a, **_kw):
    ci = pd.DataFrame(
        {"oil_well": ["O1"], "ppd_well": ["P1"], "CI_value": [4.2], "ci_cat": ["weak"]}
    )
    return pd.DataFrame(), ci


def _stub_calc_corr(*_a, **_kw):
    return pd.DataFrame(
        {"oil_well": ["O1"], "ppd_well": ["P1"], "corr_cat": ["weak"]}
    )


def _stub_run_final_mix(*_a, **_kw):
    return pd.DataFrame(
        {"oil_well": ["O1"], "ppd_well": ["P1"], "final_cat": ["weak"]}
    )


# ----------------------------------------------------------------------
# фикстура: подмена всех тяжёлых шагов + CLEAN/START каталоги
# ----------------------------------------------------------------------
@pytest.fixture
def patched_main(tmp_path, monkeypatch):
    """
    Возвращает уже-загруженный модуль *main* с пропатченными
    зависимостями.  Каждый тест получает свежий экземпляр.
    """
    # Импортируем модуль с нуля, чтобы не тащить предыдущие патчи
    main = importlib.import_module("main")
    importlib.reload(main)

    monkeypatch.setattr(main, "build_clean_data", _stub_build_clean_data)
    monkeypatch.setattr(main, "detect_ppd_events", _stub_detect_ppd_events)
    monkeypatch.setattr(main, "build_pairs", _stub_build_pairs)
    monkeypatch.setattr(main, "build_oil_windows", _stub_build_oil_windows)
    monkeypatch.setattr(main, "compute_ci", _stub_compute_ci)
    monkeypatch.setattr(main, "calc_corr", _stub_calc_corr)
    monkeypatch.setattr(main, "run_final_mix", _stub_run_final_mix)

    # Перенаправляем каталоги into tmp_path
    clean_dir = tmp_path / "clean_data"
    clean_dir.mkdir()
    start_dir = tmp_path / "start_data"
    start_dir.mkdir()

    monkeypatch.setattr(main, "CLEAN_DIR", clean_dir, raising=False)
    monkeypatch.setattr(main, "START_DIR", start_dir, raising=False)

    # Чистый in-memory режим
    monkeypatch.setattr(main, "DEBUG", False, raising=False)

    return main, clean_dir, start_dir


# ----------------------------------------------------------------------
# 1. Без ground-truth
# ----------------------------------------------------------------------
def test_main_produces_csv(patched_main):
    main, clean_dir, _ = patched_main

    # gt выключаем
    main.final_filter_by_gt = False

    # запускаем
    main.main()

    out_csv = clean_dir / "final_result.csv"
    assert out_csv.exists(), "main должен записать final_result.csv"

    df = pd.read_csv(out_csv)
    assert list(df.columns) == ["oil_well", "ppd_well", "final_cat"]
    assert df.iloc[0]["final_cat"] == "weak"


# ----------------------------------------------------------------------
# 2. С ground-truth + TOTALS
# ----------------------------------------------------------------------
def test_main_merges_ground_truth(patched_main, monkeypatch):
    main, clean_dir, start_dir = patched_main

    # включаем фильтр
    main.final_filter_by_gt = True

    # кладём искусственный ground_truth.csv
    gt = pd.DataFrame(
        {
            "well": ["O1"],
            "ppd_well": ["P1"],
            "expected": ["weak"],
            "acceptable": [""],
        }
    )
    gt_path = start_dir / "ground_truth.csv"
    gt.to_csv(gt_path, index=False)

    # запускаем
    main.main()

    out_csv = clean_dir / "final_result.csv"
    df = pd.read_csv(out_csv)

    # 1) должны появиться expected/acceptable
    assert {"expected", "acceptable"}.issubset(df.columns)

    # 2) строка TOTALS добавлена
    assert "TOTALS" in set(df["oil_well"])

    # 3) категория совпала с expected
    row = df[df["oil_well"] == "O1"].iloc[0]
    assert row["final_cat"] == row["expected"] == "weak"
