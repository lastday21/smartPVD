"""
SmartPVD · oil_windows — unit-тесты

Проверяем: загрузчики CSV, построение словаря pairs,
ядро _build_oil_windows_df и фасад build_oil_windows.
"""

from __future__ import annotations
from pathlib import Path
from textwrap import dedent
from io import StringIO

import pandas as pd
import pytest

import oil_windows as ow


# ───────────────────────── helpers ──────────────────────────

def _mini_csv(text: str) -> StringIO:
    """Удобно создавать CSV-«файлы» прямо в тесте."""
    return StringIO(dedent(text).lstrip())


# ------------------------------------------------------------------
# 1) _read_csv_dates
# ------------------------------------------------------------------

def test_read_csv_dates_parses_and_renames(tmp_path: Path):
    p = tmp_path / "a.csv"
    p.write_text("mydate,val\n01.01.2025,2\n", encoding="utf-8")
    df = ow._read_csv_dates(p, date_cols={"mydate": "date"})
    assert df["date"].dtype == "datetime64[ns]"
    assert list(df.columns) == ["date", "val"]


# ------------------------------------------------------------------
# 2) _load_oil_df
# ------------------------------------------------------------------

def test_load_oil_df_str_well(tmp_path: Path):
    p = tmp_path / "oil.csv"
    p.write_text("well,date,q_oil,p_oil\n1,02.01.2025,10,100\n", encoding="utf-8")
    df = ow._load_oil_df(p)
    assert df["well"].dtype == object
    assert df["date"].dtype == "datetime64[ns]"


# ------------------------------------------------------------------
# 3) _load_ppd_events
# ------------------------------------------------------------------

def test_load_ppd_events_rename(tmp_path: Path):
    p = tmp_path / "ppd.csv"
    p.write_text("well,start_date,end_date\nP1,01.01.25,03.01.25\n", encoding="utf-8")
    df = ow._load_ppd_events(p)
    assert {"ppd_well", "start", "end"}.issubset(df.columns)
    assert df["ppd_well"].iloc[0] == "P1"


# ------------------------------------------------------------------
# 4) _load_pairs
# ------------------------------------------------------------------

def test_load_pairs_builds_mapping():
    csv = _mini_csv("""
        ppd_well,oil_well
        P1,O1
        P1,O2
        P2,O2
    """)
    mapping = ow._load_pairs(csv)
    assert mapping == {"P1": ["O1", "O2"], "P2": ["O2"]}


# ------------------------------------------------------------------
# 5) _build_oil_windows_df — ядро
# ------------------------------------------------------------------

@pytest.fixture
def synthetic_data():
    ppd_events = pd.DataFrame({
        "ppd_well": ["P1", "P2"],
        "start": pd.to_datetime(["2025-01-01", "2025-01-05"]),
        "end":   pd.to_datetime(["2025-01-03", "2025-01-07"]),
    })

    dates = pd.date_range("2025-01-01", periods=10)
    oil_df = pd.concat([
        pd.DataFrame({"well": "O1", "date": dates,
                      "q_oil": 10, "p_oil": 100}),
        pd.DataFrame({"well": "O2", "date": dates,
                      "q_oil": 12, "p_oil": 105}),
    ], ignore_index=True)

    mapping = {"P1": ["O1", "O2"], "P2": ["O2"]}
    return ppd_events, oil_df, mapping


@pytest.fixture(autouse=True)
def stub_window_selector(monkeypatch):
    """Подменяем тяжёлые функции на простые заглушки."""
    def fake_select_response_window(ev, series):
        start = series.index.min()
        return start, start + pd.Timedelta(days=2)  # всегда три дня

    def fake_make_window_passport(series, start, end):
        return {
            "q_start": series["q_oil"].loc[start],
            "p_start": series["p_oil"].loc[start],
            "q_end":   series["q_oil"].loc[end],
            "p_end":   series["p_oil"].loc[end],
        }

    monkeypatch.setattr(ow, "select_response_window", fake_select_response_window)
    monkeypatch.setattr(ow, "make_window_passport", fake_make_window_passport)


def test_build_oil_windows_df_basic(synthetic_data):
    ppd_events, oil_df, mapping = synthetic_data
    df = ow._build_oil_windows_df(ppd_events, oil_df, mapping)

    # должно выйти 3 строки: O1-P1, O2-P1, O2-P2
    assert len(df) == 3
    assert set(df["ppd_well"]) == {"P1", "P2"}
    assert set(df["well"]) == {"O1", "O2"}

    expected_cols = [
        "well", "oil_start", "q_start", "p_start",
        "oil_end", "q_end", "p_end", "duration_days_oil",
        "ppd_well", "ppd_start", "ppd_end", "duration_days_ppd",
    ]
    assert list(df.columns) == expected_cols


def test_build_oil_windows_df_skips_missing_series(synthetic_data):
    ppd_events, oil_df, mapping = synthetic_data
    mapping["P1"].append("O3")  # O3 данных нет
    df = ow._build_oil_windows_df(ppd_events, oil_df, mapping)
    assert "O3" not in set(df["well"])


# ------------------------------------------------------------------
# 6) build_oil_windows — фасад
# ------------------------------------------------------------------

def test_build_oil_windows_facade(tmp_path: Path, synthetic_data):
    ppd_events, oil_df, _ = synthetic_data
    pairs_df = pd.DataFrame({
        "ppd_well": ["P1", "P1", "P2"],
        "oil_well": ["O1", "O2", "O2"],
    })

    res = ow.build_oil_windows(
        ppd_events_df=ppd_events,
        oil_df=oil_df,
        pairs_df=pairs_df,
        save_csv=False,
    )

    mapping = {"P1": ["O1", "O2"], "P2": ["O2"]}
    expected = ow._build_oil_windows_df(ppd_events, oil_df, mapping)

    pd.testing.assert_frame_equal(
        res.sort_index(axis=1).reset_index(drop=True),
        expected.sort_index(axis=1).reset_index(drop=True),
    )
