"""
1. _mean                       – простая математика, но лучше убедиться.
2. _detect_ppd_events_df       – ядро детекции «скачков» приёма.
3. detect_ppd_events           – фасад, который умеет читать/писать CSV.

Чтобы не гонять длинные ряды, патчим параметры события:
PPD_WINDOW_SIZE = 3, PPD_MIN_EVENT_DAYS = 2, PPD_REL_THRESH = 0.10.
"""

from pathlib import Path

import pandas as pd
import pytest

import events_PPD as ep


# ────────────────────────────────────────────────────────────
#  Автоматическое «смягчение» параметров алгоритма
# ────────────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def _patch_params(monkeypatch):
    """Ставим маленькие цифры, чтобы тесты шли быстро."""
    monkeypatch.setattr(ep, "PPD_WINDOW_SIZE", 3, raising=False)
    monkeypatch.setattr(ep, "PPD_MIN_EVENT_DAYS", 2, raising=False)
    monkeypatch.setattr(ep, "PPD_REL_THRESH", 0.10, raising=False)


# ────────────────────────────────────────────────────────────
# 1. _mean
# ────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "seq, expected",
    [([1, 2, 3], 2.0), ([10.0], 10.0), ([], 0.0)],
)
def test_mean_basic(seq, expected):
    """Пустой список → 0.0, иначе обычное среднее."""
    assert ep._mean(seq) == expected


# ────────────────────────────────────────────────────────────
# 2. _detect_ppd_events_df
# ────────────────────────────────────────────────────────────
@pytest.fixture
def make_ppd_df():
    """Фабрика коротких рядов q_ppd для одной скважины."""
    def _factory(values):
        dates = pd.date_range("2025-01-01", periods=len(values), freq="D")
        return pd.DataFrame({"well": "W1", "date": dates, "q_ppd": values})
    return _factory


def test_detect_event_found(make_ppd_df):
    """Падение расхода ≥10 % на 7 суток → ровно одно событие."""
    q = [100, 100, 100, 80, 80, 80, 80, 80, 80, 80]
    events = ep._detect_ppd_events_df(make_ppd_df(q))

    assert len(events) == 1
    evt = events.iloc[0]

    # в исходном коде даты строковые ─ переводим, чтобы сравнить корректно
    start = pd.to_datetime(evt["start_date"], dayfirst=True)
    end   = pd.to_datetime(evt["end_date"],   dayfirst=True)

    assert start == pd.Timestamp("2025-01-04")
    assert end   == pd.Timestamp("2025-01-10")
    assert evt["baseline_before"] == 100
    assert evt["baseline_during"] == 80
    assert evt["duration_days"] == 7



def test_detect_event_none_on_flat_series(make_ppd_df):
    """Ровный ряд → событий нет."""
    events = ep._detect_ppd_events_df(make_ppd_df([100] * 10))
    assert events.empty


# ────────────────────────────────────────────────────────────
# 3. detect_ppd_events (фасад)
# ────────────────────────────────────────────────────────────
def test_detect_wrapper_returns_core_result(make_ppd_df):
    """Передаём DataFrame напрямую — фасад обязан дать тот же вывод."""
    df = make_ppd_df([100, 100, 100, 70, 70, 70, 70])
    core = ep._detect_ppd_events_df(df)
    wrap = ep.detect_ppd_events(ppd_df=df, save_csv=False)
    pd.testing.assert_frame_equal(core, wrap)


def test_detect_wrapper_raises_for_missing_file(tmp_path: Path):
    """Если CSV не существует, ждём FileNotFoundError."""
    bogus = tmp_path / "no_file.csv"
    with pytest.raises(FileNotFoundError):
        ep.detect_ppd_events(csv_path=bogus, save_csv=False)
