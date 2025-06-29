"""
* _get_event_dates            — вытаскиваем даты из произвольного
                                 «PPD-события» (namedtuple / Series / row)
* select_response_window      — рассчитываем отклик нефтянки
* make_window_passport        — формируем «паспорт» расчётного окна
"""

from collections import namedtuple
import pandas as pd
import pytest

import window_selector as ws


# ---------------------------------------------------------------------------
# Глобальный фикстур-патч: ужимаем константы, чтобы тесты работали быстро
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _patch_defaults(monkeypatch):
    """
    Делаем условия «компактными», чтобы не плодить лишних строк
    и не ждать лишние секунды в расчётах.
    """
    monkeypatch.setattr(ws, "LAG_DAYS", 1, raising=False)
    monkeypatch.setattr(ws, "OIL_CHECK_DAYS", 5, raising=False)
    monkeypatch.setattr(ws, "OIL_DELTA_P_THRESH", 3.0, raising=False)
    monkeypatch.setattr(ws, "OIL_EXTEND_DAYS", 10, raising=False)


# ---------------------------------------------------------------------------
# _get_event_dates
# ---------------------------------------------------------------------------

def test_get_event_dates_variants():
    """Берём даты из трёх разных представлений события."""
    t0, t1 = pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-05")

    # 1) namedtuple
    Ev = namedtuple("Ev", "start end")
    assert ws._get_event_dates(Ev(t0, t1)) == (t0, t1)

    # 2) pandas.Series c 'start' / 'end'
    ser = pd.Series({"start": t0, "end": t1})
    assert ws._get_event_dates(ser) == (t0, t1)

    # 3) строка DataFrame с 'start_date' / 'end_date'
    row = pd.Series({"start_date": t0, "end_date": t1})
    assert ws._get_event_dates(row) == (t0, t1)

    # 4) неверный объект → AttributeError
    with pytest.raises(AttributeError):
        ws._get_event_dates(object())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# select_response_window
# ---------------------------------------------------------------------------

def _make_series(start: str, days: int, p_start: float, delta: list[float] | float):
    """
    Генерирует мини-DataFrame c 'p_oil' и 'q_oil' для одной скважины.
    *delta* — либо одно число (константное смещение), либо список по дням.
    """
    idx = pd.date_range(start, periods=days, freq="D")
    if isinstance(delta, list):
        p = [p_start + d for d in delta]
    else:
        p = [p_start + delta] * days
    q = [10] * days
    return pd.DataFrame({"p_oil": p, "q_oil": q, "well": "W-1"}, index=idx)


def test_select_basic_window_without_extension():
    """
    ΔP меньше порога — должно вернуться базовое окончание (base_end).
    """
    ev = namedtuple("Ev", "start end")(
        pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-03")
    )
    series = _make_series("2025-01-02", 3, 100.0, [0, 1, -1])

    win = ws.select_response_window(ev, series)
    assert win == (pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-04"))


def test_select_with_extension_and_long_ppd_event():
    """
    ΔP ≥ порога — окно расширяется до OIL_EXTEND_DAYS,
    т.к. событие ППД длиннее и не обрезает cand_end.
    """
    ev = namedtuple("Ev", "start end")(
        pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-25")
    )
    # прыжок давления 7 атм ⇒ расширяем
    series = _make_series("2025-01-02", 15, 100.0,
                          [0] + [7] * 14)  # delta_p_max = 7 ≥ 3

    win = ws.select_response_window(ev, series)
    # oil_start = 02-01, +9 дней (10 суток) = 11-01
    assert win == (pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-11"))


def test_select_trim_to_series_end():
    """
    cand_end выходит за предел имеющихся данных —
    функция обязана «обрезать» конец.
    """
    ev = namedtuple("Ev", "start end")(
        pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-20")
    )
    series = _make_series("2025-01-02", 5, 100.0, 5)  # всего 5 суток

    win = ws.select_response_window(ev, series)
    # конец равен последнему дню в серии
    assert win == (pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-06"))


@pytest.mark.parametrize(
    "series",
    [
        pd.DataFrame(),                                  # пустой DF
        _make_series("2025-01-10", 5, 100.0, 0).drop(columns="p_oil"),  # нет давления
    ],
)
def test_select_returns_none_on_insufficient_data(series):
    """Должен вернуться None, если данных недостаточно."""
    ev = namedtuple("Ev", "start end")(
        pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-05")
    )
    assert ws.select_response_window(ev, series) is None


# ---------------------------------------------------------------------------
# make_window_passport
# ---------------------------------------------------------------------------

def test_passport_happy_path():
    """
    Формируем паспорт и сверяем ключевые поля.
    """
    dates = pd.date_range("2025-01-02", periods=3, freq="D")
    df = pd.DataFrame(
        {
            "p_oil": [100, 101, 102],
            "q_oil": [10, 11, 12],
            "field": "Южное",
            "well": "W-77",
        },
        index=dates,
    )
    start, end = dates[0], dates[-1]
    passport = ws.make_window_passport(df, start, end)

    assert passport["well"] == "W-77"
    assert passport["oil_start"] == start
    assert passport["oil_end"] == end
    assert passport["duration_days_oil"] == 3
    # дополнительное поле «field» должно сохраниться
    assert passport["field"] == "Южное"


def test_passport_returns_none_when_dates_missing():
    """Если в DF нет строки на oil_end, получаем None."""
    df = _make_series("2025-01-02", 2, 100.0, 0)  # только 2 дня
    start, end = pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-04")
    assert ws.make_window_passport(df, start, end) is None
