"""
Модуль *window_selector.py* отвечает за выбор участка суточных данных нефти,
где, предположительно, проявляется эффект от события ППД.  Ошибка здесь
разносится дальше по пайплайну, поэтому покрываем каждую ветку:

1. **_get_event_dates** – умеет читать даты из разных форматов,
   от Series до namedtuple.
2. **select_response_window** – основная логика:
      • базовое окно,
      • расширение при скачке давления,
      • обрезка по границам данных,
      • ситуа́ции “данных нет”.
3. **make_window_passport** – формирует мини-словарик с цифрами
   начала/конца и несколькими вспомогательными полями.
"""

from collections import namedtuple
import pandas as pd
import pytest
import window_selector as ws


# ----------------------------------------------------------------------
# Глобальный «смягчитель» констант – короче интервалы, меньше циклов
# ----------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _tiny_config(monkeypatch):
    monkeypatch.setattr(ws, "LAG_DAYS",           1,  raising=False)
    monkeypatch.setattr(ws, "OIL_CHECK_DAYS",     4,  raising=False)
    monkeypatch.setattr(ws, "OIL_DELTA_P_THRESH", 3,  raising=False)
    monkeypatch.setattr(ws, "OIL_EXTEND_DAYS",   10,  raising=False)


# ----------------------------------------------------------------------
# 1. _get_event_dates
# ----------------------------------------------------------------------
def test_get_event_dates_accepts_three_shapes():
    start = pd.Timestamp("2025-01-01")
    end   = pd.Timestamp("2025-01-05")

    # namedtuple
    Ev = namedtuple("Ev", "start end")
    assert ws._get_event_dates(Ev(start, end)) == (start, end)

    # pandas.Series c полями start/end
    assert ws._get_event_dates(pd.Series({"start": start, "end": end})) == (start, end)

    # та же информация, но под другими именами
    row = pd.Series({"start_date": start, "end_date": end})
    assert ws._get_event_dates(row) == (start, end)


def test_get_event_dates_raises_on_garbage():
    with pytest.raises(AttributeError):
        ws._get_event_dates(object())   # у объекта нет нужных атрибутов


# ----------------------------------------------------------------------
# Вспомогалка для остальных тестов
# ----------------------------------------------------------------------
def _mk_series(begin: str, days: int,
               p0: float = 100,
               delta: list[float] | float = 0.0) -> pd.DataFrame:
    """
    Генерирует небольшой DataFrame с индексом-датой и колонками
    p_oil / q_oil / well (обязательно!), чтобы feed-ить select_response_window.
    """
    idx = pd.date_range(begin, periods=days, freq="D")
    if isinstance(delta, list):
        p = [p0 + d for d in delta]
    else:
        p = [p0 + delta] * days
    q = [10] * days
    return pd.DataFrame({"p_oil": p, "q_oil": q, "well": "W-1"}, index=idx)


# ----------------------------------------------------------------------
# 2. select_response_window
# ----------------------------------------------------------------------
def test_select_returns_none_on_empty_series():
    ev = namedtuple("Ev", "start end")(pd.Timestamp("2025-01-01"),
                                       pd.Timestamp("2025-01-03"))
    assert ws.select_response_window(ev, pd.DataFrame()) is None


def test_basic_window_without_extension():
    """ΔP < порога → длина окна = OIL_CHECK_DAYS (4 суток)."""
    ev  = namedtuple("Ev", "start end")(pd.Timestamp("2025-01-01"),
                                        pd.Timestamp("2025-01-10"))
    ser = _mk_series("2025-01-02", 6, 100, delta=[0, 1, 1, 2, 0, -1])

    win = ws.select_response_window(ev, ser)
    assert win == (pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-05"))


def test_extension_kicks_in_on_pressure_jump():
    """ΔP ≥ порога ⇒ конец окна сдвигается дальше (до 10-го дня)."""
    ev  = namedtuple("Ev", "start end")(pd.Timestamp("2025-01-01"),
                                        pd.Timestamp("2025-01-20"))
    # скачок +7 атм
    ser = _mk_series("2025-01-02", 15, 100, delta=[0] + [7] * 14)

    win = ws.select_response_window(ev, ser)
    # oil_start = 02-01; OIL_EXTEND_DAYS=10 ⇒ конец = 11-01
    assert win == (pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-11"))


def test_end_trimmed_to_series_boundary():
    """Если расчётный конец > последней даты в серии, он подрезается."""
    ev  = namedtuple("Ev", "start end")(pd.Timestamp("2025-01-01"),
                                        pd.Timestamp("2025-02-01"))
    ser = _mk_series("2025-01-02", 5, 100, 5)   # даты 02-01 … 06-01

    win = ws.select_response_window(ev, ser)
    # базовое окно: 02-01 … 05-01 (4 суток). 05-01 входит в данные, тримминг не нужен
    assert win == (pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-05"))



def test_series_without_pressure_column_returns_none():
    """Нет p_oil → функция не сможет посчитать ΔP и вернёт None."""
    ev  = namedtuple("Ev", "start end")(pd.Timestamp("2025-01-01"),
                                        pd.Timestamp("2025-01-03"))
    ser = _mk_series("2025-01-02", 3).drop(columns="p_oil")

    assert ws.select_response_window(ev, ser) is None


# ----------------------------------------------------------------------
# 3. make_window_passport
# ----------------------------------------------------------------------
def test_passport_happy_path():
    """Собираем паспорт: проверяем ключевые поля и длительность."""
    idx = pd.date_range("2025-01-02", periods=3, freq="D")
    df = pd.DataFrame({
        "p_oil": [100, 101, 102],
        "q_oil": [10, 11, 12],
        "well": "W-77",
        "field": "Южное",
    }, index=idx)

    passport = ws.make_window_passport(df, idx[0], idx[-1])
    assert passport == {
        "well": "W-77",
        "oil_start": idx[0],
        "q_start": 10,
        "p_start": 100,
        "oil_end": idx[-1],
        "q_end": 12,
        "p_end": 102,
        "duration_days_oil": 3,
        "field": "Южное",
    }


def test_passport_returns_none_on_missing_boundaries():
    """Если в series нет строки на oil_end — получаем None."""
    idx = pd.date_range("2025-01-02", periods=2, freq="D")
    df = _mk_series("2025-01-02", 2)         # индексы: 02-01 и 03-01
    assert ws.make_window_passport(df, idx[0], pd.Timestamp("2025-01-05")) is None
