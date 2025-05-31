import pytest
import pandas as pd
from datetime import datetime, timedelta

from events_PPD import detect_ppd_events
from config import PPD_MIN_EVENT_DAYS

def make_ppd_dataframe(well_name, start_date_str, values):
    start = datetime.strptime(start_date_str, "%Y-%m-%d")
    dates = [start + timedelta(days=i) for i in range(len(values))]
    return pd.DataFrame({
        'well': [well_name] * len(values),
        'date': dates,
        'q_ppd': values
    })


def test_no_events_all_healthy():
    """
    Все точки «здоровые» (никакого отклонения ≥20%), поэтому событий не должно быть.
    """
    df = make_ppd_dataframe("W1", "2025-01-01", [100] * 40)
    events_df = detect_ppd_events(df)
    assert events_df.empty


def test_single_event_exactly_7_devs():
    """
    5 здоровых дней (100), затем 7 dev-дней (200), потом 10 дней (100).
    Получаются два события:
      1) 06-02-2025–12-02-2025 с baseline_before=100, baseline_during=200, min_q=max_q=200, duration=7
      2) 13-02-2025–22-02-2025 с baseline_before=200, baseline_during=100, min_q=max_q=100, duration=10
    """
    healthy = [100] * 5
    devs = [200] * PPD_MIN_EVENT_DAYS
    tail = [100] * 10
    values = healthy + devs + tail
    df = make_ppd_dataframe("W2", "2025-02-01", values)
    events_df = detect_ppd_events(df)

    assert len(events_df) == 2

    ev1 = events_df.iloc[0]
    assert ev1['start_date'] == "06-02-2025"
    assert ev1['baseline_before'] == 100
    assert ev1['baseline_during'] == 200
    assert ev1['min_q'] == 200
    assert ev1['max_q'] == 200
    assert ev1['end_date'] == "12-02-2025"
    assert ev1['duration_days'] == 7

    ev2 = events_df.iloc[1]
    assert ev2['start_date'] == "13-02-2025"
    assert ev2['baseline_before'] == 200
    assert ev2['baseline_during'] == 100
    assert ev2['min_q'] == 100
    assert ev2['max_q'] == 100
    assert ev2['end_date'] == "22-02-2025"
    assert ev2['duration_days'] == 10


def test_multiple_back_to_back_events():
    """
    5 здоровых (100), 7 dev1 (200), 7 dev2 (50), потом 5 здоровых (200).
    Должны получиться два события.
    """
    healthy = [100] * 5
    devs1 = [200] * PPD_MIN_EVENT_DAYS
    devs2 = [50] * PPD_MIN_EVENT_DAYS
    tail = [200] * 5
    values = healthy + devs1 + devs2 + tail
    df = make_ppd_dataframe("W3", "2025-03-01", values)
    events_df = detect_ppd_events(df)

    assert len(events_df) == 2

    ev1 = events_df.iloc[0]
    assert ev1['start_date'] == "06-03-2025"
    assert ev1['baseline_before'] == 100
    assert ev1['baseline_during'] == 200
    assert ev1['min_q'] == 200
    assert ev1['max_q'] == 200
    assert ev1['end_date'] == "12-03-2025"
    assert ev1['duration_days'] == 7

    ev2 = events_df.iloc[1]
    assert ev2['start_date'] == "13-03-2025"
    assert ev2['baseline_before'] == 200
    assert ev2['baseline_during'] == 50
    assert ev2['min_q'] == 50
    assert ev2['max_q'] == 50
    assert ev2['end_date'] == "19-03-2025"
    assert ev2['duration_days'] == 7


def test_filtering_with_gradual_decline():
    """
    5 здоровых (100), затем [80,60,40,20,0×7], потом 5 zero.
    Dev-блок лишь после семи подряд нулей: одно событие.
    """
    healthy = [100] * 5
    decline = [80, 60, 40, 20] + [0] * 7
    values = healthy + decline + [0] * 5
    df = make_ppd_dataframe("W4", "2025-04-01", values)
    events_df = detect_ppd_events(df)

    assert len(events_df) == 1
    ev = events_df.iloc[0]
    assert ev['start_date'] == "10-04-2025"
    assert ev['baseline_before'] == 100
    assert ev['baseline_during'] == 0
    assert ev['min_q'] == 0
    assert ev['max_q'] == 0
    assert ev['end_date'] == "21-04-2025"
    assert ev['duration_days'] == 12


def test_date_format_and_integer_rounding():
    """
    3 healthy (0), 7 dev (50), 2 healthy (0).
    Проверка формата дат и что все числовые поля — Python int.
    """
    values = [0] * 3 + [50] * 7 + [0] * 2
    df = make_ppd_dataframe("W5", "2025-05-01", values)
    events_df = detect_ppd_events(df)
    ev = events_df.iloc[0]

    assert ev['start_date'] == "04-05-2025"
    assert ev['end_date'] == "10-05-2025"
    assert isinstance(ev['baseline_before'], int)
    assert isinstance(ev['baseline_during'], int)
    assert isinstance(ev['min_q'], int)
    assert isinstance(ev['max_q'], int)
    assert ev['baseline_before'] == 0
    assert ev['baseline_during'] == 50
    assert ev['min_q'] == 50
    assert ev['max_q'] == 50


if __name__ == "__main__":
    pytest.main()
