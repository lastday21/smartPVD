"""

Модуль-тесты для блока A (выбор окна отклика) и функции «паспорт окна».
Здесь **не используется** реальный prod-датасет — мы создаём
минимальные синтетические ряды «на лету», чтобы быстро прогнать все ветви
условий и убедиться, что логика работает.


"""

from collections import namedtuple

import pandas as pd
import pytest

from window_selector import select_response_window, make_window_passport
from oil_windows import build_oil_windows                     # noqa: F401  # импорт нужен IDE для подсветки

# -----------------------------------------------------------------------------
# Вспомогательные генераторы ---------------------------------------------------
# -----------------------------------------------------------------------------


def _synthetic_series(
    days: int,
    q_base: float = 100.0,
    p_base: float = 50.0,
    *,
    p_jump: float = 0.0,
) -> pd.DataFrame:
    """
    Собрать DataFrame с индексом-датой длиной *days*.

    • `q_oil` — константа *q_base*
    • `p_oil` — константа *p_base* до 5-го дня (index 4),
      далее + *p_jump* (эмуляция ΔP внутри базового окна).

    Используем минимальный набор колонок, который нужен
    `select_response_window` / `make_window_passport`.
    """
    idx = pd.date_range("2025-01-01", periods=days, freq="D")
    p = [p_base] * days
    if p_jump:
        # добавляем скачок давления, начиная с 5-го дня
        for i in range(4, days):
            p[i] += p_jump
    return pd.DataFrame({"q_oil": q_base, "p_oil": p}, index=idx)


# минимальный «событийный» namedtuple, чтобы не тянуть полноценный класс
Ev = namedtuple("Ev", ["ppd_well", "start", "end"])

# -----------------------------------------------------------------------------
# Тесты select_response_window -------------------------------------------------
# -----------------------------------------------------------------------------

@pytest.mark.parametrize(
    "ppd_len, p_jump, expected",
    [
        # -------- короткие события (<10 суток) ------------------------------
        (8, 20, 8),   # ΔP ≥ 5, окно упирается в PPD_end+lag
        (8, 2, 8),    # ΔP < 5, та же длина

        # -------- PPD чуть длиннее базового окна ----------------------------
        (9, 4, 9),    # ΔP < 5, обрезка по PPD_end+lag
        (12, 5, 12),  # ΔP ≥ 5, extend→30, но обрезка по PPD_end+lag

        # -------- длинное PPD, ΔP ниже порога --------------------------------
        (15, 2, 10),  # базовое окно 10 сут
        (21, 3, 10),  # ещё длиннее, ΔP всё ещё < 5

        # -------- длинное PPD, ΔP выше порога --------------------------------
        (21, 6, 21),  # extend→30, обрезка до PPD_end+lag (=21)
        (30, 0, 10),  # ΔP = 0, остаётся базовое 10 сут
        (30, 5, 30),  # ΔP ≥ 5, упираемся в максимум 30

        # -------- очень длинное событие -------------------------------------
        (40, 10, 30),  # ΔP ≥ 5, ограничение max 30 сут
    ],
)
def test_select_response_window(ppd_len, p_jump, expected):
    """
    Проверяем, что (oil_end - oil_start + 1) == *expected* для
    разных сочетаний длительности события и амплитуды ΔP.
    """
    ppd_start = pd.Timestamp("2025-01-01")
    ppd_end = ppd_start + pd.Timedelta(days=ppd_len - 1)

    # synthetic series длиной >= max(expected, ppd_len+2)
    series = _synthetic_series(max(expected, ppd_len + 2), p_jump=p_jump)

    event = Ev("575", ppd_start, ppd_end)

    win = select_response_window(event, series)
    assert win is not None, "Window unexpectedly None"
    oil_start, oil_end = win

    actual = (oil_end - oil_start).days + 1
    assert (
        actual == expected
    ), f"got {actual}, expected {expected} (ppd_len={ppd_len}, p_jump={p_jump})"


# -----------------------------------------------------------------------------
# Тест make_window_passport ----------------------------------------------------
# -----------------------------------------------------------------------------

def test_make_window_passport_vals():
    """
    Проверяем, что make_window_passport корректно вытягивает
    значения q/p на границах и считает длительность.
    """
    series = _synthetic_series(12, q_base=110, p_base=50, p_jump=0)
    oil_start = series.index[1]
    oil_end = series.index[10]

    passport = make_window_passport(series, oil_start, oil_end)

    # sanity-чек на наличие и правильность полей
    assert passport is not None
    assert passport["q_start"] == 110
    assert passport["p_end"] == 50
    assert passport["duration_days_oil"] == 10
