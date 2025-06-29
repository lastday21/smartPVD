from __future__ import annotations

"""
Этот модуль содержит две основные функции:
  1. select_response_window(ev_ppd, series) -> (oil_start, oil_end) | None
     Определяет границы окна отклика на основе данных о событии ППД и ряда
     показателей нефтяной скважины.
  2. make_window_passport(series, oil_start, oil_end) -> dict | None
     Формирует «паспорт» окна: ключевые данные о скважине (дебит, давление,
     длительность) на границах рассчитанного периода.

Аргумент ev_ppd может быть представлен:
  • namedtuple или pandas.Series с атрибутами `start` и `end`
  • pandas.DataFrame-строка с колонками `start_date` и `end_date`
"""

from typing import Optional, Tuple, Dict, Any
import pandas as pd

# -----------------------------------------------------------------------------
# Константы порогов (обычно их определяют в config.py) ----------------------
# -----------------------------------------------------------------------------
try:
    from config import (
        LAG_DAYS,           # задержка между началом ППД-события и реакцией нефтянки (в сутках)
        OIL_CHECK_DAYS,     # базовый размер окна (10 суток)
        OIL_DELTA_P_THRESH, # порог изменения давления (5 атм), после которого расширяем окно
        OIL_EXTEND_DAYS,    # расширенный размер окна (30 суток) при ΔP ≥ OIL_DELTA_P_THRESH
    )
except ImportError:
    # Если config.py не найден (например, в контексте pytest),
    # задаём значения по умолчанию для корректного тестирования.
    LAG_DAYS = 1
    OIL_CHECK_DAYS = 10
    OIL_DELTA_P_THRESH = 5.0
    OIL_EXTEND_DAYS = 30


# -----------------------------------------------------------------------------
# Вспомогательные функции ------------------------------------------------------
# -----------------------------------------------------------------------------

def _get_event_dates(ev_ppd) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Извлекает даты начала и конца события ППД.
    Работает с объектом ev_ppd, который может быть:
      • namedtuple / Series с полями `start` и `end`
      • DataFrame-строка с колонками `start_date` и `end_date`

    Возвращает кортеж (start, end) в виде pd.Timestamp.
    Выбрасывает AttributeError, если не находит нужных полей.
    """
    # пытаемся получить атрибуты start/end
    start_raw = getattr(ev_ppd, "start", getattr(ev_ppd, "start_date", None))
    end_raw = getattr(ev_ppd, "end", getattr(ev_ppd, "end_date", None))
    if start_raw is None or end_raw is None:
        raise AttributeError("ev_ppd must have start/end or start_date/end_date")
    # приводим к pd.Timestamp для дальнейших вычислений
    return pd.to_datetime(start_raw), pd.to_datetime(end_raw)


# -----------------------------------------------------------------------------
# Основные функции ------------------------------------------------------------
# -----------------------------------------------------------------------------

def select_response_window(
    ev_ppd,
    series: pd.DataFrame
) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Рассчитывает границы окна отклика нефтянной скважины на событие ППД.

    Логика:
      1. Сдвигаем начало окна на LAG_DAYS от ppd_start.
      2. Определяем базовое окончание base_end = oil_start + (OIL_CHECK_DAYS - 1),
         но не дальше чем ppd_end + LAG_DAYS.
      3. Извлекаем столбец "p_oil" на отрезке [oil_start:base_end] и считаем
         максимальное абсолютное изменение давления относительно p0.
      4. Если ΔP ≥ OIL_DELTA_P_THRESH, расширяем окончание до oil_start + (OIL_EXTEND_DAYS - 1).
      5. Ограничиваем окончание не позднее чем ppd_end + LAG_DAYS (чтобы окно не выходило за событие ППД).
      6. Если найденное сырцовое окончание cand_end выходит за пределы доступных дат series,
         обрезаем его до series.index.max(). Если oil_start > series.index.max(), возвращаем None.

    Аргументы:
      ev_ppd: объект с информацией о событии ППД (start, end или start_date, end_date).
      series: pd.DataFrame с индексом типа pd.DatetimeIndex и колонкой "p_oil"
              (приёмное давление) и, предположительно, колонкой "q_oil" (но для выбора окна нужен только "p_oil").

    Возвращает:
      • (oil_start, oil_end) – кортеж pd.Timestamp, если окно получилось построить
      • None – в случае, если series пуст или отсутствуют данные давления в нужном диапазоне.
    """
    # Получаем чистые pd.Timestamp для начала и конца события ППД
    ppd_start, ppd_end = _get_event_dates(ev_ppd)

    # Проверка, что данные по скважине есть
    if series.empty:
        return None

    # 1. Начало окна = ppd_start + LAG_DAYS
    oil_start = ppd_start + pd.Timedelta(days=LAG_DAYS)

    # 2. Базовое окончание: 0–(OIL_CHECK_DAYS-1) дней после oil_start,
    #    но не дальше чем конец события ППД + LAG_DAYS на конце
    base_end = oil_start + pd.Timedelta(days=OIL_CHECK_DAYS - 1)
    base_end = min(base_end, ppd_end + pd.Timedelta(days=LAG_DAYS))

    # 3. Берём данные давления на отрезке [oil_start:base_end]
    try:
        p_window = series.loc[oil_start:base_end, "p_oil"]
    except KeyError:
        # если хотя бы одна из границ не существует в index, или нет колонки "p_oil"
        return None

    if p_window.empty:
        # Если в базовом окне нет строк (например, пробел в данных) – выходим
        return None

    # Вычисляем максимальное абсолютное отклонение давления от начальной точки
    delta_p_max = (p_window - p_window.iloc[0]).abs().max()

    # 4. Изначально кандидатная граница совпадает с base_end
    cand_end = base_end
    # Если изменение ≥ порога, расширяем окно до OIL_EXTEND_DAYS
    if delta_p_max >= OIL_DELTA_P_THRESH:
        cand_end = oil_start + pd.Timedelta(days=OIL_EXTEND_DAYS - 1)

    # 5. Не позволяем окну выходить за конец события ППД + LAG_DAYS
    cand_end = min(cand_end, ppd_end + pd.Timedelta(days=LAG_DAYS))

    # 6. Проверяем, что cand_end не выходит за доступный диапазон series.index
    if cand_end > series.index.max():
        # обрезаем до последней доступной даты
        cand_end = series.index.max()

    # Если начало окна тоже позже, чем любые даты в series, возвращаем None
    if oil_start > series.index.max():
        return None

    # Возвращаем рассчитанные границы
    return oil_start, cand_end


def make_window_passport(
    series: pd.DataFrame,
    oil_start: pd.Timestamp,
    oil_end: pd.Timestamp
) -> Optional[Dict[str, Any]]:
    """
    Формирует «паспорт» окна отклика, доставая из series ключевые параметры:
      • well (имя скважины) – строковый тип
      • oil_start / oil_end – границы окна
      • q_start / q_end   – дебит на границах
      • p_start / p_end   – давление на границах
      • duration_days_oil – длительность окна (days)

    Дополнительно (опционально) берёт из row_start поля "field", "Куст" или "cluster",
    если они присутствуют в DataFrame, и добавляет их в словарь.

    Аргументы:
      series: DataFrame, где индекс – pd.DatetimeIndex, содержащий хотя бы даты oil_start и oil_end,
              а в колонках есть "q_oil", "p_oil" и, возможно, "field", "Куст", "cluster".
      oil_start, oil_end: pd.Timestamp, границы окна, которые должны присутствовать в series.index.

    Возвращает:
      • dict с полями (см. пример ниже) или
      • None, если series не содержит oil_start или oil_end среди индексов.
    """
    try:
        # Пытаемся достать строки в точных датах oil_start и oil_end
        row_start = series.loc[oil_start]
        row_end = series.loc[oil_end]
    except KeyError:
        # Если в series нет данных на этих датах, ничего не возвращаем
        return None

    # Считаем длительность в днях (включая оба конца)
    duration_days = (oil_end - oil_start).days + 1

    # Основной паспорт с обязательными полями
    passport: Dict[str, Any] = {
        "well": str(row_start.get("well", "")),
        "oil_start": oil_start,
        "q_start": row_start.get("q_oil", None),
        "p_start": row_start.get("p_oil", None),
        "oil_end": oil_end,
        "q_end": row_end.get("q_oil", None),
        "p_end": row_end.get("p_oil", None),
        "duration_days_oil": duration_days,
    }

    # Дополнительные поля: если в series есть "field", "Куст" или "cluster", копируем их
    for extra in ("field", "Куст", "cluster"):
        if extra in row_start:
            passport[extra] = row_start[extra]

    return passport
