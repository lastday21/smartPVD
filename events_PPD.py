import pandas as pd
from typing import List, Dict
from config import PPD_BASELINE_DAYS, PPD_REL_THRESH, PPD_MIN_EVENT_DAYS

def detect_ppd_events(ppd_df: pd.DataFrame) -> pd.DataFrame:
    """
    Обнаружение событий PPD с динамической сменой режима (baseline).

    Алгоритм для каждой скважины:
    1. Инициализация baseline = медиана первых PPD_BASELINE_DAYS значений q_ppd.
    2. Проход по датам: если |q - baseline|/baseline >= порог — старт события.
       - Накопить подряд дни, пока условие верно.
       - Если длительность >= PPD_MIN_EVENT_DAYS, зафиксировать событие:
         * baseline_before
         * baseline_during = медиана q внутри события
         * min_q, max_q, relative_change_max
       - Обновить baseline = baseline_during
    3. Если после завершения события нет нового события >= PPD_BASELINE_DAYS дней подряд,
       пересчитать baseline = медиана последних PPD_BASELINE_DAYS значений q_ppd до текущей даты.
    4. Продолжить до конца ряда.

    Возвращает DataFrame событий с колонками:
    ['well', 'start_date', 'end_date', 'duration_days',
     'baseline_before', 'baseline_during', 'min_q', 'max_q', 'relative_change']
    """
    events: List[Dict] = []

    for well, grp in ppd_df.groupby('well'):
        series = grp.sort_values('date').set_index('date')['q_ppd']
        dates = series.index.to_list()
        values = series.values.tolist()
        n = len(dates)
        i = PPD_BASELINE_DAYS  # начнём с дня, следующего после первых PPD_BASELINE_DAYS точек
        # initial baseline_before: median of first PPD_BASELINE_DAYS
        baseline = pd.Series(values[:PPD_BASELINE_DAYS]).median()
        last_event_end_idx = PPD_BASELINE_DAYS - 1

        while i < n:
            q = values[i]
            rel = abs(q - baseline) / baseline if baseline > 0 else 0.0
            # старт нового события?
            if rel >= PPD_REL_THRESH:
                start_idx = i
                window_vals = [q]
                rel_vals = [rel]
                i += 1
                # накапливаем подряд дни события
                while i < n:
                    q_i = values[i]
                    rel_i = abs(q_i - baseline) / baseline if baseline > 0 else 0.0
                    if rel_i < PPD_REL_THRESH:
                        break
                    window_vals.append(q_i)
                    rel_vals.append(rel_i)
                    i += 1
                end_idx = i - 1
                duration = end_idx - start_idx + 1
                # подтверждаем событие по длительности
                if duration >= PPD_MIN_EVENT_DAYS:
                    med_during = pd.Series(window_vals).median()
                    events.append({
                        'well':            well,
                        'start_date':      dates[start_idx],
                        'end_date':        dates[end_idx],
                        'duration_days':   duration,
                        'baseline_before': baseline,
                        'baseline_during': med_during,
                        'min_q':           min(window_vals),
                        'max_q':           max(window_vals),
                        'relative_change': max(rel_vals),
                    })
                    # обновляем baseline режим
                    baseline = med_during
                    last_event_end_idx = end_idx
                    continue
                # иначе шум — игнорируем и продолжаем с i
            # если не событие — проверяем, не пора ли переподсчитать baseline
            if i - last_event_end_idx >= PPD_BASELINE_DAYS:
                # пересчёт baseline из последних PPD_BASELINE_DAYS точек до idx i
                window_start = i - PPD_BASELINE_DAYS
                baseline = pd.Series(values[window_start:i]).median()
                last_event_end_idx = i - 1
            i += 1

    return pd.DataFrame(events)