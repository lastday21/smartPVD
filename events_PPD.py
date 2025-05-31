# events_PPD.py

import pandas as pd
import numpy as np
from datetime import timedelta
from config import PPD_REL_THRESH, PPD_MIN_EVENT_DAYS, PPD_WINDOW_SIZE

def detect_ppd_events(ppd_df: pd.DataFrame) -> pd.DataFrame:
    """
    Обнаруживает события по PPD для каждой скважины.

    Вход:
        ppd_df: pd.DataFrame с колонками:
            - 'well'  : идентификатор скважины (любой hashable, например строка)
            - 'date'  : pd.Timestamp или строка в формате 'YYYY-MM-DD' (вЫбрать единый формат)
            - 'q_ppd' : float или int — измеренное значение PPD для этой скважины в этот день.

        Предполагается, что:
        1) Данные уже очищены (нет дубликатов (well, date) и нет NaN в q_ppd).
        2) Для каждой скважины в `ppd_df` есть ровно по одной записи на каждую дату (или даты без замера пропущены).

    Выход:
        pd.DataFrame со следующими колонками:
            - 'well'
            - 'start_date'        : дата начала события (pd.Timestamp)
            - 'end_date'          : дата конца события (pd.Timestamp)
            - 'baseline_before'   : baseline (среднее) до старта события
            - 'baseline_during'   : baseline (среднее) по первым PPD-значениям dev_list (PPD_MIN_EVENT_DAYS точек)
            - 'min_q'             : минимальное q_ppd в итоговом окне (window) к моменту закрытия события
            - 'max_q'             : максимальное q_ppd в итоговом окне (window) к моменту закрытия события
            - 'duration_days'     : длительность события (end_date − start_date + 1)
    """
    # Убедимся, что date в datetime
    if not np.issubdtype(ppd_df['date'].dtype, np.datetime64):
        ppd_df = ppd_df.copy()
        ppd_df['date'] = pd.to_datetime(ppd_df['date'])

    # Результирующий список словарей
    all_events = []

    # Проходим по каждой скважине отдельно
    for well, group in ppd_df.groupby('well'):
        # Сортируем по дате
        dfw = group.sort_values('date').reset_index(drop=True)
        dates = dfw['date'].tolist()
        values = dfw['q_ppd'].tolist()

        # Скользящее окно «здоровых» точек: хранит список кортежей (date, q_ppd)
        window = []  # [(pd.Timestamp, float), ...]
        # Список подрядных «dev»-точек
        devs = []    # [(pd.Timestamp, float), ...]

        in_event = False
        start_date = None
        baseline_before = None
        baseline_during = None

        def compute_baseline(vals_list):
            # vals_list — список float
            return float(np.mean(vals_list)) if len(vals_list) > 0 else 0.0

        # Проходим по всем измерениям этой скважины
        for today, q in zip(dates, values):
            # 1) вычисляем текущий baseline (если окно не пусто)
            if window:
                current_baseline = compute_baseline([v for (_, v) in window])
            else:
                # Если окно пустое (первый день), baseline = текущее измерение
                current_baseline = q

            # 2) вычисляем rel = |q − baseline| / baseline
            if current_baseline == 0.0:
                # при baseline = 0: если q == 0 → rel = 0, иначе → rel = бесконечность
                rel = 0.0 if q == 0.0 else float('inf')
            else:
                rel = abs(q - current_baseline) / current_baseline

            if not in_event:
                # Если мы ещё не в событии:
                if rel < PPD_REL_THRESH:
                    # «здоровый» день
                    devs.clear()
                    window.append((today, q))
                    # поддерживаем размер окна не больше PPD_WINDOW_SIZE
                    if len(window) > PPD_WINDOW_SIZE:
                        window.pop(0)
                else:
                    # «dev»-день
                    devs.append((today, q))
                    if len(devs) < PPD_MIN_EVENT_DAYS:
                        # ещё не накоплено 7 подряд dev-дней
                        continue

                    # накопилось PPD_MIN_EVENT_DAYS подряд dev-дней → старт события
                    start_date = devs[0][0]
                    baseline_before = current_baseline
                    baseline_during = compute_baseline([v for (_, v) in devs])

                    # очищаем окно и заполняем именно этими PPD_MIN_EVENT_DAYS точками
                    window.clear()
                    window.extend(devs)

                    in_event = True
                    devs.clear()

            else:
                # Мы «внутри» события
                current_baseline = compute_baseline([v for (_, v) in window])
                if current_baseline == 0.0:
                    rel = 0.0 if q == 0.0 else float('inf')
                else:
                    rel = abs(q - current_baseline) / current_baseline

                if rel < PPD_REL_THRESH:
                    # «здоровый» в пределах события
                    devs.clear()
                    window.append((today, q))
                    if len(window) > PPD_WINDOW_SIZE:
                        window.pop(0)
                else:
                    # «dev»-день внутри события
                    devs.append((today, q))
                    if len(devs) < PPD_MIN_EVENT_DAYS:
                        continue

                    # накопилось PPD_MIN_EVENT_DAYS подряд dev-дней → закрываем событие
                    end_date = devs[0][0] - timedelta(days=1)
                    # собираем метрики
                    min_q = min(v for (_, v) in window)
                    max_q = max(v for (_, v) in window)
                    duration = (end_date - start_date).days + 1

                    all_events.append({
                        'well': well,
                        'start_date': start_date,
                        'end_date': end_date,
                        'baseline_before': baseline_before,
                        'baseline_during': baseline_during,
                        'min_q': min_q,
                        'max_q': max_q,
                        'duration_days': duration
                    })

                    # Запускаем следующее событие сразу:
                    # 7 dev-точек переносятся в окно, они уже «начало» нового события
                    window.clear()
                    window.extend(devs)

                    start_date = devs[0][0]
                    baseline_before = current_baseline
                    baseline_during = compute_baseline([v for (_, v) in devs])

                    devs.clear()
                    in_event = True  # остаёмся «внутри» нового события

        # В конце, если мы всё ещё «внутри события» и данных больше нет,
        # фиксируем его до последнего дня
        if in_event and window:
            end_date = window[-1][0]
            min_q = min(v for (_, v) in window)
            max_q = max(v for (_, v) in window)
            duration = (end_date - start_date).days + 1

            all_events.append({
                'well': well,
                'start_date': start_date,
                'end_date': end_date,
                'baseline_before': baseline_before,
                'baseline_during': baseline_during,
                'min_q': min_q,
                'max_q': max_q,
                'duration_days': duration
            })

    # Собираем всё в единый DataFrame
    events_df = pd.DataFrame(all_events, columns=[
        'well', 'start_date', 'end_date',
        'baseline_before', 'baseline_during',
        'min_q', 'max_q', 'duration_days'
    ])
    return events_df

ppd_df = pd.read_csv('clean_data/ppd_clean.csv')
ppd_df['date'] = pd.to_datetime(ppd_df['date'], dayfirst=True)  # если даты в формате DD.MM.YYYY

events_df = detect_ppd_events(ppd_df)
events_df.to_csv('ppd_all_wells_events.csv', index=False)
