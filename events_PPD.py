import os
import pandas as pd
import numpy as np
from datetime import timedelta
from config import PPD_REL_THRESH, PPD_MIN_EVENT_DAYS, PPD_WINDOW_SIZE

def detect_ppd_events(ppd_df: pd.DataFrame) -> pd.DataFrame:
    """
    Обнаруживает события по PPD с учётом:
    - скользящего окна (до PPD_WINDOW_SIZE) здоровых и dev-точек,
    - фильтрации dev-точек скользящим средним внутри devs,
    - накопления глобальных min/max внутри события,
    - и приводит все числовые метрики (baseline_before, baseline_during, min_q, max_q) к целым.

    Вход:
        ppd_df: pd.DataFrame с колонками:
            - 'well'  : идентификатор скважины
            - 'date'  : pd.Timestamp или строка формата YYYY-MM-DD
            - 'q_ppd' : float или int — измеренное значение PPD

        Предполагается, что данные очищены:
        нет NaN в 'q_ppd', даты корректны;
        для каждой скважины есть записи последовательно по датам.

    Выход:
        pd.DataFrame со столбцами:
          'well', 'start_date', 'end_date',
          'baseline_before', 'baseline_during',
          'min_q', 'max_q', 'duration_days'
          где baseline_before, baseline_during, min_q, max_q — целые int.
    """

    # Убедимся, что столбец date в формате datetime
    if not np.issubdtype(ppd_df['date'].dtype, np.datetime64):
        ppd_df = ppd_df.copy()
        ppd_df['date'] = pd.to_datetime(ppd_df['date'], dayfirst=True)

    all_events = []

    def compute_baseline(vals_list):
        """Возвращает среднее по списку чисел, либо 0.0, если список пуст."""
        return float(np.mean(vals_list)) if len(vals_list) > 0 else 0.0

    # Группируем по каждой скважине
    for well, group in ppd_df.groupby('well'):
        dfw = group.sort_values('date').reset_index(drop=True)
        dates = dfw['date'].tolist()
        values = dfw['q_ppd'].tolist()

        window = []     # FIFO: последние здоровые + начальные dev-точки (максимум PPD_WINDOW_SIZE)
        devs = []       # подрядные dev-точки (фильтруемые скользящим средним внутри devs)
        in_event = False
        start_date = None
        baseline_before = None
        baseline_during = None

        # Глобальные экстремумы (min/max) в пределах текущего события
        event_min = None
        event_max = None

        for today, q in zip(dates, values):
            # 1) Вычисляем текущее baseline_during как среднее по window (или q, если window пуст)
            if window:
                baseline_during = compute_baseline([v for (_, v) in window])
            else:
                baseline_during = q

            # 2) Вычисляем rel относительно baseline_during
            if baseline_during == 0.0:
                rel = 0.0 if q == 0.0 else float('inf')
            else:
                rel = abs(q - baseline_during) / baseline_during

            # 3) Фаза «нет события»: накопление до первой серии из PPD_MIN_EVENT_DAYS dev-дней
            if not in_event:
                if rel < PPD_REL_THRESH:
                    # Здоровый день до старта
                    devs.clear()
                    window.append((today, q))
                    if len(window) > PPD_WINDOW_SIZE:
                        window.pop(0)
                else:
                    # Dev-день до старта: фильтрация внутри devs
                    devs.append((today, q))
                    while devs:
                        mean_devs = compute_baseline([v for (_, v) in devs])
                        if mean_devs == 0.0:
                            rel_dev = 0.0 if q == 0.0 else float('inf')
                        else:
                            rel_dev = abs(q - mean_devs) / mean_devs

                        if rel_dev < PPD_REL_THRESH:
                            break
                        devs.pop(0)

                    if len(devs) < PPD_MIN_EVENT_DAYS:
                        continue  # ждём накопления PPD_MIN_EVENT_DAYS подряд dev-точек

                    # --- Старт первого события ---
                    start_date = devs[0][0]
                    baseline_before = baseline_during

                    # Инициализируем global min/max по первым PPD_MIN_EVENT_DAYS dev-точкам
                    values_initial_dev = [v for (_, v) in devs]
                    event_min = min(values_initial_dev)
                    event_max = max(values_initial_dev)

                    # Очищаем window и переносим туда эти PPD_MIN_EVENT_DAYS точек
                    window.clear()
                    window.extend(devs)

                    in_event = True
                    devs.clear()
                    # После старта события, переходим к обработке как «in_event»

            # 4) Фаза «внутри события»
            if in_event:
                #   a) Пересчитываем baseline_during = среднее window
                baseline_during = compute_baseline([v for (_, v) in window])

                #   b) Определяем, healthy или dev внутри события
                if rel < PPD_REL_THRESH:
                    # Здоровый день внутри события
                    devs.clear()
                    window.append((today, q))
                    if len(window) > PPD_WINDOW_SIZE:
                        window.pop(0)
                    # Обновляем global min/max всего события
                    event_min = q if event_min is None else min(event_min, q)
                    event_max = q if event_max is None else max(event_max, q)
                    continue

                # Dev-день внутри события: добавляем + фильтрация
                devs.append((today, q))
                while devs:
                    mean_devs = compute_baseline([v for (_, v) in devs])
                    if mean_devs == 0.0:
                        rel_dev = 0.0 if q == 0.0 else float('inf')
                    else:
                        rel_dev = abs(q - mean_devs) / mean_devs

                    if rel_dev < PPD_REL_THRESH:
                        break
                    devs.pop(0)

                # Каждая dev-точка тоже обновляет global min/max
                event_min = q if event_min is None else min(event_min, q)
                event_max = q if event_max is None else max(event_max, q)

                if len(devs) < PPD_MIN_EVENT_DAYS:
                    continue

                # --- Закрываем текущее событие (вторая подрядная серия из PPD_MIN_EVENT_DAYS dev-дней) ---
                end_date = devs[0][0] - timedelta(days=1)
                duration = (end_date - start_date).days + 1

                win_vals = [v for (_, v) in window]
                min_q = int(round(min(win_vals)))
                max_q = int(round(max(win_vals)))

                all_events.append({
                    'well': well,
                    'start_date': start_date.strftime('%d-%m-%Y'),
                    'end_date': end_date.strftime('%d-%m-%Y'),
                    # Приводим baseline_before и baseline_during к целым
                    'baseline_before': int(round(baseline_before)),
                    'baseline_during': int(round(baseline_during)),
                    'min_q': min_q,
                    'max_q': max_q,
                    'duration_days': duration
                })

                # --- Запускаем новое событие из этих PPD_MIN_EVENT_DAYS dev-точек ---
                window.clear()
                window.extend(devs)

                start_date = devs[0][0]
                baseline_before = baseline_during
                baseline_during = compute_baseline([v for (_, v) in devs])

                # Инициализируем global min/max для нового события
                values_new_dev = [v for (_, v) in devs]
                event_min = min(values_new_dev)
                event_max = max(values_new_dev)

                devs.clear()
                in_event = True  # остаёмся «внутри» нового события

        # 5) В конце, если всё ещё in_event, фиксируем незакрытое событие
        if in_event and window:
            end_date = window[-1][0]
            duration = (end_date - start_date).days + 1

            win_vals = [v for (_, v) in window]
            min_q = int(round(min(win_vals)))
            max_q = int(round(max(win_vals)))

            all_events.append({
                'well': well,
                'start_date': start_date.strftime('%d-%m-%Y'),
                'end_date': end_date.strftime('%d-%m-%Y'),
                'baseline_before': int(round(baseline_before)),
                'baseline_during': int(round(baseline_during)),
                'min_q': min_q,
                'max_q': max_q,
                'duration_days': duration
            })

    # Собираем результаты в DataFrame
    events_df = pd.DataFrame(all_events, columns=[
        'well', 'start_date', 'end_date',
        'baseline_before', 'baseline_during',
        'min_q', 'max_q', 'duration_days'
    ])
    for col in ['baseline_before', 'baseline_during', 'min_q', 'max_q']:
        # 1) Достаём все значения, превращаем каждый в Python int
        py_list = [int(x) for x in events_df[col].tolist()]
        # 2) Присваиваем обратно и сразу меняем тип столбца на object
        events_df[col] = pd.Series(py_list, dtype=object)
    return events_df




if __name__ == "__main__":
    # -----------------------------------------------
    # Когда модуль запускается напрямую, он автоматически:
    # 1) читает «clean_data/ppd_clean.csv»
    # 2) запускает detect_ppd_events()
    # 3) сохраняет результат в «ppd_events.csv»
    # -----------------------------------------------

    # Предполагаем, что файл лежит по пути «clean_data/ppd_clean.csv»
    input_path = os.path.join("clean_data", "ppd_clean.csv")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Не найден файл: {input_path}")

    # Читаем входные данные
    ppd_df = pd.read_csv(input_path)
    # Если столбец date не в datetime, приводим:
    if not np.issubdtype(ppd_df["date"].dtype, np.datetime64):
        # Попробуем догадаться: чаще всего в формате «DD.MM.YYYY»
        try:
            ppd_df["date"] = pd.to_datetime(ppd_df["date"], dayfirst=True)
        except Exception:
            ppd_df["date"] = pd.to_datetime(ppd_df["date"])

    # Запускаем обнаружение событий
    events_df = detect_ppd_events(ppd_df)

    # Сохраняем результат в «ppd_events.csv» рядом с этим скриптом
    output_path = "ppd_events.csv"
    events_df.to_csv(output_path, index=False)
    print(f"Обнаружено событий: {len(events_df)}. Сохранено в файл: {output_path}")
