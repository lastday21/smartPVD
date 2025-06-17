
"""
Что делает модуль
-----------------
1. Принимает суточный DataFrame *ppd_df* с колонками
   «well, date, q_ppd» (расход) и, опционально, «p_cust» (кустовое давление).
2. Для каждой скважины ищет периоды резкого отклонения дебита («dev-дни»)
   относительно скользящего baseline.
3. Когда накапливается серия из **PPD_MIN_EVENT_DAYS** подряд dev-дней,
   фиксируется начало события; окончание определяется аналогично.
4. Для каждого события рассчитываются:
   – *baseline_before* (дебит до события)
   – *baseline_during* (дебит по ходу события)
   – *min_q* / *max_q*  экстремумы внутри события
   – *duration_days*   длительность события
   Все четыре показателя округляются до целого `int`.
5. Возвращает DataFrame **ppd_events** со столбцами
   `['well','start_date','end_date','baseline_before','baseline_during',
     'min_q','max_q','duration_days']`.

Константы `PPD_REL_THRESH`, `PPD_MIN_EVENT_DAYS`, `PPD_WINDOW_SIZE`
подтягиваются из *config.py* — они задают чувствительность алгоритма.

Колонки на входе (`ppd_daily`):
    well | date | q_ppd | … (остальные не используются)
Колонки на выходе (`ppd_events`):
    well, start_date, end_date,
    baseline_before, baseline_during,
    min_q, max_q, duration_days
"""

import os
from datetime import timedelta

import numpy as np
import pandas as pd

from config import PPD_REL_THRESH, PPD_MIN_EVENT_DAYS, PPD_WINDOW_SIZE


def detect_ppd_events(ppd_df: pd.DataFrame) -> pd.DataFrame:
    """
    Анализирует суточный ряд q_ppd каждой скважины и формирует таблицу событий.

    Параметры
    ----------
    ppd_df : pd.DataFrame
        Обязательные колонки:
            • 'well'  – идентификатор скважины (str / int)
            • 'date'  – дата измерения (pd.Timestamp или строка 'DD.MM.YYYY')
            • 'q_ppd' – расход приёма, м³/сут (float / int)

        Предполагается, что данные уже очищены предобработкой:
        даты валидны, NaN удалены, строки отсортированы по дате внутри каждой
        скважины.

    Возвращает
    ----------
    pd.DataFrame
        Столбцы:
        'well', 'start_date', 'end_date',
        'baseline_before', 'baseline_during',
        'min_q', 'max_q', 'duration_days'
    """
    # --- 0. Убедимся, что date – это datetime --------------------------------
    if not np.issubdtype(ppd_df["date"].dtype, np.datetime64):
        ppd_df = ppd_df.copy()
        ppd_df["date"] = pd.to_datetime(ppd_df["date"], dayfirst=True)

    all_events = []

    def _mean(lst):
        """Небольшой helper: среднее списка или 0.0, если список пуст."""
        return float(np.mean(lst)) if lst else 0.0

    # --- 1. Проходим по всем скважинам ---------------------------------------
    for well, group in ppd_df.groupby("well"):
        dfw = group.sort_values("date").reset_index(drop=True)
        dates = dfw["date"].tolist()
        values = dfw["q_ppd"].tolist()

        window = []      # скользящее окно (healthy + dev), макс. PPD_WINDOW_SIZE
        devs = []        # подрядные dev-дни (фильтруются внутри цикла)

        in_event = False
        start_date = None
        baseline_before = None
        baseline_during = None

        event_min = None
        event_max = None

        for today, q in zip(dates, values):
            # 1.1 Текущий baseline_during = среднее по window
            baseline_during = _mean([v for (_, v) in window]) if window else q

            # 1.2 Относительное отклонение от baseline
            if baseline_during == 0.0:
                rel = 0.0 if q == 0.0 else float("inf")
            else:
                rel = abs(q - baseline_during) / baseline_during

            # ---------------- Фаза «вне события» ------------------------------
            if not in_event:
                if rel < PPD_REL_THRESH:          # healthy-день
                    devs.clear()
                    window.append((today, q))
                    if len(window) > PPD_WINDOW_SIZE:
                        window.pop(0)
                    continue

                # dev-день до старта: фильтрация внутри devs
                devs.append((today, q))
                while devs:
                    mean_devs = _mean([v for (_, v) in devs])
                    rel_dev = (abs(q - mean_devs) / mean_devs) if mean_devs else 0.0
                    if rel_dev < PPD_REL_THRESH:
                        break
                    devs.pop(0)

                if len(devs) < PPD_MIN_EVENT_DAYS:
                    continue  # ждем накопления серии dev-дней

                # ---- старт события ------------------------------------------
                start_date = devs[0][0]
                baseline_before = baseline_during

                # инициализация экстремумов по первым dev-точкам
                values_init = [v for (_, v) in devs]
                event_min = min(values_init)
                event_max = max(values_init)

                window.clear()
                window.extend(devs)
                in_event = True
                devs.clear()
                # далее переходим к блоку «in_event»

            # ---------------- Фаза «внутри события» ---------------------------
            if in_event:
                baseline_during = _mean([v for (_, v) in window])

                if rel < PPD_REL_THRESH:          # healthy-день внутри события
                    devs.clear()
                    window.append((today, q))
                    if len(window) > PPD_WINDOW_SIZE:
                        window.pop(0)
                    event_min = min(event_min, q) if event_min is not None else q
                    event_max = max(event_max, q) if event_max is not None else q
                    continue

                # dev-день внутри события
                devs.append((today, q))
                while devs:
                    mean_devs = _mean([v for (_, v) in devs])
                    rel_dev = (abs(q - mean_devs) / mean_devs) if mean_devs else 0.0
                    if rel_dev < PPD_REL_THRESH:
                        break
                    devs.pop(0)

                event_min = min(event_min, q) if event_min is not None else q
                event_max = max(event_max, q) if event_max is not None else q

                if len(devs) < PPD_MIN_EVENT_DAYS:
                    continue

                # ---- закрываем событие --------------------------------------
                end_date = devs[0][0] - timedelta(days=1)
                duration = (end_date - start_date).days + 1

                win_vals = [v for (_, v) in window]
                min_q = int(round(min(win_vals)))
                max_q = int(round(max(win_vals)))

                all_events.append({
                    "well": well,
                    "start_date": start_date.strftime("%d-%m-%Y"),
                    "end_date": end_date.strftime("%d-%m-%Y"),
                    "baseline_before": int(round(baseline_before)),
                    "baseline_during": int(round(baseline_during)),
                    "min_q": min_q,
                    "max_q": max_q,
                    "duration_days": duration,
                })

                # --- новая инициализация для возможного следующего события ---
                window.clear()
                window.extend(devs)
                start_date = devs[0][0]
                baseline_before = baseline_during
                baseline_during = _mean([v for (_, v) in devs])

                values_new_dev = [v for (_, v) in devs]
                event_min = min(values_new_dev)
                event_max = max(values_new_dev)

                devs.clear()
                in_event = True  # остаёмся «внутри» нового события

        # --- 2. Конец ряда: фиксируем незакрытое событие ----------------------
        if in_event and window:
            end_date = window[-1][0]
            duration = (end_date - start_date).days + 1
            win_vals = [v for (_, v) in window]
            min_q = int(round(min(win_vals)))
            max_q = int(round(max(win_vals)))

            all_events.append({
                "well": well,
                "start_date": start_date.strftime("%d-%m-%Y"),
                "end_date": end_date.strftime("%d-%m-%Y"),
                "baseline_before": int(round(baseline_before)),
                "baseline_during": int(round(baseline_during)),
                "min_q": min_q,
                "max_q": max_q,
                "duration_days": duration,
            })

    # --- 3. Собираем финальный DataFrame -------------------------------------
    events_df = pd.DataFrame(all_events, columns=[
        "well", "start_date", "end_date",
        "baseline_before", "baseline_during",
        "min_q", "max_q", "duration_days",
    ])

    # столбцы-инт вершём в object, чтобы сохранить «точно int» без float-хвостов
    for col in ("baseline_before", "baseline_during", "min_q", "max_q"):
        events_df[col] = events_df[col].astype(object)

    return events_df


# ---------------------------------------------------------------------------
# CLI-режим: запуск «как скрипт»
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # 1) входной CSV с суточными PPD-рядами
    input_path = os.path.join("clean_data", "ppd_clean.csv")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Не найден файл: {input_path}")

    ppd_df = pd.read_csv(input_path)
    # приведение date → datetime при необходимости
    if not np.issubdtype(ppd_df["date"].dtype, np.datetime64):
        try:
            ppd_df["date"] = pd.to_datetime(ppd_df["date"], dayfirst=True)
        except Exception:
            ppd_df["date"] = pd.to_datetime(ppd_df["date"])

    # 2) детекция событий
    events_df = detect_ppd_events(ppd_df)

    # 3) вывод
    output_path = os.path.join("clean_data", "ppd_events.csv")
    events_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Обнаружено событий: {len(events_df)}  →  {output_path}")