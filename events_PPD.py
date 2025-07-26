"""

========================  О Б З О Р  М О Д У Л Я  ========================
Модуль выделяет интервалы резкого изменения суточного расхода приёма **q_ppd**
по каждой скважине. Алгоритм не зависит от конкретного формата хранения
данных — вся тяжёлая логика инкапсулирована в «чистом ядре»
:pyfunc:`_detect_ppd_events_df`, которое работает **только** с
:pymod:`pandas.DataFrame` и не обращается к файловой системе.

* **Чистое ядро** — `_detect_ppd_events_df(ppd_df)` → `ppd_events`.
* **Универсальный интерфейс** — `detect_ppd_events(...)`, который по выбору
  принимает входные DataFrame-ы *или* загружает их из CSV, а также умеет
  сохранять результат в `clean_data/`.
* **CLI-режим** — запуск модуля как скрипта сохраняет 100 % прежнюю
  функциональность: читает `clean_data/ppd_clean.csv`, детектирует события,
  пишет `clean_data/ppd_events.csv`.

Все параметры чувствительности алгоритма (`PPD_REL_THRESH`,
`PPD_MIN_EVENT_DAYS`, `PPD_WINDOW_SIZE`) берутся из глобального
:mypy:`config.py`.

==========================================================================
"""
from __future__ import annotations
from datetime import timedelta
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from config import (
    PPD_REL_THRESH,       # относительный порог отклонения дебита
    PPD_MIN_EVENT_DAYS,   # минимальная длина серии dev-дней для фиксации события
    PPD_WINDOW_SIZE,      # ширина «скользящего» окна baseline
)

# Папка, где по умолчанию лежат/создаются CSV с промежуточными данными
CLEAN_DIR = Path("clean_data")
CLEAN_DIR.mkdir(exist_ok=True)

# ----------------------------------------------------------------------
#  В С П О М О Г А Т Е Л Ь Н Ы Е  Ф У Н К Ц И И
# ----------------------------------------------------------------------

def _mean(seq: List[float]) -> float:
    """Среднее значение списка *seq*. Если список пуст, возвращает 0.0.

    Функция вынесена отдельно, чтобы избежать дублирования и повысить
    читаемость основного алгоритма.
    """
    return float(np.mean(seq)) if seq else 0.0


# ======================================================================
#                       Ч И С Т О Е   Я Д Р О
# ======================================================================

def _detect_ppd_events_df(ppd_df: pd.DataFrame) -> pd.DataFrame:
    """Определяет интервалы «событий» (аномальной динамики **q_ppd**).

    Алгоритм пошагово сопоставим с первоначальной реализацией *events_PPD.py* —
    разница лишь в том, что полностью исключён файловый ввод/вывод.

    Параметры
    ----------
    ppd_df : pd.DataFrame
        Очищенный суточный ряд приёма жидкости (ППД). Требуемые столбцы::

            well | date | q_ppd [| ...]

        * **well**  — идентификатор скважины (str / int).
        * **date**  — дата наблюдения (`datetime64[ns]`).
        * **q_ppd** — суточный дебит приёма, м³/сут (int / float).

        Датафрейм должен быть предварительно отсортирован по «date» внутри
        каждой скважины, пропуски заполнены.

    Возвращает
    ----------
    pd.DataFrame
        Таблица событий (аналог прежнего `ppd_events.csv`) со столбцами::

            well | start_date | end_date | baseline_before | baseline_during |
            min_q | max_q | duration_days

        Все числовые показатели округлены до `int` без дробной части.
    """
    # --- 0. Приводим колонку date к dtype datetime -------------------------
    if not np.issubdtype(ppd_df["date"].dtype, np.datetime64):
        ppd_df = ppd_df.copy()
        ppd_df["date"] = pd.to_datetime(ppd_df["date"], dayfirst=True, errors="coerce")

    events_out: List[dict] = []

    # --- 1. Обходим каждую скважину отдельно ------------------------------
    for well, group in ppd_df.groupby("well", sort=False):
        # Внутренние локальные структуры
        dfw = group.sort_values("date").reset_index(drop=True)
        dates: List[pd.Timestamp] = dfw["date"].tolist()
        values: List[float] = dfw["q_ppd"].astype(float).tolist()

        # «Скользящее» окно последних точек (healthy + dev)
        window: List[Tuple[pd.Timestamp, float]] = []
        # Массив dev‑дней подряд
        devs: List[Tuple[pd.Timestamp, float]] = []

        in_event = False                   # флаг «мы внутри события»
        start_date: pd.Timestamp | None = None
        baseline_before: float | None = None

        event_min: float | None = None     # экстремумы внутри события
        event_max: float | None = None

        for today, q in zip(dates, values):
            # 1.1 Текущий baseline как среднее по окну
            baseline_during = _mean([v for (_, v) in window]) if window else q

            # 1.2 Относительное отклонение сегодняшнего дебита от baseline
            if baseline_during == 0.0:
                rel = 0.0 if q == 0.0 else float("inf")
            else:
                rel = abs(q - baseline_during) / baseline_during

            # ============= Фаза «вне события» =============================
            if not in_event:
                # healthy‑день → расширяем окно и продолжаем
                if rel < PPD_REL_THRESH:
                    devs.clear()
                    window.append((today, q))
                    if len(window) > PPD_WINDOW_SIZE:
                        window.pop(0)
                    continue

                # dev‑день: аккумулируем и проверяем устойчивость
                devs.append((today, q))
                # Фильтрация «выбросов» внутри dev‑последовательности
                while devs:
                    mean_devs = _mean([v for (_, v) in devs])
                    rel_dev = (abs(q - mean_devs) / mean_devs) if mean_devs else 0.0
                    if rel_dev < PPD_REL_THRESH:
                        break
                    devs.pop(0)

                # Ждём, пока dev‑дней накопится достаточно для события
                if len(devs) < PPD_MIN_EVENT_DAYS:
                    continue

                # ---- СТАРТ СОБЫТИЯ --------------------------------------
                start_date = devs[0][0]
                baseline_before = baseline_during

                # Начальные экстремумы по dev‑дням
                init_vals = [v for (_, v) in devs]
                event_min = min(init_vals)
                event_max = max(init_vals)

                # Перезапуск окна
                window.clear()
                window.extend(devs)
                devs.clear()
                in_event = True

            # ============= Фаза «внутри события» =========================
            if in_event:
                baseline_during = _mean([v for (_, v) in window]) or q

                # healthy‑день внутри события
                if rel < PPD_REL_THRESH:
                    devs.clear()
                    window.append((today, q))
                    if len(window) > PPD_WINDOW_SIZE:
                        window.pop(0)
                    event_min = q if event_min is None else min(event_min, q)
                    event_max = q if event_max is None else max(event_max, q)
                    continue

                # dev‑день внутри события
                devs.append((today, q))
                while devs:
                    mean_devs = _mean([v for (_, v) in devs])
                    rel_dev = (abs(q - mean_devs) / mean_devs) if mean_devs else 0.0
                    if rel_dev < PPD_REL_THRESH:
                        break
                    devs.pop(0)

                event_min = q if event_min is None else min(event_min, q)
                event_max = q if event_max is None else max(event_max, q)

                # Проверяем условие закрытия события
                if len(devs) < PPD_MIN_EVENT_DAYS:
                    continue

                # ---- ЗАКРЫТИЕ СОБЫТИЯ -----------------------------------
                end_date = devs[0][0] - timedelta(days=1)
                duration = (end_date - start_date).days + 1

                win_vals = [v for (_, v) in window]
                min_q = int(round(min(win_vals)))
                max_q = int(round(max(win_vals)))

                events_out.append({
                    "well": well,
                    "start_date": start_date.strftime("%d-%m-%Y"),
                    "end_date": end_date.strftime("%d-%m-%Y"),
                    "baseline_before": int(round(baseline_before)),
                    "baseline_during": int(round(baseline_during)),
                    "min_q": min_q,
                    "max_q": max_q,
                    "duration_days": duration,
                })

                # --- Возможное продолжение (несколько событий подряд) ----
                window.clear()
                window.extend(devs)
                start_date = devs[0][0]
                baseline_before = baseline_during
                baseline_during = _mean([v for (_, v) in devs])
                event_min = min([v for (_, v) in devs])
                event_max = max([v for (_, v) in devs])
                devs.clear()
                in_event = True

        # --- 2. Обрезаем «хвост» незакрытого события ---------------------
        if in_event and window:
            end_date = window[-1][0]
            duration = (end_date - start_date).days + 1
            win_vals = [v for (_, v) in window]
            min_q = int(round(min(win_vals)))
            max_q = int(round(max(win_vals)))

            events_out.append({
                "well": well,
                "start_date": start_date.strftime("%d-%m-%Y"),
                "end_date": end_date.strftime("%d-%m-%Y"),
                "baseline_before": int(round(baseline_before)),
                "baseline_during": int(round(_mean(win_vals))),
                "min_q": min_q,
                "max_q": max_q,
                "duration_days": duration,
            })

    # --- 3. Финальный DataFrame ------------------------------------------
    events_df = pd.DataFrame(events_out, columns=[
        "well", "start_date", "end_date",
        "baseline_before", "baseline_during",
        "min_q", "max_q", "duration_days",
    ])

    # Конвертируем числовые столбцы в object, чтобы исключить float‑хвосты
    for col in ("baseline_before", "baseline_during", "min_q", "max_q"):
        events_df[col] = events_df[col].astype(object)

    return events_df

# ======================================================================
#                     У Н И В Е Р С А Л Ь Н Ы Й   И Н Т Е Р Ф Е Й С
# ======================================================================

def detect_ppd_events(
    *,
    ppd_df: pd.DataFrame | None = None,
    csv_path: str | Path | None = None,
    save_csv: bool = True,
) -> pd.DataFrame:
    """Обёртка над :pyfunc:`_detect_ppd_events_df` с поддержкой I/O.

    Параметры
    ----------
    ppd_df : pd.DataFrame | None, optional
        Если передан DataFrame — используется он, файлы **не** читаются.
    csv_path : str | Path | None, optional
        Путь к CSV с колонками ``well, date, q_ppd``; нужен, когда `ppd_df` не
        задан. По умолчанию берётся «clean_data/ppd_clean.csv».
    save_csv : bool, default **True**
        При *True* результат сохраняется в «clean_data/ppd_events.csv».

    Возвращает
    ----------
    pd.DataFrame
        Таблица детектированных событий (см. описание :pyfunc:`_detect_ppd_events_df`).
    """
    # 1) Получаем входной DataFrame -------------------------------------
    if ppd_df is None:
        csv_path = Path(csv_path or CLEAN_DIR / "ppd_clean.csv")
        if not csv_path.exists():
            raise FileNotFoundError(f"Файл не найден: {csv_path}")
        ppd_df = pd.read_csv(csv_path)
        # дата в datetime, если нужно
        if not np.issubdtype(ppd_df["date"].dtype, np.datetime64):
            ppd_df["date"] = pd.to_datetime(ppd_df["date"], dayfirst=True, errors="coerce")

    # 2) Чистое ядро ----------------------------------------------------
    events_df = _detect_ppd_events_df(ppd_df)

    # 3) Сохранение на диск (по желанию) --------------------------------
    if save_csv:
        out_path = CLEAN_DIR / "ppd_events.csv"
        events_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"✓ События ППД сохранены: {out_path} (N={len(events_df)})")

    return events_df

# ----------------------------------------------------------------------
#                               C L I                                   
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Запуск «как скрипт» полностью повторяет поведение старой версии
    detect_ppd_events(save_csv=True)
