from __future__ import annotations

"""
Создаёт итоговый CSV с окнами отклика нефтяных скважин на события ППД.

Файл читает три подготовленных датасета в *clean_data/*:
    • **oil_clean.csv**   – суточные ряды Q/P по нефтяным скважинам;
    • **ppd_events.csv**  – события на скважинах ППД;
    • **pairs_oil_ppd.csv** – связи «ppd_well ↔ oil_well», формируемые
      модулем ``well_pairs.py``.

Результирующий CSV **oil_windows.csv** содержит колонки ровно в требуемом
порядке:
    well, oil_start, q_start, p_start,
    oil_end, q_end, p_end,
    duration_days_oil,
    ppd_well, ppd_start, ppd_end,
    duration_days_ppd

Если в исходных данных нет `field` или `Куст`, эти колонки всё равно присутствуют,
но значения в них будут ``NaN``.
"""

from pathlib import Path
from typing import Dict, List

import pandas as pd

from window_selector import select_response_window, make_window_passport

# ---------------------------------------------------------------------------
# Пути к входным/выходным файлам
# ---------------------------------------------------------------------------
# DATA_DIR – папка, где лежат все «чистые» CSV
DATA_DIR = Path("clean_data")
# Путь к файлу со «чистыми» данными по нефти
OIL_CLEAN_PATH = DATA_DIR / "oil_clean.csv"
# Путь к файлу со списком PPD-событий
PPD_EVENTS_PATH = DATA_DIR / "ppd_events.csv"
# Файл, где пары PPD ↔ OIL сохраняются в виде двух колонок
PAIRS_PATH = DATA_DIR / "pairs_oil_ppd.csv"
# Куда мы записываем результат – CSV с готовыми окнами
WINDOWS_OUT_PATH = DATA_DIR / "oil_windows.csv"


# ---------------------------------------------------------------------------
# УТИЛИТЫ ЗАГРУЗКИ
# ---------------------------------------------------------------------------

def _read_csv_dates(path: Path, *, date_cols: Dict[str, str]) -> pd.DataFrame:
    """
    Читает CSV и конвертирует указанные колонки в datetime.

    Аргументы:
        path: путь к CSV-файлу;
        date_cols: словарь {старое_имя: новое_имя} колонок с датами.

    Поведение:
        1. Загружает весь DataFrame через pd.read_csv.
        2. Переименовывает колонки старых имён в новые (если они есть).
        3. Для каждого нового имени вызывает pd.to_datetime с dayfirst=True,
           errors="coerce" (чтобы не падать, а получить NaT для недопарсенных строк).
        4. Удаляет строки, где хотя бы одна из перечисленных новых колонок получилась NaT.

    Возвращает:
        DataFrame с гарантировано непустыми значениями в колонках дат.
    """
    df = pd.read_csv(path)
    # Переименовываем и сразу конвертируем в datetime
    for old, new in date_cols.items():
        if old in df.columns:
            df = df.rename(columns={old: new})
        # Конвертация (dayfirst=True, т.к. формат дд.мм.ГГГГ)
        df[new] = pd.to_datetime(df[new], dayfirst=True, errors="coerce")
    # Удаляем пропуски в столбцах, которые должны быть датами
    return df.dropna(subset=date_cols.values())


def _event_dates(ev):
    """
    Возвращает кортеж (start, end) для события PPD, вне зависимости
    от того, как эти поля называются в ev (start/start_date или end/end_date).

    Аргумент:
        ev: namedtuple, Series или любой объект, у которого есть
            атрибуты start и end или start_date и end_date.

    Возвращает:
        (pd.Timestamp(start), pd.Timestamp(end))
    """
    # Защищённый «get»: берём первое найденное имя из списка
    get = lambda *names: next(
        (getattr(ev, n) for n in names if hasattr(ev, n)), None
    )
    return (
        pd.to_datetime(get("start", "start_date")),
        pd.to_datetime(get("end", "end_date"))
    )


def _load_oil_df(path: Path) -> pd.DataFrame:
    """
    Загружает «чистый» oil DataFrame из CSV, конвертирует колонку date
    и приводит well к str, чтобы гарантировать единообразие.

    Ожидаемые колонки в CSV:
        date, well, q_oil, p_oil, [field, Куст, ...]

    Возвращает:
        DataFrame с колонкой 'date' в формате datetime и 'well' как строка.
    """
    # Читаем CSV, переименовывая «date» → «date»
    oil = _read_csv_dates(path, date_cols={"date": "date"})
    # Приводим скважину к строковому типу (чтобы индексация точно совпадала)
    oil["well"] = oil["well"].astype(str)
    return oil


def _load_ppd_events(path: Path) -> pd.DataFrame:
    """
    Загружает DataFrame PPD-событий из CSV, конвертирует start_date/end_date
    в колонки start/end в формате datetime и переименовывает колонку well → ppd_well.

    Ожидаемые колонки в CSV:
        well, start_date, end_date, [плюс другие, но они не используются]

    Возвращает:
        DataFrame с колонкой 'ppd_well' как str и датами в колонках start, end.
    """
    ppd = _read_csv_dates(path, date_cols={"start_date": "start", "end_date": "end"})
    ppd = ppd.rename(columns={"well": "ppd_well"})
    ppd["ppd_well"] = ppd["ppd_well"].astype(str)
    return ppd


def _load_pairs(path: Path) -> Dict[str, List[str]]:
    """
    Загружает связи между PPD-скважинами и oil-скважинами из CSV в виде словаря:
        { ppd_well: [oil_well1, oil_well2, ...], ... }

    Ожидаемые колонки в CSV:
        ppd_well, oil_well

    Возвращает:
        Словарь, где ключ – строка ppd_well, а значение – список строк oil_well.
    """
    pairs = pd.read_csv(path, dtype={"ppd_well": str, "oil_well": str})
    mapping: Dict[str, List[str]] = {}
    for ppd, oil in pairs[["ppd_well", "oil_well"]].itertuples(index=False):
        mapping.setdefault(ppd, []).append(oil)
    return mapping


# ---------------------------------------------------------------------------
# ГЛАВНЫЙ ПРОЦЕСС
# ---------------------------------------------------------------------------

def build_oil_windows(
        ppd_events: pd.DataFrame,
        oil_df: pd.DataFrame,
        mapping: Dict[str, List[str]]
) -> pd.DataFrame:
    """
    Проходит по всем событиям PPD и для каждой связанной нефтянки
    вычисляет окно отклика (oil_start, oil_end) и формирует «паспорт» этого окна.

    Аргументы:
      ppd_events: DataFrame с колонками ['ppd_well', 'start', 'end', ...]
      oil_df:      DataFrame с колонками ['well', 'date', 'q_oil', 'p_oil', ...]
      mapping:     словарь {ppd_well: [oil_well1, oil_well2, ...]}

    Логика:
      1. Конвертируем oil_df['date'] → datetime, сортируем и создаём MultiIndex (well, date).
      2. Проходимся по всем строкам ppd_events (каждое — namedtuple ev):
         a) Извлекаем ppd_start, ppd_end = _event_dates(ev).
         b) Для каждого oil_well в mapping[ev.ppd_well]:
            • Делим oil_df по oil_well → получаем series (DataFrame с индексом date).
            • Вызываем select_response_window(ev, series) → (oil_start, oil_end) или None.
            • Если None – пропускаем. Иначе:
                – Вызываем make_window_passport(series, oil_start, oil_end):
                  passport: dict с { "well", "oil_start","q_start","p_start",... }.
                – Расширяем passport полями:
                    • well                – oil_well (строка)
                    • oil_start, oil_end
                    • duration_days_oil   – (oil_end - oil_start) + 1
                    • ppd_well, ppd_start, ppd_end
                    • duration_days_ppd   – (ppd_end - ppd_start) + 1
                – Сохраняем словарь в список records.
      3. После обхода всех событий/скважин собираем DataFrame из records.
      4. Упорядочиваем колонки ровно в требуемом порядке (см. cols_order) и возвращаем.
    """
    # 1) Конвертация и индексация oil_df
    oil_df["date"] = pd.to_datetime(oil_df["date"], dayfirst=True)
    # Создаём MultiIndex: сначала по «скважине», потом по «дате»
    oil_idx = oil_df.set_index(["well", "date"]).sort_index()

    records: List[dict] = []

    # 2) Обход всех PPD-событий
    for ev in ppd_events.itertuples():
        # Извлекаем даты начала/конца события
        ppd_start, ppd_end = _event_dates(ev)

        # Смотрим, к каким нефтянкам привязан текущий ev.ppd_well
        for oil_well in mapping.get(str(ev.ppd_well), []):
            try:
                # «series» – это DataFrame для конкретной oil_well, индекс = date
                series = oil_idx.loc[oil_well]
            except KeyError:
                # Если в oil_idx нет такой скважины – пропускаем
                continue

            # Вычисляем границы окна
            win = select_response_window(ev, series)
            if win is None:
                # не получилось построить окно – пропускаем
                continue
            oil_start, oil_end = win

            # Собираем паспорт окна
            passport = make_window_passport(series, oil_start, oil_end)
            if passport is None:
                # Если не удалось получить данные на границах – пропускаем
                continue

            # Дополняем паспорт необходимыми полями:
            passport |= {
                "well": oil_well,
                "oil_start": oil_start,
                "oil_end": oil_end,
                "duration_days_oil": (oil_end - oil_start).days + 1,
                #
                "ppd_well": ev.ppd_well,
                "ppd_start": ppd_start,
                "ppd_end": ppd_end,
                "duration_days_ppd": (ppd_end - ppd_start).days + 1,
            }
            # Добавляем в список результатов
            records.append(passport)

    # 3) Формируем итоговый DataFrame и упорядочиваем колонки
    cols_order = [
        "well",
        "oil_start", "q_start", "p_start",
        "oil_end", "q_end", "p_end",
        "duration_days_oil",
        "ppd_well", "ppd_start", "ppd_end",
        "duration_days_ppd",
    ]
    return pd.DataFrame(records)[cols_order]


# ---------------------------------------------------------------------------
# CLI-блок: если этот файл запустить напрямую
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # 1) Загружаем входные таблицы
    oil_df = _load_oil_df(OIL_CLEAN_PATH)
    ppd_df = _load_ppd_events(PPD_EVENTS_PATH)
    mapping = _load_pairs(PAIRS_PATH)

    # 2) Строим окна по всем событиям
    windows_df = build_oil_windows(ppd_df, oil_df, mapping)

    # 3) Если окон нет – сообщаем об этом, иначе сохраняем CSV
    if windows_df.empty:
        print("Не найдено ни одного окна — проверьте входные данные/пары")
    else:
        windows_df.to_csv(WINDOWS_OUT_PATH, index=False)
        print("Сформировано окон:", len(windows_df))
