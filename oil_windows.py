"""oil_windows_refactored.py — построение «окон отклика» нефтяных скважин на
события ППД для конвейера *smartPVD*.

Модуль реализует тотже алгоритм, что и исходный *oil_windows.py*, но разбит
по правилам «чистое ядро + универсальный интерфейс»:

* **Чистое ядро**— :pyfunc:`_build_oil_windows_df`, принимает *только*
  `pandas.DataFrame`‑ы и словарь `mapping`, никаких файловых операций.
* **Универсальный интерфейс**— :pyfunc:`build_oil_windows`, который
  позволяет:

  1. Работать «in‑memory»: передаём готовые DataFrame‑ы, получаем результат
     (по умолчанию *save_csv=False*, чтобы в пайплайне ничего на диск не
     писалось).
  2. Работать «CLI»‑режимом: если DataFrame‑ы не переданы, функция сама
     прочитает CSV‑файлы из *clean_data/* и (при *save_csv=True*) положит
     результат в *clean_data/oil_windows.csv*— идентично старому скрипту.

CLI‑блок в конце модуля полностью повторяет прежнее поведение: запуск
`python oil_windows_refactored.py` формирует файл *oil_windows.csv*.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd

from window_selector import select_response_window, make_window_passport

# ---------------------------------------------------------------------------
# Глобальные константы / пути
# ---------------------------------------------------------------------------
DATA_DIR = Path("clean_data")
DATA_DIR.mkdir(exist_ok=True)

OIL_CLEAN_PATH = DATA_DIR / "oil_clean.csv"
PPD_EVENTS_PATH = DATA_DIR / "ppd_events.csv"
PAIRS_PATH = DATA_DIR / "pairs_oil_ppd.csv"
WINDOWS_OUT_PATH = DATA_DIR / "oil_windows.csv"


# ---------------------------------------------------------------------------
# В С П О М О Г А Т Е Л Ь Н Ы Е   Ф У Н К Ц И И   Д Л Я  I / O
# ---------------------------------------------------------------------------

def _read_csv_dates(path: Path, *, date_cols: Dict[str, str]) -> pd.DataFrame:
    """Читает CSV и конвертирует указанные колонки в *datetime* (day‑first).

    Параметры
    ----------
    path : Path
        Путь к CSV‑файлу.
    date_cols : dict[str, str]
        Словарь вида `{старое_имя : новое_имя}`. Каждая перечисленная
        колонка будет переименована (если требуется) и приведена к
        ``datetime64[ns]``.

    Возвращает
    ----------
    pd.DataFrame
        Загруженный и преобразованный датафрейм без строк с некорректными
        датами (``dropna`` по перечисленным колонкам).
    """
    df = pd.read_csv(path)
    for old, new in date_cols.items():
        if old in df.columns:
            df = df.rename(columns={old: new})
        df[new] = pd.to_datetime(df[new], dayfirst=True, errors="coerce")
    return df.dropna(subset=date_cols.values())


def _load_oil_df(path: Path) -> pd.DataFrame:
    """Загружает *oil_clean.csv* и приводит колонки к нужным типам."""
    df = _read_csv_dates(path, date_cols={"date": "date"})
    df["well"] = df["well"].astype(str)
    return df


def _load_ppd_events(path: Path) -> pd.DataFrame:
    """Загружает *ppd_events.csv* (после детекции событий)."""
    df = _read_csv_dates(path, date_cols={"start_date": "start", "end_date": "end"})
    df = df.rename(columns={"well": "ppd_well"})
    df["ppd_well"] = df["ppd_well"].astype(str)
    return df


def _load_pairs(path: Path) -> Dict[str, List[str]]:
    """Создаёт словарь `mapping` из CSV‑таблицы пар oil ↔ ppd."""
    pairs = pd.read_csv(path, dtype={"ppd_well": str, "oil_well": str})
    mapping: Dict[str, List[str]] = {}
    for ppd, oil in pairs[["ppd_well", "oil_well"]].itertuples(index=False):
        mapping.setdefault(ppd, []).append(oil)
    return mapping


# ---------------------------------------------------------------------------
#                             Ч И С Т О Е   Я Д Р О
# ---------------------------------------------------------------------------

def _build_oil_windows_df(
        ppd_events: pd.DataFrame,
        oil_df: pd.DataFrame,
        mapping: Dict[str, List[str]],
) -> pd.DataFrame:
    """Определяет окна отклика нефти на события ППД (без файловых операций).

    Параметры
    ----------
    ppd_events : pd.DataFrame
        Датафрейм, минимум со столбцами ``ppd_well, start, end`` (``datetime64``).
    oil_df : pd.DataFrame
        Суточные ряды добычи/давления нефти c колонкой ``date`` в формате
        ``datetime64[ns]`` и колонкой ``well``.
    mapping : dict[str, list[str]]
        Словарь соответствий «ppd_well → [oil_well1, …]».

    Возвращает
    ----------
    pd.DataFrame
        Готовая таблица «окон» с фиксированным порядком колонок (см. ниже).
    """
    # 1. Индексируем oil‑данные для быстрого доступа:         (well, date) → row
    oil_df = oil_df.copy()
    oil_df["date"] = pd.to_datetime(oil_df["date"], dayfirst=True, errors="coerce")
    oil_idx = oil_df.set_index(["well", "date"]).sort_index()

    cols_order = [
        "well",
        "oil_start", "q_start", "p_start",
        "oil_end", "q_end", "p_end",
        "duration_days_oil",
        "ppd_well", "ppd_start", "ppd_end",
        "duration_days_ppd",
    ]
    records: List[dict] = []

    # 2. Обходим события ППД
    for ev in ppd_events.itertuples():
        ppd_start = getattr(ev, "start")
        ppd_end = getattr(ev, "end")
        ppd_well = str(ev.ppd_well)

        # 2.1 Все oil‑скважины, связанные с данным PPD
        for oil_well in mapping.get(ppd_well, []):
            try:
                series = oil_idx.loc[oil_well]
            except KeyError:
                # В mapping скважина есть, но в oil_clean — отсутствует
                continue

            win: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = select_response_window(ev, series)
            if win is None:
                continue  # окно не найдено
            oil_start, oil_end = win

            passport = make_window_passport(series, oil_start, oil_end)
            if passport is None:
                continue  # нет данных давления/дебита внутри окна

            passport |= {
                "well": oil_well,
                "oil_start": oil_start,
                "oil_end": oil_end,
                "duration_days_oil": (oil_end - oil_start).days + 1,
                "ppd_well": ppd_well,
                "ppd_start": ppd_start,
                "ppd_end": ppd_end,
                "duration_days_ppd": (ppd_end - ppd_start).days + 1,
            }
            records.append(passport)

    return pd.DataFrame(records)[cols_order]


# ---------------------------------------------------------------------------
#                     У Н И В Е Р С А Л Ь Н Ы Й   И Н Т Е Р Ф Е Й С
# ---------------------------------------------------------------------------

def build_oil_windows(
        *,
        ppd_events_df: pd.DataFrame | None = None,
        oil_df: pd.DataFrame | None = None,
        pairs_df: pd.DataFrame | None = None,
        ppd_events_path: str | Path | None = None,
        oil_path: str | Path | None = None,
        pairs_path: str | Path | None = None,
        save_csv: bool = True,
) -> pd.DataFrame:
    """Высокоуровневая обёртка над :pyfunc:`_build_oil_windows_df`.

    В пайплайне вызывайте функцию, передавая *in‑memory* DataFrame‑ы и
    *save_csv=False*. При запуске модуля напрямую (CLI‑режим) можно опустить
    аргументы, и функция сама прочитает необходимые файлы из *clean_data/*.

    Параметры
    ----------
    ppd_events_df, oil_df, pairs_df : pd.DataFrame | None
        Уже загруженные таблицы. Если **все** они переданы, никаких файловых
        операций не выполняется.
    ppd_events_path, oil_path, pairs_path : str | Path | None
        Пути к CSV‑файлам. Используются, когда соответствующий DataFrame не
        передан. По умолчанию берутся стандартные файлы в *clean_data/*.
    save_csv : bool, default **True**
        При *True* результат будет сохранён в *clean_data/oil_windows.csv*.

    Возвращает
    ----------
    pd.DataFrame
        Таблица «окон отклика» в формате, идентичном прежнему *oil_windows.csv*.
    """
    # 1. Загружаем недостающие входы
    if ppd_events_df is None:
        ppd_events_path = Path(ppd_events_path or PPD_EVENTS_PATH)
        if not ppd_events_path.exists():
            raise FileNotFoundError(f"Не найден файл событий ППД: {ppd_events_path}")
        ppd_events_df = _load_ppd_events(ppd_events_path)
    else:
            # Если DataFrame пришёл «in-memory», приводим имена колонок и типы
            # 1) Переименовываем столбцы start_date → start, end_date → end
            if "start_date" in ppd_events_df.columns or "end_date" in ppd_events_df.columns:
                ppd_events_df = ppd_events_df.rename(
                    columns={"start_date": "start", "end_date": "end"}
                )
            # 2) Переименовываем столбец well → ppd_well
            if "well" in ppd_events_df.columns and "ppd_well" not in ppd_events_df.columns:
                ppd_events_df = ppd_events_df.rename(columns={"well": "ppd_well"})
            # 3) Конвертируем start и end в datetime
            ppd_events_df["start"] = pd.to_datetime(
                ppd_events_df["start"], dayfirst=True, errors="coerce"
            )
            ppd_events_df["end"] = pd.to_datetime(
                ppd_events_df["end"], dayfirst=True, errors="coerce"
            )
            # 4) Отбрасываем строки с некорректными датами
            ppd_events_df = ppd_events_df.dropna(subset=["start", "end"])

    if oil_df is None:
        oil_path = Path(oil_path or OIL_CLEAN_PATH)
        if not oil_path.exists():
            raise FileNotFoundError(f"Не найден файл oil_clean: {oil_path}")
        oil_df = _load_oil_df(oil_path)

    if pairs_df is None:
        pairs_path = Path(pairs_path or PAIRS_PATH)
        if not pairs_path.exists():
            raise FileNotFoundError(f"Не найден файл pairs_oil_ppd: {pairs_path}")
        pairs_df = pd.read_csv(pairs_path, dtype={"ppd_well": str, "oil_well": str})

    # 2. Строим словарь mapping
    mapping: Dict[str, List[str]] = (
        pairs_df.groupby("ppd_well", sort=False)["oil_well"].apply(list).to_dict()
    )

    # 3. Вызываем «чистое ядро»
    windows_df = _build_oil_windows_df(ppd_events_df, oil_df, mapping)

    # 4. Сохраняем результат при необходимости
    if save_csv:
        windows_df.to_csv(WINDOWS_OUT_PATH, index=False, encoding="utf-8-sig")
        print(f"✓ Сформировано окон: {len(windows_df)} → {WINDOWS_OUT_PATH}")

    return windows_df


# ---------------------------------------------------------------------------
#                                C L I
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Полностью повторяем прежнее поведение скрипта
    build_oil_windows(save_csv=True)
