"""well_pairs_refactored.py — формирование пар «oil-PPD» для проекта.

=====================  О Б Щ А Я   С Х Е М А  =====================

* **Чистое ядро** — :pyfunc:`_build_pairs_df`, которое работает ТОЛЬКО с
  переданными ``pandas.DataFrame`` и *не* выполняет файловых операций.
* **Универсальный интерфейс** — :pyfunc:`build_pairs`, допускающий два режима:
  1. *In‑memory* (default): подаём готовые DataFrame‑ы, получаем результат,
     ``save_csv=False`` — ничего не пишется на диск.
  2. *CLI/Debug*: если DF‑ы не переданы, функция читает стандартные файлы из
     *clean_data/* и (при ``save_csv=True``) сохраняет
     *clean_data/pairs_oil_ppd.csv*.
* **CLI‑блок** в конце модуля воспроизводит прежнее поведение старого скрипта:
  `python well_pairs_refactored.py` → готовый CSV с парами.

"""
from __future__ import annotations

from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd

import config

# ---------------------------------------------------------------------------
#  П У Т И   П О  У М О Л Ч А Н И Ю  (используются во внешнем интерфейсе)
# ---------------------------------------------------------------------------
DATA_DIR = Path("clean_data")
DATA_DIR.mkdir(exist_ok=True)

COORDS_PATH = DATA_DIR / "coords_clean.csv"
OIL_PATH    = DATA_DIR / "oil_clean.csv"
PPD_PATH    = DATA_DIR / "ppd_clean.csv"
OUT_PATH    = DATA_DIR / "pairs_oil_ppd.csv"

# ---------------------------------------------------------------------------
#            П О М О Щ Н И К И   Б Е З   Ф А Й Л О В О Й   I / O
# ---------------------------------------------------------------------------

def _merge_with_coords(coords: pd.DataFrame, wells: pd.Series, *, kind: str) -> pd.DataFrame:
    """Дополняет список ``wells`` координатами из ``coords``.

    Параметры
    ----------
    coords : pd.DataFrame
        Таблица с колонками ``well, x, y`` (numeric).
    wells : pd.Series
        Список/Series идентификаторов скважин (str/int).
    kind : str
        Строка-индикатор (``"oil"`` или ``"ppd"``) — используется только для
        информативного предупреждения о пропущенных скважинах.

    Возвращает
    ----------
    pd.DataFrame
        Подмножество ``coords`` только для тех ``well``, что присутствуют в
        ``wells``. Отсутствующие координаты аккуратно игнорируются.
    """
    subset = pd.DataFrame({"well": wells})
    merged = subset.merge(coords[["well", "x", "y"]], on="well", how="inner")

    return merged


def _compute_pairs_within_radius(
    oil_coords: pd.DataFrame,
    ppd_coords: pd.DataFrame,
    *,
    radius: float,
) -> pd.DataFrame:
    """Строит все пары (oil_well, ppd_well) с расстоянием ≤ *radius*.

    Параметры
    ----------
    oil_coords : pd.DataFrame
        Колонки ``well, x, y`` для нефтяных скважин.
    ppd_coords : pd.DataFrame
        Аналогичные колонки для нагнетательных скважин.
    radius : float
        Предельное расстояние поиска (в тех же единицах, что ``x, y``).

    Возвращает
    ----------
    pd.DataFrame
        Таблица с колонками ``oil_well, ppd_well, distance`` (int). Если подходящих
        пар нет — возвращается пустой DF.
    """
    oil_ids = oil_coords["well"].astype(str).values
    ppd_ids = ppd_coords["well"].astype(str).values

    oil_xy = oil_coords[["x", "y"]].to_numpy()
    ppd_xy = ppd_coords[["x", "y"]].to_numpy()

    dx = ppd_xy[:, 0][:, None] - oil_xy[:, 0][None, :]
    dy = ppd_xy[:, 1][:, None] - oil_xy[:, 1][None, :]
    dist = np.sqrt(dx ** 2 + dy ** 2)  # shape = (n_ppd, n_oil)

    rows: List[Dict[str, str | int]] = []
    for i, ppd_id in enumerate(ppd_ids):
        within = np.where(dist[i] <= radius)[0]
        for j in within:
            oil_id = oil_ids[j]
            rows.append({
                "oil_well": oil_id,
                "ppd_well": ppd_id,
                "distance": int(round(dist[i, j])),
            })

    df_pairs = pd.DataFrame(rows)
    if not df_pairs.empty:
        df_pairs = df_pairs.sort_values(["oil_well", "distance", "ppd_well"]).reset_index(drop=True)
    return df_pairs

# ---------------------------------------------------------------------------
#                           Ч И С Т О Е   Я Д Р О
# ---------------------------------------------------------------------------

def _ensure_well_str(df: pd.DataFrame) -> pd.DataFrame:
    """Приводим колонку well к строковому типу для корректных merge."""
    df["well"] = df["well"].astype(str)
    return df
def _build_pairs_df(
    coords_df: pd.DataFrame,
    oil_wells: pd.Series,
    ppd_wells: pd.Series,
    *,
    radius: float,
) -> pd.DataFrame:
    """Комбинирует вспомогательные шаги и возвращает итоговый DataFrame.

    Ни одной файловой операции здесь нет — входы/выходы только ``DataFrame``.
    """
    oil_coords = _merge_with_coords(coords_df, oil_wells, kind="oil")
    ppd_coords = _merge_with_coords(coords_df, ppd_wells, kind="ppd")
    return _compute_pairs_within_radius(oil_coords, ppd_coords, radius=radius)

# ---------------------------------------------------------------------------
#                     У Н И В Е Р С А Л Ь Н Ы Й   И Н Т Е Р Ф Е Й С
# ---------------------------------------------------------------------------

def _ensure_well_str(df: pd.DataFrame) -> pd.DataFrame:
    """Приводим колонку 'well' к строковому типу для корректных merge."""
    df["well"] = df["well"].astype(str)
    return df

def build_pairs(
    *,
    coords_df: pd.DataFrame | None = None,
    oil_df: pd.DataFrame | None = None,
    ppd_df: pd.DataFrame | None = None,
    coords_path: str | Path | None = None,
    oil_path: str | Path | None = None,
    ppd_path: str | Path | None = None,
    radius: float | None = None,
    save_csv: bool = True,
) -> pd.DataFrame:
    """Основная точка входа для формирования таблицы «пар скважин».

    Параметры
    ----------
    coords_df, oil_df, ppd_df : pd.DataFrame | None
        Already‑loaded tables. Когда переданы, файл I/O **не выполняется**.
    coords_path, oil_path, ppd_path : str | Path | None
        Пути к стандартным CSV‑файлам. Используются, если соответствующий DF не
        задан. По умолчанию: *clean_data/coords_clean.csv*, *oil_clean.csv*,
        *ppd_clean.csv*.
    radius : float | None, default *config.radius*
        Радиус поиска. Можно переопределить на лету.
    save_csv : bool, default **True**
        Сохранять результат в *clean_data/pairs_oil_ppd.csv*.

    Возвращает
    ----------
    pd.DataFrame
        Итоговая таблица ``oil_well, ppd_well, distance``.
    """
    # 1. Входы — читаем при необходимости
    if coords_df is None:
        coords_path = Path(coords_path or COORDS_PATH)
        coords_df = pd.read_csv(coords_path)
    if oil_df is None:
        oil_path = Path(oil_path or OIL_PATH)
        oil_df = pd.read_csv(oil_path)
    if ppd_df is None:
        ppd_path = Path(ppd_path or PPD_PATH)
        ppd_df = pd.read_csv(ppd_path)

    coords_df = _ensure_well_str(coords_df.copy())
    oil_df = _ensure_well_str(oil_df.copy())
    ppd_df = _ensure_well_str(ppd_df.copy())

    required_cols = {"well", "x", "y"}
    if not required_cols.issubset(coords_df.columns):
        raise RuntimeError("coords_clean.csv должен содержать колонки 'well', 'x', 'y'")
    if "well" not in oil_df.columns or "well" not in ppd_df.columns:
        raise RuntimeError("oil_clean.csv и ppd_clean.csv должны содержать колонку 'well'")

    # 2. Уникальные списки скважин
    oil_wells = oil_df["well"].drop_duplicates().reset_index(drop=True)
    ppd_wells = ppd_df["well"].drop_duplicates().reset_index(drop=True)

    # 3. Радиус
    radius = float(radius if radius is not None else config.radius)

    # 4. Чистое ядро
    pairs_df = _build_pairs_df(coords_df, oil_wells, ppd_wells, radius=radius)

    # 5. Save‑CSV (по желанию)
    if save_csv:
        pairs_df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
        print(f"✓ Сохранено {len(pairs_df)} пар → {OUT_PATH}")
    return pairs_df

# ---------------------------------------------------------------------------
#                                  C L I
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Запуск скрипта вручную (поведение идентично старому well_pairs.py)
    build_pairs(save_csv=True)
