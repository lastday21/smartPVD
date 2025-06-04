"""
well_pairs.py

Модуль для формирования пар нефтяных скважин (oil) и нагнетательных скважин (PPD) по радиусу.
В результате получается таблица с тремя колонками:
    oil_well   — идентификатор нефтянки
    ppd_well   — идентификатор ППД
    distance   — расстояние между ними, округлённое до целого

Используем:
    - coords_clean.csv — координаты всех скважин (столбцы: well, x, y)
    - oil_clean.csv    — список нефтянок (столбец: well)
    - ppd_clean.csv    — список ППД     (столбец: well)

Радиус поиска задаётся в файле config.py параметром `radius`.
Результирующий CSV сохраняется в "clean_data/pairs_oil_ppd.csv".

Пример запуска:
    python well_pairs.py
"""

import pandas as pd
import numpy as np
import config


def load_data(coords_path: str, oil_path: str, ppd_path: str):
    """
    Загружает три CSV-файла:
      - coords_clean.csv: таблица всех скважин с координатами
                         (ожидаются колонки 'well', 'x', 'y' в нижнем регистре).
      - oil_clean.csv:   список нефтянных скважин ('well').
      - ppd_clean.csv:   список нагнетательных скважин ('well').

    1) Проверяет, что coords содержит колонки 'well', 'x', 'y'.
       Если нет — выбрасывает RuntimeError.
    2) Проверяет наличие колонки 'well' в oil_clean.csv и ppd_clean.csv.
       Если нет — выбрасывает RuntimeError.
    3) Берёт из oil_clean и ppd_clean только уникальные номера скважин.

    Возвращает:
      coords     — DataFrame со столбцами ['well', 'x', 'y'].
      oil_wells  — pd.Series уникальных идентификаторов скважин из oil_clean.csv.
      ppd_wells  — pd.Series уникальных идентификаторов скважин из ppd_clean.csv.
    """
    # --- Шаг 1: загружаем coords_clean.csv и проверяем колонки ---
    coords = pd.read_csv(coords_path)
    if not {"well", "x", "y"}.issubset(coords.columns):
        raise RuntimeError("В coords_clean.csv должны быть колонки 'well', 'x' и 'y' (в нижнем регистре)")

    # --- Шаг 2: загружаем oil_clean.csv и ppd_clean.csv ---
    oil_all = pd.read_csv(oil_path)
    ppd_all = pd.read_csv(ppd_path)

    if "well" not in oil_all.columns:
        raise RuntimeError("В oil_clean.csv нет колонки 'well'")
    if "well" not in ppd_all.columns:
        raise RuntimeError("В ppd_clean.csv нет колонки 'well'")

    # --- Шаг 3: оставляем только уникальные значения well ---
    oil_wells = oil_all["well"].drop_duplicates().reset_index(drop=True)
    ppd_wells = ppd_all["well"].drop_duplicates().reset_index(drop=True)

    return coords, oil_wells, ppd_wells


def merge_with_coords(coords: pd.DataFrame, wells: pd.Series, kind: str) -> pd.DataFrame:
    """
    Объединяет (inner merge) список well-идентификаторов с таблицей coords:
      - coords      — DataFrame со столбцами ['well', 'x', 'y'].
      - wells       — pd.Series с уникальными номерами скважин.
      - kind        — строка ("oil" или "ppd") для информативного предупреждения.

    Если какие-то well из списка отсутствуют в coords, они просто игнорируются,
    но выводится предупреждение в консоль.

    Возвращает DataFrame с колонками ['well', 'x', 'y'] тех скважин, координаты которых найдены.
    """
    # Формируем вспомогательный DataFrame только с колонкой 'well'
    subset = pd.DataFrame({"well": wells})
    # Выполняем inner-объединение с coords
    merged = subset.merge(coords[["well", "x", "y"]], on="well", how="inner")

    # Определяем, какие well из wells не попали в coords
    missing = set(wells.tolist()) - set(merged["well"].tolist())
    if missing:
        print(
            f" [!] ПРЕДУПРЕЖДЕНИЕ: следующих {kind}-скважин нет в coords_clean.csv, "
            f"они будут пропущены:\n     {sorted(missing)}\n"
        )

    return merged


def compute_pairs_within_radius(
    oil_coords: pd.DataFrame,
    ppd_coords: pd.DataFrame,
    radius: float
):
    """
    Строит попарную матрицу расстояний между нефтянными (oil_coords) и нагнетательными (ppd_coords) скважинами.

    Параметры:
      - oil_coords: DataFrame с колонками ['well', 'x', 'y'] для нефтянок.
      - ppd_coords: DataFrame с колонками ['well', 'x', 'y'] для ППД.
      - radius:     радиус поиска (число, в тех же единицах, что и x,y).

    Алгоритм:
      1) Извлекаем numpy-массивы координат oil_xy (n_oil×2) и ppd_xy (n_ppd×2).
      2) Вычисляем разности по x и y, формируем матрицу расстояний dist_matrix формы (n_ppd×n_oil):
           dist_matrix[i,j] = sqrt((ppd_x[i] - oil_x[j])^2 + (ppd_y[i] - oil_y[j])^2).
      3) Для каждой строки i (ППД) находим индексы j нефтянок, у которых dist_matrix[i,j] ≤ radius.
      4) Для каждой найденной пары (i,j):
           - округляем расстояние до целого: dist_int = int(round(dist_raw)).
           - добавляем строку {"oil_well": oil_id, "ppd_well": ppd_id, "distance": dist_int}.

    Возвращает:
      DataFrame с тремя колонками:
        oil_well (int), ppd_well (int), distance (int),
      отсортированный по (oil_well, distance, ppd_well). Если ни одной пары нет, возвращает пустой DataFrame.
    """
    # Извлекаем идентификаторы скважин в виде numpy-массивов
    oil_ids_arr = oil_coords["well"].values
    ppd_ids_arr = ppd_coords["well"].values

    # Координаты (n_oil×2) и (n_ppd×2)
    oil_xy = oil_coords[["x", "y"]].to_numpy()
    ppd_xy = ppd_coords[["x", "y"]].to_numpy()

    # Разницы по x и y: формируем матрицу (n_ppd×n_oil)
    dx = ppd_xy[:, 0].reshape(-1, 1) - oil_xy[:, 0].reshape(1, -1)
    dy = ppd_xy[:, 1].reshape(-1, 1) - oil_xy[:, 1].reshape(1, -1)
    dist_matrix = np.sqrt(dx**2 + dy**2)

    rows = []
    n_ppd, n_oil = dist_matrix.shape

    # Для каждой ППД-скважины ищем все нефтянки внутри заданного радиуса
    for i in range(n_ppd):
        ppd_id = ppd_ids_arr[i]
        # Индексы oil-скважин, у которых расстояние ≤ radius
        idxs = np.where(dist_matrix[i, :] <= radius)[0]

        for j in idxs:
            oil_id = oil_ids_arr[j]
            raw_dist = dist_matrix[i, j]
            dist_int = int(round(raw_dist))  # округление до ближайшего целого
            rows.append({
                "oil_well": oil_id,
                "ppd_well": ppd_id,
                "distance": dist_int
            })

    # Собираем результат в DataFrame
    df_pairs = pd.DataFrame(rows)
    if not df_pairs.empty:
        # Сортируем: сначала по oil_well, затем по distance, затем по ppd_well
        df_pairs = df_pairs.sort_values(
            by=["oil_well", "distance", "ppd_well"]
        ).reset_index(drop=True)

    return df_pairs


def main():
    """
    Основная функция:
      1) Жёстко задаёт пути к трем входным файлам в папке clean_data:
         - coords_clean.csv
         - oil_clean.csv
         - ppd_clean.csv
      2) Читает параметр поиска `radius` из config.py.
      3) Загружает данные (load_data) и объединяет списки скважин с координатами (merge_with_coords).
      4) Вычисляет все пары (oil, PPD) с расстоянием <= radius (compute_pairs_within_radius).
      5) Сохраняет получившийся DataFrame в CSV файл clean_data/pairs_oil_ppd.csv.
      6) Печатает уведомление о завершении и предупреждение, если ни одной пары не найдено.
    """
    # Пути к исходным таблицам (жёстко пропиcаны)
    coords_path = "clean_data/coords_clean.csv"
    oil_path    = "clean_data/oil_clean.csv"
    ppd_path    = "clean_data/ppd_clean.csv"

    # Читаем радиус из config.py
    radius = config.radius

    # Загружаем и проверяем входные таблицы
    coords, oil_wells, ppd_wells = load_data(coords_path, oil_path, ppd_path)

    # Объединяем списки well-идентификаторов с таблицей coords
    oil_coords = merge_with_coords(coords, oil_wells, kind="oil")
    ppd_coords = merge_with_coords(coords, ppd_wells, kind="ppd")

    # Вычисляем пары (oil, PPD) внутри радиуса
    df_pairs = compute_pairs_within_radius(oil_coords, ppd_coords, radius)

    # Сохраняем результат в CSV
    output_path = "clean_data/pairs_oil_ppd.csv"
    df_pairs.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"\nГотово. Радиус = {radius} м. Пары (oil, ppd, distance) сохранены в '{output_path}'.")
    if df_pairs.empty:
        print(" [!] В указанном радиусе ни одна нефтянка не нашла ни одной ППД.")


if __name__ == "__main__":
    main()
