import os
import pandas as pd
import numpy as np
import pytest


from well_pairs import load_data, merge_with_coords, compute_pairs_within_radius


def write_csv(path: str, df: pd.DataFrame):
    """
    Вспомогательная функция для сохранения DataFrame в CSV (с заголовком и без индекса).
    """
    df.to_csv(path, index=False, encoding="utf-8-sig")


def test_load_data_success(tmp_path):
    """
    Проверяем, что load_data корректно читает файлы, если:
      - coords содержит колонки ['well','x','y']
      - oil/ppd содержат колонку ['well']
    """
    base = tmp_path / "data"
    base.mkdir()

    # coords_clean.csv
    coords_df = pd.DataFrame({
        "well": [1, 2, 3],
        "x": [100.0, 200.0, 300.0],
        "y": [10.0, 20.0, 30.0]
    })
    coords_path = str(base / "coords_clean.csv")
    write_csv(coords_path, coords_df)

    # oil_clean.csv (колонка well, с дубликатами)
    oil_df = pd.DataFrame({
        "well": [1, 1, 3, 3, 3]
    })
    oil_path = str(base / "oil_clean.csv")
    write_csv(oil_path, oil_df)

    # ppd_clean.csv
    ppd_df = pd.DataFrame({
        "well": [2, 2, 2]
    })
    ppd_path = str(base / "ppd_clean.csv")
    write_csv(ppd_path, ppd_df)

    # Вызываем load_data
    coords_ret, oil_wells_ret, ppd_wells_ret = load_data(coords_path, oil_path, ppd_path)

    # Проверяем, что coords_ret соответствует исходному DataFrame
    pd.testing.assert_frame_equal(coords_ret.reset_index(drop=True), coords_df)

    # oil_wells_ret и ppd_wells_ret — Series с уникальными well
    assert isinstance(oil_wells_ret, pd.Series)
    assert set(oil_wells_ret.tolist()) == {1, 3}

    assert isinstance(ppd_wells_ret, pd.Series)
    assert set(ppd_wells_ret.tolist()) == {2}


def test_load_data_missing_coords_columns(tmp_path):
    """
    Проверяем, что load_data выбрасывает RuntimeError,
    если в coords_clean.csv нет колонок 'well','x','y'.
    """
    base = tmp_path / "data2"
    base.mkdir()

    # coords без колонки 'y'
    coords_df = pd.DataFrame({
        "well": [1, 2],
        "x": [10.0, 20.0],
        # "y" отсутствует
        "z": [0, 0]
    })
    coords_path = str(base / "coords_bad.csv")
    write_csv(coords_path, coords_df)

    # oil и ppd корректные
    oil_df = pd.DataFrame({"well": [1]})
    oil_path = str(base / "oil_bad.csv")
    write_csv(oil_path, oil_df)

    ppd_df = pd.DataFrame({"well": [2]})
    ppd_path = str(base / "ppd_bad.csv")
    write_csv(ppd_path, ppd_df)

    with pytest.raises(RuntimeError) as excinfo:
        load_data(coords_path, oil_path, ppd_path)
    assert "В coords_clean.csv должны быть колонки 'well', 'x' и 'y'" in str(excinfo.value)


def test_merge_with_coords_ignores_missing(tmp_path, capsys):
    """
    Проверяем, что merge_with_coords:
      - Возвращает только те well, что есть в coords
      - Печатает предупреждение о пропущенных скважинах
    """
    # DataFrame coords
    coords_df = pd.DataFrame({
        "well": [10, 20, 30],
        "x": [0.0, 1.0, 2.0],
        "y": [0.0, 1.0, 2.0]
    })

    # wells содержит существующие и несуществующие номера
    wells = pd.Series([20, 30, 40, 50])

    # Вызываем merge_with_coords
    merged = merge_with_coords(coords_df, wells, kind="test_kind")

    # В merged только well=20 и well=30
    assert set(merged["well"].tolist()) == {20, 30}
    # Колонки — ['well','x','y']
    assert list(merged.columns) == ["well", "x", "y"]

    # Проверяем, что выведено предупреждение о пропущенных [40, 50]
    captured = capsys.readouterr()
    assert "следующих test_kind-скважин нет в coords_clean.csv" in captured.out
    assert "40" in captured.out and "50" in captured.out


def test_compute_pairs_within_radius_basic(tmp_path):
    """
    Проверяем compute_pairs_within_radius на простом примере:
    - Три нефтянки: A(0,0), B(10,0), C(0,10)
    - Две ППД: P1(0,1), P2(20,20)
    При radius=5 должна остаться только пара (A,P1) с distance=1.
    """
    oil_coords_df = pd.DataFrame({
        "well": [1, 2, 3],
        "x": [0.0, 10.0, 0.0],
        "y": [0.0, 0.0, 10.0]
    })
    ppd_coords_df = pd.DataFrame({
        "well": [100, 200],
        "x": [0.0, 20.0],
        "y": [1.0, 20.0]
    })

    df_pairs = compute_pairs_within_radius(oil_coords_df, ppd_coords_df, radius=5.0)

    # Ожидаем ровно одну пару
    assert len(df_pairs) == 1

    # Структура столбцов
    assert list(df_pairs.columns) == ["oil_well", "ppd_well", "distance"]

    row = df_pairs.iloc[0]
    assert row["oil_well"] == 1
    assert row["ppd_well"] == 100
    # distance может быть np.integer или int
    assert isinstance(row["distance"], (int, np.integer))
    assert int(row["distance"]) == 1  # округление 1.0 → 1


def test_compute_pairs_within_radius_sorting(tmp_path):
    """
    Проверяем сортировку compute_pairs_within_radius:
    - Два oil: 1(0,0), 2(0,2)
    - Две ppd: 10(0,1), 20(0,3)
    При radius=5 получаем четыре пары:
      (1,10)=1
      (1,20)=3
      (2,10)=1
      (2,20)=1
    Ожидаем порядок:
      (1,10,1), (1,20,3), (2,10,1), (2,20,1)
    """
    oil_coords_df = pd.DataFrame({
        "well": [1, 2],
        "x": [0.0, 0.0],
        "y": [0.0, 2.0]
    })
    ppd_coords_df = pd.DataFrame({
        "well": [10, 20],
        "x": [0.0, 0.0],
        "y": [1.0, 3.0]
    })

    df_pairs = compute_pairs_within_radius(oil_coords_df, ppd_coords_df, radius=5.0)

    # Ожидаем 4 строки
    assert len(df_pairs) == 4

    # Проверяем порядок (oil_well, ppd_well, distance)
    expected = [
        (1, 10, 1),
        (1, 20, 3),
        (2, 10, 1),
        (2, 20, 1),
    ]
    actual = list(df_pairs.itertuples(index=False, name=None))
    assert actual == expected


if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__)])
