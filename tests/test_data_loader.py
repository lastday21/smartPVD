import pandas as pd
import pytest

import data_loader
from data_loader import _ru, load_oil, load_ppd, _daily

# тест _ru
@pytest.mark.parametrize("inp,exp", [
    ("Обв ХАЛ", "обвхал"),
    (" f вращ  тм ", "fвращтм"),
])
def test_ru(inp, exp):
    assert _ru(inp) == exp

def test_load_oil_mapping(monkeypatch, tmp_path):
    # подготовим фиктивный DataFrame как _read_raw
    df_raw = pd.DataFrame({
        "№ п/п": [1],
        "Мест.": ["F"],
        "№ скважины": [10],
        "Куст": [5],
        "Дата": ["2023-01-01"],
        "Qж": [100],
        "Обв": [50],
        "Рприем": [80],
        "F вращ ТМ": [60],
        "Tраб(ТМ)": [12.3],
    })
    # monkeypatch для _read_raw
    monkeypatch.setattr(data_loader, "_read_raw", lambda path, sheet: df_raw.copy())
    df = load_oil(path=tmp_path/"dummy.xlsx")
    # проверяем, что названия колонок конвертированы
    assert set(["idx","field","well","cluster","date",
                "q_oil","water_cut","p_oil","freq","t_work"]) <= set(df.columns)
    # и что значения перенеслись верно
    assert df.loc[0, "q_oil"] == 100
    assert df.loc[0, "water_cut"] == 50
    assert df.loc[0, "p_oil"] == 80
    assert df.loc[0, "freq"] == 60
    assert df.loc[0, "t_work"] == pytest.approx(12.3)

def test_load_ppd_mapping(monkeypatch, tmp_path):
    df_raw = pd.DataFrame({
        "№ п/п": [2],
        "Мест.": ["G"],
        "№ скважины": [20],
        "Куст": [8],
        "Дата": ["2023-02-02"],
        "Dшт": [2.5],
        "Pкуст": [15.7],
        "Qприем.Тех": [40],
    })
    monkeypatch.setattr(data_loader, "_read_raw", lambda path, sheet: df_raw.copy())
    df = load_ppd(path=tmp_path/"dummy2.xlsx")
    assert set(["idx","field","well","cluster","date",
                "d_choke","p_cust","q_ppd"]) <= set(df.columns)
    assert df.loc[0, "d_choke"] == pytest.approx(2.5)
    assert df.loc[0, "p_cust"] == pytest.approx(15.7)
    assert df.loc[0, "q_ppd"] == pytest.approx(40)

def test_daily_resample():
    # один well, два дня, колонки
    idx = pd.to_datetime(["2023-01-01","2023-01-03"])
    s = pd.Series([1,3], index=idx, name="q_oil")
    df = pd.DataFrame({"well":[1,1],"date":idx,"q_oil":[1,3]})
    out = _daily(df, "q_oil", kind="oil")
    # должен быть 2023-01-01,02,03 с интерполяцией 1,2,3
    vals = out["q_oil"].tolist()
    assert vals == [1,2,3]
