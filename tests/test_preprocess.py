import pandas as pd
import numpy as np
import pytest

from preprocess import _num, _bf_ff, _interp_bf_ff, clean_ppd, clean_oil
from config import GAP_LIMIT, FREQ_THRESH, MIN_WORK_PPD


def test_num():
    s = pd.Series(["1 234,56", " 789", "foo", None])
    out = _num(s)
    assert out.dtype == float
    assert out.iloc[0] == pytest.approx(1234.56)
    assert out.iloc[1] == pytest.approx(789.0)
    assert np.isnan(out.iloc[2])
    assert np.isnan(out.iloc[3])


def test_bf_ff():
    idx = pd.date_range("2023-01-01", periods=5, freq="D")
    s = pd.Series([np.nan, 2, np.nan, 4, np.nan], index=idx)
    out = _bf_ff(s)
    # сначала bfill → [2,2,4,4,4], затем ffill не меняет
    assert list(out) == [2, 2, 4, 4, 4]


def test_interp_bf_ff():
    idx = pd.date_range("2023-01-01", periods=5, freq="D")
    s = pd.Series([1, np.nan, np.nan, np.nan, 5], index=idx)
    out = _interp_bf_ff(s)
    # линейная интерполяция 1→5: [1,2,3,4,5]
    assert list(out) == [1, 2, 3, 4, 5]


@pytest.fixture
def ppd_complex():
    # 15 дней: фазы raw-work и простоя
    # PPD:
    # дни 0-1: p=0 → провал<5 → treated as work → fill backward from day2=8
    # дни 2-6: p=8 → work
    # дни 7-9: p=0, q>=MIN_WORK_PPD → провал< NO_PRESS_WITH_Q_LIMIT → treated as work → fill=8
    # день 10: p=12 → work
    # дни 11-14: p=0 → провал<5 → treated as work → fill=12
    dates = pd.date_range("2023-01-01", periods=15, freq="D")
    p =    [0,0,8,8,8,8,8, 0,0,0, 12,0,0,0,0]
    # дебит дающийся >= MIN_WORK_PPD (пример MIN_WORK_PPD=30)
    q =    [0,0,0,0,0,50,50,50,50,50,50,50,50,50,50]
    d =    [2]*15
    return pd.DataFrame({"well": 1, "date": dates, "p_cust": p, "q_ppd": q, "d_choke": d})

def test_clean_ppd_complex(ppd_complex):
    cln = clean_ppd(ppd_complex.copy())
    # первые 2 дня не работает из day2=0
    assert all(cln.loc[0:1, "p_cust"] == 0)
    # дни 2-6 ровно исходные 8
    assert all(cln.loc[2:6, "p_cust"] == 8)
    # дни 7-9 провал без давления но с q, fill-back из 8
    assert all(cln.loc[7:9, "p_cust"] == 8)
    # день 10 новое давление 12
    assert cln.loc[10, "p_cust"] == 12
    # дни 11-14 тех. провал<5, fill-back из 12
    assert all(cln.loc[11:14, "p_cust"] == 12)
    # дебит q_ppd совпадает
    assert all(cln.loc[0:1, "q_ppd"] == 0)
    assert all(cln.loc[2:14, "q_ppd"] == 50)

    # диаметр int
    assert cln["d_choke"].dtype == int

@pytest.fixture
def oil_complex():
    # 14 дней, фазы работы/провала/стопа
    # дни 0-4: raw_work True (freq>FREQ_THRESH) → water_cut = [5,...]
    # дни 5-7: raw False подряд 3 дня <5 → tech gap → water_cut fill-back from day4=5
    # дни 8-13: raw False подряд 6 дней >=5 → stop → water_cut = 0
    dates = pd.date_range("2023-02-01", periods=14, freq="D")
    freq   = [45,45,45,45,45, 0,0,0, 0,0,0,0,0,0]
    q_oil  = [10,10,10,10,10, 1,1,1, 0,0,0,0,0,0]
    t_work = [ 1, 1, 1, 1, 1, 0,0,0, 0,0,0,0,0,0]
    p_oil  = [20,20,20,20,20,20,20,20, 0,0,0,0,0,0]
    wc     = [5, 5, 5, 5, 5, 2,2,2, 7,7,7,7,7,7]
    return pd.DataFrame({
        "well":1, "date":dates,
        "freq":freq, "q_oil":q_oil, "t_work":t_work,
        "p_oil":p_oil, "water_cut":wc
    })

def test_clean_oil_complex(oil_complex):
    cln = clean_oil(oil_complex.copy())
    # 0-4 raw True → original wc=5
    assert list(cln.loc[0:4,"water_cut"]) == [5]*5
    # 5-7 raw False, tech gap (<5) → fill-back from day4
    assert list(cln.loc[5:7,"water_cut"]) == [0]*3
    # 8-13 raw False, gap>=5 → stop → wc=0
    assert list(cln.loc[8:13,"water_cut"]) == [0]*6
    # частота аналогично: fill-back/fill gaps, then zeros
    assert list(cln.loc[5:7,"freq"]) == [0]*3
    assert list(cln.loc[8:13,"freq"]) == [0]*6
    # p_oil аналогично
    assert list(cln.loc[5:7,"p_oil"]) == [0]*3
    assert list(cln.loc[8:13,"p_oil"]) == [0]*6
    # t_work float with one decimal
    assert isinstance(cln.loc[0,"t_work"], float)
    assert cln["t_work"].iloc[0] == pytest.approx(1.0)