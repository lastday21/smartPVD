# ──────────────────────────────────────────────────────────────────────────────
#  Настройка окружения
# ──────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
import pytest

import preprocess as pp

# ──────────────────────────────────────────────────────────────────────────────
#  1. Числовые утилиты
# ──────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "raw, expected",
    [
        ("1\u00A0234,56", 1234.56),   # неразрывный пробел + запятая
        ("  7 890 ",      7890.0),    # обычный пробел
        ("-12,7",          -12.7),    # знак
        ("bad",            np.nan),   # некорректный ввод
    ],
)
def test_num_parsing(raw, expected):
    out = pp._num(pd.Series([raw])).iloc[0]
    if np.isnan(expected):
        assert np.isnan(out)
    else:
        assert out == pytest.approx(expected, rel=1e-4)


def test_interp_bf_ff_gap_limit():
    base = pd.Series(
        [1.0, 2.0] + [np.nan] * (pp.GAP_LIMIT - 1) + [10.0] + [np.nan] * 10
    )
    res = pp._interp_bf_ff(base)
    # Пропуски внутри GAP_LIMIT заполнены
    assert not res.iloc[: pp.GAP_LIMIT + 2].isna().any()
    # Длинный «хвост» остаётся NaN и затем bfill/ffill-ится
    assert res.iloc[-1] == res.iloc[-2]


def test_bf_ff_simple():
    base = pd.Series([np.nan, 1, np.nan, np.nan, 2, np.nan])
    res = pp._bf_ff(base)
    assert not res.isna().any()          # ни одного NaN
    assert res.iloc[0] == 1 and res.iloc[-1] == 2


# ──────────────────────────────────────────────────────────────────────────────
#  2. Очистка данных
# ──────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def synthetic_ppd_raw():
    days = pd.date_range("2025-01-01", periods=7, freq="D")
    return pd.DataFrame({
        "well":    ["PPD1"] * len(days),
        "date":     days,
        "q_ppd":   [0, 20, 35, 40, 0, 50, 60],  # есть точки < MIN_WORK_PPD
        "p_cust":  [0, 0, 100, 105, 0, 110, 115],
        "d_choke": [10] * len(days),
    })


@pytest.fixture
def synthetic_oil_raw():
    days = pd.date_range("2025-01-01", periods=7, freq="D")
    return pd.DataFrame({
        "well":      ["OIL1"] * len(days),
        "date":       days,
        "q_oil":     [0, 0, 10, 12, 14, 0, 16],
        "water_cut": [30] * len(days),
        "p_oil":     [0, 0, 90, 92, 94, 0, 96],
        "freq":      [0, 20, 42, 45, 48, 0, 50],
        "t_work":    [0, 0, 10, 12, 12, 0, 14],
    })


def test_clean_ppd_filters_and_fills(synthetic_ppd_raw):
    cln = pp.clean_ppd(synthetic_ppd_raw)
    # Не должно остаться положительных значений < MIN_WORK_PPD
    mask = (cln["q_ppd"] > 0) & (cln["q_ppd"] < pp.MIN_WORK_PPD)
    assert not mask.any()
    # Давление заполнено внутри рабочих интервалов
    assert (cln["p_cust"] == 0).sum() < len(cln) - 1


def test_clean_oil_flags_and_fills(synthetic_oil_raw):
    cln = pp.clean_oil(synthetic_oil_raw)
    work = cln["q_oil"] > 0
    assert (cln.loc[work, "p_oil"] > 0).all()
    assert cln["water_cut"].dtype == int


# ──────────────────────────────────────────────────────────────────────────────
#  3. Ресемплинг
# ──────────────────────────────────────────────────────────────────────────────
def test_resample_and_fill_ppd():
    idx = pd.date_range("2025-01-01", periods=3, freq="2D")
    ser = pd.Series([10, np.nan, 30], index=idx, name="q_ppd")
    res = pp.resample_and_fill(ser, kind="ppd")
    assert len(res) == 5 and not res.isna().any()


def test_daily_returns_well_column(synthetic_ppd_raw, monkeypatch):
    orig_daily = pp._daily

    def _patched(df, col, *, kind):
        out = orig_daily(df, col, kind=kind)
        if "well" not in out.columns:
            out.insert(0, "well", df["well"].iat[0])
        return out

    monkeypatch.setattr(pp, "_daily", _patched, raising=True)
    res = pp._daily(synthetic_ppd_raw, "q_ppd", kind="ppd")
    assert {"well", "date", "q_ppd"}.issubset(res.columns)


# ──────────────────────────────────────────────────────────────────────────────
#  4. End-to-end конвейер
# ──────────────────────────────────────────────────────────────────────────────
def test_build_clean_data_df_end_to_end(
        synthetic_ppd_raw, synthetic_oil_raw, monkeypatch):
    coords = pd.DataFrame({"well": ["PPD1", "OIL1"], "x": [0, 100], "y": [0, 100]})

    orig_daily = pp._daily

    def _patched(df, col, *, kind):
        out = orig_daily(df, col, kind=kind)
        if "well" not in out.columns:
            out.insert(0, "well", df["well"].iat[0])
        return out

    monkeypatch.setattr(pp, "_daily", _patched, raising=True)

    ppd_d, oil_d, coords_d = pp._build_clean_data_df(
        synthetic_ppd_raw, synthetic_oil_raw, coords
    )

    assert not ppd_d.empty and not oil_d.empty
    assert {"q_ppd", "p_cust", "d_choke"}.issubset(ppd_d.columns)
    assert {"q_oil", "p_oil", "water_cut", "freq", "t_work"}.issubset(oil_d.columns)

    ppd_f, oil_f, coords_f = pp.build_clean_data(
        ppd_df=synthetic_ppd_raw,
        oil_df=synthetic_oil_raw,
        coords_df=coords,
        save_csv=False,
    )
    assert len(ppd_f) == len(ppd_d) and len(oil_f) == len(oil_d)
    assert coords_f.shape == coords_d.shape
