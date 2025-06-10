# hyperparam_search.py
"""
Перебор 10 985 комбинаций baseline-параметров и четырёх CI-методов.
Запуск:
    python hyperparam_search.py --n_jobs 8
Выход:
    grid_scores.csv  – все комбинации, отсортированные по exact ↓, off_by_1 ↓
"""

import os
import argparse
import itertools
import warnings

import pandas as pd
from joblib import Parallel, delayed
from importlib import reload

import config          # ваш config.py
import metrics_pro     # ваш compute_ci_with_pre()

warnings.filterwarnings("ignore", category=FutureWarning)

# ────────────────────────── 0. Аргументы CLI
parser = argparse.ArgumentParser()
parser.add_argument("--n_jobs", type=int, default=8, help="Число параллельных процессов")
parser.add_argument("--out",    default="grid_scores.csv")
args = parser.parse_args()

# ────────────────────────── 1. Ground truth PRO
gt = pd.read_csv("ground_truth_pro.csv", dtype=str)
gt["acceptable"] = (
    gt["acceptable"]
      .fillna("")
      .apply(lambda s: s.split(";") if s else [])
)
# приведём индексы к строкам без пробелов
gt["well"]     = gt["well"].astype(str).str.strip()
gt["ppd_well"] = gt["ppd_well"].astype(str).str.strip()
gt = gt.set_index(["well", "ppd_well"])

# ────────────────────────── 2. Кешируем входные CSV (читаются 1 раз)
BASE_ARGS = {
    "clean_data_dir":    "clean_data",
    "oil_windows_fname": "oil_windows.csv",
    "ppd_events_fname":  "ppd_events.csv",
    "oil_clean_fname":   "oil_clean.csv"
}
# разогреем кэш pandas
_ = pd.read_csv(os.path.join("clean_data", "oil_windows.csv"), nrows=1)

# ────────────────────────── 3. Сетка гиперпараметров
T_PRE   = [6, 18, 30]                         # 6,8,…,30  → 13
DIV_Q_P = [1.0, 4.0, 7.0] # 1.0 … 7.0 шаг 0.5 → 13
W_Q     = [0.3, 0.5, 0.7]  # 0.3 … 0.7 шаг 0.1 → 5

GRID = list(itertools.product(T_PRE, DIV_Q_P, DIV_Q_P, W_Q))  # 10 985

# ────────────────────────── 4. Категоризация CI
def categorize(ci: float) -> str:
    if ci < 2:
        return "none"
    elif ci < 4:
        return "weak"
    elif ci < 6:
        return "medium"
    else:
        return "strong"

# ────────────────────────── 5. Одна итерация
def one_run(params):
    t_pre, d_q, d_p, w_q = params
    w_p = 1.0 - w_q

    # 5.1 обновить config
    config.pre_len    = t_pre
    config.divider_q  = d_q
    config.divider_p  = d_p
    config.w_q        = w_q
    config.w_p        = w_p


    # 5.2 посчитать CI
    df = metrics_pro.compute_ci_with_pre(**BASE_ARGS)

    # 5.3 суммировать по паре и округлить до десятых
    ci_cols = ["CI_none", "CI_mean", "CI_regression", "CI_median_ewma"]
    sum_ci = (
        df.groupby(["well", "ppd_well"], as_index=False)[ci_cols]
          .sum()
          .round(1)
    )

    # привести ключи к строкам без пробелов
    sum_ci["well"]     = sum_ci["well"].astype(str).str.strip()
    sum_ci["ppd_well"] = sum_ci["ppd_well"].astype(str).str.strip()

    # 5.4 присоединить ground_truth_pro
    report = (
        sum_ci.set_index(["well", "ppd_well"])
              .join(gt, how="inner")
              .reset_index()
    )

    # 5.5 выбрать лучший метод по exact затем off-by-1
    best_method = None
    best_exact = -1
    best_off = -1

    for col in ci_cols:
        got = report[col].apply(categorize)
        exact = (got == report["expected"]).sum()
        off = sum(g in acc for g, acc in zip(got, report["acceptable"]))
        if exact > best_exact or (exact == best_exact and off > best_off):
            best_method, best_exact, best_off = col, exact, off

    return {
        "T_pre":       t_pre,
        "divider_q":   d_q,
        "divider_p":   d_p,
        "w_q":         w_q,
        "w_p":         w_p,
        "best_method": best_method,
        "exact":       best_exact,
        "off_by_1":    best_off,
        "miss":        len(report) - best_exact - best_off
    }

# ────────────────────────── 6. Запуск параллельно
print(f"Start grid search over {len(GRID)} combinations …")
results = Parallel(
    n_jobs=args.n_jobs,
    backend="loky",
    verbose=8
)(delayed(one_run)(params) for params in GRID)

# ────────────────────────── 7. Сохранение результатов
df_res = (
    pd.DataFrame(results)
      .sort_values(["exact", "off_by_1"], ascending=[False, False])
      .reset_index(drop=True)
)
df_res["w_q"] = df_res["w_q"].round(1)
df_res["w_p"] = df_res["w_p"].round(1)

df_res.to_csv(args.out, index=False, float_format="%.1f")
print(
    f"Done: {args.out}  (top exact={df_res.loc[0,'exact']}, "
    f"method={df_res.loc[0,'best_method']})"
)
