"""
Задача
------
Из всех комбинаций (`divider_q`, `divider_p`, `w_q/w_p`) подобрать ту,
которая даёт **максимум точных попаданий** (`exact`) и, при равенстве exact,
максимум «соседних» попаданий (`off-by-1`) на расширенной разметке
`ground_truth.csv`.

Что перебираем
--------------
* **divider_q** — нормировка дебита [м³/сут]; диапазон **1 … 10** с шагом **0.5**
  (19 вариантов: 1.0, 1.5, …, 10.0).
* **divider_p** — нормировка давления [атм]; тот же диапазон и шаг (1.0 … 10.0).
* **w_q**       — вес ∆Q в итоговой формуле CI; диапазон **0.3 … 0.7**
  с шагом **0.1** (то есть 0.3, 0.4, 0.5, 0.6, 0.7).
  Вес давления вычисляется как `w_p = 1 − w_q`.

Итого 19 × 19 × 5 = **1 805 комбинаций**.

Что считается
-------------
Расчёт выполняет *только* «грубый» индекс **CI_none**
(без baseline-коррекции) — самый быстрый вариант.

Файлы-источники
---------------
* `clean_data/oil_windows.csv`
* `clean_data/ppd_events.csv`
* `clean_data/oil_clean.csv`

Каждый CSV читается и кешируется ровно ОДИН раз на процесс (см. модуль `metrics.py`).

Выход
-----
`grid_scores.csv` — таблица со всеми 1 805 комбинациями, отсортированная по
`exact ↓`, затем `off_by_1 ↓`.  Колонки:

    divider_q, divider_p, w_q, w_p, exact, off_by_1, miss, method
"""

import os, argparse, itertools, warnings
import pandas as pd
from joblib import Parallel, delayed
import config, metrics            # наши локальные модули

warnings.filterwarnings("ignore", category=FutureWarning)

# ───── CLI
p = argparse.ArgumentParser()
p.add_argument("--n_jobs", type=int, default=8)
p.add_argument("--out",    default="grid_scores.csv")
args = p.parse_args()

# ───── разметка
gt = pd.read_csv("ground_truth.csv", dtype=str)
gt["acceptable"] = gt["acceptable"].fillna("").apply(lambda s: s.split(";") if s else [])
gt["well"] = gt["well"].astype(str).str.strip()
gt["ppd_well"] = gt["ppd_well"].astype(str).str.strip()
gt = gt.set_index(["well","ppd_well"])

# ───── диапазоны
DIV = [round(1+0.5*i,1) for i in range(10)]        # 1 … 10
WQ  = [round(0.3+0.1*i,1) for i in range(5)]       # 0.3 … 0.7
LMD = [300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500]                    # λ м
MODE = ["exp","linear"]

GRID = list(itertools.product(DIV, DIV, WQ, LMD, MODE))

BASE_ARGS = dict(clean_data_dir="clean_data",
                 oil_windows_fname="oil_windows.csv",
                 ppd_events_fname="ppd_events.csv",
                 oil_clean_fname="oil_clean.csv")

T1, T2, T3 = config.CI_THRESHOLDS

def categorize(ci: float) -> str:
    """
    Присваивает CI-значению одну из четырёх категорий
    по порогам из CI_THRESHOLDS.
    """
    return ("none", "weak", "medium", "strong")[
        0 if ci < T1 else 1 if ci < T2 else 2 if ci < T3 else 3
    ]

def one_run(dq,dp,wq,lmd,mode):
    config.divider_q = dq
    config.divider_p = dp
    config.w_q, config.w_p = wq, 1-wq
    config.lambda_dist, config.distance_mode = lmd, mode

    df = metrics.compute_ci_with_pre(**BASE_ARGS, methods=("none",))
    sums = (df.groupby(["well","ppd_well"],as_index=False)["CI_none"]
              .sum().round(1))
    sums["well"]=sums["well"].astype(str).str.strip()
    sums["ppd_well"]=sums["ppd_well"].astype(str).str.strip()
    rep = sums.set_index(["well","ppd_well"]).join(gt,how="inner").reset_index()

    cats = rep["CI_none"].apply(categorize)
    exact = (cats==rep["expected"]).sum()
    off   = sum(g in acc for g,acc in zip(cats,rep["acceptable"]))

    return dict(divider_q=dq, divider_p=dp,
                w_q=wq, w_p=round(1-wq,1),
                lambda_m=lmd, mode=mode,
                exact=exact, off_by_1=off,
                miss=len(rep)-exact-off, method="CI_none")

print(f"Комбинаций: {len(GRID)}")
res = Parallel(n_jobs=args.n_jobs, backend="loky", verbose=5)(
        delayed(one_run)(*g) for g in GRID)

pd.DataFrame(res).sort_values(["exact","off_by_1"],ascending=[0,0])\
    .to_csv(args.out,index=False,float_format="%.1f")
print("Сохранено",args.out)
