"""
Задача
------
Из всех комбинаций (`divider_q`, `divider_p`, `w_q/w_p`) подобрать ту,
которая даёт **максимум точных попаданий** (`exact`) и, при равенстве exact,
максимум «соседних» попаданий (`off-by-1`) на расширенной разметке
`ground_truth_pro.csv`.

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

import os
import argparse
import itertools
import warnings

import pandas as pd
from joblib import Parallel, delayed

import config          # динамически меняем divider_q/p и w_q/w_p
import metrics         # основной расчёт CI (быстрый, без baseline)

warnings.filterwarnings("ignore", category=FutureWarning)

# ─────────────── CLI ─────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--n_jobs", type=int, default=8,
                    help="Сколько процессов запускать параллельно")
parser.add_argument("--out", default="grid_scores.csv",
                    help="Имя выходного CSV")
args = parser.parse_args()

# ─────────────── ground truth ────────────────────────────────────────────────
gt = pd.read_csv("ground_truth_pro.csv", dtype=str)
gt["acceptable"] = gt["acceptable"].fillna("").apply(lambda s: s.split(";") if s else [])
gt["well"] = gt["well"].astype(str).str.strip()
gt["ppd_well"] = gt["ppd_well"].astype(str).str.strip()
gt = gt.set_index(["well", "ppd_well"])

# ─────────────── путь к данным ───────────────────────────────────────────────
BASE_ARGS = dict(
    clean_data_dir="clean_data",
    oil_windows_fname="oil_windows.csv",
    ppd_events_fname="ppd_events.csv",
    oil_clean_fname="oil_clean.csv",
)
# «разогрев» кеша pandas
_ = pd.read_csv(os.path.join("clean_data", "oil_windows.csv"), nrows=1)

# ─────────────── сетка гиперпараметров ───────────────────────────────────────
DIV_RANGE = [round(1.0 + 0.5 * i, 1) for i in range(0, 19)]   # 1.0 … 10.0
W_Q_RANGE = [round(0.3 + 0.1 * i, 1) for i in range(0, 5)]    # 0.3 … 0.7
GRID = list(itertools.product(DIV_RANGE, DIV_RANGE, W_Q_RANGE))

# ─────────────── категоризация CI → класс ───────────────────────────────────
def categorize(ci: float) -> str:
    if ci < 2:
        return "none"
    elif ci < 4:
        return "weak"
    elif ci < 6:
        return "medium"
    else:
        return "strong"

# ─────────────── одна комбинация ─────────────────────────────────────────────
def one_run(div_q, div_p, w_q):
    # записываем в config
    config.divider_q = div_q
    config.divider_p = div_p
    config.w_q = w_q
    config.w_p = 1.0 - w_q

    # считаем CI (только none)
    df = metrics.compute_ci_with_pre(**BASE_ARGS, methods=("none",))

    # суммируем по паре
    sum_ci = (
        df.groupby(["well", "ppd_well"], as_index=False)["CI_none"]
          .sum()
          .round(1)  # .1 для компактности
    )
    sum_ci["well"] = sum_ci["well"].astype(str).str.strip()
    sum_ci["ppd_well"] = sum_ci["ppd_well"].astype(str).str.strip()

    # join с разметкой
    rep = (
        sum_ci.set_index(["well", "ppd_well"])
              .join(gt, how="inner")
              .reset_index()
    )

    # точные / соседние
    cats = rep["CI_none"].apply(categorize)
    exact = (cats == rep["expected"]).sum()
    off = sum(g in acc for g, acc in zip(cats, rep["acceptable"]))

    return {
        "divider_q": div_q,
        "divider_p": div_p,
        "w_q":       w_q,
        "w_p":       round(1.0 - w_q, 1),
        "exact":     exact,
        "off_by_1":  off,
        "miss":      len(rep) - exact - off,
        "method":    "CI_none",
    }

# ─────────────── запуск параллельно ──────────────────────────────────────────
print(f"Перебираем {len(GRID)} комбинаций …")
results = Parallel(
    n_jobs=args.n_jobs,
    backend="loky",
    verbose=4,
)(delayed(one_run)(dq, dp, wq) for dq, dp, wq in GRID)

# ─────────────── сохранение ──────────────────────────────────────────────────
df_scores = (
    pd.DataFrame(results)
      .sort_values(["exact", "off_by_1"], ascending=[False, False])
      .reset_index(drop=True)
)
df_scores.to_csv(args.out, index=False, float_format="%.1f")
print(f"Файл {args.out} сохранён.   top-exact = {df_scores.loc[0,'exact']}")
