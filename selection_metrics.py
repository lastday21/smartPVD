"""
selection_metrics.py — Grid-search гиперпараметров и CI-порогов

Назначение
----------
Перебирает комбинации гиперпараметров:
  - divider_q, divider_p  (нормализация дебита/давления)
  - w_q, w_p              (веса ∆Q/∆P)
  - lambda_dist, distance_mode  (затухание по расстоянию)

и CI-порогов:
  - T1 (none→weak), T2 (weak→medium), T3 (medium→strong)

Для каждой комбинации:
  1) Вычисляет CI_none один раз (metrics.compute_ci_with_pre).
  2) Агрегирует CI_none по парам (well, ppd_well).
  3) Для всех (T1,T2,T3) быстро рассчитывает метрики
     exact / off_by_1 / miss / total.

Результат сохраняется в CSV с колонками:
  divider_q, divider_p, w_q, w_p,
  lambda_m, mode,
  T1, T2, T3,
  exact, off_by_1, miss, total,
  method="CI_none"

CLI
---
  python selection_metrics.py \
    --n_jobs <int>       # число параллельных процессов
    --out <path.csv>     # выходной файл (по умолчанию grid_scores.csv)

Конфигурация (config.py)
------------------------
# CI-пороговые границы (tuple из трёх чисел)
CI_THRESHOLDS = (T1, T2, T3)

# Затухание по расстоянию
lambda_dist   = <float>   # например 1400.0
distance_mode = "linear"  # или "exp"

# Параметры предобработки и детекции событий (metrics.py):
PPD_FILE            = Path("…xlsx")   # исходные Excel-файлы
OIL_FILE            = Path("…xlsx")
COORD_FILE          = Path("…xlsx")
GAP_LIMIT           = <int>
FREQ_THRESH         = <float>
MIN_WORK_PPD        = <int>
NO_PRESS_WITH_Q_LIMIT = <int>
radius              = <float>
PPD_WINDOW_SIZE     = <int>
PPD_REL_THRESH      = <float>
PPD_MIN_EVENT_DAYS  = <int>
LAG_DAYS            = <int>
OIL_CHECK_DAYS      = <int>
OIL_DELTA_P_THRESH  = <float>
OIL_EXTEND_DAYS     = <int>

# Дефолтные гиперпараметры (можно при желании перенести из скрипта):
DIV_LIST = [1.0, 1.5, …, 6.0]
WQ_LIST  = [0.4, 0.5, 0.6]
LMD_LIST = [400, 500, …, 1400]
MODE_LIST= ["linear", "exp"]
T1_LIST  = [0.5, 1.0, 1.5, 2.0]
T2_LIST  = [3.0, 4.0, 5.0]
T3_LIST  = [6.0, 7.0, 8.0]

"""

import argparse
import itertools
import warnings
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

import config
import metrics

warnings.filterwarnings("ignore", category=FutureWarning)

# ─────────────────────────────────────────────────────────────────────────────
# Модульный уровень: загрузка gt и ключевые функции для тестов
GT_PATH = Path("ground_truth.csv")
try:
    gt = pd.read_csv(GT_PATH, dtype=str)
    gt["acceptable"] = (
        gt["acceptable"].fillna("")
          .apply(lambda s: [x.strip() for x in s.split(";")] if s else [])
    )
    for col in ("well", "ppd_well"):
        gt[col] = gt[col].astype(str).str.strip()
    gt = gt.set_index(["well", "ppd_well"])
except FileNotFoundError:
    # для тестов: пустая разметка с корректным MultiIndex
    empty_index = pd.MultiIndex.from_tuples([], names=["well", "ppd_well"])
    gt = pd.DataFrame(columns=["expected", "acceptable"], index=empty_index)

# Базовые аргументы для compute_ci_with_pre
BASE_ARGS = dict(
    clean_data_dir     = "clean_data",
    oil_windows_fname  = "oil_windows.csv",
    ppd_events_fname   = "ppd_events.csv",
    oil_clean_fname    = "oil_clean.csv"
)

# CI-пороги по умолчанию (тесты могут переопределять THRESHOLDS)
T1_LIST   = [0.5, 1.0, 1.5, 2.0]
T2_LIST   = [3.0, 4.0, 5.0]
T3_LIST   = [6.0, 7.0, 8.0]
THRESHOLDS = [
    (t1, t2, t3)
    for t1 in T1_LIST for t2 in T2_LIST for t3 in T3_LIST
    if t1 < t2 < t3
]

def categorize_by(ci_val: float, t1: float, t2: float, t3: float) -> str:
    """
    Присваивает категорию CI по порогам.
    """
    return ("none", "weak", "medium", "strong")[
        0 if ci_val < t1 else 1 if ci_val < t2 else 2 if ci_val < t3 else 3
    ]

def prepare_rep(sums_df: pd.DataFrame) -> pd.DataFrame:
    """
    Объединяет агрегированные CI (sums_df) с разметкой gt.
    Возвращает DataFrame с колонками ['well','ppd_well','CI_none','expected','acceptable'].
    """
    for col in ("well", "ppd_well"):
        sums_df[col] = sums_df[col].astype(str).str.strip()
    rep = sums_df.set_index(["well","ppd_well"]).join(gt, how="inner").reset_index()
    return rep

def one_hyper_run(params):
    """
    Выполняет один прогон для params=(divider_q,divider_p,w_q,lambda_dist,mode):
    1. Настраивает config
    2. Вычисляет CI_none один раз
    3. Агрегирует суммы по (well,ppd_well)
    4. Для каждого порогового набора из THRESHOLDS
       считается exact/off_by_1/miss
    Возвращает список словарей с результатами.
    """
    dq, dp, wq, lmd, mode = params
    # настраиваем config
    config.divider_q     = dq
    config.divider_p     = dp
    config.w_q, config.w_p = wq, round(1-wq,1)
    config.lambda_dist   = lmd
    config.distance_mode = mode

    # тяжёлый вызов compute_ci_with_pre
    ci_df = metrics.compute_ci_with_pre(**BASE_ARGS, methods=("none",))
    sums_df = ci_df.groupby(["well","ppd_well"], as_index=False)["CI_none"].sum().round(1)
    rep = prepare_rep(sums_df)

    rows = []
    for t1, t2, t3 in THRESHOLDS:
        cats    = rep["CI_none"].apply(lambda x: categorize_by(x, t1, t2, t3))
        exact   = (cats == rep["expected"]).sum()
        off_by1 = sum(c in acc for c, acc in zip(cats, rep["acceptable"]))
        miss    = len(rep) - exact - off_by1
        rows.append({
            "divider_q": dq,
            "divider_p": dp,
            "w_q": wq,
            "w_p": round(1-wq,1),
            "lambda_m": lmd,
            "mode": mode,
            "T1": t1,
            "T2": t2,
            "T3": t3,
            "exact": exact,
            "off_by_1": off_by1,
            "miss": miss,
            "method": "CI_none",
        })
    return rows
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # CLI-параметры
    p = argparse.ArgumentParser()
    p.add_argument("--n_jobs", type=int, default=8, help="число параллельных потоков")
    p.add_argument("--out", default="grid_scores.csv", help="имя выходного файла")
    args = p.parse_args()

    # гиперпараметры для перебора
    DIV  = [1 + 0.5 * i for i in range(11)]
    WQ   = [0.4, 0.5, 0.6]
    LMD  = [400 + 100 * i for i in range(11)]
    MODE = ["linear"]

    # CI-пороги
    T1s = [0.5, 1.0, 1.5, 2.0]
    T2s = [3.0, 4.0, 5.0]
    T3s = [6.0, 7.0, 8.0]
    global THRESHOLDS
    THRESHOLDS = [(a,b,c) for a,b,c in itertools.product(T1s,T2s,T3s) if a<b<c]

    HYPER_GRID = list(itertools.product(DIV, DIV, WQ, LMD, MODE))

    print(f"Запускаем {len(HYPER_GRID)} hyper-комбинаций × {len(THRESHOLDS)} порогов…")
    res_nested = Parallel(n_jobs=args.n_jobs, backend="loky", verbose=5)(
        delayed(one_hyper_run)(h) for h in HYPER_GRID
    )

    flat_results = [r for sub in res_nested for r in sub]
    output_df = pd.DataFrame(flat_results).sort_values(["exact","off_by_1"], ascending=[False,False])
    output_df.to_csv(args.out, index=False, float_format="%.1f")
    print(f"Сохранено {args.out} ({len(output_df)} записей)")

if __name__ == "__main__":
    main()