#!/usr/bin/env python
"""
check_result/check_ci.py
Сверяем CI с ground_truth и формируем один отчёт ci_check_report.csv.
"""

import pandas as pd, config
from pathlib import Path

# ──────────────────────── расположение файлов ─────────────────────────
ROOT         = Path(__file__).resolve().parents[1]      # <проект>/…
GROUND_TRUTH = ROOT / "ground_truth.csv"
CI_RESULTS   = ROOT / "clean_data" / "ci_results.csv"
OUT_REPORT   = Path(__file__).with_name("ci_check_report.csv")

# ──────────────────────── загрузка данных ─────────────────────────────
gt = pd.read_csv(GROUND_TRUTH, dtype=str)
gt["acceptable"] = gt["acceptable"].fillna("").apply(
    lambda s: [x.strip() for x in s.split(";")] if s else []
)

ci = pd.read_csv(CI_RESULTS, dtype={"well": str, "ppd_well": str})

# чистим пробелы в ключах
for col in ("well", "ppd_well"):
    gt[col] = gt[col].astype(str).str.strip()
    ci[col] = ci[col].astype(str).str.strip()

# ──────────────────────── суммируем CI_none ───────────────────────────
agg = ci.groupby(["well", "ppd_well"], as_index=False)["CI_none"].sum().round(1)

# ──────────────────────── объединяем с GT ─────────────────────────────
df = agg.merge(gt, on=["well", "ppd_well"], how="inner")

# ──────────────────────── категория CI ────────────────────────────────
try:                       # пробуем забрать функцию из config
    categorize = config.categorize
except AttributeError:     # иначе строим на основе порогов в конфиге
    T1, T2, T3 = getattr(config, "CI_THRESHOLDS", (2, 4, 6))
    def categorize(ci):
        return ("none", "weak", "medium", "strong")[
            0 if ci < T1 else 1 if ci < T2 else 2 if ci < T3 else 3
        ]

df["pred_cat"] = df["CI_none"].apply(categorize)

# ──────────────────────── считаем метрики ─────────────────────────────
exact     = (df["pred_cat"] == df["expected"]).sum()
off_by_1  = sum(df.apply(lambda r: r["pred_cat"] in r["acceptable"], axis=1))
miss      = len(df) - exact - off_by_1
total     = len(df)

# ──────────────────────── формируем отчёт ─────────────────────────────
report_cols = ["well", "ppd_well", "CI_none", "expected", "acceptable"]
df_report   = df[report_cols].copy()

accuracy = round((exact + off_by_1) / total, 2)

# дописываем строку-итог
summary = pd.DataFrame([{
    "well":        "TOTALS",
    "ppd_well":    config.distance_mode,            # exp / linear
    "CI_none":     f"exact={exact}",
    "expected":    f"off={off_by_1};",
    "acceptable":  f"miss={miss};all={total}, accuracy={accuracy}"
}])
df_report = pd.concat([df_report, summary], ignore_index=True)
df_report.to_csv(OUT_REPORT, index=False)



# ──────────────────────── вывод в консоль ─────────────────────────────
print(f"{exact},{off_by_1},{miss},{total}, точность {accuracy}")

