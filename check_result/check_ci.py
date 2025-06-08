"""
check_ci.py

Скрипт для проверки CI-результатов против ground_truth.csv.

1. Читает clean_data/ci_results_pro.csv с колонками:
   well, ppd_well, CI_none, CI_mean, CI_regression, CI_median_ewma
2. Суммирует каждый из CI_* по паре (well, ppd_well)
3. Округляет сумму до целых
4. Сравнивает результаты с разметкой из ground_truth.csv (колонки well, ppd_well, expected, acceptable)
5. Выводит ci_check_pro_report.csv с колонками:
   well, ppd_well, CI_none, CI_mean, CI_regression, CI_median_ewma, expected, acceptable

Использование:
    python check_ci.py
"""
import pandas as pd

# 1. Загрузка ground truth из CSV
gt_df = pd.read_csv("ground_truth.csv", dtype=str)
gt_df['acceptable'] = gt_df['acceptable'].str.split(';')

# 2. Загрузка CI результатов
ci_df = pd.read_csv(
    "../clean_data/ci_results_pro.csv",
    dtype={"well": str, "ppd_well": str}
)

# 3. Суммируем каждый CI_* по паре
sum_df = ci_df.groupby(["well","ppd_well"], as_index=False).agg({
    'CI_none': 'sum',
    'CI_mean': 'sum',
    'CI_regression': 'sum',
    'CI_median_ewma': 'sum'
})

# 4. Округляем суммы до целых
for col in ['CI_none','CI_mean','CI_regression','CI_median_ewma']:
    sum_df[col] = sum_df[col].round(0).astype(int)

# 5. Мёрджим с разметкой
df_report = sum_df.merge(gt_df, on=['well','ppd_well'], how='inner')

# 6. Сохраняем итоговый отчет
report_cols = [
    'well','ppd_well',
    'CI_none','CI_mean','CI_regression','CI_median_ewma',
    'expected','acceptable'
]
df_report.to_csv("ci_check_pro_report.csv", index=False, columns=report_cols)

print("ci_check_pro_report.csv создан с результатами проверки.")
