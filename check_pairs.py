import pandas as pd

# Замените пути на реальные в вашей файловой структуре:
gt = pd.read_csv("ground_truth.csv", dtype=str)                          # ваша разметка
ci = pd.read_csv("clean_data/ci_results_pro.csv", dtype=str)             # итог CI

# Строгий cleanup ключей
gt["well"]     = gt["well"].astype(str).str.strip()
gt["ppd_well"] = gt["ppd_well"].astype(str).str.strip()
ci["well"]     = ci["well"].astype(str).str.strip()
ci["ppd_well"] = ci["ppd_well"].astype(str).str.strip()

# Множества пар
gt_pairs = set(zip(gt["well"], gt["ppd_well"]))
ci_pairs = set(zip(ci["well"], ci["ppd_well"]))

# Пары, которые есть в ground_truth, но нет в CI
missing = sorted(gt_pairs - ci_pairs)

print(f"Всего пар в разметке: {len(gt_pairs)}")
print(f"Найдено CI-результатов для: {len(ci_pairs)} уникальных пар")
print(f"Пропавших пар: {len(missing)}")
print("Список пропавших пар (well, ppd_well):")
for w, p in missing:
    print(f"  {w}, {p}")
