"""
main.py — конвейер SmartPVD «от Excel до final_result.csv» через pandas.DataFrame.

Режимы:
    DEBUG = True   — сохранять промежуточные CSV для отладки;
    DEBUG = False  — чистый in-memory (только финальный CSV).
    final_filter_by_gt = True - включить проверку по GT(только для тестового варианта)

Пайплайн:
    1) Предобработка исходных Excel/CSV → ppd_daily, oil_daily, coords_df
    2) Детекция PPD-событий → ppd_events
    3) Формирование пар PPD ↔ OIL → pairs_df
    4) Построение «окон» отклика нефти → oil_windows_df
    5) Расчёт CI (Confidence Index) → detail_df, ci_agg_df
    6) Корреляционный анализ → corr_df
    7) Гибридный финальный микс → final_df
    8) опциональное использование GT
    9) Сохранение final_result.csv
    10) Вывод в консоль
"""

from pathlib import Path
import pandas as pd

from config import radius, FUSION_DIST_LIMIT
from preprocess import build_clean_data
from events_PPD import detect_ppd_events
from well_pairs import build_pairs
from oil_windows import build_oil_windows
from metrics import compute_ci
from correl import calc_corr
from final_mix import run_final_mix

# ── Г Л О Б А Л Ь Н Ы Е  П А Р А М Е Т Р Ы ─────────────────────────────
DEBUG = True                  # сохранять промежуточные CSV
final_filter_by_gt = True     # использовать ground_truth.csv

CLEAN_DIR = Path("clean_data"); CLEAN_DIR.mkdir(exist_ok=True)
START_DIR = Path("start_data")

# ───────────────────────────────────────────────────────────────────────
def main() -> None:
    # 1) Предобработка
    ppd_daily, oil_daily, coords_df = build_clean_data(save_csv=DEBUG)
    ppd_daily["well"] = ppd_daily["well"].astype(str)
    oil_daily["well"] = oil_daily["well"].astype(str)
    coords_df["well"] = coords_df["well"].astype(str)

    # 2) События ППД
    ppd_events = detect_ppd_events(ppd_df=ppd_daily, save_csv=DEBUG)

    # 3) Пары скважин
    pairs_df = build_pairs(
        coords_df=coords_df,
        oil_df=oil_daily,
        ppd_df=ppd_daily,
        radius=radius,
        save_csv=DEBUG
    )

    # 4) Окна отклика нефти
    oil_windows_df = build_oil_windows(
        ppd_events_df=ppd_events,
        oil_df=oil_daily,
        pairs_df=pairs_df,
        save_csv=DEBUG
    )

    # 5) CI
    gt_path = START_DIR / "ground_truth.csv"
    gt_df = pd.read_csv(gt_path, dtype=str) if gt_path.exists() else None

    _, ci_agg_df = compute_ci(
        df={
            "oil_windows": oil_windows_df,
            "ppd_events":  ppd_events,
            "oil_clean":   oil_daily,
            "pairs":       pairs_df,
            "ground_truth": gt_df,
        },
        methods=("none",),
        save_csv=DEBUG,
        filter_by_gt=False,
    )

    # 6) Корреляция
    corr_df = calc_corr(
        df={
            "ppd_clean":  ppd_daily,
            "oil_clean":  oil_daily,
            "ppd_events": ppd_events,
            "oil_windows": oil_windows_df,
            "pairs_df":   pairs_df,
        },
        save_csv=DEBUG
    )

    # 7) Финальный микс
    if final_filter_by_gt and gt_df is not None:
        allowed_pairs = set(zip(gt_df["well"].astype(str), gt_df["ppd_well"].astype(str)))
    else:
        allowed_pairs = None

    final_df = run_final_mix(
        corr_df=corr_df,
        ci_df=ci_agg_df,
        pairs_df=pairs_df,
        allowed_pairs=allowed_pairs,
        dist_limit=FUSION_DIST_LIMIT,
        filter_by_gt=final_filter_by_gt
    )

    # 8) Если работаем с GT — merge expected/acceptable и строка TOTALS
    show_summary = False
    if final_filter_by_gt and gt_df is not None:
        # 1) merge expected / acceptable
        gt_df = gt_df.rename(columns={"well": "oil_well"})
        gt_df["acceptable"] = (
            gt_df.get("acceptable", "")
            .fillna("")
            .apply(lambda s: [x.strip() for x in s.split(";")] if s else [])
        )
        final_df = final_df.merge(
            gt_df[["oil_well", "ppd_well", "expected", "acceptable"]],
            on=["oil_well", "ppd_well"],
            how="left",
        )

        # 2) метрики
        exact = int((final_df["final_cat"] == final_df["expected"]).sum())

        def _is_nearby(r):
            acc = r["acceptable"]
            return isinstance(acc, list) and r["final_cat"] in acc

        nearby = int(final_df.apply(_is_nearby, axis=1).sum())
        total = int(len(final_df))
        miss = total - exact - nearby
        accuracy = round(exact / total, 2) if total else 0.0

        # 3) итоговая строка без лишних колонок
        summary_text = (
            f"exact={exact},nearby={nearby},miss={miss},"
            f"all={total},accuracy={accuracy:.2f}"
        )
        summary_row = {col: "" for col in final_df.columns}
        summary_row.update({
            "oil_well": "TOTALS",
            "final_cat": summary_text,
        })
        final_df = pd.concat([final_df, pd.DataFrame([summary_row])],
                             ignore_index=True, sort=False)
        show_summary = True

    # ─── 9) Сохраняем результат ───────────────────────────────────────────
    out_path = CLEAN_DIR / "final_result.csv"
    final_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    # ─── 10) Вывод в консоль ──────────────────────────────────────────────
    if show_summary:
        print(f"[MAIN] {out_path} | {summary_text}")
    else:
        print(f"[MAIN] Готово: {out_path} (строк = {len(final_df)})")


if __name__ == "__main__":
    main()
