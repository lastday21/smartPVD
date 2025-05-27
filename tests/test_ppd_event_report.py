# tests/test_ppd_event_report.py

import pandas as pd
from pathlib import Path
import pytest

from events_PPD import detect_ppd_events
from config import PPD_BASELINE_DAYS, PPD_REL_THRESH, PPD_MIN_EVENT_DAYS

# Пути
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
PPD_CSV      = PROJECT_ROOT / "clean_data" / "ppd_clean.csv"
TEST_DIR     = Path(__file__).parent
REPORT_DIR   = TEST_DIR / "reports"

def build_reports(events_well: pd.DataFrame):
    """
    Сохраняет три формата отчёта в REPORT_DIR:
    - CSV
    - Markdown (ручная сборка |-таблицы)
    - Excel
    """
    # CSV
    out_csv = REPORT_DIR / "ppd_events_575.csv"
    events_well.to_csv(out_csv, index=False)

    # Markdown
    out_md = REPORT_DIR / "ppd_events_575.md"
    headers = list(events_well.columns)
    lines = []
    # шапка
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    # строки
    for _, row in events_well.iterrows():
        vals = [row[h] for h in headers]
        # приводим NaN, pd.NA в пустую строку
        str_vals = ["" if pd.isna(v) else str(v) for v in vals]
        lines.append("| " + " | ".join(str_vals) + " |")
    out_md.write_text("\n".join(lines), encoding="utf-8")

    # Excel
    out_xlsx = REPORT_DIR / "ppd_events_575.xlsx"
    events_well.to_excel(out_xlsx, index=False)


def test_ppd_event_report_for_well_575():
    # --- подготовка ---
    REPORT_DIR.mkdir(exist_ok=True)
    assert PPD_CSV.exists(), f"Не найден файл: {PPD_CSV}"

    # Читаем очищенные данные
    ppd_df = pd.read_csv(PPD_CSV, parse_dates=['date'])
    # Убедимся, что столбец well — int
    ppd_df['well'] = ppd_df['well'].astype(int)

    # Обнаруживаем события
    events = detect_ppd_events(ppd_df)
    events_well = events[events['well'] == 575].reset_index(drop=True)

    # --- сохраняем отчёты ---
    build_reports(events_well)

    # --- pytest-проверки ---
    # 1) Есть события
    assert not events_well.empty, "Не найдено PPD-событий для скважины 575"
    # 2) Правильные колонки
    expected = {
        'well', 'start_date', 'end_date', 'duration_days',
        'baseline', 'min_q', 'max_q', 'relative_change'
    }
    miss = expected - set(events_well.columns)
    assert not miss, f"В отчёте отсутствуют колонки: {miss}"
    # 3) Файлы созданы
    assert (REPORT_DIR / "ppd_events_575.csv").exists()
    assert (REPORT_DIR / "ppd_events_575.md").exists()
    assert (REPORT_DIR / "ppd_events_575.xlsx").exists()


if __name__ == '__main__':
    # Быстрый анализ без pytest
    REPORT_DIR.mkdir(exist_ok=True)
    ppd_df = pd.read_csv(PPD_CSV, parse_dates=['date'])
    ppd_df['well'] = ppd_df['well'].astype(int)

    events = detect_ppd_events(ppd_df)
    events_well = events[events['well'] == 575]

    # Сохраняем все три отчёта
    build_reports(events_well)

    # И выводим Markdown-таблицу в консоль
    pd.set_option('display.expand_frame_repr', False)
    print("\nPPD-события для скважины 575:\n")
    # тут вручную печатаем первые 10 строк
    headers = list(events_well.columns)
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in events_well.head(10).iterrows():
        vals = [row[h] for h in headers]
        str_vals = ["" if pd.isna(v) else str(v) for v in vals]
        print("| " + " | ".join(str_vals) + " |")
