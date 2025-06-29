"""
Модуль *final_mix.py* «склеивает» две независимые оценки влияния
— корреляционную (corr_cat) и CI-категорию (ci_cat) — в одну
финальную метку **final_cat**.  Ошибка здесь фатальна: отчёт уйдёт
аналитикам с неверными выводами.  Поэтому раскладываем алгоритм
на атомы и проверяем каждую ветку логики.

Что именно проверяем
--------------------
1. Удаление служебной строки **TOTALS**.
2. Базовая «побеждает сильнейший» — impact перекрывает weak.
3. Дальние пары → понижение категории (dist > dist_limit).
4. Спец-правило «обоим weak, но близко» → impact.
5. Спец-правило «ci none + corr weak, но очень далеко» → none.
6. Фильтрация по ground-truth (allowed_pairs).

Для скорости ставим `dist_limit = 100`, `CORR_DIST_LIMIT = 150`
"""

from __future__ import annotations

from typing import List, Tuple

import pandas as pd
import pytest

import final_mix as fm


# ------------------------------------------------------------------ #
#  Общая фикстура: патчим дистанционные пороги
# ------------------------------------------------------------------ #
@pytest.fixture(autouse=True)
def _patch_limits(monkeypatch):
    monkeypatch.setattr(fm, "FUSION_DIST_LIMIT", 100, raising=False)
    monkeypatch.setattr(fm, "CORR_DIST_LIMIT", 150, raising=False)


# ------------------------------------------------------------------ #
#  Хелпер — строим минимальные входные таблицы
# ------------------------------------------------------------------ #
def _make_rows(rows: List[Tuple[str, str, str, str, float]]):
    """
    rows: (oil, ppd, corr_cat, ci_cat, dist)
    возвращает corr_df, ci_df, pairs_df
    """
    corr = pd.DataFrame(
        [{"oil_well": o, "ppd_well": p, "corr_cat": c} for o, p, c, _, _ in rows]
        + [{"oil_well": "TOTALS", "ppd_well": "TOTALS", "corr_cat": "none"}]  # служебная
    )
    ci = pd.DataFrame(
        [{"oil_well": o, "ppd_well": p, "ci_cat": ci} for o, p, _, ci, _ in rows]
        + [{"oil_well": "TOTALS", "ppd_well": "TOTALS", "ci_cat": "none"}]
    )
    pairs = pd.DataFrame(
        [{"oil_well": o, "ppd_well": p, "distance": d} for o, p, _, _, d in rows]
    )
    return corr, ci, pairs


# ------------------------------------------------------------------ #
#  1. TOTALS удаляется
# ------------------------------------------------------------------ #
def test_totals_row_removed():
    corr, ci, pairs = _make_rows([("O1", "P1", "weak", "weak", 50)])
    out = fm.run_final_mix(corr, ci, pairs, dist_limit=100, filter_by_gt=False)
    assert "TOTALS" not in set(out["oil_well"])


# ------------------------------------------------------------------ #
#  2. Базовое правило: сильнейший побеждает
# ------------------------------------------------------------------ #
def test_strongest_wins_near_pair():
    rows = [("O1", "P1", "impact", "weak", 50)]
    corr, ci, pairs = _make_rows(rows)
    out = fm.run_final_mix(corr, ci, pairs, dist_limit=100, filter_by_gt=False)
    assert out.loc[0, "final_cat"] == "impact"


# ------------------------------------------------------------------ #
#  3. Далёкая пара → понижение
# ------------------------------------------------------------------ #
@pytest.mark.parametrize(
    "ci, corr, dist, expected",
    [
        ("none",  "impact", 200, "none"),   # шаг −2
        ("weak",  "impact", 200, "weak"),   # шаг −1
    ],
)
def test_distant_pair_downgrade(ci, corr, dist, expected):
    corr_df, ci_df, pairs_df = _make_rows([("O2", "P2", corr, ci, dist)])
    out = fm.run_final_mix(corr_df, ci_df, pairs_df, dist_limit=100, filter_by_gt=False)
    assert out.loc[0, "final_cat"] == expected


# ------------------------------------------------------------------ #
#  4. Спец-правило: обе weak, но ближе 550 м => impact
# ------------------------------------------------------------------ #
def test_two_weak_close_becomes_impact():
    rows = [("O3", "P3", "weak", "weak", 80)]  # 80 < 550
    corr, ci, pairs = _make_rows(rows)
    out = fm.run_final_mix(corr, ci, pairs, dist_limit=100, filter_by_gt=False)
    assert out.loc[0, "final_cat"] == "impact"


# ------------------------------------------------------------------ #
#  5. ci none + corr weak + очень далеко → none
# ------------------------------------------------------------------ #
def test_none_plus_weak_far_becomes_none():
    rows = [("O4", "P4", "weak", "none", 200)]  # > CORR_DIST_LIMIT (150)
    corr, ci, pairs = _make_rows(rows)
    out = fm.run_final_mix(corr, ci, pairs, dist_limit=100, filter_by_gt=False)
    assert out.loc[0, "final_cat"] == "none"


# ------------------------------------------------------------------ #
#  6. Фильтрация по ground-truth
# ------------------------------------------------------------------ #
def test_filter_by_ground_truth():
    rows = [
        ("O1", "P1", "weak",   "weak",   50),
        ("O2", "P2", "impact", "impact", 50),
    ]
    corr, ci, pairs = _make_rows(rows)

    # оставляем только пару (O2,P2)
    allowed = {("O2", "P2")}

    out = fm.run_final_mix(
        corr, ci, pairs,
        allowed_pairs=allowed,
        dist_limit=100,
        filter_by_gt=True,
    )

    assert set(zip(out.oil_well, out.ppd_well)) == allowed
