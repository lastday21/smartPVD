## 1. О чём проект

**SmartPVD** — компактный Python-фреймворк для оценки влияния  
поддерживающей закачки (ППД) на добычу нефти.  
Он превращает «сырые» Excel/CSV-файлы в готовый отчёт  
с категорией влияния (`none` / `weak` / `impact`) по каждой паре  
добывающая – нагнетательная скважина.
Запуск происходит в main.py для прогона алгоритма от начала до конца. Для тестирования работы модулей есть возможность запускать их отдельно.
В config.py основные параметры для расчета всего алгоритма. Для их подбора используется sweep_runner, который подбирает 
параметры под наилучший результат при наличии эталона - ground_truth.cvs.

---

## 2. Структура репозитория

```text
SmartPVD/
├─ clean_data/          ← все промежуточные и финальные CSV  
├─ start_data/          ← входные Excel/CSV и ground_truth.csv (опц.)  
├─ tests/               ← PyTest-набор (≈100% покрытия)  
├─ config.py            ← глобальные константы и пути  
├─ preprocess.py        ← очистка и нормализация суточных рядов  
├─ events_PPD.py        ← детектор событий ППД  
├─ well_pairs.py        ← подбор пар «добыча – ППД» по координатам  
├─ window_selector.py   ← расчёт одного окна отклика нефти  
├─ oil_windows.py       ← пакетная обёртка над window_selector  
├─ metrics.py           ← расчёт CI (Confidence Index)  
├─ correl.py            ← расчёт корреляций и best-CCF  
├─ final_mix.py         ← объединение CI + CORR → final_cat  
├─ sweep_runner.py      ← перебор параметров (Optuna-ready)  
└─ main.py              ← точка входа: `python -m main`  
```

---

## 3. Установка

1. Клонировать репозиторий и перейти в папку:
   ```bash
   git clone https://github.com/lastday21/smartPVD.git
   cd smartPVD
   ```
2. Создать и активировать виртуальное окружение:
   ```bash
   python -m venv venv
   # macOS/Linux
   source venv/bin/activate
   # Windows
   venv\Scripts\activate
   ```
3. Установить зависимости и прогнать тесты:
   ```bash
   pip install -r requirements.txt
   pytest -q
   ```
> **Требуется** Python 3.10+ и `pandas >= 2.0`.

---

## 4. Конвейер обработки шаг за шагом

| №  | Шаг                   | Модуль / файл                      | Вход → Выход                                                    | Описание                                                                                                                 |
|----|-----------------------|------------------------------------|-----------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------|
| 0  | Старт                 | `main.py`                          | `start_data/*`                                                  | Оркестрация, парсинг флагов, базовое логирование                                                                         |
| 1  | Очистка данных        | `preprocess.py`                    | Excel → `ppd_clean.csv`<br>`oil_clean.csv`<br>`coords_clean.csv` | Нормализация дат, NaN, интерполяция, фильтрация «мертвых» значений                                                       |
| 2  | Подбор пар            | `well_pairs.py`                    | `coords_clean.csv` → `pairs_oil_ppd.csv`                       | Для каждой нефтяной скважины поиск ППД-скважин в радиусе `PAIR_RADIUS`                                                   |
| 3  | Детекция событий ППД  | `events_PPD.py`                    | `ppd_clean.csv` → `ppd_events.csv`                              | «Скользящее окно»: падение расхода > `REL_THRESH` на `MIN_EVENT_DAYS`                                                    |
| 4  | Окна отклика нефти    | `oil_windows.py` + `window_selector.py` | `ppd_events.csv` + `oil_clean.csv` → `oil_windows.csv`          | Окно через `LAG_DAYS`, длительность `OIL_CHECK_DAYS`, продление при ΔP_oil ≥ порога, обрезка                             |
| 5  | Confidence Index (CI) | `metrics.py`                       | `oil_windows.csv` + `ppd_events.csv` + `pairs_oil_ppd.csv` → `ci_results.csv` | Расчёт CI, затухание по дистанции (`linear`/`exp`), категоризация                                                        |
| 6  | Корреляции            | `correl.py`                        | `ppd_clean.csv`, `oil_clean.csv` → `corr_results.csv`           | Spearman по событиям и best-CCF (кросс-корреляция ∆-рядов)                                                               |
| 7  | Финальный микс        | `final_mix.py`                     | `ci_results.csv`, `corr_results.csv`, `pairs_oil_ppd.csv` → `final_result.csv` | Объединение CI + CORR, штраф за дальность, спец-правила                                                                  |
| 8  | Ground Truth (опц.)   | `final_mix.py`                     | merge с `start_data/ground_truth.csv`                          | Эталон, для подбора параметрво и проверки точности.При `final_filter_by_gt=True` фильтрация и добавление строки `TOTALS` |
| 9  | Завершение            | —                                  | CSV в `clean_data/`                                            | Готовые результаты для BI-аналитики или Excel Pivot                                                                      |

---

## 5. Тесты

- Каталог `tests/` содержит более 250 проверок ключевой логики.  
- Запуск:
  ```bash
  pytest -q
  ```

---
