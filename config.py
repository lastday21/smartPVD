from pathlib import Path

# Источники
PPD_FILE      = Path("Data/Параметры ППД вынгапур 2.0.xlsx")
OIL_FILE      = Path("Data/Параметры нефтяной вынгапур 2.0.xlsx")
COORD_FILE    = Path("Data/Координаты2.0.xlsx")

# Предобработка
GAP_LIMIT     = 2       # дней для интерполяции
FREQ_THRESH   = 41.0    # порог частоты для шумовых Q=0

# Детекция PPD-событий
PPD_BASELINE_DAYS    = 30      # окно для базовой линии
PPD_REL_THRESH       = 0.20    # падение ≥20%
PPD_MIN_EVENT_DAYS  = 7       #сколько минимум дней должно держаться событие

# Детекция «нефтянки» после PPD-события
OIL_CHECK_DAYS       = 10      # смотрим первые 10 дней
OIL_DELTA_P_THRESH   = 5.0     # изменение ΔPприем ≥5 → автоматическое продление
OIL_EXTEND_DAYS      = 30      # максимально продлеваем анализ до 30 дней
MAX_EVENT_DAYS       = 30      # общий предел длины события

# Метрики импакта
IMPACT_Q_DIVISOR     = 15.0
IMPACT_P_DIVISOR     = 5.0