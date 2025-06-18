from pathlib import Path

# Источники
PPD_FILE      = Path("start_data/Параметры ППД вынгапур 2.0.xlsx")
OIL_FILE      = Path("start_data/Параметры нефтяной вынгапур 2.0.xlsx")
COORD_FILE    = Path("start_data/Координаты2.0.xlsx")

# Предобработка
GAP_LIMIT     = 5       # дней для интерполяции
FREQ_THRESH   = 41.0    # порог частоты для шумовых Q=0
MIN_WORK_PPD = 30    # порог шумовых замеров
NO_PRESS_WITH_Q_LIMIT = 80

#Составление пар
radius = 2000.0

# Детекция PPD-событий
PPD_WINDOW_SIZE     = 30      # окно для базовой линии
PPD_REL_THRESH       = 0.20    # падение ≥20%
PPD_MIN_EVENT_DAYS  = 7       #сколько минимум дней должно держаться событие

# Детекция «нефтянки» после PPD-события
LAG_DAYS = 1                   # задержка от PPD до реакции нефтянки, сут
OIL_CHECK_DAYS       = 10      # смотрим первые 10 дней
OIL_DELTA_P_THRESH   = 5.0     # изменение ΔPприем ≥5 → автоматическое продление
OIL_EXTEND_DAYS      = 30      # максимально продлеваем анализ до 30 дней

# --- затухание по расстоянию ---
lambda_dist   = 900       # м – расстояние, на котором влияние падает ~в e раз
distance_mode = "linear"       # "exp"  (e^{-d/λ})  или  "linear"  (max(0,1-d/λ))

#Метрики для расчета корреляции
CORR_THRESHOLDS: tuple[float, float, float] = (0.05, 0.30, 0.60)    # Пороги корреляции (|ρ|)  none/weak, weak/medium, medium/strong
MAX_LAG = 30    #CCF дней
PENALTY_NEG      = 1      # отрицательный знак ρ
PENALTY_ONE_EVT  = 2      # n_events == 1
PENALTY_TWO_EVT  = 1      # n_events == 2
MIN_POINTS_CCF = 60       # Минимум точек для надёжного ρ CCF

# Метрики для расчета CI
T_pre = 14 # Число дней предокна для baseline, уже не актуально


divider_q: float = 1.0 # Делители для расчета CI Δq
divider_p: float = 1.5 # Делители для расчета CI Δp
w_q: float = 0.6 # Веса для расчёта CI (сумма w_q + w_p должна быть = 1.0)
w_p: float = 0.4 # Веса для расчёта CI (сумма w_q + w_p должна быть = 1.0)

CI_THRESHOLDS = (0.5, 3, 8)  # none <0.5, weak <3, medium <8, strong ≥8

#Для финального рассчета
FUSION_DIST_LIMIT = 900

PPD_SHEET_NAME = "Скважины"

OIL_SHEET_NAME = "Скважины"