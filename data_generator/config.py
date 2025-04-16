import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
RESULTS_DIR = BASE_DIR / 'results'

class Config:
    # Размеры наборов
    SMALL_DATASET_SIZE = 10_000  # ~500k записей в stats
    BIG_DATASET_SIZE = 500_000  # ~25M записей в stats

    # Вероятности
    COMPANY_RATIO = 0.2  # Доля юр.лиц
    ANOMALY_RATIO = 0.01  # Доля аномалий

    # Диапазоны
    PSX_RANGE = (1000, 99999)  # Диапазон ID оборудования
    SESSION_DURATION = (3600, 86400)  # Длительность сессии (сек)

    DATE_FORMATS = [
        "%d/%m/%Y %H:%M:%S",
        "%d-%m-%Y %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%m/%d/%Y %H:%M:%S"
    ]


os.makedirs(DATA_DIR / 'small', exist_ok=True)
os.makedirs(DATA_DIR / 'big', exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)