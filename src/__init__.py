"""Road-accident severity modeling package."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "UK_Accident.csv"
MODELS_DIR = PROJECT_ROOT / "models"
CONFIGS_DIR = PROJECT_ROOT / "configs"
REPORTS_DIR = PROJECT_ROOT / "reports"
RESULTS_DIR = REPORTS_DIR / "results"
FIGURES_DIR = REPORTS_DIR / "figures"

RANDOM_STATE = 42
CLASS_NAMES = ["Fatal", "Serious", "Slight"]
TARGET_MAP = {1: 0, 2: 1, 3: 2}
__version__ = "0.2.0"
