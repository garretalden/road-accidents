"""Project-wide paths, seeds, and column lists."""

from pathlib import Path

RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_CSV_PATH = DATA_DIR / "raw" / "UK_Accident.csv"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
BASELINE_MODELS_DIR = MODELS_DIR / "baseline"
EXPERIMENTS_MODELS_DIR = MODELS_DIR / "experiments"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Class-index → human label (after LabelEncoder maps severities 1/2/3 → 0/1/2)
CLASS_NAMES = ["Fatal", "Serious", "Slight"]

# Columns dropped during cleaning: identifiers, geo, admin, redundant, or
# post-accident information not available at prediction time.
DROP_COLUMNS = [
    "Accident_Index",
    "Unnamed: 0",
    "Location_Easting_OSGR",
    "Location_Northing_OSGR",
    "Longitude",
    "Latitude",
    "Local_Authority_(District)",
    "Local_Authority_(Highway)",
    "1st_Road_Number",
    "2nd_Road_Number",
    "Pedestrian_Crossing-Human_Control",
    "Special_Conditions_at_Site",
    "Carriageway_Hazards",
    "LSOA_of_Accident_Location",
    "Year",
    "Junction_Control",
    "Police_Force",
    "Number_of_Casualties",
    "Did_Police_Officer_Attend_Scene_of_Accident",
]

CATEGORICAL_FEATURES = [
    "Road_Type",
    "Light_Conditions",
    "Weather_Conditions",
    "Road_Surface_Conditions",
    "1st_Road_Class",
    "2nd_Road_Class",
    "Pedestrian_Crossing-Physical_Facilities",
    "Day_of_Week",
    "Urban_or_Rural_Area",
]

SEASON_MAP = {
    1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring", 6: "Summer",
    7: "Summer", 8: "Summer", 9: "Fall", 10: "Fall", 11: "Fall", 12: "Winter",
}

RUSH_HOURS = frozenset({7, 8, 9, 16, 17, 18})

# Downsampling targets (post LabelEncoder mapping: 0=Fatal, 1=Serious, 2=Slight)
DOWNSAMPLE_TARGETS = {2: 60_000, 1: 60_000}  # Fatal (0) untouched
