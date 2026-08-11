"""Streamlit demo: pick pre-crash conditions, see predicted severity probabilities.

Uses the XGBoost model (highest macro-F1 on the class project) and the fitted
preprocessor from training so single-row inference matches the training pipeline.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# pyarrow's default (jemalloc) memory pool segfaults if first touched from a
# background thread, which is exactly what Streamlit's ScriptRunner does when
# rendering a DataFrame/chart. Forcing the system allocator avoids the crash.
os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer

from road_accidents.config import BASELINE_MODELS_DIR, CLASS_NAMES, MODELS_DIR, SPEED_LIMITS
from road_accidents.features import hour_to_cyclical, is_rush_hour, month_to_season
from road_accidents.preprocessing import ONE_HOT_FEATURES

st.set_page_config(page_title="UK Accident Severity Predictor", page_icon=None, layout="centered")


@st.cache_resource
def load_artifacts() -> tuple[object, ColumnTransformer]:
    model = joblib.load(BASELINE_MODELS_DIR / "xgb.joblib")
    preprocessor: ColumnTransformer = joblib.load(MODELS_DIR / "preprocessor.joblib")
    return model, preprocessor


def _categories(preprocessor: ColumnTransformer, column: str) -> list:
    idx = ONE_HOT_FEATURES.index(column)
    return list(preprocessor.named_transformers_["cat"].categories_[idx])


def main() -> None:
    st.title("UK Road Accident Severity Predictor")
    st.markdown(
        "Choose the pre-crash conditions below and see the model's predicted severity "
        "probabilities. Model: XGBoost trained on ~1.5M UK Department for Transport records "
        "(2005–2018). See the [repository README](README.md) for caveats on the fatal class."
    )

    try:
        model, preprocessor = load_artifacts()
    except FileNotFoundError:
        st.error(
            "Trained model not found. Run `make prepare && make train` first."
        )
        st.stop()

    with st.form("prediction"):
        col1, col2 = st.columns(2)

        with col1:
            speed_limit = st.selectbox(
                "Speed limit (mph)",
                options=SPEED_LIMITS,
                index=min(3, len(SPEED_LIMITS) - 1),
            )
            number_of_vehicles = st.number_input(
                "Number of vehicles involved", min_value=1, max_value=15, value=2
            )
            urban_or_rural = st.selectbox(
                "Urban or rural area",
                options=_categories(preprocessor, "Urban_or_Rural_Area"),
                format_func=lambda v: {1: "Urban", 2: "Rural", 3: "Unallocated"}.get(v, str(v)),
            )
            day_of_week = st.selectbox(
                "Day of week (1=Sunday)",
                options=_categories(preprocessor, "Day_of_Week"),
                index=3,
            )
            hour = st.slider("Hour of day", 0, 23, 17)
            month = st.slider("Month", 1, 12, 6)

        with col2:
            light_conditions = st.selectbox(
                "Light conditions", _categories(preprocessor, "Light_Conditions")
            )
            weather = st.selectbox(
                "Weather conditions", _categories(preprocessor, "Weather_Conditions")
            )
            road_surface = st.selectbox(
                "Road surface conditions", _categories(preprocessor, "Road_Surface_Conditions")
            )
            road_type = st.selectbox("Road type", _categories(preprocessor, "Road_Type"))
            first_road_class = st.selectbox(
                "1st road class (1=Motorway .. 6=Unclassified)",
                _categories(preprocessor, "1st_Road_Class"),
            )
            second_road_class = st.selectbox(
                "2nd road class (-1 if none)",
                _categories(preprocessor, "2nd_Road_Class"),
            )
            ped_crossing = st.selectbox(
                "Pedestrian crossing (physical)",
                _categories(preprocessor, "Pedestrian_Crossing-Physical_Facilities"),
            )

        submitted = st.form_submit_button("Predict severity")

    if not submitted:
        return

    hour_sin, hour_cos = hour_to_cyclical(hour)
    row = {
        "Number_of_Vehicles": number_of_vehicles,
        "Speed_limit": speed_limit,
        "Season": month_to_season(month),
        "hour_sin": float(hour_sin),
        "hour_cos": float(hour_cos),
        "rush_hour": is_rush_hour(hour),
        "Road_Type": road_type,
        "Light_Conditions": light_conditions,
        "Weather_Conditions": weather,
        "Road_Surface_Conditions": road_surface,
        "1st_Road_Class": first_road_class,
        "2nd_Road_Class": second_road_class,
        "Pedestrian_Crossing-Physical_Facilities": ped_crossing,
        "Day_of_Week": day_of_week,
        "Urban_or_Rural_Area": urban_or_rural,
    }

    X_row = preprocessor.transform(pd.DataFrame([row]))
    probs = model.predict_proba(X_row)[0]

    st.subheader("Predicted severity probabilities")
    st.bar_chart(pd.DataFrame({"probability": probs}, index=CLASS_NAMES))

    top_class = int(np.argmax(probs))
    st.metric("Most likely class", CLASS_NAMES[top_class], f"{probs[top_class]:.1%}")

    st.info(
        "Heads up: this dataset is heavily skewed toward Slight accidents. "
        "The model rarely predicts Fatal with high confidence even when true — "
        "read the README's 'What I'd do next' section for context."
    )


if __name__ == "__main__":
    main()
