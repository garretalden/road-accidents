"""Streamlit demo: pick pre-crash conditions, see predicted severity probabilities.

Uses the XGBoost model (highest macro-F1 on the class project) and the fitted
encoders from training so single-row inference matches the training pipeline.
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

from road_accidents.config import BASELINE_MODELS_DIR, CLASS_NAMES, MODELS_DIR, RUSH_HOURS, SEASON_MAP
from road_accidents.encoding import FittedEncoders, transform_row

st.set_page_config(page_title="UK Accident Severity Predictor", page_icon=None, layout="centered")


@st.cache_resource
def load_artifacts() -> tuple[object, FittedEncoders]:
    model = joblib.load(BASELINE_MODELS_DIR / "xgb.joblib")
    encoders: FittedEncoders = joblib.load(MODELS_DIR / "encoders.joblib")
    return model, encoders


def _categories(encoders: FittedEncoders, column: str) -> list:
    from road_accidents.config import CATEGORICAL_FEATURES

    idx = CATEGORICAL_FEATURES.index(column)
    return list(encoders.ohe.categories_[idx])


def main() -> None:
    st.title("UK Road Accident Severity Predictor")
    st.markdown(
        "Choose the pre-crash conditions below and see the model's predicted severity "
        "probabilities. Model: XGBoost trained on ~1.5M UK Department for Transport records "
        "(2005–2018). See the [repository README](README.md) for caveats on the fatal class."
    )

    try:
        model, encoders = load_artifacts()
    except FileNotFoundError:
        st.error(
            "Trained model not found. Run `make prepare && make train` first."
        )
        st.stop()

    with st.form("prediction"):
        col1, col2 = st.columns(2)

        with col1:
            speed_limit_options = sorted(int(v) for v in encoders.ordinal.categories_[0])
            speed_limit = st.selectbox(
                "Speed limit (mph)",
                options=speed_limit_options,
                index=min(2, len(speed_limit_options) - 1),
            )
            number_of_vehicles = st.number_input(
                "Number of vehicles involved", min_value=1, max_value=15, value=2
            )
            urban_or_rural = st.selectbox(
                "Urban or rural area",
                options=_categories(encoders, "Urban_or_Rural_Area"),
                format_func=lambda v: {1: "Urban", 2: "Rural", 3: "Unallocated"}.get(v, str(v)),
            )
            day_of_week = st.selectbox(
                "Day of week (1=Sunday)",
                options=_categories(encoders, "Day_of_Week"),
                index=3,
            )
            hour = st.slider("Hour of day", 0, 23, 17)
            month = st.slider("Month", 1, 12, 6)

        with col2:
            light_conditions = st.selectbox(
                "Light conditions", _categories(encoders, "Light_Conditions")
            )
            weather = st.selectbox(
                "Weather conditions", _categories(encoders, "Weather_Conditions")
            )
            road_surface = st.selectbox(
                "Road surface conditions", _categories(encoders, "Road_Surface_Conditions")
            )
            road_type = st.selectbox("Road type", _categories(encoders, "Road_Type"))
            first_road_class = st.selectbox(
                "1st road class (1=Motorway .. 6=Unclassified)",
                _categories(encoders, "1st_Road_Class"),
            )
            second_road_class = st.selectbox(
                "2nd road class (-1 if none)",
                _categories(encoders, "2nd_Road_Class"),
            )
            ped_crossing = st.selectbox(
                "Pedestrian crossing (physical)",
                _categories(encoders, "Pedestrian_Crossing-Physical_Facilities"),
            )

        submitted = st.form_submit_button("Predict severity")

    if not submitted:
        return

    row = {
        "Number_of_Vehicles": number_of_vehicles,
        "Speed_limit": speed_limit,
        "Month": month,
        "Season": SEASON_MAP[month],
        "hour_sin": float(np.sin(2 * np.pi * hour / 24)),
        "hour_cos": float(np.cos(2 * np.pi * hour / 24)),
        "rush_hour": int(hour in RUSH_HOURS),
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

    X_row = transform_row(row, encoders)
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
