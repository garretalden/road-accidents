"""Interactive prediction-time severity classification demo."""
# ruff: noqa: E402
from __future__ import annotations
import os
from pathlib import Path

os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from src import CLASS_NAMES
from src.category_labels import format_category_option
from src.deployment import DEFAULT_MODEL_ARTIFACT, matching_threshold_config
from src.evaluation import apply_fatal_threshold
from src.features import hour_to_cyclical, is_rush_hour, month_to_season
from src.preprocessing import ONE_HOT_FEATURES

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEED_LIMITS = [10, 15, 20, 30, 40, 50, 60, 70]
st.set_page_config(page_title="UK Road Accident Severity", page_icon="🚧", layout="centered")


@st.cache_resource
def load_model(path: Path):
    return joblib.load(path)


def categories(model, column: str) -> list:
    index = ONE_HOT_FEATURES.index(column)
    return list(model.named_steps["preprocessor"].named_transformers_["cat"].categories_[index])


def main() -> None:
    st.title("UK road-accident severity")
    st.caption("Predictions from static context and contemporaneous conditions")
    st.warning(
        "Educational portfolio demo only. This model estimates severity conditional on a "
        "reported collision; it does not predict whether a collision will occur. Scores are "
        "uncalibrated and must not guide emergency response or other operational decisions."
    )
    relative_model = os.environ.get("ROAD_ACCIDENT_MODEL", DEFAULT_MODEL_ARTIFACT)
    model_path = PROJECT_ROOT / relative_model
    if not model_path.exists():
        st.error(f"Model not found: `{relative_model}`. Follow the retraining sequence in README.md.")
        st.stop()
    model = load_model(model_path)
    with st.form("prediction"):
        left, right = st.columns(2)
        with left:
            speed = st.selectbox("Speed limit (mph)", SPEED_LIMITS, index=3)
            area = st.selectbox(
                "Urban or rural area",
                categories(model, "Urban_or_Rural_Area"),
                format_func=lambda value: format_category_option("Urban_or_Rural_Area", value),
            )
            day = st.selectbox(
                "Day of week",
                categories(model, "Day_of_Week"),
                index=3,
                format_func=lambda value: format_category_option("Day_of_Week", value),
            )
            hour = st.slider("Hour of day", 0, 23, 17)
            month = st.slider("Month", 1, 12, 6)
        with right:
            light = st.selectbox("Light conditions", categories(model, "Light_Conditions"))
            weather = st.selectbox("Weather conditions", categories(model, "Weather_Conditions"))
            surface = st.selectbox("Road surface", categories(model, "Road_Surface_Conditions"))
            road_type = st.selectbox("Road type", categories(model, "Road_Type"))
            first_class = st.selectbox(
                "Primary road class",
                categories(model, "1st_Road_Class"),
                format_func=lambda value: format_category_option("1st_Road_Class", value),
            )
            second_class = st.selectbox(
                "Secondary road class",
                categories(model, "2nd_Road_Class"),
                format_func=lambda value: format_category_option("2nd_Road_Class", value),
            )
            crossing = st.selectbox(
                "Physical pedestrian crossing", categories(model, "Pedestrian_Crossing-Physical_Facilities")
            )
        submitted = st.form_submit_button("Estimate severity")
    if not submitted:
        return
    hour_sin, hour_cos = hour_to_cyclical(hour)
    row = pd.DataFrame([{
        "Speed_limit": speed, "Season": month_to_season(month),
        "hour_sin": hour_sin, "hour_cos": hour_cos, "rush_hour": is_rush_hour(hour),
        "Road_Type": road_type, "Light_Conditions": light, "Weather_Conditions": weather,
        "Road_Surface_Conditions": surface, "1st_Road_Class": first_class,
        "2nd_Road_Class": second_class, "Pedestrian_Crossing-Physical_Facilities": crossing,
        "Day_of_Week": day, "Urban_or_Rural_Area": area,
    }])
    probabilities = model.predict_proba(row)[0]
    st.subheader("Uncalibrated class scores")
    st.bar_chart(pd.DataFrame({"Score": probabilities}, index=CLASS_NAMES))
    st.metric("Argmax prediction", CLASS_NAMES[int(np.argmax(probabilities))])
    threshold_config = matching_threshold_config(relative_model)
    if threshold_config is not None:
        threshold = float(threshold_config["threshold"])
        adjusted = apply_fatal_threshold(probabilities.reshape(1, -1), threshold)[0]
        st.metric("Threshold-adjusted decision", CLASS_NAMES[int(adjusted)])
        st.caption(f"Fatal threshold {threshold:.4f}, selected from training OOF predictions.")
    else:
        st.info("No validated Fatal threshold is associated with this selected model.")


if __name__ == "__main__":
    main()
