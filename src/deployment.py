"""Deployment configuration helpers shared by the interactive demo and tests."""

from __future__ import annotations

import json
from pathlib import Path

from . import CONFIGS_DIR

DEFAULT_MODEL_ARTIFACT = "models/baseline_xgb.joblib"
THRESHOLD_CONFIG_BY_MODEL = {
    "models/baseline_xgb.joblib": "baseline_fatal_threshold.json",
    "models/tuned_xgb.joblib": "fatal_threshold.json",
}


def matching_threshold_config(
    model_artifact: str, configs_dir: Path = CONFIGS_DIR
) -> dict | None:
    """Return a ready threshold only when it belongs to the selected model."""
    filename = THRESHOLD_CONFIG_BY_MODEL.get(model_artifact)
    if filename is None:
        return None
    path = configs_dir / filename
    if not path.exists():
        return None
    config = json.loads(path.read_text())
    if config.get("status") != "ready":
        return None
    if config.get("model_artifact") != model_artifact:
        return None
    if "threshold" not in config:
        return None
    return config
