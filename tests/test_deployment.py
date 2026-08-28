import json

from src.deployment import DEFAULT_MODEL_ARTIFACT, matching_threshold_config


def test_baseline_is_the_prespecified_default_model():
    assert DEFAULT_MODEL_ARTIFACT == "models/baseline_xgb.joblib"


def test_threshold_config_must_be_ready_and_match_the_model(tmp_path):
    baseline = {
        "status": "ready",
        "model_artifact": "models/baseline_xgb.joblib",
        "threshold": 0.25,
    }
    (tmp_path / "baseline_fatal_threshold.json").write_text(json.dumps(baseline))
    assert matching_threshold_config(DEFAULT_MODEL_ARTIFACT, tmp_path) == baseline

    baseline["status"] = "pending_retrain"
    (tmp_path / "baseline_fatal_threshold.json").write_text(json.dumps(baseline))
    assert matching_threshold_config(DEFAULT_MODEL_ARTIFACT, tmp_path) is None
    assert matching_threshold_config("models/weighted_xgb.joblib", tmp_path) is None

    tuned = {
        "status": "ready",
        "model_artifact": "models/tuned_xgb.joblib",
        "threshold": 0.5,
    }
    (tmp_path / "fatal_threshold.json").write_text(json.dumps(tuned))
    assert matching_threshold_config("models/tuned_xgb.joblib", tmp_path) == tuned


def test_threshold_config_rejects_artifact_mismatch(tmp_path):
    config = {
        "status": "ready",
        "model_artifact": "models/tuned_xgb.joblib",
        "threshold": 0.5,
    }
    (tmp_path / "baseline_fatal_threshold.json").write_text(json.dumps(config))
    assert matching_threshold_config(DEFAULT_MODEL_ARTIFACT, tmp_path) is None
