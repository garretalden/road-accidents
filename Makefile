.PHONY: setup data preflight test eda train-baseline train-weighted train-tuned train-interpolated train-joint train-ordinal train-all evaluate error-analysis report app all full-retrain clean
.NOTPARALLEL: full-retrain train-all

setup:
	uv sync --group dev

data:
	@test -f data/raw/UK_Accident.csv || (echo "Place UK_Accident.csv in data/raw/; see data/README.md" && exit 1)
	@echo "data/raw/UK_Accident.csv is ready"

preflight: data
	uv run python scripts/validate_data.py

test:
	uv run python -m pytest

eda: data
	uv run python scripts/generate_eda.py

train-baseline: data
	uv run python scripts/train_baseline.py

train-weighted: data
	uv run python scripts/train_weighted_xgb.py

train-tuned: data
	uv run python scripts/train_tuned_xgb.py

train-interpolated: data
	uv run python scripts/train_interpolated_weight_xgb.py

train-joint: data
	uv run python scripts/train_joint_tuned_xgb.py

train-ordinal: data
	uv run python scripts/train_ordinal_xgb.py

train-all: train-baseline train-weighted train-tuned train-interpolated train-ordinal

evaluate: data
	uv run python scripts/evaluate_models.py

error-analysis: data
	uv run python scripts/generate_error_analysis.py

report: evaluate error-analysis

app:
	uv run python -m streamlit run app/streamlit_app.py

all: full-retrain

# Intentionally sequential: later stages consume artifacts produced by earlier ones.
full-retrain: data
	$(MAKE) test
	$(MAKE) preflight
	$(MAKE) clean
	$(MAKE) eda
	$(MAKE) train-baseline
	$(MAKE) train-weighted
	$(MAKE) train-tuned
	$(MAKE) train-interpolated
	$(MAKE) train-ordinal
	$(MAKE) train-joint
	$(MAKE) evaluate
	$(MAKE) error-analysis

clean:
	rm -f models/baseline_xgb.joblib models/weighted_xgb.joblib models/tuned_xgb.joblib models/interpolated_weight_xgb.joblib
	rm -f models/experiments/xgb_joint_tuned.joblib
	rm -f models/ordinal/serious_or_worse.joblib models/ordinal/fatal.joblib
	rm -f reports/results/*.csv reports/results/*.json reports/figures/*.png
	rm -f reports/figures/feature_distributions/*.png
	rm -f reports/figures/eda/*.png
	rm -f reports/results/eda/*.csv reports/results/*.md
