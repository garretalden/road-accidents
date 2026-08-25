.PHONY: setup data test train-baseline train-weighted train-tuned train-interpolated train-ordinal train-all evaluate error-analysis report app all clean

setup:
	uv sync --group dev

data:
	@test -f data/raw/UK_Accident.csv || (echo "Place UK_Accident.csv in data/raw/; see data/README.md" && exit 1)
	@echo "data/raw/UK_Accident.csv is ready"

test:
	uv run python -m pytest

train-baseline: data
	uv run python scripts/train_baseline.py

train-weighted: data
	uv run python scripts/train_weighted_xgb.py

train-tuned: data
	uv run python scripts/train_tuned_xgb.py

train-interpolated: data
	uv run python scripts/train_interpolated_weight_xgb.py

train-ordinal: data
	uv run python scripts/train_ordinal_xgb.py

train-all: train-baseline train-weighted train-tuned train-interpolated train-ordinal

evaluate: data
	uv run python scripts/evaluate_models.py

error-analysis: data
	uv run python scripts/generate_error_analysis.py

report: evaluate error-analysis

app:
	uv run streamlit run app/streamlit_app.py

all: train-all report

clean:
	rm -f models/baseline_xgb.joblib models/weighted_xgb.joblib models/tuned_xgb.joblib models/interpolated_weight_xgb.joblib
	rm -f models/ordinal/serious_or_worse.joblib models/ordinal/fatal.joblib
	rm -f reports/results/*.csv reports/results/*.json reports/figures/*.png
	rm -f reports/figures/feature_distributions/*.png
