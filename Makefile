.PHONY: setup data prepare train train-baseline experiment validate-xgb tune-xgb threshold-xgb ordinal-xgb figures app all test clean

setup:
	uv sync

data:
	uv run python scripts/download_data.py

prepare:
	uv run python scripts/prepare_data.py

train: train-baseline

train-baseline:
	uv run python scripts/train_baseline.py

experiment:
	uv run python scripts/train_experiment.py $(NAME)

validate-xgb:
	uv run python scripts/validate_xgb.py

tune-xgb:
	uv run python scripts/tune_xgb_weighted.py

threshold-xgb:
	uv run python scripts/threshold_xgb_fatal.py

ordinal-xgb:
	uv run python scripts/validate_xgb_ordinal.py

figures:
	uv run python scripts/make_figures.py

app:
	uv run streamlit run app.py

test:
	uv run pytest

all: prepare train figures

clean:
	rm -rf data/processed/*.parquet models/baseline/*.joblib models/experiments/*.joblib \
		reports/figures/*.png reports/baseline_results.json reports/experiments_results.json \
		reports/xgb_validation_results.json reports/xgb_validation_results.md \
		reports/xgb_tuning_results.json reports/xgb_tuning_results.md \
		reports/xgb_fatal_threshold_results.json reports/xgb_fatal_threshold_results.md \
		reports/xgb_ordinal_results.json reports/xgb_ordinal_results.md \
		models/experiments/xgb_weighted_tuned_fatal_threshold.json
