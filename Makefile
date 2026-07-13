.PHONY: setup data prepare train figures app all test clean

setup:
	uv sync

data:
	uv run python scripts/download_data.py

prepare:
	uv run python scripts/prepare_data.py

train:
	uv run python scripts/train_models.py

figures:
	uv run python scripts/make_figures.py

app:
	uv run streamlit run app.py

test:
	uv run pytest

all: prepare train figures

clean:
	rm -rf data/processed/*.parquet models/*.joblib reports/figures/*.png reports/results.json
