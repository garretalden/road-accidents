# Data

This project uses the UK Department for Transport road-safety collision data
distributed as `UK_Accident.csv` by the Kaggle **DFT Accident Data** dataset.

Place the file at:

```text
data/raw/UK_Accident.csv
```

The final training run is pinned to this audited source:

- Size: 449,667,447 bytes
- SHA-256: `a387b49d22a06191bcec5bc0c46c29094a62cd2c57a361b69dbdce94cc922799`

Run `make preflight` to verify the file, deduplicated cohort, class counts,
represented years, and deterministic split before training.

The CSV is intentionally gitignored. The modeling code recreates one frozen,
stratified 80/20 train/test split with random seed 42 directly from the raw
file; no processed dataset is committed.

Before any fields are removed, the pipeline drops exact substantive duplicate
records by comparing every source column except the exported `Unnamed: 0` row
index. It does not deduplicate on `Accident_Index`, because many values in this
distribution were truncated to scientific notation and collide across
otherwise different records.

Only information available at the intended prediction time is retained,
including contemporaneous weather, lighting, and surface conditions. In particular,
`Number_of_Vehicles`, `Number_of_Casualties`, severity, police attendance, and
other post-collision fields are excluded before feature engineering or model
fitting.
