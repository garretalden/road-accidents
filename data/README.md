# Data

This project uses the UK Department for Transport road-safety collision data
distributed as `UK_Accident.csv` by the Kaggle **DFT Accident Data** dataset.

Place the file at:

```text
data/raw/UK_Accident.csv
```

The CSV is intentionally gitignored. The modeling code recreates one frozen,
stratified 80/20 train/test split with random seed 42 directly from the raw
file; no processed dataset is committed.

Only information available before a collision is retained. In particular,
`Number_of_Vehicles`, `Number_of_Casualties`, severity, police attendance, and
other post-collision fields are excluded before feature engineering or model
fitting.
