# Experiments

New models go here, separate from the fixed `baseline/` (the original
class-project models).

1. Copy `_template.py` to `<your_module>.py` (no leading underscore).
2. Set `NAME`, `SLUG`, and implement `train(X_train, y_train) -> TrainResult`.
3. Run:

   ```bash
   make experiment NAME=<your_module>
   ```

This evaluates on the same held-out test set as baseline, saves the model to
`models/experiments/<SLUG>.joblib`, and upserts your result into
`reports/experiments_results.json` — re-running an experiment only replaces
its own row; it never touches `reports/baseline_results.json` or other
experiments' results.

## Choosing a class-balancing strategy

By default an experiment trains on the same *downsampled* train set as
baseline (Slight/Serious undersampled — see `DOWNSAMPLE_TARGETS` in
`road_accidents.config`). Set `BALANCE = "full"` in your module to instead
train on the full, natural-distribution train set and handle imbalance via
weighting inside `train()`:

- Logistic Regression / Random Forest: pass `class_weight="balanced"` to the
  estimator's constructor.
- XGBoost: it has no multiclass `class_weight` argument — compute per-sample
  weights with `sklearn.utils.class_weight.compute_sample_weight("balanced",
  y_train)` and pass them as `sample_weight` to `fit()`.

See `xgb_class_weighted.py` for a complete example of the "full" + sample
weighting path.

## Leakage-safe fixed XGBoost comparison

The historical weighted experiment is frozen. To evaluate its selected model
and the reproduced baseline XGBoost without another hyperparameter search, run:

```bash
make validate-xgb
```

This runs five-fold stratified cross-validation on the training split with a
fresh preprocessor in every fold. Only after both models complete CV does it
fit them on their full training inputs and evaluate the held-out test set once.
Results are written to `reports/xgb_validation_results.json` and `.md`; fitted
self-contained pipelines are saved under `models/experiments/`.
