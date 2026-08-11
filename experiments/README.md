# Experiments

New models go here, separate from the fixed `baseline/` (the original
class-project models).

1. Copy `_template.py` to `<your_module>.py` (no leading underscore).
2. Set `NAME`, `SLUG`, and implement `train(X_train, y_train) -> TrainResult`.
3. Run:

   ```bash
   make experiment NAME=<your_module>
   ```

This trains on the same preprocessed data as baseline, evaluates on the same
held-out test set, saves the model to `models/experiments/<SLUG>.joblib`, and
upserts your result into `reports/experiments_results.json` — re-running an
experiment only replaces its own row; it never touches
`reports/baseline_results.json` or other experiments' results.
