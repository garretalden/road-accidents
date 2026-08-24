# UK Road Accident Severity

An end-to-end machine-learning study of whether a reported UK road collision
will be **Fatal**, **Serious**, or **Slight**, using only information that exists
before the collision. The project emphasizes rare-class evaluation, leakage-safe
model selection, reproducibility, and honest discussion of operational limits.

The data covers roughly 1.5 million Department for Transport records from
2005–2018. Four XGBoost strategies are compared: a downsampled baseline, a
class-weighted model, a tuned class-weighted model, and a cumulative-binary
ordinal formulation.

## Why this version is being retrained

An audit found that the earlier models included `Number_of_Vehicles`, meaning
vehicles involved in the collision. That value is not available before a
collision and violated the project's prediction premise. It has now been
removed from data cleaning, preprocessing, the app, tests, and every model
configuration. The incompatible model binaries and performance claims were
deleted rather than presented as valid results.

## Modeling safeguards

- One deterministic stratified 80/20 split (`random_state=42`)
- Fold-local preprocessing in every cross-validation run
- Hyperparameter selection using training folds only
- Fatal-threshold selection from out-of-fold training probabilities only
- Training, tuning, and threshold-selection scripts never compute test metrics
- Held-out evaluation and error analysis run only after model selection is frozen
- Self-contained model pipelines that accept raw pre-accident fields
- Macro F1 and Fatal precision/recall reported alongside accuracy

## Reproduce the project

Python 3.11 is required. `uv` is recommended; `requirements.txt` is included
for standard pip environments.

```bash
make setup
make data
```

Place `UK_Accident.csv` under `data/raw/` before running `make data`; see
[data/README.md](data/README.md).

Retrain models in order:

```bash
make train-baseline     # downsampled reference
make train-weighted     # fixed class-weighted model
make train-tuned        # tuning + 5-fold validation + OOF threshold; about 1 hour
make train-ordinal      # two cumulative binary models; uses tuned parameters
make evaluate           # the only stage that evaluates the frozen test split
make error-analysis     # report, confusion matrices, PR, SHAP, distributions
```

`make train-all` runs the four training stages. `make report` runs evaluation
and error analysis after artifacts exist. Avoid `make all` unless you intend to
run the complete, potentially long pipeline.

## Interactive demo

```bash
make app
```

The app defaults to `models/tuned_xgb.joblib`, displays raw class probabilities,
and shows a threshold-adjusted decision only when the threshold config matches
the selected artifact. After reviewing the regenerated comparison, choose a
different model without editing code:

```bash
ROAD_ACCIDENT_MODEL=models/weighted_xgb.joblib make app
```

Until retraining is complete, the app intentionally displays a clear missing-
artifact message instead of loading an obsolete model.

## Repository layout

```text
app/                 Streamlit interface
configs/             Versioned model and threshold configurations
src/                 Reusable data, modeling, evaluation, and plotting code
scripts/             Six reproducible pipeline entry points
models/              Generated self-contained model artifacts
reports/results/      CV, tuning, threshold, and held-out comparison outputs
reports/figures/      Publication-ready diagnostics
notebooks/            EDA and error-analysis narratives
tests/                Feature, preprocessing, and prediction contracts
```

## Portfolio outputs

After retraining, [reports/modeling_report.md](reports/modeling_report.md)
contains the model comparison, raw and normalized confusion matrices, Fatal
precision–recall analysis, the leakage-safe threshold tradeoff, global and
Fatal-specific SHAP interpretation, feature-distribution overlaps, and model
limitations.

## Limitations

The source contains police-reported personal-injury collisions, not exposure
data for uneventful journeys, so predictions are conditional on a collision
having occurred. Weather, lighting, and surface conditions require a known
location and contemporaneous observations. Probabilities are not claimed to be
calibrated risks.

## Credits and license

Original class project by Garret Fantini, Stanley Jin, and Yuliya Solyanyk at
Penn CIS 545. Portfolio refactor and reproducibility work by Garret Fantini.

Released under the [MIT License](LICENSE).
