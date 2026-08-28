# UK Road Accident Severity

An end-to-end machine-learning study of whether a reported UK road collision
will be **Fatal**, **Serious**, or **Slight**, using only information available
at prediction time. The project emphasizes rare-class evaluation, leakage-safe
model selection, reproducibility, and honest discussion of operational limits.

The source contains roughly 1.5 million Department for Transport rows from
2005–2007 and 2009–2014. Six XGBoost strategies are compared: a downsampled baseline, a
class-weighted model, a tuned class-weighted model, a tuned model with optimized
class-weight interpolation, a joint hyperparameter-and-weight tuning experiment,
and a cumulative-binary ordinal formulation.

> **Retraining complete:** the pipeline removes 34,155 duplicate substantive
> records before splitting. All models, evaluations, thresholds, figures, and
> error-analysis outputs have been regenerated from the deduplicated cohort.

## Data and leakage corrections

An audit found that the earlier models included `Number_of_Vehicles`, meaning
vehicles involved in the collision. That value is not available before a
collision and violated the project's prediction premise. It has now been
removed from data cleaning, preprocessing, the app, tests, and every model
configuration. The incompatible model binaries and performance claims were
discarded, and every reported strategy was retrained using only the corrected
prediction-time feature contract.

A later data audit also found 34,155 complete duplicate copies when all source
fields except the non-substantive `Unnamed: 0` export index were compared. The
much larger repeated-`Accident_Index` count is not a valid duplicate count:
many identifiers were truncated to scientific notation in the distributed
CSV. Deduplication therefore compares complete substantive records and runs
before feature removal, missing-value cleaning, or train/test splitting.

## Current results

No model is uniformly strongest. On the untouched deduplicated test split, the
downsampled baseline has the highest default macro F1 at 0.341. The jointly
tuned model has the highest default Fatal F1 at 0.086 and 0.776 accuracy, but
its Serious F1 is only 0.038.

Applying the baseline's Fatal threshold of 0.255, selected from out-of-fold
training predictions, raises held-out macro F1 to 0.351 and Fatal F1 to 0.095.
This operating point retrieves 26.6% of Fatal cases at 5.8% precision, so 94.2%
of Fatal alerts are false positives. The results therefore describe a tradeoff
map, not a model suitable for autonomous severity decisions. See the
[modeling report](reports/modeling_report.md) for the full comparison and
[`reports/results/`](reports/results/) for machine-readable outputs.

## Modeling safeguards

- One deterministic stratified 80/20 split (`random_state=42`)
- Fold-local preprocessing in every cross-validation run
- Hyperparameter selection using training folds only
- Fatal-threshold selection from out-of-fold training probabilities only
- Training, tuning, and threshold-selection scripts never compute test metrics
- Held-out evaluation and error analysis run only after model selection is frozen
- Self-contained model pipelines that accept raw prediction-time fields
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
make train-baseline     # downsampled reference + OOF threshold
make train-weighted     # fixed class-weighted model
make train-tuned        # tuning + 5-fold validation + OOF threshold; about 1 hour
make train-interpolated # two-stage 3-fold search over class-weight interpolation
make train-ordinal      # two cumulative binary models; uses tuned parameters
make train-joint        # 20-candidate joint hyperparameter/alpha search; run separately
make evaluate           # the only stage that evaluates the frozen test split
make error-analysis     # figures + machine-readable metrics; curated report is preserved
make eda                # model-independent EDA figures + data-quality tables
```

To run the complete sequence—including the expensive joint search—in the
required order:

```bash
make full-retrain
```

This command runs the unit suite and a read-only data preflight before deleting
old artifacts or starting training. The preflight pins the exact audited CSV,
its deduplicated cohort, and the deterministic split counts.
If a later training stage is interrupted, resume from that individual target;
rerunning `make full-retrain` intentionally cleans completed artifacts first.

`make train-all` runs all five training stages. `make report` runs evaluation
and error analysis after artifacts exist. Avoid `make all` unless you intend to
run the complete, potentially long pipeline.

The expensive joint experiment is intentionally excluded from `make train-all`.
Run it explicitly, watch its candidate/fold progress in the terminal, and only
then evaluate the frozen model on the untouched test split:

```bash
make train-joint
make evaluate
```

The interpolation search freezes the tuned XGBoost hyperparameters and selects
`alpha` using training folds only. It first searches 0.0–1.0 in 0.2 steps, then
searches within ±0.15 of the coarse winner in 0.05 steps. Results are written to
`reports/results/xgb_weight_alpha_search.csv` and `.json`.

The joint search samples alpha across the full 0–1 interval together with the
XGBoost hyperparameters, selects by three-fold mean macro-F1, validates the
frozen winner with five folds, and saves it under
`models/experiments/xgb_joint_tuned.joblib`. Its JSON and Markdown reports are
written under `reports/results/` when `make train-joint` completes.

## Interactive demo

```bash
make app
```

The app defaults to the prespecified downsampled baseline, displays uncalibrated
class scores, and loads a threshold only when a ready configuration belongs to
the selected artifact. Choose a different model without editing code:

```bash
ROAD_ACCIDENT_MODEL=models/weighted_xgb.joblib make app
```

If the selected artifact is absent, the app displays a clear missing-artifact
message instead of attempting a prediction.

## Repository layout

```text
app/                 Streamlit interface
configs/             Versioned model and threshold configurations
src/                 Reusable data, modeling, evaluation, and plotting code
scripts/             Reproducible training, validation, and reporting entry points
models/              Generated self-contained model artifacts
reports/results/      CV, tuning, threshold, and held-out comparison outputs
reports/figures/      Publication-ready diagnostics
notebooks/            EDA and error-analysis narratives
tests/                Feature, preprocessing, and prediction contracts
```

## Portfolio outputs

`make eda` audits the raw CSV and writes model-independent descriptive figures
to `reports/figures/eda/` and report-ready tables to `reports/results/eda/`.
Substantive plots use the complete cleaned analytical cohort; raw fields are
used only to document dimensions, missingness, invalid values, and field-removal
decisions. The figures describe patterns among recorded accidents and do not
represent exposure-adjusted accident risk.

`make error-analysis` uses the prespecified downsampled baseline to generate raw
and normalized confusion matrices, Fatal precision–recall analysis, its
leakage-safe threshold tradeoff, global and Fatal-specific SHAP interpretation,
and true-severity feature distribution overlaps. Machine-readable outputs and a
generated Markdown companion live under `reports/results/`. The manually curated
[modeling report](reports/modeling_report.md) is never overwritten by the script.

## Limitations

The source contains police-reported personal-injury collisions, not exposure
data for uneventful journeys, so predictions are conditional on a collision
having occurred. Weather, lighting, and surface conditions require a known
location and contemporaneous observations. Probabilities are not claimed to be
calibrated risks.

## Credits and license

Project, analysis, modeling, and portfolio refactor by Garret Fantini for
Penn CIS 545 and subsequent independent development.

Released under the [MIT License](LICENSE).
