# UK road-accident severity modeling report

<!-- Add author, date, course/organization, and repository metadata here. -->
Garret Fantini
8/27/2026
Extension of course project for UPenn CIS 545: Big Data Analytics

## Executive summary (½ page)

<!--
State:
- The research question
- The UK DfT dataset and study period
- The leakage-safe redesign and retraining decision
- The most important model and error-analysis findings
- The practical conclusion

Suggested conclusion:
No model is suitable as an automated severity decision system, but the study
demonstrates how class weighting and threshold selection can support screening
when missed Fatal cases are more costly than false alerts.
-->

In this project, we seek to explore the extent to which temporal and road-related conditions can predict the severity of road accidents. Our data
covers over 1.5 million road accidents in United Kingdom, spanning from ____ to _____. 

** most important findings

** practical conclusion

This study initially began as a group project for CIS 545, in which we explored the dataset, engineered new features, and fit three models: logistic regression, random forest, and XGBoost. In converting the project to a reproducible portfolio study and expanding upon it, I omitted several variables that were initially included, such as `Number_of_Vehicles`, as to not leak post-accident information into the model. The baseline models and subsequent models were retrained thereafter.

The revised project retains the original motivation and exploratory foundation while adding a formal prediction-time feature contract, leakage-safe cross-validation, rare-class evaluation, class-weight and threshold experiments, held-out error analysis, and reproducible training pipelines.

## Problem definition and scope (½–1 page)

<!--
Define the target classes: Fatal, Serious, and Slight.

Clarify that predictions are:
- Conditional on a collision having occurred
- Based only on information available before the collision
- Not calibrated estimates of crash or fatality risk
-->

## Data and exploratory analysis (1–1½ pages)

<!--
Cover:
- Approximately 1.5 million UK DfT collision records
- The actual CSV's nine represented years: 2005–2007 and 2009–2014
- Raw-to-cleaned dimensions and missing-data handling
- Severe class imbalance
- Important temporal and road-context patterns
- Why accuracy is misleading when Fatal cases are only about 1.3% of records

Use descriptive language about recorded collisions; do not describe these
within-collision patterns as exposure-adjusted risk or causal effects.
-->

### Recorded accidents by severity

![Recorded accidents by Fatal, Serious, and Slight severity in the full cleaned cohort](figures/eda/severity_distribution.png)

### Temporal distribution of recorded accidents

![Monthly, seasonal, weekday, and hourly distributions of recorded accidents](figures/eda/temporal_distributions.png)

### Road context and severity composition

![Severity composition by speed limit, road type, and urban or rural context](figures/eda/road_context_by_severity.png)

## Feature contract and leakage audit (¾ page)

<!--
Explain:
- Why Number_of_Vehicles, Number_of_Casualties, police attendance, and other
  post-collision fields were excluded
- Which pre-collision road, time, weather, lighting, and location-context fields remain
- Date/time engineering: month, season, cyclical hour, and rush-hour indicators
- Why earlier models, artifacts, and performance claims were discarded and retrained

Use the audit tables under reports/results/eda/ as supporting evidence.
-->

## Experimental design (1 page)

<!--
Document:
- Frozen stratified 80/20 train/test split
- random_state = 42
- Fold-local preprocessing
- Training-only cross-validation and model selection
- Out-of-fold threshold selection using training data only
- A single final evaluation on the untouched test split

Define:
- Macro F1 as the primary balanced three-class metric
- Fatal precision, recall, and F1
- Per-class F1
- Accuracy as a secondary descriptive metric
-->

## Modeling approaches (1–1½ pages)

<!-- Briefly motivate each strategy. Move grids and full candidate results to the appendix. -->

### Downsampled XGBoost baseline

<!-- Add motivation, balancing approach, and role as the reference model. -->

### Fixed class-weighted XGBoost

<!-- Add motivation and contrast with discarding majority-class observations. -->

### Tuned class-weighted XGBoost

<!-- Add motivation and summarize the training-only hyperparameter search. -->

### Interpolated class-weight XGBoost

<!-- Add motivation for searching between unweighted and fully balanced class weights. -->

### Joint hyperparameter-and-class-weight tuned XGBoost

<!-- Add motivation for optimizing model capacity and weighting together. -->

### Cumulative-binary ordinal XGBoost

<!-- Add motivation for exploiting the ordered Fatal–Serious–Slight target structure. -->

## Results and model tradeoffs (1½–2 pages)

<!--
Interpret results by objective rather than declaring a universal winner:
- Downsampled baseline: highest default held-out macro F1 (0.344)
- Joint-tuned: highest accuracy (0.775) and Fatal F1 (0.086), but Serious F1 = 0.041
- Interpolated weights: more balanced high-recall option; Fatal recall = 0.552
- Fully class-weighted and ordinal models: retrieve more Fatal cases but create many false positives
- Every strategy remains constrained by class overlap

Do not call the highest-accuracy model "best" without naming the objective.
-->

### Held-out model comparison

![Held-out macro F1 and Fatal F1 across the six model strategies](figures/model_comparison.png)

### Compact held-out metrics

| Model | Accuracy | Macro F1 | Fatal precision | Fatal recall | Fatal F1 | Serious F1 | Slight F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Downsampled baseline | 0.582 | **0.344** | 0.083 | 0.041 | 0.055 | **0.262** | 0.714 |
| Fixed class-weighted | 0.540 | 0.319 | 0.033 | 0.630 | 0.063 | 0.185 | 0.708 |
| Tuned class-weighted | 0.546 | 0.324 | 0.035 | 0.578 | 0.066 | 0.197 | 0.710 |
| Interpolated class weight | 0.653 | 0.340 | 0.038 | 0.552 | 0.070 | 0.146 | 0.803 |
| Joint hyperparameter and weight tuned | **0.775** | 0.337 | 0.049 | 0.380 | **0.086** | 0.041 | **0.884** |
| Cumulative-binary ordinal | 0.652 | 0.297 | 0.032 | **0.642** | 0.061 | 0.018 | 0.812 |

## Threshold analysis and error analysis (1–1½ pages)

<!--
Explain:
- Threshold selection changes the operating point rather than inherently improving the model
- The displayed threshold diagnostics use the downsampled baseline
- The baseline threshold of 0.27 increases Fatal recall while precision remains low
- Operational implications: alert volume, missed Fatal cases, and false alarms
- The reference confusion matrix shows ordinary multiclass predictions unless explicitly noted
-->

### Fatal precision–recall curve — downsampled baseline

![Fatal precision-recall curve for the downsampled baseline on the held-out test set](figures/fatal_precision_recall.png)

### Fatal threshold tradeoff — downsampled baseline

![Training out-of-fold threshold tradeoff for the downsampled baseline](figures/fatal_threshold_tradeoff.png)

### Normalized confusion matrix — downsampled baseline

![Row-normalized held-out confusion matrix for the downsampled baseline](figures/confusion_matrix_normalized.png)

## Interpretability and class overlap (1 page)

<!--
Highlight:
- Speed limit as the strongest Fatal-related feature
- Hour of day and road class as additional contributors
- Strong distribution overlap among Fatal, Serious, and Slight observations
- SHAP values describe model associations, not causal effects
-->

### Fatal-specific SHAP effects — downsampled baseline

![Fatal-specific SHAP summary for the downsampled baseline](figures/shap_fatal.png)

### Speed-limit overlap by true severity

![Speed-limit distributions and Fatal-Serious overlap by true severity](figures/feature_distributions/speed_limit.png)

## Limitations, ethics, and appropriate use (1 page)

<!--
Cover:
- Police-reported personal-injury collisions only
- No exposure data for journeys without collisions
- Historical data from 2005–2007 and 2009–2014, the missing years, and possible temporal drift
- Location and contemporaneous conditions may not be available early enough for every use case
- Uncalibrated probabilities
- Consequences of false positives and false negatives
- No external or time-based validation
-->

## Conclusion and next steps (½ page)

<!--
Answer the research question directly.

Recommended next steps:
- Time-based validation
- Probability calibration
- Additional pre-collision exposure features
- Evaluation against an explicitly defined operational cost function
-->

# Appendices

## Appendix A — Reproducibility commands

<!-- Add environment setup, data preparation, EDA, training, evaluation, and report commands. -->

## Appendix B — Complete feature contract

<!-- Add retained features, engineered features, excluded fields, and exclusion rationales. -->

## Appendix C — Hyperparameters and tuning candidates

<!--
Add frozen configurations, search spaces, selected candidates, and links to:
- reports/results/xgb_tuning_results.json
- reports/results/xgb_weight_alpha_search.json
- reports/results/xgb_joint_tuning_results.json
-->

## Appendix D — Additional exploratory figures

### Weekday-by-hour accident counts

![Weekday-by-hour heatmap of recorded accident counts](figures/eda/weekday_hour_heatmap.png)

### Road class and severity composition

![Severity composition by first and second road class](figures/eda/road_class_by_severity.png)

### Environmental conditions and severity composition

![Severity composition by lighting, weather, and road-surface conditions](figures/eda/environment_by_severity.png)

### Pedestrian-crossing facilities and severity composition

![Severity composition by pedestrian-crossing facilities](figures/eda/crossing_facilities_by_severity.png)

### Spearman correlation matrix

![Spearman correlations among ordered and numeric analytical fields](figures/eda/spearman_correlation.png)

### Categorical association matrix

![Bias-corrected Cramer's V associations among categorical analytical fields](figures/eda/categorical_associations.png)

## Appendix E — Additional model diagnostics

### Raw confusion matrix — downsampled baseline

![Held-out confusion matrix counts for the downsampled baseline](figures/confusion_matrix.png)

### Global SHAP importance — downsampled baseline

![Global SHAP importance across all severity classes for the downsampled baseline](figures/shap_global.png)

## Appendix F — Full cross-validation results

<!-- Add or link the complete fold-level results from reports/results/cv_results.csv. -->
