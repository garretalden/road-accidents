# UK road-accident severity modeling report

Garret Fantini<br>
August 28, 2026<br>
Extension of a course project for UPenn CIS 545: Big Data Analytics

> **Retraining status — complete:** A post-report audit identified 34,155
> duplicate substantive records that could cross the original random train/test
> split. The pipeline now deduplicates before splitting, and every model,
> threshold, metric, and figure reported below was regenerated from the
> deduplicated cohort.

## Executive summary

I investigated how well road, time, weather, and lighting conditions available at the intended prediction time can distinguish Fatal, Serious, and Slight outcomes. The source file contains 1,504,150 UK Department for Transport rows from nine represented years—2005–2007 and 2009–2014. Removing 34,155 duplicate substantive copies and 150 remaining records with missing retained values produces an analytical cohort of 1,469,845 collisions. Fatal outcomes account for 1.30%, making accuracy a poor standalone measure of performance.

This portfolio extension also corrects a leakage problem in the original course project. The earlier feature set included `Number_of_Vehicles`, which is known only after a collision. I defined a prediction-time feature contract, removed post-collision fields, discarded incompatible model artifacts and claims, and retrained every reported strategy using fold-local preprocessing and training-only model selection.

No model was uniformly strongest. The downsampled baseline achieved the highest default held-out macro F1 (0.341), while the jointly tuned model achieved the highest accuracy (0.776) and default Fatal F1 (0.086) at the cost of a Serious F1 of only 0.038. The interpolated-weight model offered a more balanced high-recall operating profile, recovering 55.0% of Fatal cases at default prediction, but its Fatal precision was only 3.7%. Applying the baseline's training-selected Fatal threshold raised held-out macro F1 to 0.351 and Fatal F1 to 0.095, although 94.2% of Fatal alerts were false positives. These results are not adequate for an automated severity decision system. They instead show how class weighting and threshold selection expose tradeoffs that could inform a carefully governed screening workflow when missed Fatal cases are more costly than false alerts.

## Problem definition and scope

Each row represents a police-reported personal-injury collision labeled Fatal, Serious, or Slight. The task is three-class classification conditional on a collision having occurred. It is not a model of whether a collision will happen, because the dataset contains no exposure observations for journeys without collisions.

Only information intended to be available at prediction time is admitted to the model. This includes road type and class, speed limit, pedestrian-crossing facilities, day and time, urban or rural context, season, and contemporaneous lighting, weather, and road-surface conditions. Any application of the model must supply those contemporaneous observations. Geographic coordinates and administrative identifiers are not model inputs because the course-project scope excluded them to limit implementation complexity, not because they are inherently unavailable or post-collision leakage.

The outputs are classification scores, not calibrated probabilities of fatality or crash risk. Downsampling and class weighting deliberately alter the effective training distribution, and no Platt, isotonic, or other calibration procedure was performed.

## Data and exploratory analysis

The raw CSV contains 1,504,150 rows and 33 columns. Comparing every source field except the non-substantive `Unnamed: 0` export index identifies 34,155 duplicate copies, including 34,144 from 2012. Deduplication retains the first copy in source order and leaves 1,469,995 rows. The separate count of 576,763 repeated `Accident_Index` values does not measure duplicate collisions: 554,052 rows contain identifiers rendered in scientific notation, collapsing them into only 372 distinct strings. Records sharing an identifier but differing in any substantive field are retained. Only `Time` (117 rows) and `Pedestrian_Crossing-Physical_Facilities` (34 rows) are missing among retained fields; after deduplication, their overlap leaves 150 rows with at least one missing retained value. Complete-case cleaning therefore leaves 1,469,845 observations. The audit found no invalid severity, weekday, road-class, speed-limit, urban/rural, date, time, or blank retained-string values.

The represented years are 2005, 2006, 2007, 2009, 2010, 2011, 2012, 2013, and 2014. The absence of 2008 and all years after 2014 means that the file should not be described as a continuous 2005–2018 panel.

### Recorded accidents by severity

![Recorded accidents by Fatal, Serious, and Slight severity in the full cleaned cohort](figures/eda/severity_distribution.png)

After deduplication and complete-case cleaning, the data contains 19,039 Fatal collisions (1.30%), 198,894 Serious collisions (13.53%), and 1,251,912 Slight collisions (85.17%). A classifier that favored the dominant class could therefore achieve high accuracy while offering little value on Fatal or Serious cases. This imbalance motivates macro F1 and per-class metrics.

### Temporal distribution of recorded accidents

![Monthly, seasonal, weekday, and hourly distributions of recorded accidents](figures/eda/temporal_distributions.png)

Recorded collisions are most common in the late afternoon, with a visible peak around 17:00, and Friday has the largest weekday count (241,579). November is the highest-volume month (135,623), while fall has the largest seasonal total. These are collision counts rather than exposure-normalized rates: differences in traffic volume could explain part of each pattern.

### Road context and severity composition

![Severity composition by speed limit, road type, and urban or rural context](figures/eda/road_context_by_severity.png)

Most recorded collisions occur at 30 mph limits (942,830; 64.1%) and on single carriageways (1,100,559; 74.9%). Rural collisions have a larger Fatal share than urban collisions (2.35% versus 0.71%), and the Fatal share is highest among well-populated speed-limit groups at 60 mph (3.14%). Collisions recorded in darkness with no street lighting also have a larger Fatal share (4.36%) than daylight collisions. These are within-collision severity proportions, not causal estimates or evidence that a setting creates more collisions per journey.

## Feature contract and leakage audit

The original analysis used `Number_of_Vehicles`, which describes the collision after it occurred. It could improve apparent prediction while violating the intended prediction-time contract. I removed it together with `Number_of_Casualties` and police-attendance status, then invalidated and retrained the earlier models rather than carrying forward contaminated results.

The final model contract retains road type; first and second road class; speed limit; physical pedestrian-crossing facilities; day of week; urban/rural context; lighting; weather; and road-surface conditions. Date and time are transformed before fitting: month maps to one of four seasons, hour maps to sine and cosine components to preserve its cyclical structure, and hours 7–9 and 16–18 produce a rush-hour indicator. Categorical inputs and season are one-hot encoded, while speed limit and the engineered time fields pass through unchanged. Every transformation is fitted inside the model pipeline.

Row identifiers are excluded because they are not model features. Coordinates, road numbers, local-authority fields, police-force codes, and LSOA identifiers were excluded to simplify the course-project feature contract; a future spatiotemporal experiment could add latitude/longitude numerically and encode administrative fields within training folds. Four excluded fields also have substantial missingness in the 1,504,150-row source file: `Carriageway_Hazards` is missing in 1,476,900 rows (98.19%), `Special_Conditions_at_Site` in 1,467,568 (97.57%), `Junction_Control` in 602,835 (40.08%), and `LSOA_of_Accident_Location` in 108,238 (7.20%). The last field is additionally a high-cardinality geographic identifier.

`Pedestrian_Crossing-Human_Control` is excluded because it has almost no variation: 1,495,269 rows (99.41%) are recorded as `None within 50 metres`, while only 8,864 rows contain either of the other two recorded categories and 17 are missing. `Year` was excluded with the geographic fields to limit course-project complexity. This choice also happens to reduce reliance on period-specific historical patterns, but that was not the original exclusion rationale. Month and hour remain represented through the engineered season and time-of-day features because they describe recurring conditions relevant to the prediction task.

## Experimental design

The pipeline creates one deterministic stratified 80/20 split with random seed 42 after deduplication: 1,175,876 training rows and 293,969 untouched test rows. The test set contains 3,808 Fatal, 39,779 Serious, and 250,382 Slight collisions. All preprocessing, resampling, weighting, hyperparameter selection, and threshold selection occur on training data. Cross-validation fits a fresh preprocessing-and-model pipeline within each fold, preventing validation observations from influencing learned category encodings or resampling.

Hyperparameter and weight searches use training-fold macro F1. Selected configurations receive fresh validation where specified and are then fitted to the complete training split. The evaluation script is the only stage that compares every frozen model on the held-out test set. Fatal thresholds are selected from five-fold out-of-fold training probabilities; test labels do not determine the threshold.

Macro F1 is the primary summary metric because it gives equal weight to Fatal, Serious, and Slight F1 scores. Fatal precision measures how many Fatal predictions are correct, Fatal recall measures how many actual Fatal cases are retrieved, and Fatal F1 balances those quantities. Per-class F1 exposes performance hidden by an aggregate. Accuracy remains useful descriptively but is secondary because Slight cases dominate the data.

## Modeling approaches

### Downsampled XGBoost baseline

The reference strategy keeps all Fatal training rows while capping Serious and Slight observations at 60,000 each within the resampling pipeline. This reduces majority-class dominance by discarding observations. Its fixed XGBoost configuration uses 200 trees, depth 7, and learning rate 0.07.

### Fixed class-weighted XGBoost

The fixed weighted model retains every training observation and assigns balanced sample weights inversely related to class frequency. It tests whether weighting can improve minority retrieval without throwing away most Slight collisions. Its fixed configuration uses 200 trees, depth 6, and learning rate 0.05.

### Tuned class-weighted XGBoost

This strategy keeps fully balanced sample weights but samples 12 hyperparameter candidates deterministically. Three-fold training CV selects the candidate with the highest mean macro F1, followed by fresh five-fold validation. The winner uses 600 trees, depth 7, learning rate 0.07, subsample 0.7, column subsample 0.6, minimum child weight 5, gamma 0.1, and L2 regularization 5. Its fresh validation macro F1 is 0.3243 ± 0.0009.

### Interpolated class-weight XGBoost

Full inverse-frequency weighting may overcorrect, so this model interpolates between unit and balanced sample weights using `1 + alpha × (balanced_weight − 1)`. A coarse three-fold search over alpha from 0 to 1 is followed by a fine search around the winner. With the tuned XGBoost parameters fixed, alpha 0.90 achieves the strongest training-fold macro F1 (0.3417) and is fitted to the full training split.

### Joint hyperparameter-and-class-weight tuned XGBoost

The joint experiment samples 20 combinations of alpha and XGBoost parameters with seed 42, selects by three-fold mean macro F1, and validates the frozen winner with five fresh folds. Candidate 6 wins with alpha 0.610, 1,000 trees, depth 6, and learning rate 0.125; its fresh validation macro F1 is 0.3375 ± 0.0010. Joint tuning seeks a better interaction between model capacity and minority weighting than the sequential searches provide.

### Cumulative-binary ordinal XGBoost

The ordinal formulation uses two balanced binary models with the tuned XGBoost parameters: one estimates Serious-or-worse and the other estimates Fatal. Their outputs are reconciled to preserve cumulative ordering and converted into Fatal, Serious, and Slight probabilities. This approach encodes the natural severity order, but separate binary objectives do not guarantee better three-class discrimination.

## Results and model tradeoffs

### Held-out model comparison

![Held-out macro F1 and Fatal F1 across the six model strategies](figures/model_comparison.png)

### Compact held-out metrics

| Model | Accuracy | Macro F1 | Fatal precision | Fatal recall | Fatal F1 | Serious F1 | Slight F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Downsampled baseline | 0.586 | **0.341** | 0.080 | 0.030 | 0.043 | **0.262** | 0.718 |
| Fixed class-weighted | 0.540 | 0.319 | 0.033 | 0.634 | 0.064 | 0.184 | 0.708 |
| Tuned class-weighted | 0.544 | 0.322 | 0.034 | 0.574 | 0.065 | 0.194 | 0.708 |
| Interpolated class weight | 0.651 | 0.339 | 0.037 | 0.550 | 0.070 | 0.145 | 0.802 |
| Joint hyperparameter and weight tuned | **0.776** | 0.336 | 0.049 | 0.377 | **0.086** | 0.038 | **0.884** |
| Cumulative-binary ordinal | 0.652 | 0.297 | 0.032 | **0.639** | 0.061 | 0.019 | 0.811 |

The downsampled baseline has the highest default test macro F1 at 0.341 and the highest Serious F1 at 0.262, but its Fatal recall is only 3.0%. The jointly tuned model has the highest accuracy (0.776), default Fatal F1 (0.086), and Slight F1 (0.884). Its accuracy comes largely from retrieving Slight cases while identifying only 2.08% of Serious cases, producing a Serious F1 of 0.038. It is therefore not a universal winner.

The interpolated-weight model is a more balanced high-recall option: it reaches 55.0% Fatal recall, 0.339 macro F1, and 0.651 accuracy. Fixed class weighting retrieves 63.4% of Fatal cases, and the ordinal model retrieves 63.9%, but their Fatal precision is approximately 3%. The ordinal model identifies only 1.0% of Serious cases. In practical terms, higher Fatal recall is obtained by labeling many non-Fatal collisions as Fatal.

The differences among objectives are substantive: choosing by accuracy favors Slight performance, choosing by macro F1 favors the downsampled baseline, and choosing by Fatal recall favors the ordinal or fixed-weighted strategies. None combines strong precision and recall across all three classes, so deployment would require an explicit error-cost policy rather than a generic claim that one model is “best.”

## Threshold analysis and error analysis

Threshold and error analysis use the downsampled baseline as a prespecified diagnostic reference, independent of which strategy ultimately leads a held-out metric. Its ordinary argmax predictions identify 113 of 3,808 Fatal test cases, for 3.0% recall and 8.0% precision. The normalized confusion matrix also shows 53.3% Serious recall and 60.3% Slight recall, illustrating that the reference model separates the classes weakly.

### Fatal precision–recall curve — downsampled baseline

![Fatal precision-recall curve for the downsampled baseline on the held-out test set](figures/fatal_precision_recall.png)

The Fatal average precision is 0.045. The threshold of 0.255 was selected by macro F1 using five-fold out-of-fold training predictions, not the test set. Applied to held-out Fatal scores, it raises Fatal recall to 26.6% while precision falls to 5.8%. Its held-out macro F1 is 0.351 and Fatal F1 is 0.095, both higher than any default argmax operating point in the six-model comparison; Serious F1 falls from 0.262 to 0.242.

### Fatal threshold tradeoff — downsampled baseline

![Training out-of-fold threshold tradeoff for the downsampled baseline](figures/fatal_threshold_tradeoff.png)

At 0.255, 17,435 of 293,969 test observations (5.93%) trigger a Fatal alert. Of these, 1,012 are actual Fatal cases and 16,423 are false alerts; 2,796 Fatal cases remain missed. Thus, 94.2% of alerts are false positives even though nearly three quarters of Fatal cases are still not retrieved. The threshold changes the operating point—it does not create new discriminatory information or make the scores calibrated.

### Normalized confusion matrix — downsampled baseline

![Row-normalized held-out confusion matrix for the downsampled baseline](figures/confusion_matrix_normalized.png)

The operational acceptability of this tradeoff cannot be decided from model metrics alone. A screening application would need a defined intervention, alert-handling capacity, costs for false alarms and missed cases, and validation that all required conditions are available at decision time.

## Interpretability and class overlap

### Fatal-specific SHAP effects — downsampled baseline

![Fatal-specific SHAP summary for the downsampled baseline](figures/shap_fatal.png)

TreeSHAP values were computed for a deterministic sample of 2,000 held-out rows. After aggregating one-hot components back to their source fields, speed limit has the largest mean absolute Fatal SHAP value (0.399), followed by second road class (0.213), hour of day (0.204), first road class (0.144), and road type (0.115). These values rank influence on this fitted model; they do not establish causal effects of changing a road or environmental condition.

### Speed-limit overlap by true severity

![Speed-limit distributions and Fatal-Serious overlap by true severity](figures/feature_distributions/speed_limit.png)

The strongest model feature still overlaps substantially across true classes. Speed-limit distributions have probability-mass overlap of 0.740 between Fatal and Serious cases and 0.676 between Fatal and Slight cases. Hour-of-day overlap is still greater—0.888 for Fatal versus Serious—and road type overlap is 0.943. These shared distributions help explain why reweighting changes prediction frequency more readily than it produces clean class separation.

## Limitations, ethics, and appropriate use

The dataset includes police-reported personal-injury collisions only. It omits uneventful journeys and therefore cannot support exposure-adjusted claims about where or when collisions are more likely. Reporting and coding practices may vary, and the nine represented years are historical and non-contiguous. There is no external or time-based validation to test geographic transfer or temporal drift.

The retained features are limited to the current contract. Fields excluded for missingness may still contain information in the subset of records where they are observed, and excluding `Year` prevents the model from representing genuine temporal changes as well as discouraging reliance on non-transferable period effects. Geographic coordinates are not modeled, and contemporaneous weather, lighting, surface, or location-context inputs may not be known in every early-response setting. The scores are uncalibrated, and the project does not establish that a threshold transfers to a new period or agency.

False negatives could withhold attention from genuinely Fatal collisions; false positives could overwhelm responders, delay other work, or create inequitable resource allocation if model errors vary across places or populations. Given the low precision and incomplete validation, the models should not autonomously determine severity or allocate emergency services. At most, they provide evidence for designing a prospective, human-supervised screening study with monitoring and an explicit cost function.

## Conclusion and next steps

Pre-collision road and environmental conditions contain some signal about recorded collision severity, but the six XGBoost strategies do not separate Fatal, Serious, and Slight outcomes reliably enough for automated decisions. The strongest default macro F1 is 0.341; the baseline's training-selected Fatal threshold raises it to 0.351, but 94.2% of its Fatal alerts are false positives. Every attempt to retrieve more Fatal cases therefore produces substantial false-positive costs or loses performance on Serious cases. The main result is a tradeoff map, not a deployable winner.

The next steps should be time-based validation on newer data, external validation across jurisdictions, and probability calibration performed strictly within training folds. Additional pre-collision exposure features—such as traffic volume or journey-level denominators—would be needed to address risk rather than severity conditional on a collision. Any operational evaluation should define the intervention, capacity constraints, and relative costs of missed Fatal cases, false Fatal alerts, and confusion between Serious and Slight outcomes before selecting a model or threshold.

# Appendices

## Appendix A — Reproducibility commands

Python 3.11 and the raw `UK_Accident.csv` file under `data/raw/` are required.
The audited file has SHA-256 `a387b49d22a06191bcec5bc0c46c29094a62cd2c57a361b69dbdce94cc922799`.

To regenerate every artifact in the required sequence, run:

```bash
make full-retrain
```

The individual stages are:

```bash
make setup
make data
make eda
make train-baseline
make train-weighted
make train-tuned
make train-interpolated
make train-ordinal
make train-joint
make evaluate
make error-analysis
make test
```

The joint search is intentionally separate from `make train-all`; it must finish before the held-out comparison is regenerated with `make evaluate`.

## Appendix B — Complete feature contract

### Model inputs

| Group | Features | Transformation |
| --- | --- | --- |
| Road context | `Road_Type`, `1st_Road_Class`, `2nd_Road_Class`, `Urban_or_Rural_Area` | One-hot encoded |
| Conditions | `Light_Conditions`, `Weather_Conditions`, `Road_Surface_Conditions` | One-hot encoded |
| Crossing context | `Pedestrian_Crossing-Physical_Facilities` | One-hot encoded |
| Calendar | `Day_of_Week`, engineered `Season` | One-hot encoded |
| Numeric | `Speed_limit`, `hour_sin`, `hour_cos`, `rush_hour` | Passed through |

`Date` supplies month and season, while `Time` supplies hour, cyclical components, and rush-hour status. Raw `Date`, `Time`, and intermediate `Month` are not passed to XGBoost.

### Excluded source fields

| Rationale | Fields |
| --- | --- |
| Post-collision leakage | `Number_of_Vehicles`, `Number_of_Casualties`, `Did_Police_Officer_Attend_Scene_of_Accident` |
| Row or collision identifiers | `Unnamed: 0`, `Accident_Index` |
| Geographic coordinates | `Location_Easting_OSGR`, `Location_Northing_OSGR`, `Longitude`, `Latitude` |
| High-cardinality geographic or administrative identifiers | `Police_Force`, `Local_Authority_(District)`, `Local_Authority_(Highway)`, `1st_Road_Number`, `2nd_Road_Number`, `LSOA_of_Accident_Location` |
| Substantial missingness | `Junction_Control` (40.08%), `Special_Conditions_at_Site` (97.57%), `Carriageway_Hazards` (98.19%), `LSOA_of_Accident_Location` (7.20%; also a high-cardinality geographic identifier) |
| Near-zero variation | `Pedestrian_Crossing-Human_Control` (99.41% recorded as `None within 50 metres`) |
| Temporal index outside the intended condition-based feature contract | `Year` |

## Appendix C — Hyperparameters and tuning candidates

### Fixed and selected configurations

| Model | Trees | Depth | Learning rate | Subsample | Column sample | Minimum child weight | Gamma | L1 | L2 | Weighting |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Downsampled baseline | 200 | 7 | 0.070 | 1.000 | 1.000 | 1 | — | — | — | Retain all Fatal; cap Serious/Slight at 60,000 each |
| Fixed class-weighted | 200 | 6 | 0.050 | 0.700 | 0.800 | 5 | — | — | — | Balanced sample weights |
| Tuned class-weighted | 600 | 7 | 0.070 | 0.700 | 0.600 | 5 | 0.1 | 0 | 5 | Balanced sample weights |
| Interpolated weights | 600 | 7 | 0.070 | 0.700 | 0.600 | 5 | 0.1 | 0 | 5 | Alpha 0.900 |
| Joint tuned | 1,000 | 6 | 0.125 | 0.832 | 0.942 | 3 | 0.5 | 0.282 | 0.522 | Alpha 0.610 |
| Ordinal cumulative | 600 | 7 | 0.070 | 0.700 | 0.600 | 5 | 0.1 | 0 | 5 | Balanced binary sample weights |

### Tuned class-weighted search

All candidates use 600 trees and balanced weights. Complete precision is retained in [the JSON results](results/xgb_tuning_results.json).

| Candidate | Depth | Rate | Child weight | Subsample | Column sample | Gamma | L1 | L2 | CV macro F1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4 | 0.05 | 3 | 0.8 | 0.9 | 0.05 | 0.0 | 5 | 0.3179 |
| 2 | 5 | 0.09 | 5 | 0.9 | 0.8 | 0.00 | 0.5 | 5 | 0.3208 |
| 3 | 5 | 0.05 | 8 | 0.6 | 0.6 | 0.00 | 0.5 | 20 | 0.3195 |
| 4 | 5 | 0.05 | 3 | 0.8 | 0.6 | 0.10 | 0.5 | 5 | 0.3196 |
| 5 | 5 | 0.05 | 8 | 0.9 | 0.9 | 0.00 | 0.0 | 20 | 0.3197 |
| 6 | 7 | 0.03 | 1 | 0.6 | 0.7 | 0.10 | 0.1 | 5 | 0.3239 |
| 7 | 6 | 0.09 | 3 | 0.6 | 0.7 | 0.05 | 0.1 | 10 | 0.3227 |
| 8 | 5 | 0.07 | 5 | 0.8 | 0.9 | 0.00 | 0.1 | 5 | 0.3207 |
| **9** | **7** | **0.07** | **5** | **0.7** | **0.6** | **0.10** | **0.0** | **5** | **0.3250** |
| 10 | 5 | 0.05 | 1 | 0.8 | 0.8 | 0.05 | 0.0 | 5 | 0.3201 |
| 11 | 6 | 0.09 | 5 | 0.8 | 0.6 | 0.05 | 0.5 | 20 | 0.3222 |
| 12 | 4 | 0.07 | 5 | 0.8 | 0.9 | 0.00 | 0.0 | 5 | 0.3184 |

### Class-weight interpolation search

| Stage | Alpha | Macro F1 | Fatal precision | Fatal recall | Fatal F1 | Serious F1 | Slight F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Coarse | 0.00 | 0.3067 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.9199 |
| Coarse | 0.20 | 0.3177 | 0.0920 | 0.0205 | 0.0335 | 0.0004 | 0.9193 |
| Coarse | 0.40 | 0.3368 | 0.0628 | 0.1852 | 0.0938 | 0.0070 | 0.9095 |
| Coarse | 0.60 | 0.3343 | 0.0492 | 0.3654 | 0.0867 | 0.0286 | 0.8875 |
| Coarse | 0.80 | 0.3394 | 0.0405 | 0.4874 | 0.0747 | 0.0965 | 0.8469 |
| Coarse | 1.00 | 0.3250 | 0.0350 | 0.5549 | 0.0659 | 0.1980 | 0.7112 |
| Fine | 0.65 | 0.3341 | 0.0467 | 0.4015 | 0.0837 | 0.0386 | 0.8800 |
| Fine | 0.70 | 0.3352 | 0.0444 | 0.4346 | 0.0806 | 0.0540 | 0.8710 |
| Fine | 0.75 | 0.3370 | 0.0424 | 0.4637 | 0.0777 | 0.0727 | 0.8605 |
| Fine | 0.80 | 0.3394 | 0.0405 | 0.4874 | 0.0747 | 0.0965 | 0.8469 |
| Fine | 0.85 | 0.3410 | 0.0389 | 0.5092 | 0.0723 | 0.1218 | 0.8289 |
| **Fine** | **0.90** | **0.3417** | **0.0375** | **0.5271** | **0.0701** | **0.1513** | **0.8037** |
| Fine | 0.95 | 0.3374 | 0.0364 | 0.5452 | 0.0682 | 0.1787 | 0.7654 |

Complete standard deviations and full-precision values are available in [the interpolation results](results/xgb_weight_alpha_search.csv).

### Joint hyperparameter-and-weight search

All candidates use 1,000 trees. Parameter and metric precision is rounded here; full values are in [the joint-search report](results/xgb_joint_tuning_results.json).

| Candidate | Alpha | Depth | Rate | Child | Subsample | Columns | Gamma | L1 | L2 | Macro F1 | Fatal F1 | Serious F1 | Slight F1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.3745 | 7 | 0.0962 | 3 | 0.9532 | 0.9828 | 0.5 | 0.0038 | 0.1360 | 0.3350 | 0.0881 | 0.0061 | 0.9107 |
| 2 | 0.6011 | 6 | 0.1412 | 3 | 0.7565 | 0.8978 | 2.0 | 0.0047 | 0.2643 | 0.3347 | 0.0835 | 0.0343 | 0.8863 |
| 3 | 0.5248 | 4 | 0.0576 | 8 | 0.6817 | 0.8012 | 0.0 | 3.9985 | 0.3433 | 0.3331 | 0.0932 | 0.0085 | 0.8974 |
| 4 | 0.6184 | 5 | 0.0660 | 12 | 0.6728 | 0.7839 | 1.0 | 0.1767 | 0.2468 | 0.3319 | 0.0860 | 0.0252 | 0.8844 |
| 5 | 0.9489 | 4 | 0.0435 | 12 | 0.8891 | 0.9880 | 0.25 | 0.0071 | 0.3586 | 0.3301 | 0.0658 | 0.1583 | 0.7662 |
| **6** | **0.6100** | **6** | **0.1250** | **3** | **0.8320** | **0.9416** | **0.5** | **0.2823** | **0.5215** | **0.3392** | **0.0829** | **0.0513** | **0.8833** |
| 7 | 0.5467 | 4 | 0.0953 | 3 | 0.9727 | 0.7147 | 0.25 | 2.0415 | 2.3757 | 0.3317 | 0.0913 | 0.0093 | 0.8946 |
| 8 | 0.0885 | 7 | 0.1387 | 12 | 0.7749 | 0.7186 | 0.0 | 0.0101 | 8.0714 | 0.3098 | 0.0081 | 0.0017 | 0.9196 |
| 9 | 0.2809 | 7 | 0.0363 | 1 | 0.7882 | 0.8399 | 0.0 | 0.0011 | 0.9425 | 0.3308 | 0.0744 | 0.0013 | 0.9168 |
| 10 | 0.2935 | 5 | 0.0831 | 5 | 0.7755 | 0.6549 | 0.5 | 0.7127 | 0.1480 | 0.3333 | 0.0831 | 0.0007 | 0.9161 |
| 11 | 0.1159 | 7 | 0.0390 | 5 | 0.8464 | 0.9521 | 1.0 | 0.1539 | 0.4287 | 0.3084 | 0.0054 | 0.0000 | 0.9199 |
| 12 | 0.3829 | 4 | 0.0926 | 5 | 0.6589 | 0.9901 | 0.5 | 0.0858 | 0.9634 | 0.3362 | 0.0939 | 0.0037 | 0.9109 |
| 13 | 0.1079 | 6 | 0.0377 | 12 | 0.9144 | 0.6610 | 0.0 | 0.0084 | 0.8796 | 0.3084 | 0.0054 | 0.0000 | 0.9199 |
| 14 | 0.2288 | 6 | 0.0811 | 3 | 0.6869 | 0.6769 | 0.5 | 0.2039 | 0.4789 | 0.3240 | 0.0524 | 0.0013 | 0.9184 |
| 15 | 0.4565 | 5 | 0.1208 | 3 | 0.9674 | 0.7265 | 1.0 | 0.0028 | 0.6605 | 0.3343 | 0.0959 | 0.0021 | 0.9048 |
| 16 | 0.2721 | 3 | 0.1040 | 5 | 0.7961 | 0.8767 | 0.0 | 0.0011 | 1.4971 | 0.3314 | 0.0767 | 0.0001 | 0.9173 |
| 17 | 0.2221 | 5 | 0.0327 | 1 | 0.7389 | 0.6920 | 1.0 | 0.0312 | 0.1410 | 0.3225 | 0.0483 | 0.0000 | 0.9191 |
| 18 | 0.2469 | 3 | 0.0367 | 1 | 0.7939 | 0.8937 | 0.5 | 0.0097 | 17.6693 | 0.3265 | 0.0609 | 0.0000 | 0.9185 |
| 19 | 0.0331 | 4 | 0.0324 | 12 | 0.7347 | 0.7708 | 0.0 | 0.0646 | 18.5358 | 0.3066 | 0.0000 | 0.0000 | 0.9199 |
| 20 | 0.6721 | 7 | 0.0326 | 8 | 0.9358 | 0.9166 | 0.5 | 4.3416 | 0.8274 | 0.3321 | 0.0822 | 0.0373 | 0.8769 |

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

| Model | Fold | Macro F1 | Accuracy | Fatal F1 | Serious F1 | Slight F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed class-weighted | 1 | 0.3186 | 0.5413 | 0.0624 | 0.1831 | 0.7103 |
| Fixed class-weighted | 2 | 0.3182 | 0.5398 | 0.0638 | 0.1822 | 0.7085 |
| Fixed class-weighted | 3 | 0.3203 | 0.5420 | 0.0635 | 0.1871 | 0.7104 |
| Fixed class-weighted | 4 | 0.3207 | 0.5423 | 0.0660 | 0.1860 | 0.7102 |
| Fixed class-weighted | 5 | 0.3193 | 0.5404 | 0.0640 | 0.1852 | 0.7087 |
| Tuned class-weighted | 1 | 0.3237 | 0.5462 | 0.0644 | 0.1965 | 0.7102 |
| Tuned class-weighted | 2 | 0.3234 | 0.5462 | 0.0654 | 0.1945 | 0.7102 |
| Tuned class-weighted | 3 | 0.3240 | 0.5453 | 0.0653 | 0.1971 | 0.7095 |
| Tuned class-weighted | 4 | 0.3255 | 0.5479 | 0.0672 | 0.1978 | 0.7114 |
| Tuned class-weighted | 5 | 0.3249 | 0.5474 | 0.0666 | 0.1974 | 0.7108 |
| Ordinal cumulative | 1 | 0.2992 | 0.6564 | 0.0606 | 0.0227 | 0.8144 |
| Ordinal cumulative | 2 | 0.2996 | 0.6562 | 0.0607 | 0.0237 | 0.8143 |
| Ordinal cumulative | 3 | 0.2999 | 0.6567 | 0.0611 | 0.0244 | 0.8144 |
| Ordinal cumulative | 4 | 0.3002 | 0.6545 | 0.0630 | 0.0248 | 0.8128 |
| Ordinal cumulative | 5 | 0.2997 | 0.6558 | 0.0629 | 0.0225 | 0.8137 |
| Interpolated weights | 1 | 0.3407 | 0.6544 | 0.0688 | 0.1495 | 0.8037 |
| Interpolated weights | 2 | 0.3411 | 0.6562 | 0.0699 | 0.1483 | 0.8050 |
| Interpolated weights | 3 | 0.3434 | 0.6538 | 0.0715 | 0.1561 | 0.8025 |
| Joint tuned | 1 | 0.3362 | 0.7744 | 0.0807 | 0.0450 | 0.8829 |
| Joint tuned | 2 | 0.3369 | 0.7751 | 0.0809 | 0.0463 | 0.8835 |
| Joint tuned | 3 | 0.3378 | 0.7751 | 0.0837 | 0.0464 | 0.8832 |
| Joint tuned | 4 | 0.3390 | 0.7765 | 0.0849 | 0.0480 | 0.8840 |
| Joint tuned | 5 | 0.3376 | 0.7750 | 0.0842 | 0.0455 | 0.8831 |
| Downsampled baseline | 1 | 0.3355 | 0.5930 | 0.0185 | 0.2643 | 0.7237 |
| Downsampled baseline | 2 | 0.3355 | 0.5948 | 0.0163 | 0.2650 | 0.7253 |
| Downsampled baseline | 3 | 0.3331 | 0.5876 | 0.0146 | 0.2667 | 0.7181 |
| Downsampled baseline | 4 | 0.3341 | 0.5859 | 0.0218 | 0.2634 | 0.7171 |
| Downsampled baseline | 5 | 0.3359 | 0.5980 | 0.0156 | 0.2634 | 0.7287 |

Full-precision fold metrics are available in [`cv_results.csv`](results/cv_results.csv).
