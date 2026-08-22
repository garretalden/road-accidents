# Leakage-safe XGBoost validation

Five-fold stratified cross-validation is performed only on the training split. Each fold fits a fresh preprocessor. The held-out test set is evaluated once after model selection is complete.

## Cross-validation summary

| Model | Macro F1 | Fatal precision | Fatal recall | Fatal F1 |
| --- | ---: | ---: | ---: | ---: |
| XGBoost baseline (validated) | 0.355 ± 0.002 | 0.110 ± 0.009 | 0.024 ± 0.003 | 0.040 ± 0.004 |
| XGBoost (class-weighted, validated) | 0.336 ± 0.001 | 0.036 ± 0.000 | 0.636 ± 0.007 | 0.068 ± 0.001 |

## Per-class cross-validation metrics

| Model | Class | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: |
| XGBoost baseline (validated) | Fatal | 0.110 ± 0.009 | 0.024 ± 0.003 | 0.040 ± 0.004 |
| XGBoost baseline (validated) | Serious | 0.195 ± 0.001 | 0.601 ± 0.005 | 0.294 ± 0.001 |
| XGBoost baseline (validated) | Slight | 0.902 ± 0.001 | 0.613 ± 0.006 | 0.730 ± 0.004 |
| XGBoost (class-weighted, validated) | Fatal | 0.036 ± 0.000 | 0.636 ± 0.007 | 0.068 ± 0.001 |
| XGBoost (class-weighted, validated) | Serious | 0.185 ± 0.002 | 0.299 ± 0.005 | 0.229 ± 0.003 |
| XGBoost (class-weighted, validated) | Slight | 0.905 ± 0.000 | 0.585 ± 0.002 | 0.711 ± 0.002 |

## Final untouched-test evaluation

| Model | Macro F1 | Accuracy | Fatal precision | Fatal recall | Fatal F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| XGBoost baseline (validated) | 0.363 | 0.606 | 0.088 | 0.050 | 0.064 |
| XGBoost (class-weighted, validated) | 0.335 | 0.545 | 0.035 | 0.642 | 0.067 |

Classes: 0=Fatal, 1=Serious, 2=Slight. Standard deviations use `ddof=1`.
