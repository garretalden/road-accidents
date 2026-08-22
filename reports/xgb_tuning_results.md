# Weighted XGBoost tuning results

Hyperparameters were selected using three-fold stratified CV on training data only. The winner was then evaluated with five-fold CV and fitted on all training data before the final test set was evaluated once.

## Five-fold CV comparison

| Model | Macro F1 | Fatal precision | Fatal recall | Fatal F1 |
| --- | ---: | ---: | ---: | ---: |
| XGBoost baseline (validated) | 0.355 ± 0.002 | 0.110 ± 0.009 | 0.024 ± 0.003 | 0.040 ± 0.004 |
| XGBoost (class-weighted, validated) | 0.336 ± 0.001 | 0.036 ± 0.000 | 0.636 ± 0.007 | 0.068 ± 0.001 |
| XGBoost (class-weighted, tuned) | 0.339 ± 0.001 | 0.037 ± 0.000 | 0.602 ± 0.007 | 0.069 ± 0.001 |

## Five-fold per-class metrics

| Model | Class | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: |
| XGBoost baseline (validated) | Fatal | 0.110 ± 0.009 | 0.024 ± 0.003 | 0.040 ± 0.004 |
| XGBoost baseline (validated) | Serious | 0.195 ± 0.001 | 0.601 ± 0.005 | 0.294 ± 0.001 |
| XGBoost baseline (validated) | Slight | 0.902 ± 0.001 | 0.613 ± 0.006 | 0.730 ± 0.004 |
| XGBoost (class-weighted, validated) | Fatal | 0.036 ± 0.000 | 0.636 ± 0.007 | 0.068 ± 0.001 |
| XGBoost (class-weighted, validated) | Serious | 0.185 ± 0.002 | 0.299 ± 0.005 | 0.229 ± 0.003 |
| XGBoost (class-weighted, validated) | Slight | 0.905 ± 0.000 | 0.585 ± 0.002 | 0.711 ± 0.002 |
| XGBoost (class-weighted, tuned) | Fatal | 0.037 ± 0.000 | 0.602 ± 0.007 | 0.069 ± 0.001 |
| XGBoost (class-weighted, tuned) | Serious | 0.185 ± 0.001 | 0.316 ± 0.005 | 0.233 ± 0.002 |
| XGBoost (class-weighted, tuned) | Slight | 0.905 ± 0.000 | 0.591 ± 0.002 | 0.715 ± 0.002 |

## Final untouched-test comparison

| Model | Macro F1 | Accuracy | Fatal precision | Fatal recall | Fatal F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| XGBoost baseline (validated) | 0.363 | 0.606 | 0.088 | 0.050 | 0.064 |
| XGBoost (class-weighted, validated) | 0.335 | 0.545 | 0.035 | 0.642 | 0.067 |
| XGBoost (class-weighted, tuned) | 0.339 | 0.551 | 0.037 | 0.614 | 0.069 |

## Selected configuration

Candidate 7 won by mean macro F1. Final tree count is the median early-stopped tree count across its search folds.

```json
{
  "colsample_bytree": 0.6063865008880857,
  "gamma": 0.05,
  "learning_rate": 0.07249770948732696,
  "max_depth": 6,
  "min_child_weight": 5,
  "reg_alpha": 0.5,
  "reg_lambda": 20,
  "subsample": 0.6693458614031088,
  "n_estimators": 600
}
```
