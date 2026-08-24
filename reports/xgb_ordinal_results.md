# Ordinal cumulative XGBoost results

Two full-data, class-weighted binary XGBoost models estimate `P(Y >= Serious)` and `P(Y = Fatal)`. Each fold fits a fresh copy of the existing preprocessor. Independent cumulative probabilities are projected to the constraint `P(Fatal) <= P(Y >= Serious)` before class probabilities are derived.

## Binary five-fold validation

| Binary task | Precision | Recall | F1 | Accuracy |
| --- | ---: | ---: | ---: | ---: |
| Serious or Fatal vs Slight | 0.223 ± 0.001 | 0.613 ± 0.003 | 0.327 ± 0.001 | 0.625 ± 0.001 |
| Fatal vs Serious or Slight | 0.033 ± 0.000 | 0.684 ± 0.006 | 0.064 ± 0.001 | 0.740 ± 0.002 |

## Combined ordinal five-fold validation

| Metric | Value |
| --- | ---: |
| Macro F1 | 0.294 ± 0.001 |

| Class | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Fatal | 0.031 ± 0.000 | 0.724 ± 0.005 | 0.059 ± 0.000 |
| Serious | 0.207 ± 0.006 | 0.015 ± 0.001 | 0.028 ± 0.003 |
| Slight | 0.890 ± 0.000 | 0.718 ± 0.002 | 0.795 ± 0.001 |

Raw cumulative-order violations corrected: 260,340 of 1,203,200 OOF predictions (21.64%).

## Final untouched-test comparison

| Model | Macro F1 | Accuracy | Fatal P/R/F1 | Serious P/R/F1 | Slight P/R/F1 |
| --- | ---: | ---: | --- | --- | --- |
| XGBoost (class-weighted, tuned) | 0.339 | 0.551 | 0.037 / 0.614 / 0.069 | 0.186 / 0.315 / 0.234 | 0.905 / 0.588 / 0.713 |
| XGBoost (ordinal cumulative, class-weighted) | 0.291 | 0.618 | 0.030 / 0.727 / 0.058 | 0.202 / 0.012 / 0.023 | 0.890 / 0.713 / 0.792 |

No thresholds were tuned. Binary probabilities use the classifiers' native outputs; final class predictions use argmax after monotone projection.
