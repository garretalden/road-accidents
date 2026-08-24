# UK road-accident severity modeling report

The repository has been reorganized around a leakage-safe pre-accident feature
contract. Previous results were removed because they used `Number_of_Vehicles`,
which is only known after a collision.

Run the documented training sequence in the project README to regenerate the
cross-validation results, held-out model comparison, selected Fatal threshold,
and error-analysis figures. This report will then be replaced with the complete
modeling narrative by `make error-analysis`.
