# UK road-accident severity modeling report

The repository has been reorganized around a leakage-safe pre-accident feature
contract. Previous results were removed because they used `Number_of_Vehicles`,
which is only known after a collision.

This file is the manually curated modeling narrative. Running
`make error-analysis` does not overwrite it. The workflow regenerates the
analysis figures and writes machine-readable metrics to `reports/results/`,
including a generated Markdown companion that can be reviewed and selectively
incorporated here.

Generated companion: [error_analysis_generated.md](results/error_analysis_generated.md)
