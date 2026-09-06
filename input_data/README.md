# Input Data

This folder contains the de-identified manuscript input tables used for the
microplot-aware model evaluation.

Expected files:

- `training_90.csv`
  - The grouped 90% training partition.
  - Includes all rows from training microplots only.
  - Includes `cv_fold`, the grouped 5-fold cross-validation assignment used
    within the training set.
- `holdout_test_10.csv`
  - The untouched grouped 10% holdout test partition.
  - No microplot/unit in this file appears in `training_90.csv`.
- `training_90_grouped5fold_unit_assignments.csv`
  - Unit-level summary of the grouped cross-validation fold assignments.
- `input_dataset_summary.csv`
  - Row, unit, and feature-count summary for the shared input tables.

The `BB_rating` column contains the original 6-class BLB severity label. The
`BB_rating_4class` column contains the collapsed 4-class label used for the
4-category analyses.

For grouped 5-fold cross-validation, rows with `cv_fold == k` are used as the
validation set for fold `k`; all other rows in `training_90.csv` are used as
that fold's training set.
