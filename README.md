# Rice BLB Multimodal Phenotyping Workflow

This bundle contains the scripts needed to reproduce the publication workflow for the rice bacterial blight manuscript, with explicit safeguards against microplot leakage between training and test data.

## What This Bundle Covers

The bundle is organized around four parts of the analysis pipeline:

1. Group-aware data partitioning
   - all observations from the same microplot/unit stay in the same partition.
2. Leakage-aware rebinning
   - simulated-annealing VI thresholds can be refit using training-only data.
3. Final model evaluation
   - grouped 5-fold cross-validation on the 80% training partition, followed by a single evaluation on the untouched 20% holdout test set.
4. Publication tables
   - manuscript-ready Table 2 and supplementary cluster-ratio summary tables.

## Folder Layout

- `01_legacy_processing/`
  - Historical full-dataset binning scripts retained for transparency.
  - These are not the preferred scripts for the reviewer-corrected manuscript workflow.
- `02_grouped_split_and_cv/`
  - Scripts to create the microplot-safe 80/20 holdout split, grouped 5-fold assignments, and traceability sheets.
- `03_rebinning/`
  - Training-only simulated-annealing threshold fitting and rebinned dataset construction.
- `04_models/`
  - Publication model runners.
- `05_tables/`
  - Scripts that format model outputs into manuscript-ready tables.

## Recommended Publication Workflow

Run the scripts in this order.

## Shared Input Dataset

Before running the workflow, place one de-identified combined input table in `input_data/`:

- `complete_dataset.csv`
  Add the complete pre-partition dataset here in the public input folder. This file should contain all observations together before train/test splitting, while retaining the original 6-class `BB_rating` labels and the metadata columns needed by the scripts.

The grouped split script derives the 4-class version internally by collapsing BLB ratings 1 and 3 into class 3, and ratings 7 and 9 into class 9, so separate raw train/test source tables are not required.

### 1. Create the grouped holdout split

Script:
- `02_grouped_split_and_cv/create_grouped_holdout_split.R`

Purpose:
- creates the 80% training and 20% holdout test split.
- keeps each microplot/unit in only one partition.
- preserves severity-class balance as closely as possible.

### 2. Create grouped 5-fold assignments on the 80% training partition

Script:
- `02_grouped_split_and_cv/create_grouped_5fold_cv_assignments.R`

Purpose:
- creates grouped folds for model development on the training partition only.
- no unit is shared between train and validation inside any fold.

### 3. Build traceability sheets

Script:
- `02_grouped_split_and_cv/make_training_traceability_sheets.R`

Purpose:
- exports unit/date/source-file lookup sheets so model rows can be traced back to raw point-cloud files.

### 4. Refit VI binning thresholds using training-only data

Script:
- `03_rebinning/build_grouped_microplotsafe_rebinned_datasets.R`

Purpose:
- refits simulated-annealing thresholds without using holdout-test samples.
- exports:
  - fold-specific rebinned datasets for grouped cross-validation workflows.
  - full-training-threshold rebinned datasets for final 80/20 holdout evaluation.

Note:
- use this script when you want threshold fitting itself to be confined to training data.

### 5. Run the main Table 2 models

Script:
- `04_models/run_revised_table2_rf_svm_cv_holdout_models.R`

Purpose:
- runs the final Table 2 model suite.
- supports:
  - `RandomForest`
  - `SVM`
  - `XGBoost`
- supports feature sets used in the manuscript:
  - `average VI`
  - `binned VI`
  - `cluster ratio`
  - `cluster ratio + binned VI`
  - `cluster ratio + binned VI + structural traits`
  - `cluster ratio + binned VI (XGB top30)`
  - `cluster ratio + binned VI (XGB top30) + structural`

What the script does:
- performs grouped 5-fold CV on the 80% training partition.
- tunes parameters inside each outer fold.
- applies SMOTE only to training data.
- records mean, SD, and 95% CI across folds.
- chooses consensus parameters across outer folds.
- trains a final model on the full 80% training set.
- evaluates once on the untouched 20% holdout test set.
- writes:
  - per-model fold metrics
  - per-model ordinal metrics
  - final holdout predictions
  - final parameter files
  - `table2_model_summary_detailed.csv`
  - `table2_model_final_parameters.csv`
  - `table2_model_feature_sets.csv`

### 6. Run the supplementary cluster-ratio-only SVM comparison

Script:
- `04_models/run_cluster_ratio_only_svm_k_compare_grouped_cv_holdout.R`

Purpose:
- compares cluster-ratio-only SVM models using different cluster granularities.
- current intended use is the 8C, 12C, and 16C supplementary comparison.
- writes:
  - `cluster_ratio_only_svm_cv_holdout_summary.csv`
  - `cluster_ratio_only_svm_final_parameters.csv`
  - `cluster_ratio_only_svm_feature_sets.csv`
  - `supplementary_table_cluster_ratio_svm.csv`
  - `supplementary_table_cluster_ratio_svm_manuscript_ready.csv`

### 7. Format manuscript tables

Scripts:
- `05_tables/make_table2_revised.py`

Purpose:
- formats the detailed Table 2 summary into manuscript-ready output.
- writes:
  - `table2_expanded.csv`
  - `table2_manuscript_ready.csv`

## Key Modeling Assumptions

The publication model runners are written to preserve the reviewer-requested safeguards:

- no microplot/unit overlap between train and test.
- grouped 5-fold CV on the training partition only.
- SMOTE applied only after fold assignment and only to the training portion of a fold.
- parameter tuning performed inside training data only.
- final model trained once on the full 80% training set and evaluated once on the untouched 20% holdout.
- ordinal metrics are recorded alongside standard classification metrics:
  - quadratic weighted Cohen's kappa (`QWK`)
  - mean absolute error in class units (`MAE`)

## Dataset Expectations

The model scripts can be pointed to different datasets through environment variables.

The main Table 2 runner expects:
- a 4-class training file
- a 4-class holdout test file
- a 6-class training file
- a 6-class holdout test file
- optional grouped fold-assignment files for the 4-class and 6-class training partitions

If fold assignments are not already attached to the training data, the script can:
- merge them from an assignment file, or
- generate grouped 5-fold assignments directly from the training dataset.

For full Table 2 reruns, use datasets that contain the feature columns needed by the selected models. In particular:
- `average VI` models require point-cloud average VI columns such as `pcavg_ndvi`, `pcavg_sri`, etc.
- cluster-ratio models require `cluster_*_ratio` columns.
- structural models require `structural_*` columns.

## Main Environment Variables

### Table 2 runner

- `JANI_ROOT`
  - root folder containing `jani_stuff`
- `MODEL_SPEC_FILE`
  - model specification CSV
- `MODEL_OUTPUT_ROOT`
  - output folder for Table 2 model runs
- `FOUR_TRAIN_PATH`
- `FOUR_TEST_PATH`
- `SIX_TRAIN_PATH`
- `SIX_TEST_PATH`
- `FOUR_CV_ASSIGN_PATH`
- `SIX_CV_ASSIGN_PATH`
- `MODEL_IDS`
  - optional comma-separated subset of models to run

### Cluster-ratio supplementary runner

- `REVISION_ROOT`
  - root folder containing `Binned_VI_structural_08172026` and `Train_test_master`
- `OUT_ROOT`
  - output folder for the cluster-ratio supplementary run

## Example Commands

### Table 2 models

```bash
Rscript 04_models/run_revised_table2_rf_svm_cv_holdout_models.R
python3 05_tables/make_table2_revised.py
```

### Table 2 models using explicit dataset paths

```bash
FOUR_TRAIN_PATH=/path/to/fourclass_grouped_train.csv \
FOUR_TEST_PATH=/path/to/fourclass_grouped_test.csv \
SIX_TRAIN_PATH=/path/to/sixclass_grouped_train.csv \
SIX_TEST_PATH=/path/to/sixclass_grouped_test.csv \
FOUR_CV_ASSIGN_PATH=/path/to/fourclass_grouped5fold_assignments.csv \
SIX_CV_ASSIGN_PATH=/path/to/sixclass_grouped5fold_assignments.csv \
Rscript 04_models/run_revised_table2_rf_svm_cv_holdout_models.R
```

### Cluster-ratio supplementary comparison

```bash
REVISION_ROOT=/path/to/revision \
OUT_ROOT=/path/to/cluster_ratio_only_svm_runs \
Rscript 04_models/run_cluster_ratio_only_svm_k_compare_grouped_cv_holdout.R
```

## Files Most Relevant For Submission

If you only need the scripts most directly tied to the manuscript submission, these are the core ones:

- `02_grouped_split_and_cv/create_grouped_holdout_split.R`
- `02_grouped_split_and_cv/create_grouped_5fold_cv_assignments.R`
- `03_rebinning/build_grouped_microplotsafe_rebinned_datasets.R`
- `04_models/run_revised_table2_rf_svm_cv_holdout_models.R`
- `04_models/run_cluster_ratio_only_svm_k_compare_grouped_cv_holdout.R`
- `05_tables/make_table2_revised.py`

## Legacy Files

The following scripts are retained for historical continuity, but they should not be treated as the main publication workflow unless you specifically need to reproduce older exploratory runs:

- `01_legacy_processing/bin_bb_rating_sa.R`
- `01_legacy_processing/compute_bin_proportions.R`
- `04_models/run_grouped5cv_binned_vi_only_svm_rebinned.R`
- `04_models/run_final_holdout_4class_binned_vi_svm_compare.R`
- `04_models/table2_grouped5cv_microplotsafe_suite.R`
