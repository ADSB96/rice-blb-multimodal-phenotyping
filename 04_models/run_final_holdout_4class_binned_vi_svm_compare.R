#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(data.table)
  library(e1071)
  library(FNN)
})

set.seed(123)
setDTthreads(0)

base_dir <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)
rebinned_root <- file.path(base_dir, "jani_stuff", "grouped_microplotsafe_rebinned_datasets")
input_root <- file.path(rebinned_root, "holdout_fulltrain_fit")
cv_model_root <- file.path(rebinned_root, "grouped5cv_binned_vi_only_svm_rebinned", "4class_binned_vi_svm_rebinned_grouped5cv_trainonly")
out_root <- file.path(rebinned_root, "final_holdout_4class_binned_vi_svm_compare")
dir.create(out_root, recursive = TRUE, showWarnings = FALSE)

script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_path <- if (length(script_arg) > 0L) {
  normalizePath(sub("^--file=", "", script_arg[1]), winslash = "/", mustWork = FALSE)
} else {
  NA_character_
}

safe_unique <- function(dt) {
  unique(dt, by = names(dt))
}

sanitize_feature_frame <- function(dt, feature_cols) {
  out <- copy(dt)
  for (col in feature_cols) {
    vals <- suppressWarnings(as.numeric(out[[col]]))
    vals[!is.finite(vals)] <- NA_real_
    out[[col]] <- vals
  }
  out
}

scale_train_test <- function(train_x, test_x) {
  center <- colMeans(train_x)
  scale <- apply(train_x, 2, sd)
  scale[is.na(scale) | scale == 0] <- 1

  train_scaled <- sweep(sweep(train_x, 2, center, "-"), 2, scale, "/")
  test_scaled <- sweep(sweep(test_x, 2, center, "-"), 2, scale, "/")

  list(train = train_scaled, test = test_scaled, center = center, scale = scale)
}

smote_generate <- function(x_cls, n_new, k = 5L) {
  n <- nrow(x_cls)
  p <- ncol(x_cls)
  if (n_new <= 0L) return(matrix(numeric(0), nrow = 0L, ncol = p))

  if (n < 2L) {
    out <- matrix(rep(x_cls[1, ], each = n_new), nrow = n_new, byrow = FALSE)
    noise <- matrix(rnorm(n_new * p, mean = 0, sd = 1e-6), nrow = n_new, ncol = p)
    return(out + noise)
  }

  k_use <- min(k, n - 1L)
  kn <- FNN::get.knn(x_cls, k = k_use)
  out <- matrix(0, nrow = n_new, ncol = p)

  for (i in seq_len(n_new)) {
    idx <- sample.int(n, 1L)
    nn_idx <- sample(kn$nn.index[idx, ], 1L)
    gap <- runif(1)
    out[i, ] <- x_cls[idx, ] + gap * (x_cls[nn_idx, ] - x_cls[idx, ])
  }

  out
}

apply_smote <- function(dt, target_col, feature_cols, seed = 123L) {
  set.seed(seed)
  dt <- copy(dt)
  counts_before <- dt[, .N, by = .(label = get(target_col))][order(label)]
  target_n <- max(counts_before$N)

  parts <- list(dt)
  classes <- sort(unique(dt[[target_col]]))

  for (cls in classes) {
    n_now <- counts_before[label == cls, N]
    n_new <- target_n - n_now
    if (length(n_now) == 0L || n_new <= 0L) next

    x_cls <- as.matrix(dt[get(target_col) == cls, ..feature_cols])
    mode(x_cls) <- "numeric"
    synth_x <- smote_generate(x_cls, n_new = n_new, k = 5L)

    synth_dt <- as.data.table(synth_x)
    setnames(synth_dt, feature_cols)
    synth_dt[, (target_col) := cls]
    parts[[length(parts) + 1L]] <- synth_dt
  }

  smote_dt <- rbindlist(parts, use.names = TRUE, fill = TRUE)
  counts_after <- smote_dt[, .N, by = .(label = get(target_col))][order(label)]

  list(data = smote_dt, counts_before = counts_before, counts_after = counts_after, target_n = target_n)
}

compute_metrics <- function(actual, predicted, levels_all) {
  cm <- table(
    factor(as.character(actual), levels = as.character(levels_all)),
    factor(as.character(predicted), levels = as.character(levels_all))
  )

  supports <- rowSums(cm)
  precision <- recall <- f1 <- numeric(length(levels_all))

  for (i in seq_along(levels_all)) {
    tp <- cm[i, i]
    fp <- sum(cm[, i]) - tp
    fn <- sum(cm[i, ]) - tp
    p <- if ((tp + fp) == 0) 0 else tp / (tp + fp)
    r <- if ((tp + fn) == 0) 0 else tp / (tp + fn)
    precision[i] <- p
    recall[i] <- r
    f1[i] <- if ((p + r) == 0) 0 else (2 * p * r) / (p + r)
  }

  list(
    confusion = cm,
    accuracy = sum(diag(cm)) / sum(cm),
    precision_weighted = sum(precision * supports) / sum(supports),
    recall_weighted = sum(recall * supports) / sum(supports),
    f1_weighted = sum(f1 * supports) / sum(supports),
    macro_f1 = mean(f1),
    per_class = data.table(
      rating = as.character(levels_all),
      support = as.integer(supports),
      precision = precision,
      recall = recall,
      f1 = f1
    )
  )
}

build_confusion_dt <- function(cm) {
  cm_dt <- as.data.table(as.matrix(cm), keep.rownames = "actual")
  setnames(cm_dt, old = names(cm_dt)[-1], new = paste0("pred_", names(cm_dt)[-1]))
  cm_dt
}

append_model_log <- function(rows) {
  log_file <- file.path(base_dir, "jani_stuff", "model_performance_log.csv")
  now_str <- format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z")
  rows[, refreshed_at := now_str]

  if (file.exists(log_file)) {
    existing <- fread(log_file)
    existing <- existing[!run_key %in% rows$run_key]
    combined <- rbindlist(list(existing, rows), use.names = TRUE, fill = TRUE)
  } else {
    combined <- rows
  }
  fwrite(combined, log_file)
}

train_file <- file.path(input_root, "fourclass_grouped_train_80_fulltrain_threshold_binned.csv")
test_file <- file.path(input_root, "fourclass_grouped_test_20_fulltrain_threshold_binned.csv")
params_file <- file.path(cv_model_root, "4class_binned_vi_svm_rebinned_grouped5cv_trainonly_selected_params.csv")
fold_metrics_file <- file.path(cv_model_root, "4class_binned_vi_svm_rebinned_grouped5cv_trainonly_fold_metrics.csv")

for (p in c(train_file, test_file, params_file, fold_metrics_file)) {
  if (!file.exists(p)) stop("Missing required input: ", p)
}

train_dt <- fread(train_file)
test_dt <- fread(test_file)
param_dt <- fread(params_file)
fold_metrics_dt <- fread(fold_metrics_file)

feature_cols <- grep("_bin[1-6]$", names(train_dt), value = TRUE)
target_levels <- sort(unique(train_dt$BB_rating))

train_dt[, unit_norm := as.character(unit_norm)]
test_dt[, unit_norm := as.character(unit_norm)]

overlap_n <- length(intersect(unique(train_dt$unit_norm), unique(test_dt$unit_norm)))
if (overlap_n != 0L) stop("Group leakage detected between grouped train and holdout test")

train_dt <- sanitize_feature_frame(train_dt, feature_cols)
test_dt <- sanitize_feature_frame(test_dt, feature_cols)
required_cols <- c("unit_norm", "timestamp_chr", "BB_rating", feature_cols)
train_dt <- train_dt[complete.cases(train_dt[, ..required_cols])]
test_dt <- test_dt[complete.cases(test_dt[, ..required_cols])]

x_train <- as.matrix(train_dt[, ..feature_cols])
x_test <- as.matrix(test_dt[, ..feature_cols])
mode(x_train) <- "numeric"
mode(x_test) <- "numeric"

scaled <- scale_train_test(x_train, x_test)
train_ml <- as.data.table(scaled$train)
setnames(train_ml, feature_cols)
train_ml[, BB_rating := train_dt$BB_rating]

sm <- apply_smote(train_ml, target_col = "BB_rating", feature_cols = feature_cols, seed = 40001L)
train_smote <- sm$data

best_fold_row <- merge(
  param_dt,
  fold_metrics_dt[, .(fold, accuracy, macro_f1)],
  by = "fold",
  all.x = TRUE,
  sort = FALSE
)[order(-accuracy, -macro_f1, fold)][1]

consensus_cost <- as.numeric(names(sort(table(param_dt$cost), decreasing = TRUE)[1]))
consensus_gamma <- as.numeric(names(sort(table(param_dt$gamma), decreasing = TRUE)[1]))

model_specs <- list(
  list(
    model_id = "4class_binned_vi_svm_holdout_bestfoldparams",
    model_name = "4-class binned VI SVM on full 80% train with best-fold CV parameters",
    cost = best_fold_row$cost,
    gamma = best_fold_row$gamma,
    param_source = sprintf("best outer fold = %d", best_fold_row$fold)
  ),
  list(
    model_id = "4class_binned_vi_svm_holdout_consensusparams",
    model_name = "4-class binned VI SVM on full 80% train with consensus CV parameters",
    cost = consensus_cost,
    gamma = consensus_gamma,
    param_source = "consensus across 5 CV folds (modal cost and modal gamma)"
  )
)

results <- list()
perf_rows <- list()

for (spec in model_specs) {
  model_dir <- file.path(out_root, spec$model_id)
  dir.create(model_dir, recursive = TRUE, showWarnings = FALSE)
  cat(sprintf("Training model: %s\n", spec$model_id))

  svm_fit <- svm(
    x = train_smote[, ..feature_cols],
    y = factor(train_smote$BB_rating, levels = target_levels),
    kernel = "radial",
    type = "C-classification",
    cost = spec$cost,
    gamma = spec$gamma,
    scale = FALSE
  )

  pred <- predict(svm_fit, newdata = scaled$test)

  pred_dt <- data.table(
    unit_norm = test_dt$unit_norm,
    timestamp_chr = test_dt$timestamp_chr,
    folder = test_dt$folder,
    source_file = test_dt$source_file,
    threshold_source = if ("threshold_source" %in% names(test_dt)) test_dt$threshold_source else NA_character_,
    actual_BLB_rating = test_dt$BB_rating,
    predicted_BLB_rating = as.integer(as.character(pred))
  )

  met <- compute_metrics(pred_dt$actual_BLB_rating, pred_dt$predicted_BLB_rating, target_levels)
  metrics_dt <- data.table(
    evaluation = "untouched_grouped_holdout_test",
    n_train_rows = nrow(train_dt),
    n_test_rows = nrow(test_dt),
    n_train_groups = uniqueN(train_dt$unit_norm),
    n_test_groups = uniqueN(test_dt$unit_norm),
    n_features = length(feature_cols),
    accuracy = met$accuracy,
    precision_weighted = met$precision_weighted,
    recall_weighted = met$recall_weighted,
    f1_score = met$f1_weighted,
    macro_f1 = met$macro_f1
  )

  input_keep <- unique(c(
    "unit_norm", "timestamp_chr", "folder", "source_file", "unit", "timestamp",
    "BB_rating", "threshold_source", feature_cols
  ))
  input_keep <- input_keep[input_keep %in% names(train_dt)]
  fwrite(train_dt[, ..input_keep], file.path(model_dir, paste0(spec$model_id, "_train_input_dataset.csv")))
  fwrite(test_dt[, ..input_keep], file.path(model_dir, paste0(spec$model_id, "_test_input_dataset.csv")))
  fwrite(data.table(feature = feature_cols), file.path(model_dir, paste0(spec$model_id, "_used_features.csv")))
  fwrite(metrics_dt, file.path(model_dir, paste0(spec$model_id, "_metrics.csv")))
  fwrite(pred_dt, file.path(model_dir, paste0(spec$model_id, "_test_predictions.csv")))
  fwrite(build_confusion_dt(met$confusion), file.path(model_dir, paste0(spec$model_id, "_confusion_matrix.csv")))
  fwrite(met$per_class, file.path(model_dir, paste0(spec$model_id, "_per_class_metrics.csv")))
  fwrite(sm$counts_before, file.path(model_dir, paste0(spec$model_id, "_smote_counts_before.csv")))
  fwrite(sm$counts_after, file.path(model_dir, paste0(spec$model_id, "_smote_counts_after.csv")))
  fwrite(data.table(
    model_id = spec$model_id,
    model_name = spec$model_name,
    algorithm = "SVM",
    severity_categories = 4L,
    feature_label = "binned VI",
    feature_count = length(feature_cols),
    split = "full 80% grouped train, tested on untouched 20% grouped holdout",
    param_source = spec$param_source,
    cost = spec$cost,
    gamma = spec$gamma,
    threshold_file = file.path("jani_stuff", "grouped_microplotsafe_rebinned_datasets", "holdout_fulltrain_fit", "full_training_only_sa_thresholds.csv"),
    smote_strategy = "SMOTE on full training set only, to majority class size",
    heldout_test_used = TRUE,
    train_test_unit_overlap = overlap_n,
    random_seed = 123L,
    smote_seed = 40001L,
    source_script = basename(script_path)
  ), file.path(model_dir, paste0(spec$model_id, "_config.csv")))

  summary_lines <- c(
    spec$model_name,
    "",
    "Training/evaluation design:",
    "- Full grouped 80% training partition used for model fitting",
    "- Untouched grouped 20% holdout test set used once for final evaluation",
    "- Only binned VI features were used",
    "- Scaling estimated from training set only and applied to holdout test set",
    "- SMOTE applied on the full training set only",
    "",
    sprintf("Hyperparameters: cost=%s, gamma=%s", spec$cost, spec$gamma),
    sprintf("Parameter source: %s", spec$param_source),
    sprintf("Train/test unit overlap: %d", overlap_n),
    "",
    sprintf("Training rows: %d", nrow(train_dt)),
    sprintf("Test rows: %d", nrow(test_dt)),
    sprintf("Training groups: %d", uniqueN(train_dt$unit_norm)),
    sprintf("Test groups: %d", uniqueN(test_dt$unit_norm)),
    sprintf("Feature count: %d", length(feature_cols)),
    "",
    "SMOTE counts before:",
    paste(capture.output(print(sm$counts_before)), collapse = "\n"),
    "",
    "SMOTE counts after:",
    paste(capture.output(print(sm$counts_after)), collapse = "\n"),
    "",
    "Holdout metrics:",
    paste(capture.output(print(metrics_dt)), collapse = "\n"),
    "",
    "Per-class metrics:",
    paste(capture.output(print(met$per_class)), collapse = "\n")
  )
  writeLines(summary_lines, file.path(model_dir, paste0(spec$model_id, "_summary.txt")))
  if (!is.na(script_path) && file.exists(script_path)) {
    file.copy(script_path, file.path(model_dir, basename(script_path)), overwrite = TRUE)
  }

  results[[length(results) + 1L]] <- data.table(
    model_id = spec$model_id,
    model_name = spec$model_name,
    parameter_strategy = spec$param_source,
    cost = spec$cost,
    gamma = spec$gamma,
    accuracy = met$accuracy,
    precision_weighted = met$precision_weighted,
    recall_weighted = met$recall_weighted,
    f1_score = met$f1_weighted,
    macro_f1 = met$macro_f1,
    output_dir = model_dir
  )

  perf_rows[[length(perf_rows) + 1L]] <- data.table(
    run_key = file.path("jani_stuff", "grouped_microplotsafe_rebinned_datasets", "final_holdout_4class_binned_vi_svm_compare", spec$model_id),
    model_name = spec$model_name,
    algorithm = "SVM",
    split = "grouped holdout 80/20 final test",
    smote_strategy = "SMOTE on full training set only",
    feature_set = "binned VI (4-class final holdout)",
    accuracy = met$accuracy,
    precision_weighted = met$precision_weighted,
    recall_weighted = met$recall_weighted,
    f1_weighted = met$f1_weighted,
    macro_f1 = met$macro_f1,
    metric_source = "metrics_file",
    metrics_file = file.path("jani_stuff", "grouped_microplotsafe_rebinned_datasets", "final_holdout_4class_binned_vi_svm_compare", spec$model_id, paste0(spec$model_id, "_metrics.csv")),
    predictions_file = file.path("jani_stuff", "grouped_microplotsafe_rebinned_datasets", "final_holdout_4class_binned_vi_svm_compare", spec$model_id, paste0(spec$model_id, "_test_predictions.csv")),
    confusion_file = file.path("jani_stuff", "grouped_microplotsafe_rebinned_datasets", "final_holdout_4class_binned_vi_svm_compare", spec$model_id, paste0(spec$model_id, "_confusion_matrix.csv"))
  )
}

comparison_dt <- rbindlist(results, use.names = TRUE, fill = TRUE)
setorder(comparison_dt, -accuracy, -macro_f1, model_id)
fwrite(comparison_dt, file.path(out_root, "final_holdout_4class_binned_vi_svm_comparison.csv"))

comparison_lines <- c(
  "Final 4-class binned VI SVM holdout comparison",
  "",
  "Both models were trained on the same rebinned 80% training partition,",
  "used the same training-only scaling, and used the same SMOTE-augmented training data.",
  "They differ only in the selected SVM hyperparameters.",
  "",
  "Best-fold parameters:",
  sprintf("- fold = %d", best_fold_row$fold),
  sprintf("- cost = %s", best_fold_row$cost),
  sprintf("- gamma = %s", best_fold_row$gamma),
  "",
  "Consensus parameters:",
  sprintf("- cost mode across folds = %s", consensus_cost),
  sprintf("- gamma mode across folds = %s", consensus_gamma),
  "",
  "Comparison table:",
  paste(capture.output(print(comparison_dt)), collapse = "\n")
)
writeLines(comparison_lines, file.path(out_root, "comparison_summary.txt"))

append_model_log(rbindlist(perf_rows, use.names = TRUE, fill = TRUE))

readme_lines <- c(
  "Final holdout 4-class binned VI SVM comparison",
  "",
  "This folder compares two final SVMs trained on the full grouped 80% training partition",
  "and evaluated once on the untouched grouped 20% holdout test set.",
  "",
  "Models:",
  "- Best-fold CV parameters",
  "- Consensus CV parameters"
)
writeLines(readme_lines, file.path(out_root, "README.txt"))

cat("Saved final holdout comparison in:\n")
cat(out_root, "\n")
