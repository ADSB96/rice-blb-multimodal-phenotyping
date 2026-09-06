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
cv_root <- file.path(rebinned_root, "cv_fold_specific")
out_root <- file.path(rebinned_root, "grouped5cv_binned_vi_only_svm_rebinned")
dir.create(out_root, recursive = TRUE, showWarnings = FALSE)

script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_path <- if (length(script_arg) > 0L) {
  normalizePath(sub("^--file=", "", script_arg[1]), winslash = "/", mustWork = FALSE)
} else {
  NA_character_
}

log_path <- file.path(out_root, "run.log")
log_con <- file(log_path, open = "wt")
sink(log_con, split = TRUE)
sink(log_con, type = "message")
on.exit({
  try(sink(type = "message"), silent = TRUE)
  try(sink(), silent = TRUE)
  try(close(log_con), silent = TRUE)
}, add = TRUE)

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

make_group_folds <- function(group_vec, label_vec, k = 3L, seed = 123L) {
  dt <- data.table(group = as.character(group_vec), label = as.character(label_vec))
  label_levels <- sort(unique(dt$label))

  grp_counts <- dt[, .N, by = .(group, label)]
  grp_wide <- dcast(grp_counts, group ~ label, value.var = "N", fill = 0)

  missing_cols <- setdiff(label_levels, names(grp_wide))
  for (mc in missing_cols) grp_wide[, (mc) := 0]
  setcolorder(grp_wide, c("group", label_levels))

  grp_wide[, total_obs := rowSums(.SD), .SDcols = label_levels]
  set.seed(seed)
  grp_wide <- grp_wide[sample(.N)]
  setorder(grp_wide, -total_obs)

  total_counts <- colSums(as.matrix(grp_wide[, ..label_levels]))
  target_counts <- total_counts / k

  fold_counts <- matrix(0, nrow = k, ncol = length(label_levels), dimnames = list(as.character(seq_len(k)), label_levels))
  fold_sizes <- rep(0, k)
  assigned <- integer(nrow(grp_wide))

  for (i in seq_len(nrow(grp_wide))) {
    g_counts <- as.numeric(grp_wide[i, ..label_levels])
    g_size <- grp_wide$total_obs[i]
    size_after <- fold_sizes + g_size
    candidate_folds <- which(size_after == min(size_after))

    if (length(candidate_folds) > 1L) {
      class_losses <- numeric(length(candidate_folds))
      for (j in seq_along(candidate_folds)) {
        f <- candidate_folds[j]
        new_counts <- fold_counts[f, ] + g_counts
        class_losses[j] <- sum(((new_counts - target_counts)^2) / pmax(1, target_counts))
      }
      best_fold <- candidate_folds[which.min(class_losses)]
    } else {
      best_fold <- candidate_folds
    }

    assigned[i] <- best_fold
    fold_counts[best_fold, ] <- fold_counts[best_fold, ] + g_counts
    fold_sizes[best_fold] <- fold_sizes[best_fold] + g_size
  }

  grp_wide[, fold := assigned]
  grp_wide[, .(group, fold, total_obs)]
}

scale_train_test <- function(train_x, test_x) {
  center <- colMeans(train_x)
  scale <- apply(train_x, 2, sd)
  scale[is.na(scale) | scale == 0] <- 1

  train_scaled <- sweep(sweep(train_x, 2, center, "-"), 2, scale, "/")
  test_scaled <- sweep(sweep(test_x, 2, center, "-"), 2, scale, "/")

  list(train = train_scaled, test = test_scaled)
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

choose_best_svm_params <- function(train_dt, feature_cols, target_levels, outer_fold_id) {
  train_dt <- copy(train_dt)
  train_dt[, unit_norm := as.character(unit_norm)]
  inner_groups <- make_group_folds(train_dt$unit_norm, train_dt$BB_rating, k = 3L, seed = 1000L + outer_fold_id)
  inner_groups[, group := as.character(group)]
  setnames(inner_groups, "fold", "inner_fold")
  train_inner <- merge(train_dt, inner_groups, by.x = "unit_norm", by.y = "group", all.x = TRUE, sort = FALSE)

  grid <- CJ(cost = c(1, 4, 8, 16), gamma = c(0.001, 0.005, 0.01, 0.02))
  results <- vector("list", nrow(grid))

  for (g in seq_len(nrow(grid))) {
    cost_val <- grid$cost[g]
    gamma_val <- grid$gamma[g]
    inner_pred_parts <- list()

    for (inner_fold_id in sort(unique(train_inner$inner_fold))) {
      inner_train <- train_inner[inner_fold != inner_fold_id]
      inner_valid <- train_inner[inner_fold == inner_fold_id]
      if (nrow(inner_train) == 0L || nrow(inner_valid) == 0L) next
      if (uniqueN(inner_train$BB_rating) < 2L) next

      x_train <- as.matrix(inner_train[, ..feature_cols])
      x_valid <- as.matrix(inner_valid[, ..feature_cols])
      mode(x_train) <- "numeric"
      mode(x_valid) <- "numeric"

      scaled <- scale_train_test(x_train, x_valid)
      train_ml <- as.data.table(scaled$train)
      setnames(train_ml, feature_cols)
      train_ml[, BB_rating := inner_train$BB_rating]

      sm <- apply_smote(
        train_ml,
        target_col = "BB_rating",
        feature_cols = feature_cols,
        seed = 10000L + outer_fold_id * 100L + inner_fold_id * 10L + g
      )
      if (uniqueN(sm$data$BB_rating) < 2L) next

      svm_fit <- tryCatch(
        svm(
          x = sm$data[, ..feature_cols],
          y = factor(sm$data$BB_rating, levels = target_levels),
          kernel = "radial",
          type = "C-classification",
          cost = cost_val,
          gamma = gamma_val,
          scale = FALSE
        ),
        error = function(e) NULL
      )
      if (is.null(svm_fit)) next

      pred <- tryCatch(
        predict(svm_fit, newdata = scaled$test),
        error = function(e) NULL
      )
      if (is.null(pred)) next

      inner_pred_parts[[length(inner_pred_parts) + 1L]] <- data.table(
        actual = inner_valid$BB_rating,
        predicted = as.character(pred)
      )
    }

    if (length(inner_pred_parts) == 0L) {
      results[[g]] <- data.table(
        cost = cost_val,
        gamma = gamma_val,
        macro_f1 = -Inf,
        accuracy = -Inf,
        f1_weighted = -Inf,
        successful_inner_folds = 0L
      )
      next
    }

    pooled <- rbindlist(inner_pred_parts, use.names = TRUE, fill = TRUE)
    met <- compute_metrics(pooled$actual, pooled$predicted, target_levels)
    results[[g]] <- data.table(
      cost = cost_val,
      gamma = gamma_val,
      macro_f1 = met$macro_f1,
      accuracy = met$accuracy,
      f1_weighted = met$f1_weighted,
      successful_inner_folds = length(inner_pred_parts)
    )
  }

  res_dt <- rbindlist(results)
  if (all(!is.finite(res_dt$macro_f1))) {
    return(data.table(
      cost = 1,
      gamma = 0.005,
      macro_f1 = NA_real_,
      accuracy = NA_real_,
      f1_weighted = NA_real_,
      successful_inner_folds = 0L
    ))
  }

  setorder(res_dt, -macro_f1, -accuracy, -f1_weighted, cost, gamma)
  res_dt[1]
}

load_fold_bundle <- function(dataset_tag) {
  dataset_dir <- file.path(cv_root, dataset_tag)
  if (!dir.exists(dataset_dir)) stop("Missing dataset directory: ", dataset_dir)

  fold_ids <- sprintf("%02d", 1:5)
  out <- list()
  for (fid in fold_ids) {
    fold_dir <- file.path(dataset_dir, paste0("fold_", fid))
    train_file <- file.path(fold_dir, sprintf("%s_fold_%s_train_binned.csv", dataset_tag, fid))
    valid_file <- file.path(fold_dir, sprintf("%s_fold_%s_validation_binned.csv", dataset_tag, fid))
    if (!file.exists(train_file) || !file.exists(valid_file)) {
      stop("Missing fold bundle for ", dataset_tag, " fold ", fid)
    }
    out[[as.character(as.integer(fid))]] <- list(
      train_dt = fread(train_file),
      test_dt = fread(valid_file),
      train_file = train_file,
      test_file = valid_file
    )
  }
  out
}

run_svm_from_folds <- function(fold_bundle, cfg) {
  model_dir <- file.path(out_root, cfg$model_id)
  dir.create(model_dir, recursive = TRUE, showWarnings = FALSE)
  cat(sprintf("Starting model: %s\n", cfg$model_id))

  feature_cols <- cfg$feature_cols
  target_levels <- sort(unique(unlist(lapply(fold_bundle, function(x) x$test_dt$BB_rating))))
  pred_parts <- list()
  fold_metrics <- list()
  param_rows <- list()
  overlap_checks <- character()
  smote_summaries <- character()
  oof_input_rows <- list()

  for (outer_fold in sort(as.integer(names(fold_bundle)))) {
    cat(sprintf("  %s outer fold %d/5\n", cfg$model_id, outer_fold))
    train_fold <- sanitize_feature_frame(copy(fold_bundle[[as.character(outer_fold)]]$train_dt), feature_cols)
    test_fold <- sanitize_feature_frame(copy(fold_bundle[[as.character(outer_fold)]]$test_dt), feature_cols)
    train_fold[, unit_norm := as.character(unit_norm)]
    test_fold[, unit_norm := as.character(unit_norm)]

    required_cols <- c("unit_norm", "timestamp_chr", "BB_rating", feature_cols)
    train_fold <- train_fold[complete.cases(train_fold[, ..required_cols])]
    test_fold <- test_fold[complete.cases(test_fold[, ..required_cols])]

    overlap_n <- length(intersect(unique(train_fold$unit_norm), unique(test_fold$unit_norm)))
    overlap_checks <- c(overlap_checks, sprintf("Fold %d unit overlap: %d", outer_fold, overlap_n))
    if (overlap_n != 0L) stop("Group leakage detected in fold ", outer_fold, " for ", cfg$model_id)

    best_params <- choose_best_svm_params(train_fold, feature_cols, target_levels, outer_fold)
    cat(sprintf("    selected params cost=%s gamma=%s\n", best_params$cost, best_params$gamma))
    param_rows[[length(param_rows) + 1L]] <- data.table(
      fold = outer_fold,
      cost = best_params$cost,
      gamma = best_params$gamma,
      inner_macro_f1 = best_params$macro_f1,
      inner_accuracy = best_params$accuracy
    )

    x_train <- as.matrix(train_fold[, ..feature_cols])
    x_test <- as.matrix(test_fold[, ..feature_cols])
    mode(x_train) <- "numeric"
    mode(x_test) <- "numeric"

    scaled <- scale_train_test(x_train, x_test)
    train_ml <- as.data.table(scaled$train)
    setnames(train_ml, feature_cols)
    train_ml[, BB_rating := train_fold$BB_rating]

    sm <- apply_smote(train_ml, target_col = "BB_rating", feature_cols = feature_cols, seed = 20000L + outer_fold)
    smote_summaries <- c(
      smote_summaries,
      sprintf(
        "Fold %d SMOTE target per class: %d | before: %s | after: %s",
        outer_fold,
        sm$target_n,
        paste(sprintf("%s=%s", sm$counts_before$label, sm$counts_before$N), collapse = ", "),
        paste(sprintf("%s=%s", sm$counts_after$label, sm$counts_after$N), collapse = ", ")
      )
    )

    svm_fit <- tryCatch(
      svm(
        x = sm$data[, ..feature_cols],
        y = factor(sm$data$BB_rating, levels = target_levels),
        kernel = "radial",
        type = "C-classification",
        cost = best_params$cost,
        gamma = best_params$gamma,
        scale = FALSE
      ),
      error = function(e) NULL
    )
    if (is.null(svm_fit)) {
      svm_fit <- svm(
        x = sm$data[, ..feature_cols],
        y = factor(sm$data$BB_rating, levels = target_levels),
        kernel = "radial",
        type = "C-classification",
        cost = 1,
        gamma = 0.005,
        scale = FALSE
      )
    }

    pred <- tryCatch(
      predict(svm_fit, newdata = scaled$test),
      error = function(e) NULL
    )
    if (is.null(pred)) {
      pred <- predict(svm_fit, newdata = as.data.frame(scaled$test))
    }

    fold_pred <- data.table(
      fold = outer_fold,
      unit_norm = test_fold$unit_norm,
      timestamp_chr = test_fold$timestamp_chr,
      folder = test_fold$folder,
      source_file = test_fold$source_file,
      threshold_source = if ("threshold_source" %in% names(test_fold)) test_fold$threshold_source else NA_character_,
      actual_BLB_rating = test_fold$BB_rating,
      predicted_BLB_rating = as.integer(as.character(pred))
    )
    pred_parts[[length(pred_parts) + 1L]] <- fold_pred
    oof_input_rows[[length(oof_input_rows) + 1L]] <- copy(test_fold)

    fold_met <- compute_metrics(fold_pred$actual_BLB_rating, fold_pred$predicted_BLB_rating, target_levels)
    fold_metrics[[length(fold_metrics) + 1L]] <- data.table(
      fold = outer_fold,
      n_train = nrow(train_fold),
      n_test = nrow(test_fold),
      n_train_groups = uniqueN(train_fold$unit_norm),
      n_test_groups = uniqueN(test_fold$unit_norm),
      smote_target_per_class = sm$target_n,
      accuracy = fold_met$accuracy,
      precision_weighted = fold_met$precision_weighted,
      recall_weighted = fold_met$recall_weighted,
      f1_weighted = fold_met$f1_weighted,
      macro_f1 = fold_met$macro_f1
    )
  }

  pred_dt <- rbindlist(pred_parts, use.names = TRUE, fill = TRUE)
  fold_metrics_dt <- rbindlist(fold_metrics, use.names = TRUE, fill = TRUE)
  params_dt <- rbindlist(param_rows, use.names = TRUE, fill = TRUE)
  oof_input_dt <- safe_unique(rbindlist(oof_input_rows, use.names = TRUE, fill = TRUE))
  overall <- compute_metrics(pred_dt$actual_BLB_rating, pred_dt$predicted_BLB_rating, target_levels)

  overall_dt <- data.table(
    evaluation = "grouped_5fold_cv_oof",
    n_rows = nrow(oof_input_dt),
    n_groups = uniqueN(oof_input_dt$unit_norm),
    n_features = length(feature_cols),
    accuracy = overall$accuracy,
    precision_weighted = overall$precision_weighted,
    recall_weighted = overall$recall_weighted,
    f1_score = overall$f1_weighted,
    macro_f1 = overall$macro_f1,
    fold_accuracy_mean = mean(fold_metrics_dt$accuracy),
    fold_accuracy_sd = sd(fold_metrics_dt$accuracy),
    fold_macro_f1_mean = mean(fold_metrics_dt$macro_f1),
    fold_macro_f1_sd = sd(fold_metrics_dt$macro_f1)
  )

  used_features_dt <- data.table(feature = feature_cols)
  input_keep <- unique(c(
    "unit_norm", "timestamp_chr", "folder", "source_file", "unit", "timestamp",
    "BB_rating", "cv_fold", "threshold_source", feature_cols
  ))
  input_keep <- input_keep[input_keep %in% names(oof_input_dt)]
  model_input_dt <- oof_input_dt[, ..input_keep]

  fwrite(model_input_dt, file.path(model_dir, paste0(cfg$model_id, "_input_dataset.csv")))
  fwrite(used_features_dt, file.path(model_dir, paste0(cfg$model_id, "_used_features.csv")))
  fwrite(overall_dt, file.path(model_dir, paste0(cfg$model_id, "_metrics.csv")))
  fwrite(fold_metrics_dt, file.path(model_dir, paste0(cfg$model_id, "_fold_metrics.csv")))
  fwrite(pred_dt, file.path(model_dir, paste0(cfg$model_id, "_oof_predictions.csv")))
  fwrite(params_dt, file.path(model_dir, paste0(cfg$model_id, "_selected_params.csv")))
  fwrite(build_confusion_dt(overall$confusion), file.path(model_dir, paste0(cfg$model_id, "_confusion_matrix.csv")))
  fwrite(data.table(
    model_id = cfg$model_id,
    model_name = cfg$model_name,
    algorithm = "SVM",
    severity_categories = cfg$severity_categories,
    feature_label = cfg$feature_label,
    feature_count = length(feature_cols),
    split = "grouped 5-fold CV on 90% training partition only",
    smote_strategy = "SMOTE within training folds only",
    heldout_test_used = FALSE,
    outer_fold_source = "precomputed microplot-safe fold-specific rebinned datasets",
    threshold_strategy = "fold-specific thresholds fit outside each held-out fold",
    random_seed = 123L,
    source_script = basename(script_path)
  ), file.path(model_dir, paste0(cfg$model_id, "_config.csv")))

  summary_lines <- c(
    cfg$model_name,
    "",
    "Evaluation: grouped 5-fold cross-validation by unit_norm on the 90% training partition only",
    "The separate grouped holdout test set was not used.",
    "Only binned VI features were used.",
    "SMOTE was applied only within each training fold.",
    "Thresholds were fit outside each held-out fold and applied fold-safely.",
    "",
    sprintf("Rows analyzed: %d", nrow(oof_input_dt)),
    sprintf("Unique microplots/groups: %d", uniqueN(oof_input_dt$unit_norm)),
    sprintf("Feature count: %d", length(feature_cols)),
    "",
    "Leakage checks:",
    overlap_checks,
    "",
    "Fold-safe SMOTE summaries:",
    smote_summaries,
    "",
    "Selected SVM parameters by outer fold:",
    paste(capture.output(print(params_dt)), collapse = "\n"),
    "",
    "Fold metrics:",
    paste(capture.output(print(fold_metrics_dt)), collapse = "\n"),
    "",
    "Overall pooled out-of-fold metrics:",
    paste(capture.output(print(overall_dt)), collapse = "\n"),
    "",
    "Per-class metrics:",
    paste(capture.output(print(overall$per_class)), collapse = "\n")
  )
  writeLines(summary_lines, file.path(model_dir, paste0(cfg$model_id, "_summary.txt")))
  if (!is.na(script_path) && file.exists(script_path)) {
    file.copy(script_path, file.path(model_dir, basename(script_path)), overwrite = TRUE)
  }
  cat(sprintf("Completed model: %s\n", cfg$model_id))

  data.table(
    model_id = cfg$model_id,
    model_name = cfg$model_name,
    algorithm = "SVM",
    severity_categories = cfg$severity_categories,
    feature_label = cfg$feature_label,
    output_dir = model_dir,
    accuracy = overall$accuracy,
    precision_weighted = overall$precision_weighted,
    recall_weighted = overall$recall_weighted,
    f1_score = overall$f1_weighted,
    macro_f1 = overall$macro_f1
  )
}

append_model_log <- function(result_dt) {
  log_file <- file.path(base_dir, "jani_stuff", "model_performance_log.csv")
  now_str <- format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z")
  rows <- result_dt[, .(
    run_key = file.path("jani_stuff", "grouped_microplotsafe_rebinned_datasets", "grouped5cv_binned_vi_only_svm_rebinned", model_id),
    model_name = model_name,
    algorithm = algorithm,
    split = "grouped5cv on 90% training only",
    smote_strategy = "SMOTE within training folds only",
    feature_set = paste0(feature_label, " (", ifelse(severity_categories == 4L, "4-class", "6-class"), ")"),
    accuracy = accuracy,
    precision_weighted = precision_weighted,
    recall_weighted = recall_weighted,
    f1_weighted = f1_score,
    macro_f1 = macro_f1,
    metric_source = "metrics_file",
    metrics_file = file.path("jani_stuff", "grouped_microplotsafe_rebinned_datasets", "grouped5cv_binned_vi_only_svm_rebinned", model_id, paste0(model_id, "_metrics.csv")),
    predictions_file = file.path("jani_stuff", "grouped_microplotsafe_rebinned_datasets", "grouped5cv_binned_vi_only_svm_rebinned", model_id, paste0(model_id, "_oof_predictions.csv")),
    confusion_file = file.path("jani_stuff", "grouped_microplotsafe_rebinned_datasets", "grouped5cv_binned_vi_only_svm_rebinned", model_id, paste0(model_id, "_confusion_matrix.csv")),
    refreshed_at = now_str
  )]

  if (file.exists(log_file)) {
    existing <- fread(log_file)
    existing <- existing[!run_key %in% rows$run_key]
    combined <- rbindlist(list(existing, rows), use.names = TRUE, fill = TRUE)
  } else {
    combined <- rows
  }
  fwrite(combined, log_file)
}

six_master <- fread(file.path(cv_root, "sixclass", "sixclass_grouped_train_90_cv_validation_master_binned.csv"), nrows = 0)
feature_cols <- grep("_bin[1-6]$", names(six_master), value = TRUE)

cfgs <- list(
  list(
    dataset_tag = "fourclass",
    model_id = "4class_binned_vi_svm_rebinned_grouped5cv_trainonly",
    model_name = "4-class binned VI SVM on rebinned microplot-safe grouped 5-fold CV training partition",
    severity_categories = 4L,
    feature_label = "binned VI"
  ),
  list(
    dataset_tag = "sixclass",
    model_id = "6class_binned_vi_svm_rebinned_grouped5cv_trainonly",
    model_name = "6-class binned VI SVM on rebinned microplot-safe grouped 5-fold CV training partition",
    severity_categories = 6L,
    feature_label = "binned VI"
  )
)

results <- list()
for (cfg in cfgs) {
  cfg$feature_cols <- feature_cols
  fold_bundle <- load_fold_bundle(cfg$dataset_tag)
  results[[length(results) + 1L]] <- run_svm_from_folds(fold_bundle, cfg)
}

result_dt <- rbindlist(results, use.names = TRUE, fill = TRUE)
fwrite(result_dt, file.path(out_root, "grouped5cv_binned_vi_only_svm_rebinned_results_summary.csv"))
append_model_log(result_dt)

readme_lines <- c(
  "Grouped 5-fold SVM reruns on leakage-safe rebinned datasets",
  "",
  "Scope:",
  "- 4-class binned VI only SVM",
  "- 6-class binned VI only SVM",
  "",
  "Important:",
  "- Only the grouped 90% training partition was used.",
  "- The separate grouped holdout test set was not used.",
  "- SMOTE was applied only within the outer training folds.",
  "- Outer fold thresholds were the new leakage-safe fold-specific thresholds."
)
writeLines(readme_lines, file.path(out_root, "README.txt"))

cat("Saved SVM reruns in:\n")
cat(out_root, "\n")
