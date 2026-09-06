#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(data.table)
  library(e1071)
  library(FNN)
})

set.seed(123)
setDTthreads(0)

script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_path <- if (length(script_arg) > 0L) {
  normalizePath(sub("^--file=", "", script_arg[1]), winslash = "/", mustWork = FALSE)
} else {
  NA_character_
}
script_dir <- if (is.na(script_path)) normalizePath(getwd(), winslash = "/", mustWork = TRUE) else dirname(script_path)
bundle_root <- normalizePath(file.path(script_dir, ".."), winslash = "/", mustWork = FALSE)
project_root <- normalizePath(file.path(bundle_root, ".."), winslash = "/", mustWork = FALSE)

revision_root <- Sys.getenv(
  "REVISION_ROOT",
  unset = file.path(project_root, "revision")
)
out_root <- Sys.getenv(
  "OUT_ROOT",
  unset = file.path(project_root, "cluster_ratio_only_svm_grouped_cv_holdout_outputs")
)

data_root <- file.path(revision_root, "Binned_VI_structural_08172026")
master_root <- file.path(revision_root, "Train_test_master")
dir.create(out_root, recursive = TRUE, showWarnings = FALSE)

log_file <- file.path(out_root, "run.log")
log_con <- file(log_file, open = "wt")
sink(log_con, split = TRUE)
sink(log_con, type = "message")
on.exit({
  try(sink(type = "message"), silent = TRUE)
  try(sink(), silent = TRUE)
  try(close(log_con), silent = TRUE)
}, add = TRUE)

log_line <- function(...) cat(sprintf(...))

safe_unique <- function(dt) unique(dt, by = names(dt))

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

summarize_vector <- function(x) {
  x <- as.numeric(x)
  x <- x[is.finite(x)]
  n <- length(x)
  if (n == 0L) return(list(mean = NA_real_, sd = NA_real_, ci_low = NA_real_, ci_high = NA_real_, n = 0L))
  mu <- mean(x)
  s <- if (n > 1L) sd(x) else 0
  if (n > 1L) {
    se <- s / sqrt(n)
    tcrit <- qt(0.975, df = n - 1L)
    ci_low <- mu - tcrit * se
    ci_high <- mu + tcrit * se
  } else {
    ci_low <- mu
    ci_high <- mu
  }
  list(mean = mu, sd = s, ci_low = ci_low, ci_high = ci_high, n = n)
}

summarize_cv_metrics <- function(fold_metrics_dt) {
  metric_cols <- c("accuracy", "precision_weighted", "recall_weighted", "f1_weighted", "macro_f1")
  rows <- lapply(metric_cols, function(col) {
    s <- summarize_vector(fold_metrics_dt[[col]])
    data.table(metric = col, mean = s$mean, sd = s$sd, ci_low = s$ci_low, ci_high = s$ci_high, n = s$n)
  })
  rbindlist(rows)
}

mode_numeric <- function(x) {
  x <- x[is.finite(x)]
  if (!length(x)) return(NA_real_)
  ux <- sort(unique(x))
  counts <- sapply(ux, function(v) sum(x == v))
  ux[which.max(counts)]
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

      sm <- apply_smote(train_ml, target_col = "BB_rating", feature_cols = feature_cols, seed = 10000L + outer_fold_id * 100L + inner_fold_id * 10L + g)
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

      pred <- tryCatch(predict(svm_fit, newdata = scaled$test), error = function(e) NULL)
      if (is.null(pred)) next
      inner_pred_parts[[length(inner_pred_parts) + 1L]] <- data.table(actual = inner_valid$BB_rating, predicted = as.character(pred))
    }

    if (!length(inner_pred_parts)) {
      results[[g]] <- data.table(cost = cost_val, gamma = gamma_val, macro_f1 = -Inf, accuracy = -Inf, f1_weighted = -Inf, successful_inner_folds = 0L)
    } else {
      inner_pred_dt <- rbindlist(inner_pred_parts)
      met <- compute_metrics(inner_pred_dt$actual, inner_pred_dt$predicted, target_levels)
      results[[g]] <- data.table(cost = cost_val, gamma = gamma_val, macro_f1 = met$macro_f1, accuracy = met$accuracy, f1_weighted = met$f1_weighted, successful_inner_folds = length(inner_pred_parts))
    }
  }

  res_dt <- rbindlist(results)
  if (all(!is.finite(res_dt$macro_f1))) {
    return(data.table(cost = 1, gamma = 0.005, macro_f1 = NA_real_, accuracy = NA_real_, f1_weighted = NA_real_, successful_inner_folds = 0L))
  }
  setorder(res_dt, -macro_f1, -accuracy, -f1_weighted, cost, gamma)
  res_dt[1]
}

run_outer_fold <- function(cv_dt, fold_id, feature_cols, target_levels) {
  outer_train <- cv_dt[cv_fold != fold_id]
  outer_valid <- cv_dt[cv_fold == fold_id]

  best_param <- choose_best_svm_params(outer_train, feature_cols, target_levels, fold_id)

  x_train <- as.matrix(outer_train[, ..feature_cols])
  x_valid <- as.matrix(outer_valid[, ..feature_cols])
  mode(x_train) <- "numeric"
  mode(x_valid) <- "numeric"

  scaled <- scale_train_test(x_train, x_valid)
  train_ml <- as.data.table(scaled$train)
  setnames(train_ml, feature_cols)
  train_ml[, BB_rating := outer_train$BB_rating]

  sm <- apply_smote(train_ml, target_col = "BB_rating", feature_cols = feature_cols, seed = 50000L + fold_id)
  smote_before <- copy(sm$counts_before)
  smote_after <- copy(sm$counts_after)

  fit <- svm(
    x = sm$data[, ..feature_cols],
    y = factor(sm$data$BB_rating, levels = target_levels),
    kernel = "radial",
    type = "C-classification",
    cost = best_param$cost[[1]],
    gamma = best_param$gamma[[1]],
    scale = FALSE
  )
  pred <- as.character(predict(fit, newdata = scaled$test))
  met <- compute_metrics(outer_valid$BB_rating, pred, target_levels)

  list(
    fold_metrics = data.table(
      fold = fold_id,
      n_train_rows = nrow(outer_train),
      n_valid_rows = nrow(outer_valid),
      n_train_groups = uniqueN(outer_train$unit_norm),
      n_valid_groups = uniqueN(outer_valid$unit_norm),
      n_features = length(feature_cols),
      accuracy = met$accuracy,
      precision_weighted = met$precision_weighted,
      recall_weighted = met$recall_weighted,
      f1_weighted = met$f1_weighted,
      macro_f1 = met$macro_f1
    ),
    predictions = data.table(
      fold = fold_id,
      unit_norm = outer_valid$unit_norm,
      source_file = outer_valid$source_file,
      actual = outer_valid$BB_rating,
      predicted = pred
    ),
    confusion = cbind(data.table(fold = fold_id), build_confusion_dt(met$confusion)),
    per_class = cbind(data.table(fold = fold_id), met$per_class),
    params = data.table(fold = fold_id, cost = best_param$cost[[1]], gamma = best_param$gamma[[1]], feature_count = length(feature_cols), successful_inner_folds = best_param$successful_inner_folds[[1]]),
    selected_features = data.table(fold = fold_id, rank = seq_along(feature_cols), feature = feature_cols),
    smote_before = copy(smote_before)[, fold := fold_id],
    smote_after = copy(smote_after)[, fold := fold_id]
  )
}

run_final_holdout <- function(train_dt, test_dt, feature_cols, target_levels, outer_param_dt) {
  cost_val <- mode_numeric(outer_param_dt$cost)
  gamma_val <- mode_numeric(outer_param_dt$gamma)

  x_train <- as.matrix(train_dt[, ..feature_cols])
  x_test <- as.matrix(test_dt[, ..feature_cols])
  mode(x_train) <- "numeric"
  mode(x_test) <- "numeric"

  scaled <- scale_train_test(x_train, x_test)
  train_ml <- as.data.table(scaled$train)
  setnames(train_ml, feature_cols)
  train_ml[, BB_rating := train_dt$BB_rating]

  sm <- apply_smote(train_ml, target_col = "BB_rating", feature_cols = feature_cols, seed = 80001L)
  smote_before <- sm$counts_before
  smote_after <- sm$counts_after

  fit <- svm(
    x = sm$data[, ..feature_cols],
    y = factor(sm$data$BB_rating, levels = target_levels),
    kernel = "radial",
    type = "C-classification",
    cost = cost_val,
    gamma = gamma_val,
    scale = FALSE
  )
  pred <- as.character(predict(fit, newdata = scaled$test))
  met <- compute_metrics(test_dt$BB_rating, pred, target_levels)

  list(
    metrics = data.table(
      evaluation = "holdout_test",
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
    ),
    predictions = data.table(
      unit_norm = test_dt$unit_norm,
      unit = test_dt$unit,
      timestamp = if ("timestamp" %in% names(test_dt)) test_dt$timestamp else NA,
      source_file = test_dt$source_file,
      actual = test_dt$BB_rating,
      predicted = pred
    ),
    confusion = build_confusion_dt(met$confusion),
    per_class = met$per_class,
    params = data.table(
      algorithm = "SVM",
      parameter_strategy = "consensus across 5 outer folds",
      cost = cost_val,
      gamma = gamma_val,
      feature_count = length(feature_cols)
    ),
    used_features = data.table(rank = seq_along(feature_cols), feature = feature_cols),
    smote_before = smote_before,
    smote_after = smote_after
  )
}

collect_summary_row <- function(model_id, model_name, severity_categories, cluster_setting, cv_summary_dt, holdout_metrics_dt, final_param_dt, feature_count) {
  out <- data.table(
    model_id = model_id,
    model_name = model_name,
    algorithm = "SVM",
    severity_categories = severity_categories,
    cluster_setting = cluster_setting,
    feature_label = sprintf("cluster ratio (%s)", cluster_setting),
    cv_folds = cv_summary_dt[metric == "accuracy", n],
    final_feature_count = feature_count,
    holdout_accuracy = holdout_metrics_dt$accuracy[[1]],
    holdout_precision_weighted = holdout_metrics_dt$precision_weighted[[1]],
    holdout_recall_weighted = holdout_metrics_dt$recall_weighted[[1]],
    holdout_f1_weighted = holdout_metrics_dt$f1_score[[1]],
    holdout_macro_f1 = holdout_metrics_dt$macro_f1[[1]],
    final_cost = final_param_dt$cost[[1]],
    final_gamma = final_param_dt$gamma[[1]]
  )
  for (metric_name in cv_summary_dt$metric) {
    row <- cv_summary_dt[metric == metric_name]
    prefix <- paste0("cv_", metric_name)
    out[[paste0(prefix, "_mean")]] <- row$mean[[1]]
    out[[paste0(prefix, "_sd")]] <- row$sd[[1]]
    out[[paste0(prefix, "_ci_low")]] <- row$ci_low[[1]]
    out[[paste0(prefix, "_ci_high")]] <- row$ci_high[[1]]
  }
  out
}

round2 <- function(x) round(as.numeric(x), 2)

build_supplementary_row <- function(summary_row) {
  data.table(
    `Model features` = sprintf("cluster ratio (%s)", summary_row$cluster_setting[[1]]),
    `Number of severity categories` = summary_row$severity_categories[[1]],
    `Algorithm` = "SVM",
    `Accuracy` = round2(summary_row$holdout_accuracy[[1]]),
    `Precision` = round2(summary_row$holdout_precision_weighted[[1]]),
    `Recall` = round2(summary_row$holdout_recall_weighted[[1]]),
    `F1` = round2(summary_row$holdout_f1_weighted[[1]])
  )
}

model_specs <- rbindlist(list(
  data.table(model_id = "fourclass_cluster_ratio_8c_svm_grouped5cv_holdout", dataset_tag = "fourclass", severity_categories = 4L, cluster_setting = "8C"),
  data.table(model_id = "fourclass_cluster_ratio_12c_svm_grouped5cv_holdout", dataset_tag = "fourclass", severity_categories = 4L, cluster_setting = "12C"),
  data.table(model_id = "fourclass_cluster_ratio_16c_svm_grouped5cv_holdout", dataset_tag = "fourclass", severity_categories = 4L, cluster_setting = "16C"),
  data.table(model_id = "sixclass_cluster_ratio_8c_svm_grouped5cv_holdout", dataset_tag = "sixclass", severity_categories = 6L, cluster_setting = "8C"),
  data.table(model_id = "sixclass_cluster_ratio_12c_svm_grouped5cv_holdout", dataset_tag = "sixclass", severity_categories = 6L, cluster_setting = "12C"),
  data.table(model_id = "sixclass_cluster_ratio_16c_svm_grouped5cv_holdout", dataset_tag = "sixclass", severity_categories = 6L, cluster_setting = "16C")
), use.names = TRUE)
model_specs[, model_name := sprintf("%s-class cluster-ratio-only SVM (%s) with grouped 5-fold CV and final holdout", severity_categories, cluster_setting)]

summary_rows <- list()
param_rows <- list()
feature_rows <- list()
supp_rows <- list()

for (i in seq_len(nrow(model_specs))) {
  spec <- model_specs[i]
  dataset_tag <- spec$dataset_tag[[1]]
  cluster_setting <- spec$cluster_setting[[1]]
  model_id <- spec$model_id[[1]]

  trace_file <- file.path(master_root, sprintf("%s_grouped_train_90_traceability_master.csv", dataset_tag))
  train_file <- file.path(data_root, cluster_setting, sprintf("%s_grouped_train_90_fulltrain_threshold_binned_pcavg_%s.csv", dataset_tag, cluster_setting))
  test_file <- file.path(data_root, cluster_setting, sprintf("%s_grouped_test_10_fulltrain_threshold_binned_pcavg_%s.csv", dataset_tag, cluster_setting))

  log_line("\nRunning %s\n", model_id)
  log_line("  train: %s\n", train_file)
  log_line("  test:  %s\n", test_file)

  trace_dt <- fread(trace_file)[, .(source_file, cv_fold)]
  trace_dt <- safe_unique(trace_dt)
  train_dt <- fread(train_file)
  test_dt <- fread(test_file)

  train_dt <- merge(train_dt, trace_dt, by = "source_file", all.x = TRUE, sort = FALSE)
  if (train_dt[, any(is.na(cv_fold))]) stop("Missing cv_fold after merge for model ", model_id)

  overlap_units <- intersect(unique(as.character(train_dt$unit_norm)), unique(as.character(test_dt$unit_norm)))
  if (length(overlap_units) != 0L) stop("Unit overlap detected between train and holdout for model ", model_id)

  feature_cols <- grep("^cluster_[0-9]+_ratio$", names(train_dt), value = TRUE)
  feature_cols <- feature_cols[order(as.integer(sub("^cluster_([0-9]+)_ratio$", "\\1", feature_cols)))]
  if (!length(feature_cols)) stop("No cluster ratio features found for model ", model_id)

  train_dt[, unit_norm := as.character(unit_norm)]
  test_dt[, unit_norm := as.character(unit_norm)]
  train_dt[, BB_rating := as.integer(BB_rating)]
  test_dt[, BB_rating := as.integer(BB_rating)]

  train_dt <- sanitize_feature_frame(train_dt, feature_cols)
  test_dt <- sanitize_feature_frame(test_dt, feature_cols)

  required_train <- c("unit_norm", "unit", "source_file", "BB_rating", "cv_fold", feature_cols)
  required_test <- c("unit_norm", "unit", "source_file", "BB_rating", feature_cols)
  train_before <- nrow(train_dt)
  test_before <- nrow(test_dt)
  train_dt <- train_dt[complete.cases(train_dt[, ..required_train])]
  test_dt <- test_dt[complete.cases(test_dt[, ..required_test])]
  log_line("  retained rows after completeness filter: train %d/%d, test %d/%d\n", nrow(train_dt), train_before, nrow(test_dt), test_before)

  cv_dt <- copy(train_dt)
  target_levels <- sort(unique(cv_dt$BB_rating))
  fold_ids <- sort(unique(cv_dt$cv_fold))
  model_out_dir <- file.path(out_root, model_id)
  dir.create(model_out_dir, recursive = TRUE, showWarnings = FALSE)

  fold_results <- lapply(fold_ids, function(fid) run_outer_fold(cv_dt, fid, feature_cols, target_levels))

  fold_metrics_dt <- rbindlist(lapply(fold_results, `[[`, "fold_metrics"), use.names = TRUE, fill = TRUE)
  outer_param_dt <- rbindlist(lapply(fold_results, `[[`, "params"), use.names = TRUE, fill = TRUE)
  cv_predictions_dt <- rbindlist(lapply(fold_results, `[[`, "predictions"), use.names = TRUE, fill = TRUE)
  cv_confusion_dt <- rbindlist(lapply(fold_results, `[[`, "confusion"), use.names = TRUE, fill = TRUE)
  cv_per_class_dt <- rbindlist(lapply(fold_results, `[[`, "per_class"), use.names = TRUE, fill = TRUE)
  selected_features_dt <- rbindlist(lapply(fold_results, `[[`, "selected_features"), use.names = TRUE, fill = TRUE)
  cv_smote_before_dt <- rbindlist(lapply(fold_results, `[[`, "smote_before"), use.names = TRUE, fill = TRUE)
  cv_smote_after_dt <- rbindlist(lapply(fold_results, `[[`, "smote_after"), use.names = TRUE, fill = TRUE)
  cv_summary_dt <- summarize_cv_metrics(fold_metrics_dt)

  final_res <- run_final_holdout(train_dt, test_dt, feature_cols, target_levels, outer_param_dt)
  summary_row <- collect_summary_row(model_id, spec$model_name[[1]], spec$severity_categories[[1]], cluster_setting, cv_summary_dt, final_res$metrics, final_res$params, nrow(final_res$used_features))

  fwrite(spec, file.path(model_out_dir, "model_spec.csv"))
  fwrite(fold_metrics_dt, file.path(model_out_dir, "cv_fold_metrics.csv"))
  fwrite(cv_summary_dt, file.path(model_out_dir, "cv_metric_summary.csv"))
  fwrite(outer_param_dt, file.path(model_out_dir, "cv_selected_params_per_fold.csv"))
  fwrite(cv_predictions_dt, file.path(model_out_dir, "cv_predictions.csv"))
  fwrite(cv_confusion_dt, file.path(model_out_dir, "cv_confusion_matrix_by_fold.csv"))
  fwrite(cv_per_class_dt, file.path(model_out_dir, "cv_per_class_metrics_by_fold.csv"))
  fwrite(selected_features_dt, file.path(model_out_dir, "cv_selected_features_by_fold.csv"))
  fwrite(cv_smote_before_dt, file.path(model_out_dir, "cv_smote_counts_before_by_fold.csv"))
  fwrite(cv_smote_after_dt, file.path(model_out_dir, "cv_smote_counts_after_by_fold.csv"))
  fwrite(final_res$params, file.path(model_out_dir, "final_model_parameters.csv"))
  fwrite(final_res$metrics, file.path(model_out_dir, "final_holdout_metrics.csv"))
  fwrite(final_res$predictions, file.path(model_out_dir, "final_holdout_predictions.csv"))
  fwrite(final_res$confusion, file.path(model_out_dir, "final_holdout_confusion_matrix.csv"))
  fwrite(final_res$per_class, file.path(model_out_dir, "final_holdout_per_class_metrics.csv"))
  fwrite(final_res$used_features, file.path(model_out_dir, "final_model_used_features.csv"))
  fwrite(final_res$smote_before, file.path(model_out_dir, "final_smote_counts_before.csv"))
  fwrite(final_res$smote_after, file.path(model_out_dir, "final_smote_counts_after.csv"))
  fwrite(summary_row, file.path(model_out_dir, "model_summary_row.csv"))

  summary_lines <- c(
    sprintf("Model ID: %s", model_id),
    sprintf("Model name: %s", spec$model_name[[1]]),
    sprintf("Severity categories: %s", spec$severity_categories[[1]]),
    sprintf("Feature set: cluster ratio only (%s)", cluster_setting),
    "",
    "Cross-validation summary (grouped 5-fold CV on the 90% training partition):",
    capture.output(print(cv_summary_dt)),
    "",
    "Final holdout metrics:",
    capture.output(print(final_res$metrics)),
    "",
    "Final model parameters:",
    capture.output(print(final_res$params))
  )
  writeLines(summary_lines, file.path(model_out_dir, "summary.txt"))

  summary_rows[[length(summary_rows) + 1L]] <- summary_row
  param_rows[[length(param_rows) + 1L]] <- cbind(data.table(model_id = model_id), final_res$params)
  feature_rows[[length(feature_rows) + 1L]] <- cbind(data.table(model_id = model_id, severity_categories = spec$severity_categories[[1]], cluster_setting = cluster_setting), final_res$used_features)
  supp_rows[[length(supp_rows) + 1L]] <- build_supplementary_row(summary_row)
}

summary_dt <- rbindlist(summary_rows, use.names = TRUE, fill = TRUE)
param_dt <- rbindlist(param_rows, use.names = TRUE, fill = TRUE)
feature_dt <- rbindlist(feature_rows, use.names = TRUE, fill = TRUE)
supp_dt <- rbindlist(supp_rows, use.names = TRUE, fill = TRUE)
setorder(supp_dt, `Number of severity categories`, `Model features`)

fwrite(summary_dt, file.path(out_root, "cluster_ratio_only_svm_cv_holdout_summary.csv"))
fwrite(param_dt, file.path(out_root, "cluster_ratio_only_svm_final_parameters.csv"))
fwrite(feature_dt, file.path(out_root, "cluster_ratio_only_svm_feature_sets.csv"))
fwrite(supp_dt, file.path(out_root, "supplementary_table_cluster_ratio_svm.csv"))
fwrite(supp_dt, file.path(out_root, "supplementary_table_cluster_ratio_svm_manuscript_ready.csv"))

readme_lines <- c(
  "Cluster-ratio-only SVM runs for 8C, 12C, and 16C",
  "",
  "This folder contains grouped leakage-safe model runs for both 4-class and 6-class BLB severity.",
  "",
  "Pipeline:",
  "1. Uses the existing grouped 90% training / 10% holdout split from Train_test_master.",
  "2. Uses the fixed grouped 5-fold assignments (cv_fold) already defined for the 90% training partition.",
  "3. Tunes SVM cost and gamma inside each outer fold using grouped inner folds built from unit_norm.",
  "4. Applies SMOTE only to training data inside each inner fold, each outer fold, and the final full-training fit.",
  "5. Trains only on cluster-ratio features.",
  "6. Saves fold-level metrics, mean, SD, and 95% confidence intervals across the 5 held-out folds.",
  "7. Chooses consensus parameters across the 5 outer folds, then trains a final model on the full 90% training set and evaluates once on the untouched 10% holdout.",
  "",
  "Key files:",
  "- cluster_ratio_only_svm_cv_holdout_summary.csv: detailed cross-validation and final holdout summary for all six models.",
  "- cluster_ratio_only_svm_final_parameters.csv: consensus cost and gamma for the final model of each run.",
  "- cluster_ratio_only_svm_feature_sets.csv: exact cluster-ratio features used by each model.",
  "- supplementary_table_cluster_ratio_svm.csv: manuscript-style supplementary table with Accuracy, Precision, Recall, and F1.
- supplementary_table_cluster_ratio_svm_manuscript_ready.csv: duplicate of the manuscript-ready supplementary table for direct submission use.",
  "",
  sprintf("Revision root used: %s", revision_root),
  sprintf("Random seed: %d", 123)
)
writeLines(readme_lines, file.path(out_root, "README.txt"))

log_line("\nSaved outputs to %s\n", out_root)
