#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(data.table)
  library(e1071)
  library(FNN)
  library(ranger)
  library(xgboost)
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
spec_file <- Sys.getenv("MODEL_SPEC_FILE", unset = file.path(script_dir, "revised_table2_rf_svm_model_specs.csv"))
out_root <- Sys.getenv("MODEL_OUTPUT_ROOT", unset = file.path(bundle_root, "outputs", "publication_table2_grouped_cv_holdout_runs"))

default_paths <- list(
  four_train = file.path(bundle_root, "input_data", "training_90.csv"),
  four_test = file.path(bundle_root, "input_data", "holdout_test_10.csv"),
  six_train = file.path(bundle_root, "input_data", "training_90.csv"),
  six_test = file.path(bundle_root, "input_data", "holdout_test_10.csv"),
  four_cv_assign = file.path(bundle_root, "input_data", "training_90_grouped5fold_unit_assignments.csv"),
  six_cv_assign = file.path(bundle_root, "input_data", "training_90_grouped5fold_unit_assignments.csv")
)

paths <- list(
  four_train = Sys.getenv("FOUR_TRAIN_PATH", unset = default_paths$four_train),
  four_test = Sys.getenv("FOUR_TEST_PATH", unset = default_paths$four_test),
  six_train = Sys.getenv("SIX_TRAIN_PATH", unset = default_paths$six_train),
  six_test = Sys.getenv("SIX_TEST_PATH", unset = default_paths$six_test),
  four_cv_assign = Sys.getenv("FOUR_CV_ASSIGN_PATH", unset = default_paths$four_cv_assign),
  six_cv_assign = Sys.getenv("SIX_CV_ASSIGN_PATH", unset = default_paths$six_cv_assign)
)

dir.create(out_root, recursive = TRUE, showWarnings = FALSE)
shared_dir <- file.path(out_root, "shared")
dir.create(shared_dir, recursive = TRUE, showWarnings = FALSE)

if (!file.exists(spec_file)) stop("Missing model spec file: ", spec_file)
for (p in c(paths$four_train, paths$four_test, paths$six_train, paths$six_test)) {
  if (!file.exists(p)) stop("Missing required dataset: ", p)
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

log_line <- function(...) {
  cat(sprintf("[%s] ", format(Sys.time(), "%Y-%m-%d %H:%M:%S")))
  cat(sprintf(...))
}

safe_unique <- function(dt) unique(dt, by = names(dt))

collapse_to_fourclass <- function(rating_vec) {
  out <- as.integer(rating_vec)
  out[out %in% c(1L, 3L)] <- 3L
  out[out %in% c(7L, 9L)] <- 9L
  out
}

prepare_severity_labels <- function(dt, severity_categories) {
  out <- copy(dt)
  if (severity_categories == 4L) {
    if ("BB_rating_4class" %in% names(out)) {
      out[, BB_rating := as.integer(BB_rating_4class)]
    } else {
      out[, BB_rating := collapse_to_fourclass(BB_rating)]
    }
  } else {
    out[, BB_rating := as.integer(BB_rating)]
  }
  out
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

make_group_folds <- function(group_vec, label_vec, k = 5L, seed = 123L) {
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

  grp_wide[, .(unit_norm = group, cv_fold = assigned, total_obs)]
}

attach_cv_folds <- function(train_dt, assign_path = NA_character_, seed = 123L, severity_tag = "dataset") {
  out <- copy(train_dt)
  out[, unit_norm := as.character(unit_norm)]
  if ("source_file" %in% names(out)) out[, source_file := as.character(source_file)]

  if ("cv_fold" %in% names(out) && all(!is.na(out$cv_fold))) {
    fold_dt <- safe_unique(out[, .(unit_norm, source_file = if ("source_file" %in% names(out)) source_file else NA_character_, cv_fold = as.integer(cv_fold))])
    fwrite(fold_dt, file.path(shared_dir, sprintf("%s_grouped5cv_fold_assignments.csv", severity_tag)))
    return(out)
  }

  if (!is.na(assign_path) && nzchar(assign_path) && file.exists(assign_path)) {
    assign_header <- names(fread(assign_path, nrows = 0L))
    select_cols <- intersect(assign_header, c("unit_norm", "source_file", "cv_fold", "group", "fold"))
    assign_dt <- fread(assign_path, select = select_cols)
    if ("group" %in% names(assign_dt) && !"unit_norm" %in% names(assign_dt)) setnames(assign_dt, "group", "unit_norm")
    if ("fold" %in% names(assign_dt) && !"cv_fold" %in% names(assign_dt)) setnames(assign_dt, "fold", "cv_fold")
    if ("unit_norm" %in% names(assign_dt) && "cv_fold" %in% names(assign_dt)) {
      assign_dt[, unit_norm := as.character(unit_norm)]
      if ("source_file" %in% names(assign_dt)) assign_dt[, source_file := as.character(source_file)]
      merge_keys <- if ("source_file" %in% names(assign_dt) && "source_file" %in% names(out)) c("unit_norm", "source_file") else "unit_norm"
      assign_dt <- safe_unique(assign_dt[, c(merge_keys, "cv_fold"), with = FALSE])
      out <- merge(out, assign_dt, by = merge_keys, all.x = TRUE, sort = FALSE)
      if (all(!is.na(out$cv_fold))) {
        out[, cv_fold := as.integer(cv_fold)]
        fwrite(assign_dt, file.path(shared_dir, sprintf("%s_grouped5cv_fold_assignments.csv", severity_tag)))
        return(out)
      }
      warning("Fold assignment file did not cover all rows for ", severity_tag, "; generating grouped folds directly from the training data.")
      out[, cv_fold := NULL]
    }
  }

  fold_dt <- make_group_folds(out$unit_norm, out$BB_rating, k = 5L, seed = seed)
  out <- merge(out, fold_dt[, .(unit_norm, cv_fold)], by = "unit_norm", all.x = TRUE, sort = FALSE)
  fwrite(fold_dt, file.path(shared_dir, sprintf("%s_grouped5cv_fold_assignments.csv", severity_tag)))
  out[, cv_fold := as.integer(cv_fold)]
  out
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

quadratic_weighted_kappa <- function(actual_idx, predicted_idx, n_levels) {
  if (length(actual_idx) == 0L) return(NA_real_)
  obs <- matrix(0, nrow = n_levels, ncol = n_levels)
  for (i in seq_along(actual_idx)) obs[actual_idx[i], predicted_idx[i]] <- obs[actual_idx[i], predicted_idx[i]] + 1
  obs <- obs / sum(obs)
  hist_a <- rowSums(obs)
  hist_b <- colSums(obs)
  exp_mat <- outer(hist_a, hist_b)
  weights <- matrix(0, nrow = n_levels, ncol = n_levels)
  for (i in seq_len(n_levels)) {
    for (j in seq_len(n_levels)) {
      weights[i, j] <- ((i - j)^2) / ((n_levels - 1)^2)
    }
  }
  denom <- sum(weights * exp_mat)
  if (denom == 0) return(1)
  1 - (sum(weights * obs) / denom)
}

compute_ordinal_metrics <- function(actual, predicted, levels_all) {
  actual_idx <- match(as.character(actual), as.character(levels_all))
  pred_idx <- match(as.character(predicted), as.character(levels_all))
  list(
    qwk = quadratic_weighted_kappa(actual_idx, pred_idx, length(levels_all)),
    mae_class = mean(abs(actual_idx - pred_idx))
  )
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
    per_class = data.table(rating = as.character(levels_all), support = as.integer(supports), precision = precision, recall = recall, f1 = f1)
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
  if (n == 1L) return(list(mean = mu, sd = 0, ci_low = mu, ci_high = mu, n = 1L))
  s <- sd(x)
  err <- qt(0.975, df = n - 1L) * s / sqrt(n)
  list(mean = mu, sd = s, ci_low = mu - err, ci_high = mu + err, n = n)
}

mode_numeric <- function(x) {
  x <- round(as.numeric(x), 12)
  x <- x[is.finite(x)]
  if (length(x) == 0L) return(NA_real_)
  tab <- sort(table(x), decreasing = TRUE)
  as.numeric(names(tab)[1])
}

mode_integer <- function(x) {
  out <- mode_numeric(x)
  if (is.na(out)) return(NA_integer_)
  as.integer(round(out))
}

average_vi_feature_candidates <- c(
  "pcavg_ndvi", "pcavg_sri", "pcavg_psri", "pcavg_ipvi", "pcavg_gb_ndvi",
  "pcavg_gr_ndvi", "pcavg_hue", "pcavg_npci", "pcavg_greenness", "pcavg_gndvi",
  "pcavg_bndvi", "pcavg_grvi", "pcavg_gli", "pcavg_vari", "pcavg_ngbi",
  "pcavg_rgri", "pcavg_gi2", "pcavg_blb",
  "NDVI_mean", "SRI_mean", "PSRI_mean", "IPVI_mean", "GB_NDVI_mean",
  "GR_NDVI_mean", "HUE_mean", "NPCI_mean", "GREENNESS_mean", "GNDVI_mean",
  "BNDVI_mean", "GRVI_mean", "GLI_mean", "VARI_mean", "NGBI_mean",
  "RGRI_mean", "GI2_mean", "BLB_mean"
)

build_static_groups <- function(dt) {
  dt_names <- names(dt)
  avg_idx <- match(tolower(average_vi_feature_candidates), tolower(dt_names), nomatch = 0L)
  avg_cols <- unique(dt_names[avg_idx[avg_idx > 0L]])
  list(
    mean_vi_cols = avg_cols,
    bin_cols = grep("_bin[1-6]$", dt_names, value = TRUE),
    cluster_cols = grep("^cluster_[0-9]+_ratio$", dt_names, value = TRUE),
    structural_cols = grep("^structural_", dt_names, value = TRUE)
  )
}

select_xgb_top_features <- function(train_dt, candidate_features, target_col, top_n = 30L, seed = 123L) {
  work <- sanitize_feature_frame(copy(train_dt), candidate_features)
  work <- work[complete.cases(work[, c(candidate_features, target_col), with = FALSE])]
  if (nrow(work) == 0L) return(candidate_features[seq_len(min(top_n, length(candidate_features)))])
  x <- as.matrix(work[, ..candidate_features])
  mode(x) <- "numeric"
  y_levels <- sort(unique(work[[target_col]]))
  y <- match(work[[target_col]], y_levels) - 1L
  n_class <- length(y_levels)
  if (n_class < 2L) return(candidate_features[seq_len(min(top_n, length(candidate_features)))])
  dtrain <- xgboost::xgb.DMatrix(data = x, label = y)
  set.seed(seed)
  params <- list(
    objective = if (n_class == 2L) "binary:logistic" else "multi:softprob",
    eval_metric = if (n_class == 2L) "logloss" else "mlogloss",
    max_depth = 6,
    eta = 0.1,
    min_child_weight = 1,
    subsample = 0.8,
    colsample_bytree = 0.8,
    verbosity = 0
  )
  if (n_class > 2L) params$num_class <- n_class
  fit <- tryCatch(xgboost::xgb.train(params = params, data = dtrain, nrounds = 120, verbose = 0), error = function(e) NULL)
  if (is.null(fit)) return(candidate_features[seq_len(min(top_n, length(candidate_features)))])
  imp <- tryCatch(xgboost::xgb.importance(feature_names = candidate_features, model = fit), error = function(e) NULL)
  if (is.null(imp) || nrow(imp) == 0L) return(candidate_features[seq_len(min(top_n, length(candidate_features)))])
  imp$Feature[seq_len(min(top_n, nrow(imp)))]
}

choose_feature_columns <- function(model_spec, train_dt, static_groups, seed = 123L) {
  key <- model_spec$feature_key[[1]]
  if (key == "average_vi") return(static_groups$mean_vi_cols)
  if (key == "binned_vi") return(static_groups$bin_cols)
  if (key == "cluster_ratio") return(static_groups$cluster_cols)
  if (key == "cluster_plus_binned") return(unique(c(static_groups$cluster_cols, static_groups$bin_cols)))
  if (key == "cluster_plus_binned_plus_structural") return(unique(c(static_groups$cluster_cols, static_groups$bin_cols, static_groups$structural_cols)))
  if (key == "cluster_plus_binned_top30") {
    candidate <- unique(c(static_groups$cluster_cols, static_groups$bin_cols))
    return(select_xgb_top_features(train_dt, candidate, target_col = "BB_rating", top_n = 30L, seed = seed))
  }
  if (key == "cluster_plus_binned_top30_plus_structural") {
    candidate <- unique(c(static_groups$cluster_cols, static_groups$bin_cols))
    top_feats <- select_xgb_top_features(train_dt, candidate, target_col = "BB_rating", top_n = 30L, seed = seed)
    return(unique(c(top_feats, static_groups$structural_cols)))
  }
  stop("Unknown feature key: ", key)
}

prepare_train_valid <- function(train_dt, valid_dt, feature_cols) {
  keep_cols <- unique(c("unit_norm", "source_file", "BB_rating", feature_cols))
  train_work <- sanitize_feature_frame(copy(train_dt), feature_cols)
  valid_work <- sanitize_feature_frame(copy(valid_dt), feature_cols)
  train_work <- train_work[complete.cases(train_work[, keep_cols, with = FALSE])]
  valid_work <- valid_work[complete.cases(valid_work[, keep_cols, with = FALSE])]
  list(train = train_work, valid = valid_work)
}

predict_xgb_labels <- function(fit, feature_dt, feature_cols, target_levels) {
  dvalid <- xgboost::xgb.DMatrix(data = as.matrix(feature_dt[, ..feature_cols]))
  pred_raw <- predict(fit, newdata = dvalid)
  n_class <- length(target_levels)
  if (n_class == 2L) {
    pred_idx <- ifelse(pred_raw >= 0.5, 2L, 1L)
  } else {
    pred_mat <- matrix(pred_raw, ncol = n_class, byrow = TRUE)
    pred_idx <- max.col(pred_mat, ties.method = "first")
  }
  as.character(target_levels[pred_idx])
}

choose_best_svm_params <- function(train_dt, feature_cols, target_levels, outer_fold_id) {
  inner_groups <- make_group_folds(train_dt$unit_norm, train_dt$BB_rating, k = 3L, seed = 1000L + outer_fold_id)
  train_inner <- merge(copy(train_dt), inner_groups[, .(unit_norm, inner_fold = cv_fold)], by = "unit_norm", all.x = TRUE, sort = FALSE)
  grid <- CJ(cost = c(1, 4, 16), gamma = c(0.001, 0.005, 0.02))
  results <- vector("list", nrow(grid))
  for (g in seq_len(nrow(grid))) {
    cost_val <- grid$cost[g]
    gamma_val <- grid$gamma[g]
    inner_pred_parts <- list()
    for (inner_fold_id in sort(unique(train_inner$inner_fold))) {
      inner_train <- train_inner[inner_fold != inner_fold_id]
      inner_valid <- train_inner[inner_fold == inner_fold_id]
      prep <- prepare_train_valid(inner_train, inner_valid, feature_cols)
      if (nrow(prep$train) == 0L || nrow(prep$valid) == 0L) next
      x_train <- as.matrix(prep$train[, ..feature_cols])
      x_valid <- as.matrix(prep$valid[, ..feature_cols])
      mode(x_train) <- "numeric"
      mode(x_valid) <- "numeric"
      scaled <- scale_train_test(x_train, x_valid)
      train_ml <- as.data.table(scaled$train)
      setnames(train_ml, feature_cols)
      train_ml[, BB_rating := prep$train$BB_rating]
      sm <- apply_smote(train_ml, target_col = "BB_rating", feature_cols = feature_cols, seed = 10000L + outer_fold_id * 100L + inner_fold_id * 10L + g)
      fit <- tryCatch(
        svm(x = sm$data[, ..feature_cols], y = factor(sm$data$BB_rating, levels = target_levels), kernel = "radial", type = "C-classification", cost = cost_val, gamma = gamma_val, scale = FALSE),
        error = function(e) NULL
      )
      if (is.null(fit)) next
      pred <- tryCatch(as.character(predict(fit, newdata = scaled$test)), error = function(e) NULL)
      if (is.null(pred)) next
      inner_pred_parts[[length(inner_pred_parts) + 1L]] <- data.table(actual = prep$valid$BB_rating, predicted = pred)
    }
    if (length(inner_pred_parts) == 0L) {
      results[[g]] <- data.table(cost = cost_val, gamma = gamma_val, macro_f1 = -Inf, accuracy = -Inf, f1_weighted = -Inf, successful_inner_folds = 0L)
    } else {
      pred_dt <- rbindlist(inner_pred_parts)
      met <- compute_metrics(pred_dt$actual, pred_dt$predicted, target_levels)
      results[[g]] <- data.table(cost = cost_val, gamma = gamma_val, macro_f1 = met$macro_f1, accuracy = met$accuracy, f1_weighted = met$f1_weighted, successful_inner_folds = length(inner_pred_parts))
    }
  }
  res_dt <- rbindlist(results)
  if (all(!is.finite(res_dt$macro_f1))) return(data.table(cost = 1, gamma = 0.005, macro_f1 = NA_real_, accuracy = NA_real_, f1_weighted = NA_real_, successful_inner_folds = 0L))
  setorder(res_dt, -macro_f1, -accuracy, -f1_weighted, cost, gamma)
  res_dt[1]
}

choose_best_rf_params <- function(train_dt, feature_cols, target_levels, outer_fold_id) {
  inner_groups <- make_group_folds(train_dt$unit_norm, train_dt$BB_rating, k = 3L, seed = 2000L + outer_fold_id)
  train_inner <- merge(copy(train_dt), inner_groups[, .(unit_norm, inner_fold = cv_fold)], by = "unit_norm", all.x = TRUE, sort = FALSE)
  p <- length(feature_cols)
  mtry_candidates <- sort(unique(pmax(1L, as.integer(round(c(sqrt(p), p / 6, p / 4))))))
  grid <- CJ(mtry = mtry_candidates, min_node_size = c(1L, 5L, 10L))
  results <- vector("list", nrow(grid))
  for (g in seq_len(nrow(grid))) {
    mtry_val <- grid$mtry[g]
    min_node_val <- grid$min_node_size[g]
    inner_pred_parts <- list()
    for (inner_fold_id in sort(unique(train_inner$inner_fold))) {
      inner_train <- train_inner[inner_fold != inner_fold_id]
      inner_valid <- train_inner[inner_fold == inner_fold_id]
      prep <- prepare_train_valid(inner_train, inner_valid, feature_cols)
      if (nrow(prep$train) == 0L || nrow(prep$valid) == 0L) next
      train_ml <- copy(prep$train[, c(feature_cols, "BB_rating"), with = FALSE])
      sm <- apply_smote(train_ml, target_col = "BB_rating", feature_cols = feature_cols, seed = 20000L + outer_fold_id * 100L + inner_fold_id * 10L + g)
      sm$data[, BB_rating := factor(BB_rating, levels = target_levels)]
      valid_ml <- copy(prep$valid[, c(feature_cols, "BB_rating"), with = FALSE])
      valid_ml[, BB_rating := factor(BB_rating, levels = target_levels)]
      fit <- tryCatch(ranger(dependent.variable.name = "BB_rating", data = sm$data, num.trees = 1000, mtry = mtry_val, min.node.size = min_node_val, classification = TRUE, importance = "none", seed = 30000L + outer_fold_id * 100L + inner_fold_id * 10L + g), error = function(e) NULL)
      if (is.null(fit)) next
      pred_obj <- tryCatch(predict(fit, data = valid_ml), error = function(e) NULL)
      if (is.null(pred_obj)) next
      pred <- if (is.matrix(pred_obj$predictions) || is.data.frame(pred_obj$predictions)) colnames(pred_obj$predictions)[max.col(pred_obj$predictions, ties.method = "first")] else as.character(pred_obj$predictions)
      inner_pred_parts[[length(inner_pred_parts) + 1L]] <- data.table(actual = prep$valid$BB_rating, predicted = pred)
    }
    if (length(inner_pred_parts) == 0L) {
      results[[g]] <- data.table(mtry = mtry_val, min_node_size = min_node_val, macro_f1 = -Inf, accuracy = -Inf, f1_weighted = -Inf, successful_inner_folds = 0L)
    } else {
      pred_dt <- rbindlist(inner_pred_parts)
      met <- compute_metrics(pred_dt$actual, pred_dt$predicted, target_levels)
      results[[g]] <- data.table(mtry = mtry_val, min_node_size = min_node_val, macro_f1 = met$macro_f1, accuracy = met$accuracy, f1_weighted = met$f1_weighted, successful_inner_folds = length(inner_pred_parts))
    }
  }
  res_dt <- rbindlist(results)
  if (all(!is.finite(res_dt$macro_f1))) return(data.table(mtry = max(1L, round(sqrt(length(feature_cols)))), min_node_size = 5L, macro_f1 = NA_real_, accuracy = NA_real_, f1_weighted = NA_real_, successful_inner_folds = 0L))
  setorder(res_dt, -macro_f1, -accuracy, -f1_weighted, mtry, min_node_size)
  res_dt[1]
}

choose_best_xgb_params <- function(train_dt, feature_cols, target_levels, outer_fold_id) {
  inner_groups <- make_group_folds(train_dt$unit_norm, train_dt$BB_rating, k = 3L, seed = 3000L + outer_fold_id)
  train_inner <- merge(copy(train_dt), inner_groups[, .(unit_norm, inner_fold = cv_fold)], by = "unit_norm", all.x = TRUE, sort = FALSE)
  grid <- CJ(max_depth = c(4L, 6L), eta = c(0.05, 0.1), min_child_weight = c(1, 3), nrounds = c(80L, 120L), subsample = 0.8, colsample_bytree = 0.8)
  results <- vector("list", nrow(grid))
  for (g in seq_len(nrow(grid))) {
    inner_pred_parts <- list()
    for (inner_fold_id in sort(unique(train_inner$inner_fold))) {
      inner_train <- train_inner[inner_fold != inner_fold_id]
      inner_valid <- train_inner[inner_fold == inner_fold_id]
      prep <- prepare_train_valid(inner_train, inner_valid, feature_cols)
      if (nrow(prep$train) == 0L || nrow(prep$valid) == 0L) next
      train_ml <- copy(prep$train[, c(feature_cols, "BB_rating"), with = FALSE])
      sm <- apply_smote(train_ml, target_col = "BB_rating", feature_cols = feature_cols, seed = 30000L + outer_fold_id * 100L + inner_fold_id * 10L + g)
      x_train <- as.matrix(sm$data[, ..feature_cols])
      mode(x_train) <- "numeric"
      y_levels <- target_levels
      y_train <- match(sm$data$BB_rating, y_levels) - 1L
      dtrain <- xgboost::xgb.DMatrix(data = x_train, label = y_train)
      n_class <- length(y_levels)
      params <- list(
        objective = if (n_class == 2L) "binary:logistic" else "multi:softprob",
        eval_metric = if (n_class == 2L) "logloss" else "mlogloss",
        max_depth = grid$max_depth[g],
        eta = grid$eta[g],
        min_child_weight = grid$min_child_weight[g],
        subsample = grid$subsample[g],
        colsample_bytree = grid$colsample_bytree[g],
        verbosity = 0
      )
      if (n_class > 2L) params$num_class <- n_class
      fit <- tryCatch(xgboost::xgb.train(params = params, data = dtrain, nrounds = grid$nrounds[g], verbose = 0), error = function(e) NULL)
      if (is.null(fit)) next
      pred <- tryCatch(predict_xgb_labels(fit, prep$valid, feature_cols, y_levels), error = function(e) NULL)
      if (is.null(pred)) next
      inner_pred_parts[[length(inner_pred_parts) + 1L]] <- data.table(actual = prep$valid$BB_rating, predicted = pred)
    }
    if (length(inner_pred_parts) == 0L) {
      results[[g]] <- data.table(max_depth = grid$max_depth[g], eta = grid$eta[g], min_child_weight = grid$min_child_weight[g], subsample = grid$subsample[g], colsample_bytree = grid$colsample_bytree[g], nrounds = grid$nrounds[g], macro_f1 = -Inf, accuracy = -Inf, f1_weighted = -Inf, successful_inner_folds = 0L)
    } else {
      pred_dt <- rbindlist(inner_pred_parts)
      met <- compute_metrics(pred_dt$actual, pred_dt$predicted, target_levels)
      results[[g]] <- data.table(max_depth = grid$max_depth[g], eta = grid$eta[g], min_child_weight = grid$min_child_weight[g], subsample = grid$subsample[g], colsample_bytree = grid$colsample_bytree[g], nrounds = grid$nrounds[g], macro_f1 = met$macro_f1, accuracy = met$accuracy, f1_weighted = met$f1_weighted, successful_inner_folds = length(inner_pred_parts))
    }
  }
  res_dt <- rbindlist(results)
  if (all(!is.finite(res_dt$macro_f1))) return(data.table(max_depth = 6L, eta = 0.05, min_child_weight = 3, subsample = 0.8, colsample_bytree = 0.8, nrounds = 120L, macro_f1 = NA_real_, accuracy = NA_real_, f1_weighted = NA_real_, successful_inner_folds = 0L))
  setorder(res_dt, -macro_f1, -accuracy, -f1_weighted, max_depth, eta, min_child_weight, nrounds)
  res_dt[1]
}

run_outer_fold <- function(model_spec, fold_id, cv_dt, static_groups, target_levels) {
  outer_train <- cv_dt[cv_fold != fold_id]
  outer_valid <- cv_dt[cv_fold == fold_id]
  feature_cols <- choose_feature_columns(model_spec, outer_train, static_groups, seed = 40000L + fold_id)
  if (length(feature_cols) == 0L) stop("No feature columns selected for fold ", fold_id, " in model ", model_spec$model_id[[1]])
  prep <- prepare_train_valid(outer_train, outer_valid, feature_cols)
  if (nrow(prep$train) == 0L || nrow(prep$valid) == 0L) stop("No complete cases for fold ", fold_id, " in model ", model_spec$model_id[[1]])

  if (model_spec$algorithm[[1]] == "SVM") {
    best_param <- choose_best_svm_params(outer_train, feature_cols, target_levels, fold_id)
    x_train <- as.matrix(prep$train[, ..feature_cols])
    x_valid <- as.matrix(prep$valid[, ..feature_cols])
    mode(x_train) <- "numeric"
    mode(x_valid) <- "numeric"
    scaled <- scale_train_test(x_train, x_valid)
    train_ml <- as.data.table(scaled$train)
    setnames(train_ml, feature_cols)
    train_ml[, BB_rating := prep$train$BB_rating]
    sm <- apply_smote(train_ml, target_col = "BB_rating", feature_cols = feature_cols, seed = 50000L + fold_id)
    fit <- svm(x = sm$data[, ..feature_cols], y = factor(sm$data$BB_rating, levels = target_levels), kernel = "radial", type = "C-classification", cost = best_param$cost[[1]], gamma = best_param$gamma[[1]], scale = FALSE)
    pred <- as.character(predict(fit, newdata = scaled$test))
    param_row <- data.table(fold = fold_id, cost = best_param$cost[[1]], gamma = best_param$gamma[[1]], feature_count = length(feature_cols), successful_inner_folds = best_param$successful_inner_folds[[1]])
  } else if (model_spec$algorithm[[1]] == "RandomForest") {
    best_param <- choose_best_rf_params(outer_train, feature_cols, target_levels, fold_id)
    train_ml <- copy(prep$train[, c(feature_cols, "BB_rating"), with = FALSE])
    sm <- apply_smote(train_ml, target_col = "BB_rating", feature_cols = feature_cols, seed = 60000L + fold_id)
    sm$data[, BB_rating := factor(BB_rating, levels = target_levels)]
    valid_ml <- copy(prep$valid[, c(feature_cols, "BB_rating"), with = FALSE])
    valid_ml[, BB_rating := factor(BB_rating, levels = target_levels)]
    fit <- ranger(dependent.variable.name = "BB_rating", data = sm$data, num.trees = 1000, mtry = best_param$mtry[[1]], min.node.size = best_param$min_node_size[[1]], classification = TRUE, importance = "none", seed = 70000L + fold_id)
    pred_obj <- predict(fit, data = valid_ml)
    pred <- if (is.matrix(pred_obj$predictions) || is.data.frame(pred_obj$predictions)) colnames(pred_obj$predictions)[max.col(pred_obj$predictions, ties.method = "first")] else as.character(pred_obj$predictions)
    param_row <- data.table(fold = fold_id, mtry = best_param$mtry[[1]], min_node_size = best_param$min_node_size[[1]], num_trees = 1000L, feature_count = length(feature_cols), successful_inner_folds = best_param$successful_inner_folds[[1]])
  } else if (model_spec$algorithm[[1]] == "XGBoost") {
    best_param <- choose_best_xgb_params(outer_train, feature_cols, target_levels, fold_id)
    train_ml <- copy(prep$train[, c(feature_cols, "BB_rating"), with = FALSE])
    sm <- apply_smote(train_ml, target_col = "BB_rating", feature_cols = feature_cols, seed = 80000L + fold_id)
    x_train <- as.matrix(sm$data[, ..feature_cols])
    mode(x_train) <- "numeric"
    y_train <- match(sm$data$BB_rating, target_levels) - 1L
    dtrain <- xgboost::xgb.DMatrix(data = x_train, label = y_train)
    params <- list(
      objective = if (length(target_levels) == 2L) "binary:logistic" else "multi:softprob",
      eval_metric = if (length(target_levels) == 2L) "logloss" else "mlogloss",
      max_depth = best_param$max_depth[[1]],
      eta = best_param$eta[[1]],
      min_child_weight = best_param$min_child_weight[[1]],
      subsample = best_param$subsample[[1]],
      colsample_bytree = best_param$colsample_bytree[[1]],
      verbosity = 0
    )
    if (length(target_levels) > 2L) params$num_class <- length(target_levels)
    fit <- xgboost::xgb.train(params = params, data = dtrain, nrounds = best_param$nrounds[[1]], verbose = 0)
    pred <- predict_xgb_labels(fit, prep$valid, feature_cols, target_levels)
    param_row <- data.table(fold = fold_id, max_depth = best_param$max_depth[[1]], eta = best_param$eta[[1]], min_child_weight = best_param$min_child_weight[[1]], subsample = best_param$subsample[[1]], colsample_bytree = best_param$colsample_bytree[[1]], nrounds = best_param$nrounds[[1]], feature_count = length(feature_cols), successful_inner_folds = best_param$successful_inner_folds[[1]])
  } else {
    stop("Unsupported algorithm: ", model_spec$algorithm[[1]])
  }

  met <- compute_metrics(prep$valid$BB_rating, pred, target_levels)
  ord <- compute_ordinal_metrics(prep$valid$BB_rating, pred, target_levels)
  list(
    fold_metrics = data.table(fold = fold_id, n_train = nrow(prep$train), n_valid = nrow(prep$valid), n_train_groups = uniqueN(prep$train$unit_norm), n_valid_groups = uniqueN(prep$valid$unit_norm), feature_count = length(feature_cols), accuracy = met$accuracy, precision_weighted = met$precision_weighted, recall_weighted = met$recall_weighted, f1_weighted = met$f1_weighted, macro_f1 = met$macro_f1),
    ordinal_metrics = data.table(fold = fold_id, qwk = ord$qwk, mae_class = ord$mae_class),
    params = param_row,
    predictions = data.table(fold = fold_id, unit_norm = prep$valid$unit_norm, source_file = prep$valid$source_file, actual = prep$valid$BB_rating, predicted = pred),
    confusion = build_confusion_dt(met$confusion)[, fold := fold_id],
    per_class = copy(met$per_class)[, fold := fold_id],
    selected_features = data.table(fold = fold_id, rank = seq_along(feature_cols), feature = feature_cols),
    smote_before = copy(sm$counts_before)[, fold := fold_id],
    smote_after = copy(sm$counts_after)[, fold := fold_id]
  )
}

consensus_top_features <- function(selected_feature_dt, top_n = 30L) {
  if (nrow(selected_feature_dt) == 0L) return(character())
  score_dt <- copy(selected_feature_dt)[, .(freq = .N, mean_rank = mean(rank)), by = feature]
  setorder(score_dt, -freq, mean_rank, feature)
  score_dt$feature[seq_len(min(top_n, nrow(score_dt)))]
}

final_feature_columns <- function(model_spec, static_groups, outer_selected_features_dt) {
  key <- model_spec$feature_key[[1]]
  if (key == "average_vi") return(static_groups$mean_vi_cols)
  if (key == "binned_vi") return(static_groups$bin_cols)
  if (key == "cluster_ratio") return(static_groups$cluster_cols)
  if (key == "cluster_plus_binned") return(unique(c(static_groups$cluster_cols, static_groups$bin_cols)))
  if (key == "cluster_plus_binned_plus_structural") return(unique(c(static_groups$cluster_cols, static_groups$bin_cols, static_groups$structural_cols)))
  if (key == "cluster_plus_binned_top30") return(consensus_top_features(outer_selected_features_dt, top_n = 30L))
  if (key == "cluster_plus_binned_top30_plus_structural") return(unique(c(consensus_top_features(outer_selected_features_dt, top_n = 30L), static_groups$structural_cols)))
  stop("Unknown feature key: ", key)
}

run_final_holdout <- function(model_spec, train_dt, test_dt, static_groups, outer_param_dt, outer_selected_features_dt, target_levels) {
  feature_cols <- final_feature_columns(model_spec, static_groups, outer_selected_features_dt)
  if (length(feature_cols) == 0L) stop("No final feature columns for model ", model_spec$model_id[[1]])
  prep <- prepare_train_valid(train_dt, test_dt, feature_cols)
  if (nrow(prep$train) == 0L || nrow(prep$valid) == 0L) stop("No complete cases for final holdout model: ", model_spec$model_id[[1]])

  if (model_spec$algorithm[[1]] == "SVM") {
    cost_val <- mode_numeric(outer_param_dt$cost)
    gamma_val <- mode_numeric(outer_param_dt$gamma)
    x_train <- as.matrix(prep$train[, ..feature_cols])
    x_test <- as.matrix(prep$valid[, ..feature_cols])
    mode(x_train) <- "numeric"
    mode(x_test) <- "numeric"
    scaled <- scale_train_test(x_train, x_test)
    train_ml <- as.data.table(scaled$train)
    setnames(train_ml, feature_cols)
    train_ml[, BB_rating := prep$train$BB_rating]
    sm <- apply_smote(train_ml, target_col = "BB_rating", feature_cols = feature_cols, seed = 90001L)
    fit <- svm(x = sm$data[, ..feature_cols], y = factor(sm$data$BB_rating, levels = target_levels), kernel = "radial", type = "C-classification", cost = cost_val, gamma = gamma_val, scale = FALSE)
    pred <- as.character(predict(fit, newdata = scaled$test))
    param_dt <- data.table(model_id = model_spec$model_id[[1]], algorithm = "SVM", parameter_strategy = "consensus across 5 outer folds", cost = cost_val, gamma = gamma_val, feature_count = length(feature_cols))
  } else if (model_spec$algorithm[[1]] == "RandomForest") {
    mtry_val <- mode_integer(outer_param_dt$mtry)
    min_node_val <- mode_integer(outer_param_dt$min_node_size)
    train_ml <- copy(prep$train[, c(feature_cols, "BB_rating"), with = FALSE])
    sm <- apply_smote(train_ml, target_col = "BB_rating", feature_cols = feature_cols, seed = 90002L)
    sm$data[, BB_rating := factor(BB_rating, levels = target_levels)]
    test_ml <- copy(prep$valid[, c(feature_cols, "BB_rating"), with = FALSE])
    test_ml[, BB_rating := factor(BB_rating, levels = target_levels)]
    fit <- ranger(dependent.variable.name = "BB_rating", data = sm$data, num.trees = 1000, mtry = mtry_val, min.node.size = min_node_val, classification = TRUE, importance = "none", seed = 90003L)
    pred_obj <- predict(fit, data = test_ml)
    pred <- if (is.matrix(pred_obj$predictions) || is.data.frame(pred_obj$predictions)) colnames(pred_obj$predictions)[max.col(pred_obj$predictions, ties.method = "first")] else as.character(pred_obj$predictions)
    param_dt <- data.table(model_id = model_spec$model_id[[1]], algorithm = "RandomForest", parameter_strategy = "consensus across 5 outer folds", mtry = mtry_val, min_node_size = min_node_val, num_trees = 1000L, feature_count = length(feature_cols))
  } else if (model_spec$algorithm[[1]] == "XGBoost") {
    max_depth_val <- mode_integer(outer_param_dt$max_depth)
    eta_val <- mode_numeric(outer_param_dt$eta)
    min_child_val <- mode_numeric(outer_param_dt$min_child_weight)
    subsample_val <- mode_numeric(outer_param_dt$subsample)
    colsample_val <- mode_numeric(outer_param_dt$colsample_bytree)
    nrounds_val <- mode_integer(outer_param_dt$nrounds)
    train_ml <- copy(prep$train[, c(feature_cols, "BB_rating"), with = FALSE])
    sm <- apply_smote(train_ml, target_col = "BB_rating", feature_cols = feature_cols, seed = 90004L)
    x_train <- as.matrix(sm$data[, ..feature_cols])
    mode(x_train) <- "numeric"
    y_train <- match(sm$data$BB_rating, target_levels) - 1L
    dtrain <- xgboost::xgb.DMatrix(data = x_train, label = y_train)
    params <- list(
      objective = if (length(target_levels) == 2L) "binary:logistic" else "multi:softprob",
      eval_metric = if (length(target_levels) == 2L) "logloss" else "mlogloss",
      max_depth = max_depth_val,
      eta = eta_val,
      min_child_weight = min_child_val,
      subsample = subsample_val,
      colsample_bytree = colsample_val,
      verbosity = 0
    )
    if (length(target_levels) > 2L) params$num_class <- length(target_levels)
    fit <- xgboost::xgb.train(params = params, data = dtrain, nrounds = nrounds_val, verbose = 0)
    pred <- predict_xgb_labels(fit, prep$valid, feature_cols, target_levels)
    param_dt <- data.table(model_id = model_spec$model_id[[1]], algorithm = "XGBoost", parameter_strategy = "consensus across 5 outer folds", max_depth = max_depth_val, eta = eta_val, min_child_weight = min_child_val, subsample = subsample_val, colsample_bytree = colsample_val, nrounds = nrounds_val, feature_count = length(feature_cols))
  } else {
    stop("Unsupported algorithm: ", model_spec$algorithm[[1]])
  }

  met <- compute_metrics(prep$valid$BB_rating, pred, target_levels)
  ord <- compute_ordinal_metrics(prep$valid$BB_rating, pred, target_levels)
  list(
    metrics = data.table(evaluation = "holdout_test", n_train_rows = nrow(prep$train), n_test_rows = nrow(prep$valid), n_train_groups = uniqueN(prep$train$unit_norm), n_test_groups = uniqueN(prep$valid$unit_norm), n_features = length(feature_cols), accuracy = met$accuracy, precision_weighted = met$precision_weighted, recall_weighted = met$recall_weighted, f1_score = met$f1_weighted, macro_f1 = met$macro_f1),
    ordinal_metrics = data.table(evaluation = "holdout_test", n_train_rows = nrow(prep$train), n_test_rows = nrow(prep$valid), n_train_groups = uniqueN(prep$train$unit_norm), n_test_groups = uniqueN(prep$valid$unit_norm), n_features = length(feature_cols), qwk = ord$qwk, mae_class = ord$mae_class),
    predictions = data.table(unit_norm = prep$valid$unit_norm, source_file = prep$valid$source_file, actual = prep$valid$BB_rating, predicted = pred),
    confusion = build_confusion_dt(met$confusion),
    per_class = met$per_class,
    params = param_dt,
    used_features = data.table(rank = seq_along(feature_cols), feature = feature_cols),
    smote_before = sm$counts_before,
    smote_after = sm$counts_after
  )
}

summarize_cv_metrics <- function(fold_metrics_dt) {
  metric_cols <- c("accuracy", "precision_weighted", "recall_weighted", "f1_weighted", "macro_f1")
  rows <- lapply(metric_cols, function(col) {
    s <- summarize_vector(fold_metrics_dt[[col]])
    data.table(metric = col, mean = s$mean, sd = s$sd, ci_low = s$ci_low, ci_high = s$ci_high, n = s$n)
  })
  rbindlist(rows)
}

summarize_cv_ordinal_metrics <- function(ordinal_dt) {
  metric_cols <- c("qwk", "mae_class")
  rows <- lapply(metric_cols, function(col) {
    s <- summarize_vector(ordinal_dt[[col]])
    data.table(metric = col, mean = s$mean, sd = s$sd, ci_low = s$ci_low, ci_high = s$ci_high, n = s$n)
  })
  rbindlist(rows)
}

collect_summary_row <- function(model_spec, cv_summary_dt, cv_ordinal_summary_dt, holdout_metrics_dt, holdout_ordinal_dt, final_param_dt, final_feature_count) {
  out <- data.table(
    model_id = model_spec$model_id[[1]],
    model_name = model_spec$model_name[[1]],
    algorithm = model_spec$algorithm[[1]],
    severity_categories = model_spec$severity_categories[[1]],
    feature_label = model_spec$feature_label[[1]],
    cv_folds = cv_summary_dt[metric == "accuracy", n],
    final_feature_count = final_feature_count,
    holdout_accuracy = holdout_metrics_dt$accuracy[[1]],
    holdout_precision_weighted = holdout_metrics_dt$precision_weighted[[1]],
    holdout_recall_weighted = holdout_metrics_dt$recall_weighted[[1]],
    holdout_f1_weighted = holdout_metrics_dt$f1_score[[1]],
    holdout_macro_f1 = holdout_metrics_dt$macro_f1[[1]],
    holdout_qwk = holdout_ordinal_dt$qwk[[1]],
    holdout_mae_class = holdout_ordinal_dt$mae_class[[1]]
  )
  for (metric_name in cv_summary_dt$metric) {
    row <- cv_summary_dt[metric == metric_name]
    prefix <- paste0("cv_", metric_name)
    out[[paste0(prefix, "_mean")]] <- row$mean[[1]]
    out[[paste0(prefix, "_sd")]] <- row$sd[[1]]
    out[[paste0(prefix, "_ci_low")]] <- row$ci_low[[1]]
    out[[paste0(prefix, "_ci_high")]] <- row$ci_high[[1]]
  }
  for (metric_name in cv_ordinal_summary_dt$metric) {
    row <- cv_ordinal_summary_dt[metric == metric_name]
    prefix <- paste0("cv_", metric_name)
    out[[paste0(prefix, "_mean")]] <- row$mean[[1]]
    out[[paste0(prefix, "_sd")]] <- row$sd[[1]]
    out[[paste0(prefix, "_ci_low")]] <- row$ci_low[[1]]
    out[[paste0(prefix, "_ci_high")]] <- row$ci_high[[1]]
  }
  for (nm in setdiff(names(final_param_dt), c("model_id", "parameter_strategy", "algorithm", "feature_count"))) {
    out[[paste0("final_", nm)]] <- final_param_dt[[nm]][1]
  }
  out
}

run_one_model <- function(model_spec, data_map) {
  model_id <- model_spec$model_id[[1]]
  dataset_tag <- if (model_spec$severity_categories[[1]] == 4L) "four" else "six"
  cv_dt <- copy(data_map[[paste0(dataset_tag, "_cv")]])
  train_dt <- copy(data_map[[paste0(dataset_tag, "_train")]])
  test_dt <- copy(data_map[[paste0(dataset_tag, "_test")]])

  overlap_n <- length(intersect(unique(as.character(train_dt$unit_norm)), unique(as.character(test_dt$unit_norm))))
  if (overlap_n != 0L) stop("Unit overlap detected in final holdout for model ", model_id)

  static_groups <- build_static_groups(cv_dt)
  target_levels <- sort(unique(cv_dt$BB_rating))
  model_out_dir <- file.path(out_root, model_id)
  dir.create(model_out_dir, recursive = TRUE, showWarnings = FALSE)
  log_line("Running %s\n", model_id)

  fold_ids <- sort(unique(cv_dt$cv_fold))
  fold_results <- lapply(fold_ids, function(fid) run_outer_fold(model_spec, fid, cv_dt, static_groups, target_levels))

  fold_metrics_dt <- rbindlist(lapply(fold_results, `[[`, "fold_metrics"), use.names = TRUE, fill = TRUE)
  fold_ordinal_dt <- rbindlist(lapply(fold_results, `[[`, "ordinal_metrics"), use.names = TRUE, fill = TRUE)
  outer_param_dt <- rbindlist(lapply(fold_results, `[[`, "params"), use.names = TRUE, fill = TRUE)
  cv_predictions_dt <- rbindlist(lapply(fold_results, `[[`, "predictions"), use.names = TRUE, fill = TRUE)
  cv_confusion_dt <- rbindlist(lapply(fold_results, `[[`, "confusion"), use.names = TRUE, fill = TRUE)
  cv_per_class_dt <- rbindlist(lapply(fold_results, `[[`, "per_class"), use.names = TRUE, fill = TRUE)
  selected_features_dt <- rbindlist(lapply(fold_results, `[[`, "selected_features"), use.names = TRUE, fill = TRUE)
  cv_smote_before_dt <- rbindlist(lapply(fold_results, `[[`, "smote_before"), use.names = TRUE, fill = TRUE)
  cv_smote_after_dt <- rbindlist(lapply(fold_results, `[[`, "smote_after"), use.names = TRUE, fill = TRUE)
  cv_summary_dt <- summarize_cv_metrics(fold_metrics_dt)
  cv_ordinal_summary_dt <- summarize_cv_ordinal_metrics(fold_ordinal_dt)

  final_res <- run_final_holdout(model_spec, train_dt, test_dt, static_groups, outer_param_dt, selected_features_dt, target_levels)
  summary_row <- collect_summary_row(model_spec, cv_summary_dt, cv_ordinal_summary_dt, final_res$metrics, final_res$ordinal_metrics, final_res$params, nrow(final_res$used_features))

  fwrite(model_spec, file.path(model_out_dir, "model_spec.csv"))
  fwrite(fold_metrics_dt, file.path(model_out_dir, "cv_fold_metrics.csv"))
  fwrite(fold_ordinal_dt, file.path(model_out_dir, "cv_fold_ordinal_metrics.csv"))
  fwrite(cv_summary_dt, file.path(model_out_dir, "cv_metric_summary.csv"))
  fwrite(cv_ordinal_summary_dt, file.path(model_out_dir, "cv_ordinal_metric_summary.csv"))
  fwrite(outer_param_dt, file.path(model_out_dir, "cv_selected_params_per_fold.csv"))
  fwrite(cv_predictions_dt, file.path(model_out_dir, "cv_predictions.csv"))
  fwrite(cv_confusion_dt, file.path(model_out_dir, "cv_confusion_matrix_by_fold.csv"))
  fwrite(cv_per_class_dt, file.path(model_out_dir, "cv_per_class_metrics_by_fold.csv"))
  fwrite(selected_features_dt, file.path(model_out_dir, "cv_selected_features_by_fold.csv"))
  fwrite(cv_smote_before_dt, file.path(model_out_dir, "cv_smote_counts_before_by_fold.csv"))
  fwrite(cv_smote_after_dt, file.path(model_out_dir, "cv_smote_counts_after_by_fold.csv"))
  fwrite(final_res$params, file.path(model_out_dir, "final_model_parameters.csv"))
  fwrite(final_res$metrics, file.path(model_out_dir, "final_holdout_metrics.csv"))
  fwrite(final_res$ordinal_metrics, file.path(model_out_dir, "final_holdout_ordinal_metrics.csv"))
  fwrite(final_res$predictions, file.path(model_out_dir, "final_holdout_predictions.csv"))
  fwrite(final_res$confusion, file.path(model_out_dir, "final_holdout_confusion_matrix.csv"))
  fwrite(final_res$per_class, file.path(model_out_dir, "final_holdout_per_class_metrics.csv"))
  fwrite(final_res$used_features, file.path(model_out_dir, "final_model_used_features.csv"))
  fwrite(final_res$smote_before, file.path(model_out_dir, "final_smote_counts_before.csv"))
  fwrite(final_res$smote_after, file.path(model_out_dir, "final_smote_counts_after.csv"))
  fwrite(summary_row, file.path(model_out_dir, "model_summary_row.csv"))

  summary_lines <- c(
    sprintf("Model ID: %s", model_id),
    sprintf("Model name: %s", model_spec$model_name[[1]]),
    sprintf("Algorithm: %s", model_spec$algorithm[[1]]),
    sprintf("Severity categories: %s", as.character(model_spec$severity_categories[[1]])),
    sprintf("Feature set: %s", model_spec$feature_label[[1]]),
    "",
    "Cross-validation summary (5 grouped folds on training set):",
    capture.output(print(cv_summary_dt)),
    "",
    "Cross-validation ordinal summary:",
    capture.output(print(cv_ordinal_summary_dt)),
    "",
    "Final holdout metrics:",
    capture.output(print(final_res$metrics)),
    "",
    "Final holdout ordinal metrics:",
    capture.output(print(final_res$ordinal_metrics)),
    "",
    "Final model parameters:",
    capture.output(print(final_res$params))
  )
  writeLines(summary_lines, file.path(model_out_dir, "summary.txt"))

  feature_rows <- copy(final_res$used_features)
  feature_rows[, `:=`(
    model_id = model_spec$model_id[[1]],
    model_name = model_spec$model_name[[1]],
    feature_label = model_spec$feature_label[[1]],
    severity_categories = model_spec$severity_categories[[1]],
    algorithm = model_spec$algorithm[[1]]
  )]
  setcolorder(feature_rows, c("model_id", "model_name", "feature_label", "severity_categories", "algorithm", "rank", "feature"))

  list(summary_row = summary_row, final_params = final_res$params, feature_rows = feature_rows)
}

spec_dt <- fread(spec_file)
spec_dt <- spec_dt[is.na(enabled) | enabled == 1]
if ("severity_categories" %in% names(spec_dt)) spec_dt[, severity_categories := as.integer(severity_categories)]
model_filter <- trimws(strsplit(Sys.getenv("MODEL_IDS", ""), ",", fixed = TRUE)[[1]])
model_filter <- model_filter[nzchar(model_filter)]
if (length(model_filter) > 0L) spec_dt <- spec_dt[model_id %in% model_filter]
if (nrow(spec_dt) == 0L) stop("No models selected from spec file: ", spec_file)
required_spec_cols <- c("model_id", "model_name", "severity_categories", "algorithm", "feature_key", "feature_label")
missing_spec <- setdiff(required_spec_cols, names(spec_dt))
if (length(missing_spec) > 0L) stop("Missing required spec columns: ", paste(missing_spec, collapse = ", "))

log_line("Using output root: %s\n", out_root)
log_line("Using spec file: %s\n", spec_file)
log_line("Selected %d models\n", nrow(spec_dt))

four_train_dt <- fread(paths$four_train)
four_test_dt <- fread(paths$four_test)
six_train_dt <- fread(paths$six_train)
six_test_dt <- fread(paths$six_test)

for (dt in list(four_train_dt, four_test_dt, six_train_dt, six_test_dt)) {
  dt[, unit_norm := as.character(unit_norm)]
  if ("source_file" %in% names(dt)) dt[, source_file := as.character(source_file)]
  dt[, BB_rating := as.integer(BB_rating)]
}

four_train_dt <- prepare_severity_labels(four_train_dt, 4L)
four_test_dt <- prepare_severity_labels(four_test_dt, 4L)
six_train_dt <- prepare_severity_labels(six_train_dt, 6L)
six_test_dt <- prepare_severity_labels(six_test_dt, 6L)

four_cv_dt <- attach_cv_folds(four_train_dt, paths$four_cv_assign, seed = 123L, severity_tag = "fourclass")
six_cv_dt <- attach_cv_folds(six_train_dt, paths$six_cv_assign, seed = 124L, severity_tag = "sixclass")

data_map <- list(
  four_cv = four_cv_dt,
  four_train = four_train_dt,
  four_test = four_test_dt,
  six_cv = six_cv_dt,
  six_train = six_train_dt,
  six_test = six_test_dt
)

summary_rows <- vector("list", nrow(spec_dt))
param_rows <- vector("list", nrow(spec_dt))
feature_rows <- vector("list", nrow(spec_dt))
for (i in seq_len(nrow(spec_dt))) {
  spec_row <- spec_dt[i]
  res <- run_one_model(spec_row, data_map)
  summary_rows[[i]] <- res$summary_row
  param_rows[[i]] <- res$final_params
  feature_rows[[i]] <- res$feature_rows
}

summary_dt <- rbindlist(summary_rows, use.names = TRUE, fill = TRUE)
final_params_dt <- rbindlist(param_rows, use.names = TRUE, fill = TRUE)
feature_sets_dt <- rbindlist(feature_rows, use.names = TRUE, fill = TRUE)

fwrite(summary_dt, file.path(out_root, "table2_model_summary_detailed.csv"))
fwrite(summary_dt, file.path(out_root, "revised_table2_rf_svm_cv_holdout_summary.csv"))
fwrite(final_params_dt, file.path(out_root, "table2_model_final_parameters.csv"))
fwrite(final_params_dt, file.path(out_root, "revised_table2_rf_svm_final_parameters.csv"))
fwrite(feature_sets_dt, file.path(out_root, "table2_model_feature_sets.csv"))

readme_lines <- c(
  "Table 2 grouped-CV + untouched holdout model outputs",
  "",
  "What this script does:",
  "1. Uses grouped 5-fold cross-validation on the 90% training partition only.",
  "2. Keeps all rows from the same microplot/unit in the same fold.",
  "3. Applies SMOTE only within training data, never to held-out validation or test data.",
  "4. Tunes model parameters inside each outer fold.",
  "5. Saves fold-level metrics, ordinal metrics, mean, standard deviation, and 95% confidence intervals.",
  "6. Chooses consensus parameters across outer folds.",
  "7. Trains a final model on the full 90% training set and evaluates once on the untouched 10% holdout test set.",
  "8. Saves feature sets, final parameters, and manuscript-table-ready summary inputs.",
  "",
  sprintf("Model spec file used: %s", spec_file),
  sprintf("FOUR_TRAIN_PATH: %s", paths$four_train),
  sprintf("FOUR_TEST_PATH: %s", paths$four_test),
  sprintf("SIX_TRAIN_PATH: %s", paths$six_train),
  sprintf("SIX_TEST_PATH: %s", paths$six_test),
  sprintf("FOUR_CV_ASSIGN_PATH: %s", paths$four_cv_assign),
  sprintf("SIX_CV_ASSIGN_PATH: %s", paths$six_cv_assign)
)
writeLines(readme_lines, file.path(out_root, "README.txt"))

log_line("Saved detailed summary to %s\n", file.path(out_root, "table2_model_summary_detailed.csv"))
