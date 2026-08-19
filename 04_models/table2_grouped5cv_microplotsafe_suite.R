#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(data.table)
  library(e1071)
  library(ranger)
  library(FNN)
  library(xgboost)
  library(parallel)
})

set.seed(123)
setDTthreads(0)

base_dir <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)
out_root <- file.path(base_dir, "jani_stuff", "table2_grouped5cv_microplotsafe_reruns")
shared_dir <- file.path(out_root, "shared")
dir.create(out_root, showWarnings = FALSE, recursive = TRUE)
dir.create(shared_dir, showWarnings = FALSE, recursive = TRUE)

script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_path <- if (length(script_arg) > 0L) normalizePath(sub("^--file=", "", script_arg[1]), winslash = "/", mustWork = FALSE) else NA_character_
skip_avg_build <- identical(Sys.getenv("SKIP_AVG_BUILD", "0"), "1")
model_scope <- tolower(Sys.getenv("MODEL_SCOPE", "all"))
force_avg_rebuild <- identical(Sys.getenv("FORCE_AVG_REBUILD", "0"), "1")
force_rerun_model_ids <- trimws(strsplit(Sys.getenv("FORCE_RERUN_MODEL_IDS", ""), ",", fixed = TRUE)[[1]])
force_rerun_model_ids <- force_rerun_model_ids[nzchar(force_rerun_model_ids)]
only_model_ids <- trimws(strsplit(Sys.getenv("ONLY_MODEL_IDS", ""), ",", fixed = TRUE)[[1]])
only_model_ids <- only_model_ids[nzchar(only_model_ids)]

raw_vi_cols <- c(
  "ndvi", "sri", "psri", "ipvi", "gb_ndvi", "gr_ndvi", "hue", "npci",
  "greenness", "gndvi", "bndvi", "grvi", "gli", "vari", "ngbi",
  "rgri", "gi2", "blb"
)
mean_vi_cols <- c(
  "NDVI_mean", "SRI_mean", "PSRI_mean", "IPVI_mean", "GB_NDVI_mean",
  "GR_NDVI_mean", "HUE_mean", "NPCI_mean", "GREENNESS_mean", "GNDVI_mean",
  "BNDVI_mean", "GRVI_mean", "GLI_mean", "VARI_mean", "NGBI_mean",
  "RGRI_mean", "GI2_mean", "BLB_mean"
)

# Add the shared de-identified complete dataset here if rerunning this legacy workflow.
# The 4-class labels are derived internally from the original 6-class BB_rating column.
# The helper files below are placeholders showing where the corresponding fold-assignment
# and top-feature-list files should be added if this legacy suite is reused.
paths <- list(
  combined = file.path(base_dir, "input_data", "complete_dataset.csv"),
  four_folds = file.path(base_dir, "jani_stuff", "legacy_fourclass_cv_fold_assignments.csv"),
  top30_exact = file.path(base_dir, "jani_stuff", "legacy_top30_feature_list.csv")
)

for (p in paths) {
  if (!file.exists(p)) stop("Missing required input: ", p)
}

safe_unique <- function(dt) {
  unique(dt, by = names(dt))
}

collapse_to_fourclass <- function(rating_vec) {
  out <- as.integer(rating_vec)
  out[out %in% c(1L, 3L)] <- 3L
  out[out %in% c(7L, 9L)] <- 9L
  out
}

sanitize_feature_frame <- function(dt, feature_cols) {
  if (length(feature_cols) == 0L) return(dt)
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

find_needed_raw_files <- function(dt4, dt6) {
  req <- unique(rbindlist(list(
    dt4[, .(folder, source_file)],
    dt6[, .(folder, source_file)]
  ), use.names = TRUE, fill = TRUE))
  req[!is.na(folder) & !is.na(source_file)]
}

summarize_avg_vi_one <- function(folder, source_file) {
  out <- data.table(folder = folder, source_file = source_file, file_exists = FALSE)
  for (col in mean_vi_cols) out[, (col) := NA_real_]
  path <- file.path(base_dir, folder, source_file)
  if (!file.exists(path)) return(out)
  dt <- tryCatch(
    fread(path, select = raw_vi_cols, showProgress = FALSE),
    error = function(e) NULL
  )
  if (is.null(dt) || nrow(dt) == 0L) return(out)
  finite_means <- vapply(dt, function(x) {
    vals <- suppressWarnings(as.numeric(x))
    vals <- vals[is.finite(vals)]
    if (length(vals) == 0L) return(NA_real_)
    mean(vals)
  }, numeric(1))
  vals <- setNames(as.list(as.numeric(finite_means)), mean_vi_cols)
  for (nm in names(vals)) out[, (nm) := vals[[nm]]]
  out[, file_exists := TRUE]
  out
}

build_average_vi_cache <- function(req_dt, cache_path) {
  if (force_avg_rebuild && file.exists(cache_path)) {
    file.remove(cache_path)
  }

  env_cores <- suppressWarnings(as.integer(Sys.getenv("AVG_CACHE_CORES", "")))
  core_guess <- suppressWarnings(parallel::detectCores(logical = TRUE))
  if (!is.na(env_cores) && length(env_cores) > 0L && env_cores >= 1L) {
    cores <- env_cores
  } else if (length(core_guess) == 0L || is.na(core_guess) || core_guess < 2L) {
    cores <- 1L
  } else {
    cores <- min(4L, core_guess - 1L)
  }

  if (file.exists(cache_path)) {
    done_dt <- fread(cache_path)
    done_keys <- unique(done_dt[, .(folder, source_file)])
    req_dt <- req_dt[!done_keys, on = .(folder, source_file)]
    if (nrow(req_dt) == 0L) return(done_dt)
    cat(sprintf("Resuming average VI cache build: %d files remaining\n", nrow(req_dt)))
  }

  chunk_size <- 100L
  total_remaining <- nrow(req_dt)

  for (start_idx in seq(1L, total_remaining, by = chunk_size)) {
    end_idx <- min(start_idx + chunk_size - 1L, total_remaining)
    chunk <- req_dt[start_idx:end_idx]
    tasks <- split(chunk, seq_len(nrow(chunk)))

    if (cores > 1L) {
      res <- mclapply(tasks, function(x) summarize_avg_vi_one(x$folder[[1]], x$source_file[[1]]), mc.cores = cores)
    } else {
      res <- lapply(tasks, function(x) summarize_avg_vi_one(x$folder[[1]], x$source_file[[1]]))
    }

    chunk_dt <- rbindlist(res, use.names = TRUE, fill = TRUE)
    if (file.exists(cache_path)) {
      fwrite(chunk_dt, cache_path, append = TRUE)
    } else {
      fwrite(chunk_dt, cache_path)
    }
    cat(sprintf("Average VI cache progress: %d/%d files\n", end_idx, total_remaining))
  }

  fread(cache_path)
}

choose_best_svm_params <- function(train_dt, feature_cols, target_levels, outer_fold_id) {
  inner_groups <- make_group_folds(train_dt$unit_norm, train_dt$BB_rating, k = 3L, seed = 1000L + outer_fold_id)
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

copy_existing_bundle <- function(src_dir, dest_dir, model_id, model_name) {
  if (!dir.exists(src_dir)) stop("Missing existing bundle to copy: ", src_dir)
  dir.create(dest_dir, recursive = TRUE, showWarnings = FALSE)
  file.copy(list.files(src_dir, full.names = TRUE), dest_dir, recursive = TRUE, overwrite = TRUE)
  writeLines(c(
    model_name,
    "",
    "This folder was copied from an already-completed grouped 5-fold CV rerun.",
    paste("Source bundle:", src_dir)
  ), file.path(dest_dir, "README_table2_copy.txt"))

  metrics_file <- list.files(dest_dir, pattern = "_metrics\\.csv$", full.names = TRUE)
  if (length(metrics_file) == 0L) return(NULL)
  met <- fread(metrics_file[1])
  data.table(
    model_id = model_id,
    model_name = model_name,
    algorithm = if (grepl("svm", model_id, ignore.case = TRUE)) "SVM" else "RandomForest",
    severity_categories = 4L,
    feature_label = if (grepl("top30", model_id, ignore.case = TRUE)) "cluster ratio + binned VI (XGB top30) + structural" else "cluster ratio + binned VI + structural traits",
    status = "copied_existing_bundle",
    output_dir = dest_dir,
    accuracy = met$accuracy[1],
    precision_weighted = met$precision_weighted[1],
    recall_weighted = met$recall_weighted[1],
    f1_score = met$f1_score[1],
    macro_f1 = met$macro_f1[1]
  )
}

load_existing_model_result <- function(cfg) {
  if (cfg$model_id %in% force_rerun_model_ids) return(NULL)
  model_dir <- file.path(out_root, cfg$model_id)
  metrics_path <- file.path(model_dir, paste0(cfg$model_id, "_metrics.csv"))
  if (!file.exists(metrics_path)) return(NULL)

  met <- fread(metrics_path)
  data.table(
    model_id = cfg$model_id,
    model_name = cfg$model_name,
    algorithm = cfg$algorithm,
    severity_categories = cfg$severity_categories,
    feature_label = cfg$feature_label,
    status = "existing_result",
    output_dir = model_dir,
    accuracy = met$accuracy[1],
    precision_weighted = met$precision_weighted[1],
    recall_weighted = met$recall_weighted[1],
    f1_score = met$f1_score[1],
    macro_f1 = met$macro_f1[1]
  )
}

run_svm_model <- function(dt, fold_dt, cfg) {
  model_dir <- file.path(out_root, cfg$model_id)
  dir.create(model_dir, recursive = TRUE, showWarnings = FALSE)
  cat(sprintf("Starting model: %s\n", cfg$model_id))

  feature_cols <- cfg$feature_cols
  required_cols <- c("unit_norm", "timestamp_chr", "BB_rating", feature_cols)
  work_dt <- copy(dt)
  work_dt <- sanitize_feature_frame(work_dt, feature_cols)
  work_dt <- work_dt[complete.cases(work_dt[, ..required_cols])]
  work_dt <- merge(work_dt, fold_dt[, .(group, fold)], by.x = "unit_norm", by.y = "group", all.x = TRUE, sort = FALSE)
  if (any(is.na(work_dt$fold))) stop("Missing fold assignments for model ", cfg$model_id)
  setorder(work_dt, fold, unit_norm, timestamp_chr)

  target_levels <- sort(unique(work_dt$BB_rating))
  pred_parts <- list()
  fold_metrics <- list()
  param_rows <- list()
  overlap_checks <- character()
  smote_summaries <- character()

  for (outer_fold in sort(unique(work_dt$fold))) {
    cat(sprintf("  %s outer fold %d/5\n", cfg$model_id, outer_fold))
    train_fold <- work_dt[fold != outer_fold]
    test_fold <- work_dt[fold == outer_fold]

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
      pred <- predict(
        svm_fit,
        newdata = as.data.frame(scaled$test)
      )
    }
    fold_pred <- data.table(
      fold = outer_fold,
      unit_norm = test_fold$unit_norm,
      timestamp_chr = test_fold$timestamp_chr,
      folder = test_fold$folder,
      source_file = test_fold$source_file,
      genotype = if ("genotype" %in% names(test_fold)) test_fold$genotype else NA_character_,
      g_alias = if ("g_alias" %in% names(test_fold)) test_fold$g_alias else NA_character_,
      actual_BLB_rating = test_fold$BB_rating,
      predicted_BLB_rating = as.integer(as.character(pred))
    )
    pred_parts[[length(pred_parts) + 1L]] <- fold_pred

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
  overall <- compute_metrics(pred_dt$actual_BLB_rating, pred_dt$predicted_BLB_rating, sort(unique(work_dt$BB_rating)))

  overall_dt <- data.table(
    evaluation = "grouped_5fold_cv_oof",
    n_rows = nrow(work_dt),
    n_groups = uniqueN(work_dt$unit_norm),
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
  input_keep <- unique(c("unit_norm", "timestamp_chr", "folder", "source_file", "BB_rating", "genotype", "g_alias", "treatment", feature_cols))
  input_keep <- input_keep[input_keep %in% names(work_dt)]
  input_dt <- safe_unique(work_dt[, ..input_keep])

  fwrite(input_dt, file.path(model_dir, paste0(cfg$model_id, "_input_dataset.csv")))
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
    fold_assignment_source = basename(cfg$fold_path),
    smote_strategy = "SMOTE within training folds only",
    source_script = basename(script_path)
  ), file.path(model_dir, paste0(cfg$model_id, "_config.csv")))

  summary_lines <- c(
    cfg$model_name,
    "",
    "Evaluation: grouped 5-fold cross-validation by unit_norm",
    "SMOTE was applied only within each training fold.",
    "",
    sprintf("Rows analyzed: %d", nrow(work_dt)),
    sprintf("Unique microplots/groups: %d", uniqueN(work_dt$unit_norm)),
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

  if (!is.na(script_path) && file.exists(script_path)) file.copy(script_path, file.path(model_dir, basename(script_path)), overwrite = TRUE)
  cat(sprintf("Completed model: %s\n", cfg$model_id))

  data.table(
    model_id = cfg$model_id,
    model_name = cfg$model_name,
    algorithm = "SVM",
    severity_categories = cfg$severity_categories,
    feature_label = cfg$feature_label,
    status = "completed",
    output_dir = model_dir,
    accuracy = overall$accuracy,
    precision_weighted = overall$precision_weighted,
    recall_weighted = overall$recall_weighted,
    f1_score = overall$f1_weighted,
    macro_f1 = overall$macro_f1
  )
}

run_rf_model <- function(dt, fold_dt, cfg) {
  model_dir <- file.path(out_root, cfg$model_id)
  dir.create(model_dir, recursive = TRUE, showWarnings = FALSE)
  cat(sprintf("Starting model: %s\n", cfg$model_id))

  feature_cols <- cfg$feature_cols
  required_cols <- c("unit_norm", "timestamp_chr", "BB_rating", feature_cols)
  work_dt <- copy(dt)
  work_dt <- sanitize_feature_frame(work_dt, feature_cols)
  work_dt <- work_dt[complete.cases(work_dt[, ..required_cols])]
  work_dt <- merge(work_dt, fold_dt[, .(group, fold)], by.x = "unit_norm", by.y = "group", all.x = TRUE, sort = FALSE)
  if (any(is.na(work_dt$fold))) stop("Missing fold assignments for model ", cfg$model_id)
  setorder(work_dt, fold, unit_norm, timestamp_chr)

  target_levels <- sort(unique(work_dt$BB_rating))
  pred_parts <- list()
  fold_metrics <- list()
  importance_parts <- list()
  overlap_checks <- character()
  smote_summaries <- character()

  for (outer_fold in sort(unique(work_dt$fold))) {
    cat(sprintf("  %s outer fold %d/5\n", cfg$model_id, outer_fold))
    train_fold <- work_dt[fold != outer_fold]
    test_fold <- work_dt[fold == outer_fold]

    overlap_n <- length(intersect(unique(train_fold$unit_norm), unique(test_fold$unit_norm)))
    overlap_checks <- c(overlap_checks, sprintf("Fold %d unit overlap: %d", outer_fold, overlap_n))
    if (overlap_n != 0L) stop("Group leakage detected in fold ", outer_fold, " for ", cfg$model_id)

    train_ml <- train_fold[, c("BB_rating", feature_cols), with = FALSE]
    sm <- apply_smote(train_ml, target_col = "BB_rating", feature_cols = feature_cols, seed = 30000L + outer_fold)
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

    train_smote <- sm$data
    train_smote[, BB_rating := factor(BB_rating, levels = target_levels)]

    rf_fit <- ranger(
      BB_rating ~ .,
      data = train_smote,
      classification = TRUE,
      probability = FALSE,
      num.trees = 1000,
      mtry = max(1L, floor(sqrt(length(feature_cols)))),
      importance = "impurity",
      seed = 123 + outer_fold
    )

    pred <- predict(rf_fit, data = test_fold[, ..feature_cols])$predictions
    fold_pred <- data.table(
      fold = outer_fold,
      unit_norm = test_fold$unit_norm,
      timestamp_chr = test_fold$timestamp_chr,
      folder = test_fold$folder,
      source_file = test_fold$source_file,
      genotype = if ("genotype" %in% names(test_fold)) test_fold$genotype else NA_character_,
      g_alias = if ("g_alias" %in% names(test_fold)) test_fold$g_alias else NA_character_,
      actual_BLB_rating = test_fold$BB_rating,
      predicted_BLB_rating = as.integer(as.character(pred))
    )
    pred_parts[[length(pred_parts) + 1L]] <- fold_pred

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

    importance_parts[[length(importance_parts) + 1L]] <- data.table(
      fold = outer_fold,
      Feature = names(rf_fit$variable.importance),
      Importance = as.numeric(rf_fit$variable.importance)
    )
  }

  pred_dt <- rbindlist(pred_parts, use.names = TRUE, fill = TRUE)
  fold_metrics_dt <- rbindlist(fold_metrics, use.names = TRUE, fill = TRUE)
  importance_dt <- rbindlist(importance_parts, use.names = TRUE, fill = TRUE)
  importance_mean_dt <- importance_dt[, .(
    mean_importance = mean(Importance),
    sd_importance = sd(Importance)
  ), by = Feature][order(-mean_importance)]
  overall <- compute_metrics(pred_dt$actual_BLB_rating, pred_dt$predicted_BLB_rating, target_levels)

  overall_dt <- data.table(
    evaluation = "grouped_5fold_cv_oof",
    n_rows = nrow(work_dt),
    n_groups = uniqueN(work_dt$unit_norm),
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
  input_keep <- unique(c("unit_norm", "timestamp_chr", "folder", "source_file", "BB_rating", "genotype", "g_alias", "treatment", feature_cols))
  input_keep <- input_keep[input_keep %in% names(work_dt)]
  input_dt <- safe_unique(work_dt[, ..input_keep])

  fwrite(input_dt, file.path(model_dir, paste0(cfg$model_id, "_input_dataset.csv")))
  fwrite(used_features_dt, file.path(model_dir, paste0(cfg$model_id, "_used_features.csv")))
  fwrite(overall_dt, file.path(model_dir, paste0(cfg$model_id, "_metrics.csv")))
  fwrite(fold_metrics_dt, file.path(model_dir, paste0(cfg$model_id, "_fold_metrics.csv")))
  fwrite(pred_dt, file.path(model_dir, paste0(cfg$model_id, "_oof_predictions.csv")))
  fwrite(build_confusion_dt(overall$confusion), file.path(model_dir, paste0(cfg$model_id, "_confusion_matrix.csv")))
  fwrite(importance_mean_dt, file.path(model_dir, paste0(cfg$model_id, "_feature_importance.csv")))
  fwrite(data.table(
    model_id = cfg$model_id,
    model_name = cfg$model_name,
    algorithm = "RandomForest",
    severity_categories = cfg$severity_categories,
    feature_label = cfg$feature_label,
    feature_count = length(feature_cols),
    fold_assignment_source = basename(cfg$fold_path),
    smote_strategy = "SMOTE within training folds only",
    num_trees = 1000L,
    mtry = max(1L, floor(sqrt(length(feature_cols)))),
    source_script = basename(script_path)
  ), file.path(model_dir, paste0(cfg$model_id, "_config.csv")))

  summary_lines <- c(
    cfg$model_name,
    "",
    "Evaluation: grouped 5-fold cross-validation by unit_norm",
    "SMOTE was applied only within each training fold.",
    "",
    sprintf("Rows analyzed: %d", nrow(work_dt)),
    sprintf("Unique microplots/groups: %d", uniqueN(work_dt$unit_norm)),
    sprintf("Feature count: %d", length(feature_cols)),
    "",
    "Leakage checks:",
    overlap_checks,
    "",
    "Fold-safe SMOTE summaries:",
    smote_summaries,
    "",
    "Fold metrics:",
    paste(capture.output(print(fold_metrics_dt)), collapse = "\n"),
    "",
    "Overall pooled out-of-fold metrics:",
    paste(capture.output(print(overall_dt)), collapse = "\n"),
    "",
    "Per-class metrics:",
    paste(capture.output(print(overall$per_class)), collapse = "\n"),
    "",
    "Top feature importances:",
    paste(capture.output(print(importance_mean_dt[1:min(30L, .N)])), collapse = "\n")
  )
  writeLines(summary_lines, file.path(model_dir, paste0(cfg$model_id, "_summary.txt")))

  if (!is.na(script_path) && file.exists(script_path)) file.copy(script_path, file.path(model_dir, basename(script_path)), overwrite = TRUE)
  cat(sprintf("Completed model: %s\n", cfg$model_id))

  data.table(
    model_id = cfg$model_id,
    model_name = cfg$model_name,
    algorithm = "RandomForest",
    severity_categories = cfg$severity_categories,
    feature_label = cfg$feature_label,
    status = "completed",
    output_dir = model_dir,
    accuracy = overall$accuracy,
    precision_weighted = overall$precision_weighted,
    recall_weighted = overall$recall_weighted,
    f1_score = overall$f1_weighted,
    macro_f1 = overall$macro_f1
  )
}

run_xgb_model <- function(dt, fold_dt, cfg) {
  model_dir <- file.path(out_root, cfg$model_id)
  dir.create(model_dir, recursive = TRUE, showWarnings = FALSE)
  cat(sprintf("Starting model: %s\n", cfg$model_id))

  feature_cols <- cfg$feature_cols
  required_cols <- c("unit_norm", "timestamp_chr", "BB_rating", feature_cols)
  work_dt <- copy(dt)
  work_dt <- sanitize_feature_frame(work_dt, feature_cols)
  work_dt <- work_dt[complete.cases(work_dt[, ..required_cols])]
  work_dt <- merge(work_dt, fold_dt[, .(group, fold)], by.x = "unit_norm", by.y = "group", all.x = TRUE, sort = FALSE)
  if (any(is.na(work_dt$fold))) stop("Missing fold assignments for model ", cfg$model_id)
  setorder(work_dt, fold, unit_norm, timestamp_chr)

  target_levels <- sort(unique(work_dt$BB_rating))
  label_map <- data.table(
    class_value = target_levels,
    xgb_label = seq_along(target_levels) - 1L
  )

  pred_parts <- list()
  fold_metrics <- list()
  importance_parts <- list()
  overlap_checks <- character()
  smote_summaries <- character()

  for (outer_fold in sort(unique(work_dt$fold))) {
    cat(sprintf("  %s outer fold %d/5\n", cfg$model_id, outer_fold))
    train_fold <- work_dt[fold != outer_fold]
    test_fold <- work_dt[fold == outer_fold]

    overlap_n <- length(intersect(unique(train_fold$unit_norm), unique(test_fold$unit_norm)))
    overlap_checks <- c(overlap_checks, sprintf("Fold %d unit overlap: %d", outer_fold, overlap_n))
    if (overlap_n != 0L) stop("Group leakage detected in fold ", outer_fold, " for ", cfg$model_id)

    train_ml <- train_fold[, c("BB_rating", feature_cols), with = FALSE]
    sm <- apply_smote(train_ml, target_col = "BB_rating", feature_cols = feature_cols, seed = 40000L + outer_fold)
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

    x_train <- as.matrix(sm$data[, ..feature_cols])
    x_test <- as.matrix(test_fold[, ..feature_cols])
    mode(x_train) <- "numeric"
    mode(x_test) <- "numeric"
    y_train <- label_map[match(sm$data$BB_rating, class_value), xgb_label]

    dtrain <- xgb.DMatrix(data = x_train, label = y_train, missing = NA_real_)
    dtest <- xgb.DMatrix(data = x_test, missing = NA_real_)
    xgb_fit <- xgb.train(
      params = list(
        objective = "multi:softprob",
        eval_metric = "mlogloss",
        num_class = length(target_levels),
        max_depth = 6,
        eta = 0.05,
        subsample = 0.9,
        colsample_bytree = 0.9,
        min_child_weight = 1
      ),
      data = dtrain,
      nrounds = 300,
      verbose = 0
    )

    pred_prob <- matrix(predict(xgb_fit, newdata = dtest), ncol = length(target_levels), byrow = TRUE)
    pred_idx <- max.col(pred_prob, ties.method = "first") - 1L
    pred_lab <- label_map[match(pred_idx, xgb_label), class_value]

    fold_pred <- data.table(
      fold = outer_fold,
      unit_norm = test_fold$unit_norm,
      timestamp_chr = test_fold$timestamp_chr,
      folder = test_fold$folder,
      source_file = test_fold$source_file,
      genotype = if ("genotype" %in% names(test_fold)) test_fold$genotype else NA_character_,
      g_alias = if ("g_alias" %in% names(test_fold)) test_fold$g_alias else NA_character_,
      actual_BLB_rating = test_fold$BB_rating,
      predicted_BLB_rating = pred_lab
    )
    pred_parts[[length(pred_parts) + 1L]] <- fold_pred

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

    imp <- xgb.importance(model = xgb_fit, feature_names = feature_cols)
    if (nrow(imp) > 0L) {
      importance_parts[[length(importance_parts) + 1L]] <- data.table(
        fold = outer_fold,
        Feature = as.character(imp$Feature),
        Gain = as.numeric(imp$Gain),
        Cover = as.numeric(imp$Cover),
        Frequency = as.numeric(imp$Frequency)
      )
    }
  }

  pred_dt <- rbindlist(pred_parts, use.names = TRUE, fill = TRUE)
  fold_metrics_dt <- rbindlist(fold_metrics, use.names = TRUE, fill = TRUE)
  overall <- compute_metrics(pred_dt$actual_BLB_rating, pred_dt$predicted_BLB_rating, target_levels)

  if (length(importance_parts) > 0L) {
    importance_dt <- rbindlist(importance_parts, use.names = TRUE, fill = TRUE)
    importance_mean_dt <- importance_dt[, .(
      mean_gain = mean(Gain),
      sd_gain = sd(Gain),
      mean_cover = mean(Cover),
      mean_frequency = mean(Frequency)
    ), by = Feature][order(-mean_gain)]
  } else {
    importance_mean_dt <- data.table()
  }

  overall_dt <- data.table(
    evaluation = "grouped_5fold_cv_oof",
    n_rows = nrow(work_dt),
    n_groups = uniqueN(work_dt$unit_norm),
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
  input_keep <- unique(c("unit_norm", "timestamp_chr", "folder", "source_file", "BB_rating", "genotype", "g_alias", "treatment", feature_cols))
  input_keep <- input_keep[input_keep %in% names(work_dt)]
  input_dt <- safe_unique(work_dt[, ..input_keep])

  fwrite(input_dt, file.path(model_dir, paste0(cfg$model_id, "_input_dataset.csv")))
  fwrite(used_features_dt, file.path(model_dir, paste0(cfg$model_id, "_used_features.csv")))
  fwrite(overall_dt, file.path(model_dir, paste0(cfg$model_id, "_metrics.csv")))
  fwrite(fold_metrics_dt, file.path(model_dir, paste0(cfg$model_id, "_fold_metrics.csv")))
  fwrite(pred_dt, file.path(model_dir, paste0(cfg$model_id, "_oof_predictions.csv")))
  fwrite(build_confusion_dt(overall$confusion), file.path(model_dir, paste0(cfg$model_id, "_confusion_matrix.csv")))
  if (nrow(importance_mean_dt) > 0L) fwrite(importance_mean_dt, file.path(model_dir, paste0(cfg$model_id, "_feature_importance.csv")))
  fwrite(data.table(
    model_id = cfg$model_id,
    model_name = cfg$model_name,
    algorithm = "XGBoost",
    severity_categories = cfg$severity_categories,
    feature_label = cfg$feature_label,
    feature_count = length(feature_cols),
    fold_assignment_source = basename(cfg$fold_path),
    smote_strategy = "SMOTE within training folds only",
    nrounds = 300L,
    max_depth = 6L,
    eta = 0.05,
    subsample = 0.9,
    colsample_bytree = 0.9,
    min_child_weight = 1,
    source_script = basename(script_path)
  ), file.path(model_dir, paste0(cfg$model_id, "_config.csv")))

  summary_lines <- c(
    cfg$model_name,
    "",
    "Evaluation: grouped 5-fold cross-validation by unit_norm",
    "SMOTE was applied only within each training fold.",
    "",
    sprintf("Rows analyzed: %d", nrow(work_dt)),
    sprintf("Unique microplots/groups: %d", uniqueN(work_dt$unit_norm)),
    sprintf("Feature count: %d", length(feature_cols)),
    "",
    "Leakage checks:",
    overlap_checks,
    "",
    "Fold-safe SMOTE summaries:",
    smote_summaries,
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
  if (nrow(importance_mean_dt) > 0L) {
    summary_lines <- c(summary_lines, "", "Top XGBoost feature importances:", paste(capture.output(print(importance_mean_dt[1:min(30L, .N)])), collapse = "\n"))
  }
  writeLines(summary_lines, file.path(model_dir, paste0(cfg$model_id, "_summary.txt")))

  if (!is.na(script_path) && file.exists(script_path)) file.copy(script_path, file.path(model_dir, basename(script_path)), overwrite = TRUE)
  cat(sprintf("Completed model: %s\n", cfg$model_id))

  data.table(
    model_id = cfg$model_id,
    model_name = cfg$model_name,
    algorithm = "XGBoost",
    severity_categories = cfg$severity_categories,
    feature_label = cfg$feature_label,
    status = "completed",
    output_dir = model_dir,
    accuracy = overall$accuracy,
    precision_weighted = overall$precision_weighted,
    recall_weighted = overall$recall_weighted,
    f1_score = overall$f1_weighted,
    macro_f1 = overall$macro_f1
  )
}

get_feature_cols <- function(dt, feature_key, top30_exact_features) {
  cluster_cols <- grep("^cluster_[0-9]+_ratio$", names(dt), value = TRUE)
  bin_cols <- grep("^[A-Z0-9_]+_bin[1-6]$", names(dt), value = TRUE)
  struct_cols <- grep("^structural_", names(dt), value = TRUE)
  avg_cols <- intersect(mean_vi_cols, names(dt))
  top30_exact <- intersect(top30_exact_features, names(dt))

  switch(
    feature_key,
    average_vi = avg_cols,
    binned_vi = bin_cols,
    cluster_ratio = cluster_cols,
    cluster_plus_binned = unique(c(cluster_cols, bin_cols)),
    cluster_plus_binned_plus_struct = unique(c(cluster_cols, bin_cols, struct_cols)),
    top30_exact = top30_exact,
    top30_exact_plus_struct = unique(c(top30_exact, struct_cols)),
    stop("Unknown feature key: ", feature_key)
  )
}

write_root_readme <- function(summary_dt) {
  lines <- c(
    "Table 2 grouped 5-fold microplot-safe reruns",
    "",
    "Design notes:",
    "- Evaluation uses grouped 5-fold cross-validation by unit_norm.",
    "- All observations from the same microplot remain in the same fold across dates.",
    "- SMOTE is applied only within each training fold.",
    "- Four-class fold assignments were reused from the earlier grouped reruns.",
    "- Six-class fold assignments were generated once and reused across all six-class models.",
    "",
    "This folder contains one subfolder per Table 2 model."
  )
  writeLines(lines, file.path(out_root, "README.txt"))
  fwrite(summary_dt, file.path(out_root, "table2_grouped5cv_results_summary.csv"))
}

six_dt <- safe_unique(fread(paths$combined))
six_dt[, `:=`(unit_norm = as.character(unit_norm), timestamp_chr = as.character(timestamp_chr), BB_rating = as.integer(BB_rating))]
four_dt <- copy(six_dt)
four_dt[, BB_rating := collapse_to_fourclass(BB_rating)]

needed_files_dt <- find_needed_raw_files(four_dt, six_dt)
avg_cache_path <- file.path(shared_dir, "average_vi_from_raw_pointclouds.csv")
if (skip_avg_build) {
  if (model_scope == "nonaverage") {
    avg_vi_dt <- data.table(folder = character(), source_file = character())
    for (col in mean_vi_cols) avg_vi_dt[, (col) := numeric()]
  } else if (file.exists(avg_cache_path)) {
    avg_vi_dt <- fread(avg_cache_path)
  } else {
    avg_vi_dt <- data.table(folder = character(), source_file = character(), file_exists = logical())
    for (col in mean_vi_cols) avg_vi_dt[, (col) := numeric()]
  }
} else {
  avg_vi_dt <- build_average_vi_cache(needed_files_dt, avg_cache_path)
}

four_full_dt <- merge(four_dt, avg_vi_dt[, c("folder", "source_file", mean_vi_cols), with = FALSE], by = c("folder", "source_file"), all.x = TRUE, sort = FALSE)
six_full_dt <- merge(six_dt, avg_vi_dt[, c("folder", "source_file", mean_vi_cols), with = FALSE], by = c("folder", "source_file"), all.x = TRUE, sort = FALSE)
fwrite(four_full_dt, file.path(shared_dir, "fourclass_full_with_average_vi.csv"))
fwrite(six_full_dt, file.path(shared_dir, "sixclass_full_with_average_vi.csv"))

four_fold_dt <- fread(paths$four_folds)
four_fold_dt[, group := as.character(group)]
file.copy(paths$four_folds, file.path(shared_dir, basename(paths$four_folds)), overwrite = TRUE)

six_fold_path <- file.path(shared_dir, "sixclass_grouped5cv_fold_assignments.csv")
if (file.exists(six_fold_path)) {
  six_fold_dt <- fread(six_fold_path)
} else {
  six_fold_dt <- make_group_folds(six_full_dt$unit_norm, six_full_dt$BB_rating, k = 5L, seed = 123L)
  fwrite(six_fold_dt, six_fold_path)
}
six_fold_dt[, group := as.character(group)]

top30_exact_features <- as.character(fread(paths$top30_exact)[[1]])
fwrite(data.table(feature = top30_exact_features), file.path(shared_dir, "top30_exact_feature_list.csv"))

existing_copy_rows <- list()
existing_copy_rows[[1]] <- copy_existing_bundle(
  file.path(base_dir, "jani_stuff", "svm_4class_grouped5cv_clusters_binned_structural_results"),
  file.path(out_root, "4class_cluster_ratio_plus_binned_vi_plus_structural_svm"),
  "4class_cluster_ratio_plus_binned_vi_plus_structural_svm",
  "4-class cluster ratio + binned VI + structural traits SVM"
)
existing_copy_rows[[2]] <- copy_existing_bundle(
  file.path(base_dir, "jani_stuff", "rf_4class_grouped5cv_clusters_binned_structural_results"),
  file.path(out_root, "4class_cluster_ratio_plus_binned_vi_plus_structural_randomforest"),
  "4class_cluster_ratio_plus_binned_vi_plus_structural_randomforest",
  "4-class cluster ratio + binned VI + structural traits RandomForest"
)

model_configs <- rbindlist(list(
  data.table(model_id = "6class_binned_vi_randomforest", model_name = "6-class binned VI RandomForest", algorithm = "RandomForest", severity_categories = 6L, feature_key = "binned_vi", feature_label = "binned VI", dataset_tag = "6class"),
  data.table(model_id = "6class_binned_vi_svm", model_name = "6-class binned VI SVM", algorithm = "SVM", severity_categories = 6L, feature_key = "binned_vi", feature_label = "binned VI", dataset_tag = "6class"),
  data.table(model_id = "6class_binned_vi_xgboost", model_name = "6-class binned VI XGBoost", algorithm = "XGBoost", severity_categories = 6L, feature_key = "binned_vi", feature_label = "binned VI", dataset_tag = "6class"),
  data.table(model_id = "6class_cluster_ratio_plus_binned_vi_randomforest", model_name = "6-class cluster ratio + binned VI RandomForest", algorithm = "RandomForest", severity_categories = 6L, feature_key = "cluster_plus_binned", feature_label = "cluster ratio + binned VI", dataset_tag = "6class"),
  data.table(model_id = "6class_cluster_ratio_plus_binned_vi_svm", model_name = "6-class cluster ratio + binned VI SVM", algorithm = "SVM", severity_categories = 6L, feature_key = "cluster_plus_binned", feature_label = "cluster ratio + binned VI", dataset_tag = "6class"),
  data.table(model_id = "6class_average_vi_randomforest", model_name = "6-class average VI RandomForest", algorithm = "RandomForest", severity_categories = 6L, feature_key = "average_vi", feature_label = "average VI", dataset_tag = "6class"),
  data.table(model_id = "6class_average_vi_svm", model_name = "6-class average VI SVM", algorithm = "SVM", severity_categories = 6L, feature_key = "average_vi", feature_label = "average VI", dataset_tag = "6class"),
  data.table(model_id = "4class_binned_vi_randomforest", model_name = "4-class binned VI RandomForest", algorithm = "RandomForest", severity_categories = 4L, feature_key = "binned_vi", feature_label = "binned VI", dataset_tag = "4class"),
  data.table(model_id = "4class_binned_vi_svm", model_name = "4-class binned VI SVM", algorithm = "SVM", severity_categories = 4L, feature_key = "binned_vi", feature_label = "binned VI", dataset_tag = "4class"),
  data.table(model_id = "4class_binned_vi_xgboost", model_name = "4-class binned VI XGBoost", algorithm = "XGBoost", severity_categories = 4L, feature_key = "binned_vi", feature_label = "binned VI", dataset_tag = "4class"),
  data.table(model_id = "4class_cluster_ratio_randomforest", model_name = "4-class cluster ratio RandomForest", algorithm = "RandomForest", severity_categories = 4L, feature_key = "cluster_ratio", feature_label = "cluster ratio", dataset_tag = "4class"),
  data.table(model_id = "4class_cluster_ratio_svm", model_name = "4-class cluster ratio SVM", algorithm = "SVM", severity_categories = 4L, feature_key = "cluster_ratio", feature_label = "cluster ratio", dataset_tag = "4class"),
  data.table(model_id = "4class_cluster_ratio_plus_binned_vi_randomforest", model_name = "4-class cluster ratio + binned VI RandomForest", algorithm = "RandomForest", severity_categories = 4L, feature_key = "cluster_plus_binned", feature_label = "cluster ratio + binned VI", dataset_tag = "4class"),
  data.table(model_id = "4class_cluster_ratio_plus_binned_vi_svm", model_name = "4-class cluster ratio + binned VI SVM", algorithm = "SVM", severity_categories = 4L, feature_key = "cluster_plus_binned", feature_label = "cluster ratio + binned VI", dataset_tag = "4class"),
  data.table(model_id = "4class_cluster_ratio_plus_binned_vi_xgb_top30_randomforest", model_name = "4-class cluster ratio + binned VI (XGB top30) RandomForest", algorithm = "RandomForest", severity_categories = 4L, feature_key = "top30_exact", feature_label = "cluster ratio + binned VI (XGB top30)", dataset_tag = "4class"),
  data.table(model_id = "4class_cluster_ratio_plus_binned_vi_xgb_top30_svm", model_name = "4-class cluster ratio + binned VI (XGB top30) SVM", algorithm = "SVM", severity_categories = 4L, feature_key = "top30_exact", feature_label = "cluster ratio + binned VI (XGB top30)", dataset_tag = "4class"),
  data.table(model_id = "4class_cluster_ratio_plus_binned_vi_xgb_top30_plus_structural_randomforest", model_name = "4-class cluster ratio + binned VI (XGB top30) + structural RandomForest", algorithm = "RandomForest", severity_categories = 4L, feature_key = "top30_exact_plus_struct", feature_label = "cluster ratio + binned VI (XGB top30) + structural", dataset_tag = "4class"),
  data.table(model_id = "4class_cluster_ratio_plus_binned_vi_xgb_top30_plus_structural_svm", model_name = "4-class cluster ratio + binned VI (XGB top30) + structural SVM", algorithm = "SVM", severity_categories = 4L, feature_key = "top30_exact_plus_struct", feature_label = "cluster ratio + binned VI (XGB top30) + structural", dataset_tag = "4class"),
  data.table(model_id = "4class_average_vi_randomforest", model_name = "4-class average VI RandomForest", algorithm = "RandomForest", severity_categories = 4L, feature_key = "average_vi", feature_label = "average VI", dataset_tag = "4class"),
  data.table(model_id = "4class_average_vi_svm", model_name = "4-class average VI SVM", algorithm = "SVM", severity_categories = 4L, feature_key = "average_vi", feature_label = "average VI", dataset_tag = "4class")
), use.names = TRUE)

if (model_scope == "nonaverage") {
  model_configs <- model_configs[feature_key != "average_vi"]
} else if (model_scope == "averageonly") {
  model_configs <- model_configs[feature_key == "average_vi"]
}
if (length(only_model_ids) > 0L) {
  model_configs <- model_configs[model_id %in% only_model_ids]
}

summary_rows <- list()
if (!is.null(existing_copy_rows[[1]])) summary_rows[[length(summary_rows) + 1L]] <- existing_copy_rows[[1]]
if (!is.null(existing_copy_rows[[2]])) summary_rows[[length(summary_rows) + 1L]] <- existing_copy_rows[[2]]

for (i in seq_len(nrow(model_configs))) {
  cfg <- as.list(model_configs[i])
  dt_use <- if (cfg$dataset_tag == "4class") four_full_dt else six_full_dt
  fold_use <- if (cfg$dataset_tag == "4class") four_fold_dt else six_fold_dt
  fold_path <- if (cfg$dataset_tag == "4class") paths$four_folds else six_fold_path
  feature_cols <- get_feature_cols(dt_use, cfg$feature_key, top30_exact_features)
  cfg$feature_cols <- feature_cols
  cfg$fold_path <- fold_path

  if (length(feature_cols) == 0L) stop("No features resolved for model ", cfg$model_id)

  existing_res <- load_existing_model_result(cfg)
  if (!is.null(existing_res)) {
    summary_rows[[length(summary_rows) + 1L]] <- existing_res
    next
  }

  if (cfg$algorithm == "SVM") {
    res <- run_svm_model(dt_use, fold_use, cfg)
  } else if (cfg$algorithm == "RandomForest") {
    res <- run_rf_model(dt_use, fold_use, cfg)
  } else if (cfg$algorithm == "XGBoost") {
    res <- run_xgb_model(dt_use, fold_use, cfg)
  } else {
    stop("Unsupported algorithm: ", cfg$algorithm)
  }

  summary_rows[[length(summary_rows) + 1L]] <- res
}

summary_dt <- rbindlist(summary_rows, use.names = TRUE, fill = TRUE)
fwrite(summary_dt, file.path(out_root, "table2_grouped5cv_results_summary.csv"))
write_root_readme(summary_dt)

cat("Completed Table 2 grouped 5-fold reruns in:\n")
cat(out_root, "\n")
cat("Models summarized:", nrow(summary_dt), "\n")
