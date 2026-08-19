#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(data.table)
  library(parallel)
})

set.seed(123)
setDTthreads(0)

base_dir <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)
input_dir <- file.path(base_dir, "jani_stuff", "grouped_holdout_80_20_microplotsafe_split")
analysis_root <- file.path(base_dir, "jani_stuff", "grouped_microplotsafe_rebinned_datasets")
cv_root <- file.path(analysis_root, "cv_fold_specific")
holdout_root <- file.path(analysis_root, "holdout_fulltrain_fit")

dir.create(analysis_root, recursive = TRUE, showWarnings = FALSE)
dir.create(cv_root, recursive = TRUE, showWarnings = FALSE)
dir.create(holdout_root, recursive = TRUE, showWarnings = FALSE)

script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_path <- if (length(script_arg) > 0L) {
  normalizePath(sub("^--file=", "", script_arg[1]), winslash = "/", mustWork = FALSE)
} else {
  NA_character_
}

paths <- list(
  six_cv_train = file.path(input_dir, "sixclass_grouped_train_80_with_cv_fold.csv"),
  four_cv_train = file.path(input_dir, "fourclass_grouped_train_80_with_cv_fold.csv"),
  six_hold_train = file.path(input_dir, "sixclass_grouped_train_80_with_structural.csv"),
  six_hold_test = file.path(input_dir, "sixclass_grouped_test_20_with_structural.csv"),
  four_hold_train = file.path(input_dir, "fourclass_grouped_train_80_with_structural.csv"),
  four_hold_test = file.path(input_dir, "fourclass_grouped_test_20_with_structural.csv")
)

for (p in paths) {
  if (!file.exists(p)) stop("Missing required input: ", p)
}

folder_rating_levels <- c(0L, 1L, 3L, 5L, 7L, 9L)
vi_map <- data.table(
  raw = c(
    "ndvi", "sri", "psri", "ipvi", "gb_ndvi", "gr_ndvi", "hue", "npci",
    "greenness", "gndvi", "bndvi", "grvi", "gli", "vari", "ngbi", "rgri",
    "gi2", "blb"
  ),
  prefix = c(
    "NDVI", "SRI", "PSRI", "IPVI", "GB_NDVI", "GR_NDVI", "HUE", "NPCI",
    "GREENNESS", "GNDVI", "BNDVI", "GRVI", "GLI", "VARI", "NGBI", "RGRI",
    "GI2", "BLB"
  )
)

sampling_cfg <- list(
  sample_files_per_rating = 80L,
  sample_rows_per_file = 4000L,
  max_samples_per_rating_per_index = 50000L
)

sa_cfg <- list(
  maxit = 6000L,
  temp = 10,
  tmax = 10
)

core_guess <- suppressWarnings(parallel::detectCores(logical = TRUE))
mc_cores <- if (length(core_guess) == 0L || is.na(core_guess) || core_guess < 2L) {
  1L
} else {
  min(4L, core_guess - 1L)
}

default_cache_root <- file.path("/private/tmp", "within_fold_vi_binning_grouped5cv_4class_raw_vi_cache")
if (!dir.exists(default_cache_root)) {
  default_cache_root <- file.path("/private/tmp", "grouped_microplotsafe_rebinned_raw_vi_cache")
}
raw_cache_root <- Sys.getenv("RAW_CACHE_ROOT", default_cache_root)
dir.create(raw_cache_root, recursive = TRUE, showWarnings = FALSE)
cache_manifest_path <- file.path(analysis_root, "raw_vi_cache_manifest.csv")

bin_feature_cols <- as.vector(unlist(lapply(
  vi_map$prefix,
  function(prefix) paste0(prefix, "_bin", seq_len(length(folder_rating_levels)))
)))

log_line <- function(...) {
  cat(sprintf(...))
  flush.console()
}

safe_unique <- function(dt) {
  unique(dt, by = names(dt))
}

derive_folder_rating <- function(folder_vec) {
  suppressWarnings(as.integer(sub("^BB_rating_", "", as.character(folder_vec))))
}

strip_old_bin_cols <- function(dt) {
  keep_cols <- setdiff(names(dt), bin_feature_cols)
  copy(dt[, ..keep_cols])
}

append_limited <- function(current, values, limit) {
  if (length(values) == 0L || length(current) >= limit) return(current)
  remaining <- limit - length(current)
  if (length(values) > remaining) values <- sample(values, remaining)
  c(current, values)
}

clamp_sort <- function(x, lower, upper) {
  sort(pmax(lower, pmin(upper, x)))
}

ensure_strict_thresholds <- function(thresholds, lower, upper) {
  n <- length(thresholds)
  if (n <= 1L) return(sort(thresholds))
  if (!is.finite(lower) || !is.finite(upper) || lower >= upper) return(sort(thresholds))
  thr <- sort(pmax(lower, pmin(upper, thresholds)))
  if (any(!is.finite(thr)) || any(diff(thr) <= 0)) {
    return(seq(lower, upper, length.out = n + 2L)[2:(n + 1L)])
  }
  thr
}

flat_thresholds <- function(value, n_thresholds) {
  if (n_thresholds <= 0L) return(numeric())
  eps <- if (is.finite(value) && abs(value) > 1) abs(value) * 1e-6 else 1e-6
  value + seq_len(n_thresholds) * eps
}

distribution_by_bins <- function(values, thresholds, num_bins) {
  bins <- cut(values, breaks = c(-Inf, thresholds, Inf), include.lowest = TRUE, labels = FALSE)
  tabulate(bins, nbins = num_bins) / length(values)
}

multi_group_difference <- function(samples_per_group, thresholds, num_bins) {
  non_empty <- Filter(function(v) length(v) > 0L, samples_per_group)
  group_count <- length(non_empty)
  if (group_count < 2L) return(0)
  distributions <- lapply(non_empty, distribution_by_bins, thresholds = thresholds, num_bins = num_bins)
  total <- 0
  pairs <- 0L
  for (i in seq_len(group_count - 1L)) {
    for (j in seq.int(i + 1L, group_count)) {
      total <- total + sum(abs(distributions[[i]] - distributions[[j]]))
      pairs <- pairs + 1L
    }
  }
  total / pairs
}

objective_sa <- function(par, samples_per_group, lo, hi, n_bins) {
  thresholds <- clamp_sort(par, lo, hi)
  thresholds <- ensure_strict_thresholds(thresholds, lo, hi)
  score <- multi_group_difference(samples_per_group, thresholds, n_bins)
  if (length(thresholds) > 1L && is.finite(lo) && is.finite(hi) && lo < hi) {
    min_gap <- (hi - lo) * 1e-6
    if (!is.finite(min_gap) || min_gap <= 0) min_gap <- 1e-8
    gap_penalty <- sum(pmax(0, min_gap - diff(thresholds)))
  } else {
    gap_penalty <- 0
  }
  -score + (1000 * gap_penalty)
}

get_raw_vi_cache_path <- function(folder, source_file) {
  cache_dir <- file.path(raw_cache_root, folder)
  dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)
  file.path(cache_dir, sub("[.]csv$", ".rds", source_file, ignore.case = TRUE))
}

build_raw_vi_cache_one <- function(folder, source_file, raw_cols) {
  path <- file.path(base_dir, folder, source_file)
  if (!file.exists(path)) return(NULL)
  dt <- tryCatch(
    fread(path, select = raw_cols, showProgress = FALSE),
    error = function(e) NULL
  )
  if (is.null(dt)) return(NULL)

  vi_values <- setNames(vector("list", length(raw_cols)), raw_cols)
  for (vi in raw_cols) {
    vals <- suppressWarnings(as.numeric(dt[[vi]]))
    vals <- vals[is.finite(vals)]
    vi_values[[vi]] <- vals
  }

  list(
    folder = folder,
    source_file = source_file,
    n_points_total = nrow(dt),
    vi_values = vi_values
  )
}

read_raw_vi_cache <- function(folder, source_file, raw_cols) {
  cache_path <- get_raw_vi_cache_path(folder, source_file)
  if (file.exists(cache_path)) {
    return(tryCatch(readRDS(cache_path), error = function(e) NULL))
  }

  cache_obj <- build_raw_vi_cache_one(folder, source_file, raw_cols)
  if (is.null(cache_obj)) return(NULL)
  saveRDS(cache_obj, cache_path, compress = TRUE)
  cache_obj
}

build_raw_vi_cache_all <- function(file_dt, raw_cols) {
  done_dt <- if (file.exists(cache_manifest_path)) fread(cache_manifest_path) else data.table(
    folder = character(), source_file = character(), cache_path = character(),
    n_points_total = integer(), status = character()
  )
  if (nrow(done_dt) > 0L) {
    done_dt <- done_dt[file.exists(cache_path)]
  }

  done_keys <- unique(done_dt[status == "ok", .(folder, source_file)])
  todo_dt <- unique(file_dt[, .(folder, source_file)])
  if (nrow(done_keys) > 0L) {
    todo_dt <- todo_dt[!done_keys, on = .(folder, source_file)]
  }

  if (nrow(todo_dt) == 0L) {
    log_line("Raw VI cache already complete\n")
    return(invisible(done_dt))
  }

  total_todo <- nrow(todo_dt)
  log_line("Building raw VI cache for %d files\n", total_todo)

  process_one <- function(x) {
    folder <- x$folder[[1]]
    source_file <- x$source_file[[1]]
    cache_path <- get_raw_vi_cache_path(folder, source_file)
    if (file.exists(cache_path)) {
      cache_obj <- tryCatch(readRDS(cache_path), error = function(e) NULL)
      if (!is.null(cache_obj)) {
        return(data.table(
          folder = folder,
          source_file = source_file,
          cache_path = cache_path,
          n_points_total = cache_obj$n_points_total,
          status = "ok"
        ))
      }
    }
    cache_obj <- build_raw_vi_cache_one(folder, source_file, raw_cols)
    if (is.null(cache_obj)) {
      return(data.table(
        folder = folder,
        source_file = source_file,
        cache_path = cache_path,
        n_points_total = NA_integer_,
        status = "failed"
      ))
    }

    saveRDS(cache_obj, cache_path, compress = TRUE)
    data.table(
      folder = folder,
      source_file = source_file,
      cache_path = cache_path,
      n_points_total = cache_obj$n_points_total,
      status = "ok"
    )
  }

  tasks <- split(todo_dt, seq_len(nrow(todo_dt)))
  chunk_size <- max(10L, mc_cores * 10L)
  built_so_far <- 0L

  for (start_idx in seq(1L, length(tasks), by = chunk_size)) {
    end_idx <- min(start_idx + chunk_size - 1L, length(tasks))
    chunk <- tasks[start_idx:end_idx]
    chunk_rows <- if (mc_cores > 1L && length(chunk) > 1L) {
      mclapply(chunk, process_one, mc.cores = min(mc_cores, length(chunk)))
    } else {
      lapply(chunk, process_one)
    }

    chunk_dt <- rbindlist(chunk_rows, use.names = TRUE, fill = TRUE)
    done_dt <- rbindlist(list(done_dt, chunk_dt), use.names = TRUE, fill = TRUE)
    built_so_far <- end_idx
    fwrite(done_dt, cache_manifest_path)
    log_line("Raw VI cache progress %d/%d files\n", built_so_far, total_todo)
  }

  invisible(done_dt)
}

fit_thresholds_from_info <- function(train_info, output_dir, label) {
  threshold_path <- file.path(output_dir, paste0(label, "_sa_thresholds.csv"))
  sampling_path <- file.path(output_dir, paste0(label, "_sampling_summary.csv"))
  selected_path <- file.path(output_dir, paste0(label, "_threshold_training_files.csv"))

  if (file.exists(threshold_path) && file.exists(sampling_path) && file.exists(selected_path)) {
    log_line("%s: loading cached thresholds\n", label)
    return(list(
      thresholds = fread(threshold_path),
      sampling_summary = fread(sampling_path),
      selected_files = fread(selected_path)
    ))
  }

  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  work <- copy(train_info)
  if (!"folder_rating" %in% names(work)) work[, folder_rating := derive_folder_rating(folder)]

  log_line("%s: fitting simulated-annealing thresholds\n", label)
  set.seed(5000L + sum(utf8ToInt(label)))

  file_pool <- unique(work[, .(folder, source_file, folder_rating)])
  sampling_rows <- list()
  selected_rows <- list()

  sample_bank <- setNames(vector("list", nrow(vi_map)), vi_map$raw)
  for (vi in vi_map$raw) {
    sample_bank[[vi]] <- setNames(vector("list", length(folder_rating_levels)), as.character(folder_rating_levels))
    for (rk in as.character(folder_rating_levels)) sample_bank[[vi]][[rk]] <- numeric()
  }

  for (rating in folder_rating_levels) {
    avail <- file_pool[folder_rating == rating][order(folder, source_file)]
    total_avail <- nrow(avail)
    selected <- copy(avail)
    if (nrow(selected) > sampling_cfg$sample_files_per_rating) {
      selected <- selected[sample(.N, sampling_cfg$sample_files_per_rating)]
    }
    setorder(selected, folder, source_file)
    log_line(
      "%s: threshold sampling source severity %s using %d/%d files\n",
      label, as.character(rating), nrow(selected), total_avail
    )

    sampling_rows[[length(sampling_rows) + 1L]] <- data.table(
      folder_rating = rating,
      total_training_files = total_avail,
      sampled_files = nrow(selected)
    )
    if (nrow(selected) > 0L) {
      selected_rows[[length(selected_rows) + 1L]] <- copy(selected)
    }

    for (i in seq_len(nrow(selected))) {
      cache_obj <- read_raw_vi_cache(selected$folder[i], selected$source_file[i], vi_map$raw)
      if (is.null(cache_obj) || is.null(cache_obj$vi_values)) next

      for (vi in vi_map$raw) {
        values <- cache_obj$vi_values[[vi]]
        if (length(values) > sampling_cfg$sample_rows_per_file) {
          values <- sample(values, sampling_cfg$sample_rows_per_file)
        }
        sample_bank[[vi]][[as.character(rating)]] <- append_limited(
          sample_bank[[vi]][[as.character(rating)]],
          values,
          sampling_cfg$max_samples_per_rating_per_index
        )
      }
      if (i %% 10L == 0L || i == nrow(selected)) {
        log_line(
          "%s: sampled %d/%d files for source severity %s\n",
          label, i, nrow(selected), as.character(rating)
        )
      }
    }
  }

  sampling_dt <- rbindlist(sampling_rows, use.names = TRUE, fill = TRUE)
  selected_dt <- if (length(selected_rows) > 0L) {
    rbindlist(selected_rows, use.names = TRUE, fill = TRUE)
  } else {
    data.table(folder = character(), source_file = character(), folder_rating = integer())
  }

  threshold_report <- data.table(
    raw_index = vi_map$raw,
    index = vi_map$prefix,
    score = NA_real_,
    spearman_rho = NA_real_,
    direction = NA_character_,
    label_order = NA_character_
  )
  for (i in seq_len(length(folder_rating_levels) - 1L)) {
    threshold_report[[paste0("threshold_", i)]] <- NA_real_
  }

  for (row_i in seq_len(nrow(vi_map))) {
    vi_raw <- vi_map$raw[row_i]
    samples_per_group <- lapply(as.character(folder_rating_levels), function(rk) sample_bank[[vi_raw]][[rk]])
    lengths_per_group <- vapply(samples_per_group, length, integer(1))

    pooled_values_all <- unlist(samples_per_group, use.names = FALSE)
    pooled_labels_all <- rep(folder_rating_levels, lengths_per_group)
    finite_mask <- is.finite(pooled_values_all)
    pooled_values <- pooled_values_all[finite_mask]
    pooled_labels <- pooled_labels_all[finite_mask]

    if (length(pooled_values) == 0L) {
      thresholds <- rep(0, length(folder_rating_levels) - 1L)
      rho_val <- 0
      direction_val <- "ascending"
      label_order_val <- folder_rating_levels
      score_val <- NA_real_
    } else {
      lower <- min(pooled_values)
      upper <- max(pooled_values)

      if (!is.finite(lower) || !is.finite(upper) || lower == upper) {
        thresholds <- flat_thresholds(lower, length(folder_rating_levels) - 1L)
        score_val <- 0
      } else {
        init <- as.numeric(quantile(
          pooled_values,
          probs = seq.int(1L, length(folder_rating_levels) - 1L) / length(folder_rating_levels),
          na.rm = TRUE,
          type = 8
        ))
        if (length(init) != (length(folder_rating_levels) - 1L)) {
          init <- seq(lower, upper, length.out = length(folder_rating_levels) + 1L)[2:length(folder_rating_levels)]
        }
        init <- ensure_strict_thresholds(clamp_sort(init, lower, upper), lower, upper)

        result <- optim(
          par = init,
          fn = objective_sa,
          method = "SANN",
          control = sa_cfg,
          samples_per_group = samples_per_group,
          lo = lower,
          hi = upper,
          n_bins = length(folder_rating_levels)
        )
        thresholds <- ensure_strict_thresholds(clamp_sort(result$par, lower, upper), lower, upper)
        score_val <- multi_group_difference(samples_per_group, thresholds, length(folder_rating_levels))
      }

      rho_val <- suppressWarnings(cor(pooled_values, pooled_labels, method = "spearman"))
      if (!is.finite(rho_val)) rho_val <- 0
      if (rho_val >= 0) {
        direction_val <- "ascending"
        label_order_val <- folder_rating_levels
      } else {
        direction_val <- "descending"
        label_order_val <- rev(folder_rating_levels)
      }
    }

    threshold_report[row_i, score := score_val]
    threshold_report[row_i, spearman_rho := rho_val]
    threshold_report[row_i, direction := direction_val]
    threshold_report[row_i, label_order := paste(label_order_val, collapse = ">")]
    for (j in seq_len(length(folder_rating_levels) - 1L)) {
      threshold_report[row_i, (paste0("threshold_", j)) := thresholds[j]]
    }
    log_line(
      "%s: optimized %s (score=%.6f, rho=%.4f, direction=%s)\n",
      label, vi_raw, as.numeric(score_val), as.numeric(rho_val), direction_val
    )
  }

  fwrite(threshold_report, threshold_path)
  fwrite(sampling_dt, sampling_path)
  fwrite(selected_dt, selected_path)

  list(
    thresholds = threshold_report,
    sampling_summary = sampling_dt,
    selected_files = selected_dt
  )
}

build_threshold_map <- function(threshold_dt) {
  out <- vector("list", nrow(threshold_dt))
  names(out) <- threshold_dt$raw_index
  for (i in seq_len(nrow(threshold_dt))) {
    row <- threshold_dt[i]
    thr_cols <- grep("^threshold_[0-9]+$", names(row), value = TRUE)
    thr <- as.numeric(unlist(row[, ..thr_cols], use.names = FALSE))
    thr <- thr[is.finite(thr)]
    out[[row$raw_index]] <- list(
      thresholds = thr,
      label_order = as.integer(unlist(strsplit(row$label_order, ">", fixed = TRUE)))
    )
  }
  out
}

compute_binned_dataset <- function(info_dt, threshold_dt, output_path, threshold_source_label) {
  if (file.exists(output_path)) {
    log_line("Loading cached dataset: %s\n", basename(output_path))
    return(fread(output_path))
  }

  work <- strip_old_bin_cols(copy(info_dt))
  if (!"folder_rating" %in% names(work)) work[, folder_rating := derive_folder_rating(folder)]
  if (!"n_points" %in% names(work)) work[, n_points := NA_integer_]
  work[, threshold_source := threshold_source_label]

  threshold_map <- build_threshold_map(threshold_dt)
  n_bins <- length(folder_rating_levels)
  keep_cols <- names(work)

  process_one <- function(i) {
    row <- work[i]
    cache_obj <- read_raw_vi_cache(row$folder[[1]], row$source_file[[1]], vi_map$raw)

    out <- as.data.table(as.list(row[, ..keep_cols]))
    if (!is.null(cache_obj) && !is.null(cache_obj$n_points_total)) {
      out[, n_points := cache_obj$n_points_total]
    }

    for (prefix in vi_map$prefix) {
      for (k in seq_len(n_bins)) {
        out[[paste0(prefix, "_bin", k)]] <- NA_real_
      }
    }

    if (is.null(cache_obj) || is.null(cache_obj$vi_values)) return(out)

    for (row_i in seq_len(nrow(vi_map))) {
      vi_raw <- vi_map$raw[row_i]
      prefix <- vi_map$prefix[row_i]
      thr <- threshold_map[[vi_raw]]$thresholds
      vals <- cache_obj$vi_values[[vi_raw]]
      denom <- length(vals)
      if (denom == 0L) next
      bin_id <- cut(vals, breaks = c(-Inf, thr, Inf), include.lowest = TRUE, labels = FALSE)
      props <- tabulate(bin_id, nbins = n_bins) / denom
      for (k in seq_len(n_bins)) {
        out[[paste0(prefix, "_bin", k)]] <- props[k]
      }
    }

    out
  }

  chunk_size <- 100L
  chunk_parts <- vector("list", ceiling(nrow(work) / chunk_size))
  chunk_idx <- 0L

  for (start_idx in seq(1L, nrow(work), by = chunk_size)) {
    end_idx <- min(start_idx + chunk_size - 1L, nrow(work))
    tasks <- start_idx:end_idx
    res <- if (mc_cores > 1L && length(tasks) > 1L) {
      mclapply(tasks, process_one, mc.cores = min(mc_cores, length(tasks)))
    } else {
      lapply(tasks, process_one)
    }
    chunk_idx <- chunk_idx + 1L
    chunk_parts[[chunk_idx]] <- rbindlist(res, use.names = TRUE, fill = TRUE)
    log_line(
      "%s: feature progress %d/%d rows\n",
      threshold_source_label, end_idx, nrow(work)
    )
  }

  out_dt <- rbindlist(chunk_parts, use.names = TRUE, fill = TRUE)
  fwrite(out_dt, output_path)
  out_dt
}

write_manifest <- function(file_path, dt_list) {
  summary_dt <- rbindlist(dt_list, use.names = TRUE, fill = TRUE)
  fwrite(summary_dt, file_path)
}

six_cv_dt <- fread(paths$six_cv_train)
four_cv_dt <- fread(paths$four_cv_train)
six_hold_train_dt <- fread(paths$six_hold_train)
six_hold_test_dt <- fread(paths$six_hold_test)
four_hold_train_dt <- fread(paths$four_hold_train)
four_hold_test_dt <- fread(paths$four_hold_test)

needed_files_dt <- safe_unique(rbindlist(list(
  six_cv_dt[, .(folder, source_file)],
  four_cv_dt[, .(folder, source_file)],
  six_hold_train_dt[, .(folder, source_file)],
  six_hold_test_dt[, .(folder, source_file)],
  four_hold_train_dt[, .(folder, source_file)],
  four_hold_test_dt[, .(folder, source_file)]
), use.names = TRUE, fill = TRUE))

build_raw_vi_cache_all(needed_files_dt, vi_map$raw)

cv_summary_rows <- list()
for (dataset_tag in c("sixclass", "fourclass")) {
  dt <- if (dataset_tag == "sixclass") copy(six_cv_dt) else copy(four_cv_dt)
  if (!"cv_fold" %in% names(dt)) stop("Missing cv_fold column in dataset: ", dataset_tag)
  dt[, folder_rating := derive_folder_rating(folder)]
  dataset_dir <- file.path(cv_root, dataset_tag)
  dir.create(dataset_dir, recursive = TRUE, showWarnings = FALSE)

  validation_parts <- list()

  for (fold_id in sort(unique(dt$cv_fold))) {
    fold_label <- sprintf("fold_%02d", as.integer(fold_id))
    fold_dir <- file.path(dataset_dir, fold_label)
    dir.create(fold_dir, recursive = TRUE, showWarnings = FALSE)

    train_info <- dt[cv_fold != fold_id]
    valid_info <- dt[cv_fold == fold_id]

    fit_obj <- fit_thresholds_from_info(
      train_info,
      output_dir = fold_dir,
      label = paste0(dataset_tag, "_", fold_label)
    )

    train_out_path <- file.path(fold_dir, sprintf("%s_%s_train_binned.csv", dataset_tag, fold_label))
    valid_out_path <- file.path(fold_dir, sprintf("%s_%s_validation_binned.csv", dataset_tag, fold_label))

    train_binned <- compute_binned_dataset(
      train_info,
      fit_obj$thresholds,
      train_out_path,
      threshold_source_label = paste0(dataset_tag, "_", fold_label)
    )
    valid_binned <- compute_binned_dataset(
      valid_info,
      fit_obj$thresholds,
      valid_out_path,
      threshold_source_label = paste0(dataset_tag, "_", fold_label)
    )

    validation_parts[[length(validation_parts) + 1L]] <- copy(valid_binned)

    cv_summary_rows[[length(cv_summary_rows) + 1L]] <- data.table(
      export_type = "cv_fold_specific",
      dataset = dataset_tag,
      fold = fold_label,
      train_rows = nrow(train_binned),
      validation_rows = nrow(valid_binned),
      train_units = uniqueN(train_binned$unit_norm),
      validation_units = uniqueN(valid_binned$unit_norm),
      threshold_file = file.path(fold_dir, sprintf("%s_sa_thresholds.csv", paste0(dataset_tag, "_", fold_label)))
    )
  }

  validation_master <- safe_unique(rbindlist(validation_parts, use.names = TRUE, fill = TRUE))
  setorder(validation_master, unit_norm, timestamp_chr, source_file)
  master_path <- file.path(dataset_dir, sprintf("%s_grouped_train_80_cv_validation_master_binned.csv", dataset_tag))
  fwrite(validation_master, master_path)

  cv_summary_rows[[length(cv_summary_rows) + 1L]] <- data.table(
    export_type = "cv_validation_master",
    dataset = dataset_tag,
    fold = "all",
    train_rows = NA_integer_,
    validation_rows = nrow(validation_master),
    train_units = NA_integer_,
    validation_units = uniqueN(validation_master$unit_norm),
    threshold_file = master_path
  )
}

cv_summary_dt <- rbindlist(cv_summary_rows, use.names = TRUE, fill = TRUE)
fwrite(cv_summary_dt, file.path(cv_root, "cv_fold_specific_export_summary.csv"))

holdout_threshold_source <- safe_unique(rbindlist(list(
  six_hold_train_dt[, .(folder, source_file, unit_norm, timestamp_chr, folder_rating = derive_folder_rating(folder))],
  four_hold_train_dt[, .(folder, source_file, unit_norm, timestamp_chr, folder_rating = derive_folder_rating(folder))]
), use.names = TRUE, fill = TRUE))

holdout_fit <- fit_thresholds_from_info(
  holdout_threshold_source,
  output_dir = holdout_root,
  label = "full_training_only"
)

holdout_exports <- list(
  list(tag = "sixclass_grouped_train_80_fulltrain_threshold_binned", dt = six_hold_train_dt),
  list(tag = "sixclass_grouped_test_20_fulltrain_threshold_binned", dt = six_hold_test_dt),
  list(tag = "fourclass_grouped_train_80_fulltrain_threshold_binned", dt = four_hold_train_dt),
  list(tag = "fourclass_grouped_test_20_fulltrain_threshold_binned", dt = four_hold_test_dt)
)

holdout_summary_rows <- list()
for (obj in holdout_exports) {
  out_path <- file.path(holdout_root, paste0(obj$tag, ".csv"))
  out_dt <- compute_binned_dataset(
    obj$dt,
    holdout_fit$thresholds,
    out_path,
    threshold_source_label = "full_training_only"
  )

  split_label <- if (grepl("_test_", obj$tag)) "test" else "train"
  dataset_label <- if (grepl("^sixclass", obj$tag)) "sixclass" else "fourclass"
  holdout_summary_rows[[length(holdout_summary_rows) + 1L]] <- data.table(
    export_type = "grouped_holdout_fulltrain_fit",
    dataset = dataset_label,
    split = split_label,
    rows = nrow(out_dt),
    units = uniqueN(out_dt$unit_norm),
    output_file = out_path
  )
}

holdout_summary_dt <- rbindlist(holdout_summary_rows, use.names = TRUE, fill = TRUE)
fwrite(holdout_summary_dt, file.path(holdout_root, "holdout_fulltrain_export_summary.csv"))

readme_lines <- c(
  "Grouped microplot-safe rebinned datasets",
  "",
  "This folder contains two leakage-safe exports:",
  "1. cv_fold_specific/",
  "   Fold-specific simulated-annealing thresholds were re-fit using only the rows outside the held-out fold,",
  "   then applied to that fold's training and validation rows.",
  "2. holdout_fulltrain_fit/",
  "   One threshold set was fit using only the grouped 80% training split, then applied to both train and test.",
  "",
  "Old VI bin-proportion columns were removed and replaced with the newly computed leakage-safe VI bin-proportion columns.",
  "",
  sprintf("Source script: %s", ifelse(is.na(script_path), "unknown", basename(script_path))),
  sprintf("Raw VI cache root: %s", raw_cache_root)
)
writeLines(readme_lines, file.path(analysis_root, "README.txt"))

cat("Saved grouped microplot-safe rebinned datasets in:\n")
cat(analysis_root, "\n")
