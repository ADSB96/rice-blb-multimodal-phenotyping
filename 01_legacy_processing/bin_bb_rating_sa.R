#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(data.table))

set.seed(123)
setDTthreads(0)

base_dir <- getwd()
processing_dir <- file.path(base_dir, "processing")
processed_dir <- file.path(processing_dir, "processed")
dir.create(processing_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(processed_dir, recursive = TRUE, showWarnings = FALSE)

log_path <- file.path(processing_dir, "run.log")
if (file.exists(log_path)) {
  file.remove(log_path)
}

log_message <- function(msg) {
  line <- sprintf("%s %s", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), msg)
  cat(line, "\n")
  cat(line, "\n", file = log_path, append = TRUE)
}

append_limited <- function(current, values, limit) {
  if (length(values) == 0L || length(current) >= limit) {
    return(current)
  }
  remaining <- limit - length(current)
  if (length(values) > remaining) {
    values <- sample(values, remaining)
  }
  c(current, values)
}

clamp_sort <- function(x, lower, upper) {
  sort(pmax(lower, pmin(upper, x)))
}

ensure_strict_thresholds <- function(thresholds, lower, upper) {
  n <- length(thresholds)
  if (n <= 1L) {
    return(sort(thresholds))
  }
  if (!is.finite(lower) || !is.finite(upper) || lower >= upper) {
    return(sort(thresholds))
  }
  thr <- sort(pmax(lower, pmin(upper, thresholds)))
  if (any(!is.finite(thr)) || any(diff(thr) <= 0)) {
    return(seq(lower, upper, length.out = n + 2L)[2:(n + 1L)])
  }
  thr
}

flat_thresholds <- function(value, n_thresholds) {
  if (n_thresholds <= 0L) {
    return(numeric())
  }
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
  if (group_count < 2L) {
    return(0)
  }
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
    if (!is.finite(min_gap) || min_gap <= 0) {
      min_gap <- 1e-8
    }
    gap_penalty <- sum(pmax(0, min_gap - diff(thresholds)))
  } else {
    gap_penalty <- 0
  }
  -score + (1000 * gap_penalty)
}

log_message("Starting BB-rating simulated annealing binning workflow.")

bb_dirs <- list.dirs(base_dir, full.names = TRUE, recursive = FALSE)
bb_dirs <- bb_dirs[grepl("^BB_rating_[0-9]+$", basename(bb_dirs))]
if (length(bb_dirs) == 0L) {
  stop("No BB_rating_* folders found in working directory.")
}

rating_levels <- as.integer(sub("^BB_rating_", "", basename(bb_dirs)))
order_idx <- order(rating_levels)
bb_dirs <- bb_dirs[order_idx]
rating_levels <- rating_levels[order_idx]
rating_keys <- as.character(rating_levels)
num_bins <- length(rating_levels)

files_by_rating <- setNames(vector("list", length(rating_levels)), rating_keys)
for (i in seq_along(bb_dirs)) {
  files <- sort(list.files(bb_dirs[i], pattern = "\\.csv$", full.names = TRUE))
  files_by_rating[[rating_keys[i]]] <- files
}

total_files <- sum(vapply(files_by_rating, length, integer(1)))
if (total_files == 0L) {
  stop("No CSV files found inside BB_rating_* folders.")
}

first_non_empty <- NULL
for (files in files_by_rating) {
  if (length(files) > 0L) {
    first_non_empty <- files[1]
    break
  }
}
if (is.null(first_non_empty)) {
  stop("No readable CSV files found.")
}

preview <- fread(first_non_empty, nrows = 50, showProgress = FALSE)
excluded_cols <- c("x", "y", "z", "wvl1", "wvl2", "wvl3", "wvl4", "value", "BB_rating")
candidate_vi <- setdiff(names(preview), excluded_cols)
vi_cols <- candidate_vi[vapply(candidate_vi, function(col) is.numeric(preview[[col]]), logical(1))]
if (length(vi_cols) == 0L) {
  stop("No numeric vegetation-index columns found after excluding positional/spectral raw columns.")
}

log_message(sprintf(
  "Detected %d BB-rating groups (%s), %d total CSV files, and %d vegetation index columns.",
  length(rating_levels),
  paste(rating_levels, collapse = ", "),
  total_files,
  length(vi_cols)
))

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

log_message(sprintf(
  "Sampling config: files/group=%d, rows/file=%d, max_samples/group/index=%d",
  sampling_cfg$sample_files_per_rating,
  sampling_cfg$sample_rows_per_file,
  sampling_cfg$max_samples_per_rating_per_index
))
log_message(sprintf(
  "Simulated annealing config: maxit=%d, temp=%s, tmax=%s",
  sa_cfg$maxit, as.character(sa_cfg$temp), as.character(sa_cfg$tmax)
))

sample_cache_path <- file.path(processing_dir, "sample_bank.rds")
sample_bank <- NULL

if (file.exists(sample_cache_path)) {
  cache <- tryCatch(readRDS(sample_cache_path), error = function(e) NULL)
  if (!is.null(cache) &&
      identical(cache$rating_levels, rating_levels) &&
      identical(cache$vi_cols, vi_cols) &&
      is.list(cache$sample_bank)) {
    sample_bank <- cache$sample_bank
    log_message(sprintf("Loaded existing sample cache from %s", sample_cache_path))
  }
}

if (is.null(sample_bank)) {
  sample_bank <- setNames(vector("list", length(vi_cols)), vi_cols)
  for (vi in vi_cols) {
    sample_bank[[vi]] <- setNames(vector("list", length(rating_levels)), rating_keys)
    for (rk in rating_keys) {
      sample_bank[[vi]][[rk]] <- numeric()
    }
  }

  for (rk in rating_keys) {
    rating_files <- files_by_rating[[rk]]
    if (length(rating_files) == 0L) {
      next
    }
    selected_files <- rating_files
    if (length(selected_files) > sampling_cfg$sample_files_per_rating) {
      selected_files <- sample(selected_files, sampling_cfg$sample_files_per_rating)
    }

    log_message(sprintf(
      "Sampling group %s using %d/%d files.",
      rk, length(selected_files), length(rating_files)
    ))

    for (file_idx in seq_along(selected_files)) {
      file_path <- selected_files[file_idx]
      dt <- tryCatch(
        fread(file_path, select = vi_cols, showProgress = FALSE),
        error = function(e) NULL
      )
      if (is.null(dt) || nrow(dt) == 0L) {
        next
      }
      row_count <- nrow(dt)
      picked <- if (row_count > sampling_cfg$sample_rows_per_file) {
        sample.int(row_count, sampling_cfg$sample_rows_per_file)
      } else {
        seq_len(row_count)
      }
      sampled_dt <- dt[picked]
      for (vi in vi_cols) {
        values <- sampled_dt[[vi]]
        values <- values[is.finite(values)]
        sample_bank[[vi]][[rk]] <- append_limited(
          sample_bank[[vi]][[rk]],
          values,
          sampling_cfg$max_samples_per_rating_per_index
        )
      }
      if (file_idx %% 10L == 0L || file_idx == length(selected_files)) {
        log_message(sprintf(
          "Sampling progress group %s: %d/%d files",
          rk, file_idx, length(selected_files)
        ))
      }
    }
  }

  saveRDS(
    list(
      sample_bank = sample_bank,
      vi_cols = vi_cols,
      rating_levels = rating_levels,
      sampling_cfg = sampling_cfg
    ),
    sample_cache_path
  )
  log_message(sprintf("Saved sample cache to %s", sample_cache_path))
}

threshold_map <- setNames(vector("list", length(vi_cols)), vi_cols)
threshold_report <- data.table(
  index = vi_cols,
  score = NA_real_,
  spearman_rho = NA_real_,
  direction = NA_character_,
  label_order = NA_character_
)
for (i in seq_len(num_bins - 1L)) {
  threshold_report[[paste0("threshold_", i)]] <- NA_real_
}

for (vi in vi_cols) {
  samples_per_group <- lapply(rating_keys, function(rk) sample_bank[[vi]][[rk]])
  lengths_per_group <- vapply(samples_per_group, length, integer(1))

  pooled_values_all <- unlist(samples_per_group, use.names = FALSE)
  pooled_labels_all <- rep(rating_levels, lengths_per_group)
  finite_mask <- is.finite(pooled_values_all)
  pooled_values <- pooled_values_all[finite_mask]
  pooled_labels <- pooled_labels_all[finite_mask]

  if (length(pooled_values) == 0L) {
    thresholds <- rep(0, num_bins - 1L)
    rho_val <- 0
    direction_val <- "ascending"
    label_order_val <- rating_levels
    score_val <- NA_real_
  } else {
    lower <- min(pooled_values)
    upper <- max(pooled_values)
    if (!is.finite(lower) || !is.finite(upper) || lower == upper) {
      thresholds <- flat_thresholds(lower, num_bins - 1L)
      score_val <- 0
    } else {
      init <- as.numeric(quantile(
        pooled_values,
        probs = seq.int(1L, num_bins - 1L) / num_bins,
        na.rm = TRUE,
        type = 8
      ))
      if (length(init) != (num_bins - 1L)) {
        init <- seq(lower, upper, length.out = num_bins + 1L)[2:num_bins]
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
        n_bins = num_bins
      )
      thresholds <- ensure_strict_thresholds(
        clamp_sort(result$par, lower, upper),
        lower,
        upper
      )
      score_val <- multi_group_difference(samples_per_group, thresholds, num_bins)
    }

    rho_val <- suppressWarnings(cor(pooled_values, pooled_labels, method = "spearman"))
    if (!is.finite(rho_val)) {
      rho_val <- 0
    }
    if (rho_val >= 0) {
      direction_val <- "ascending"
      label_order_val <- rating_levels
    } else {
      direction_val <- "descending"
      label_order_val <- rev(rating_levels)
    }
  }

  threshold_map[[vi]] <- list(
    thresholds = thresholds,
    label_order = label_order_val,
    direction = direction_val,
    score = score_val,
    spearman_rho = rho_val
  )

  threshold_report[index == vi, score := score_val]
  threshold_report[index == vi, spearman_rho := rho_val]
  threshold_report[index == vi, direction := direction_val]
  threshold_report[index == vi, label_order := paste(label_order_val, collapse = ">")]
  for (i in seq_len(num_bins - 1L)) {
    threshold_report[index == vi, (paste0("threshold_", i)) := thresholds[i]]
  }

  log_message(sprintf(
    "Optimized %s: score=%.6f, rho=%.4f, direction=%s",
    vi, as.numeric(score_val), as.numeric(rho_val), direction_val
  ))
}

threshold_path <- file.path(processing_dir, "sa_thresholds.csv")
fwrite(threshold_report, threshold_path)
log_message(sprintf("Saved thresholds to %s", threshold_path))

files_processed <- 0L
for (dir_path in bb_dirs) {
  dir_name <- basename(dir_path)
  out_subdir <- file.path(processed_dir, dir_name)
  dir.create(out_subdir, recursive = TRUE, showWarnings = FALSE)

  csv_files <- sort(list.files(dir_path, pattern = "\\.csv$", full.names = TRUE))
  if (length(csv_files) == 0L) {
    next
  }

  log_message(sprintf("Processing %s (%d files)", dir_name, length(csv_files)))

  for (file_path in csv_files) {
    out_name <- sub("\\.csv$", "_binned.csv", basename(file_path), ignore.case = TRUE)
    out_path <- file.path(out_subdir, out_name)
    if (file.exists(out_path)) {
      files_processed <- files_processed + 1L
      if (files_processed %% 25L == 0L || files_processed == total_files) {
        log_message(sprintf("Processed %d/%d files", files_processed, total_files))
      }
      next
    }

    dt <- fread(file_path, showProgress = FALSE)

    for (vi in vi_cols) {
      thr <- threshold_map[[vi]]$thresholds
      labels <- threshold_map[[vi]]$label_order
      bin_id <- cut(dt[[vi]], breaks = c(-Inf, thr, Inf), include.lowest = TRUE, labels = FALSE)
      mapped <- labels[bin_id]
      mapped[is.na(bin_id)] <- NA_integer_
      dt[[paste0(vi, "_binned")]] <- mapped
    }

    fwrite(dt, out_path)

    files_processed <- files_processed + 1L
    if (files_processed %% 25L == 0L || files_processed == total_files) {
      log_message(sprintf("Processed %d/%d files", files_processed, total_files))
    }
  }
}

summary_path <- file.path(processing_dir, "run_summary.txt")
summary_lines <- c(
  sprintf("Run timestamp: %s", format(Sys.time(), "%Y-%m-%d %H:%M:%S")),
  sprintf("Base directory: %s", base_dir),
  sprintf("Input groups: %s", paste(sprintf("BB_rating_%d", rating_levels), collapse = ", ")),
  sprintf("Total files processed: %d", files_processed),
  sprintf("Vegetation indices binned: %s", paste(vi_cols, collapse = ", ")),
  sprintf("Threshold report: %s", threshold_path),
  sprintf("Processed output root: %s", processed_dir)
)
writeLines(summary_lines, con = summary_path)
log_message(sprintf("Saved run summary to %s", summary_path))

log_message("Completed BB-rating simulated annealing binning workflow.")
