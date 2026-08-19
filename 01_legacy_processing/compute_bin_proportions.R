#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(data.table))

setDTthreads(0)

base_dir <- getwd()
processed_root <- file.path(base_dir, "processing", "processed")
threshold_path <- file.path(base_dir, "processing", "sa_thresholds.csv")
proportions_root <- file.path(base_dir, "processing", "proportions")
master_path <- file.path(base_dir, "processing", "bin_proportions_all_files.csv")
log_path <- file.path(base_dir, "processing", "proportion_run.log")
mapping_path <- file.path(base_dir, "processing", "bin_label_mapping.csv")

if (!dir.exists(processed_root)) {
  stop("Missing processed folder: ", processed_root)
}
if (!file.exists(threshold_path)) {
  stop("Missing threshold file: ", threshold_path)
}

if (dir.exists(proportions_root)) {
  unlink(proportions_root, recursive = TRUE, force = TRUE)
}
dir.create(proportions_root, recursive = TRUE, showWarnings = FALSE)
if (file.exists(log_path)) {
  file.remove(log_path)
}

log_message <- function(msg) {
  line <- sprintf("%s %s", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), msg)
  cat(line, "\n")
  cat(line, "\n", file = log_path, append = TRUE)
}

thresholds <- fread(threshold_path)
indices <- thresholds$index
binned_cols <- paste0(indices, "_binned")

bin_label_map <- setNames(vector("list", length(indices)), indices)
for (idx in indices) {
  row <- thresholds[index == idx]
  labels <- as.integer(unlist(strsplit(row$label_order[1], ">", fixed = TRUE)))
  if (length(labels) != 6L || any(is.na(labels))) {
    stop("Invalid label_order for index: ", idx)
  }
  bin_label_map[[idx]] <- labels
}

mapping_dt <- data.table(index = toupper(indices))
for (k in 1:6) {
  mapping_dt[[paste0("bin", k, "_label")]] <- vapply(indices, function(idx) bin_label_map[[idx]][k], integer(1))
}
fwrite(mapping_dt, mapping_path)

files <- sort(list.files(
  processed_root,
  pattern = "_binned\\.csv$",
  full.names = TRUE,
  recursive = TRUE
))

if (length(files) == 0L) {
  stop("No processed *_binned.csv files found under ", processed_root)
}

log_message(sprintf(
  "Starting proportion extraction for %d files (%d indices x 6 bins).",
  length(files), length(indices)
))
log_message(sprintf("Saved bin-label mapping to %s", mapping_path))

master_rows <- vector("list", length(files))

for (i in seq_along(files)) {
  file_path <- files[i]
  parent_folder <- basename(dirname(file_path))
  file_name <- basename(file_path)

  header <- fread(file_path, nrows = 0L, showProgress = FALSE)
  present_cols <- intersect(binned_cols, names(header))

  if (length(present_cols) == 0L) {
    dt <- data.table()
    n_points <- NA_integer_
  } else {
    dt <- fread(file_path, select = present_cols, showProgress = FALSE)
    n_points <- nrow(dt)
  }

  row_dt <- data.table(
    folder = parent_folder,
    source_file = sub("_binned\\.csv$", ".csv", file_name),
    processed_file = file_name,
    n_points = n_points
  )

  for (idx in indices) {
    col <- paste0(idx, "_binned")
    idx_name <- toupper(idx)
    labels_for_bins <- bin_label_map[[idx]]
    if (!col %in% names(dt)) {
      for (k in 1:6) {
        row_dt[[paste0(idx_name, "_bin", k)]] <- NA_real_
      }
      next
    }

    vals <- dt[[col]]
    denom <- sum(!is.na(vals))
    for (k in 1:6) {
      lab <- labels_for_bins[k]
      out_col <- paste0(idx_name, "_bin", k)
      if (denom == 0L) {
        row_dt[[out_col]] <- NA_real_
      } else {
        row_dt[[out_col]] <- sum(vals == lab, na.rm = TRUE) / denom
      }
    }
  }

  out_subdir <- file.path(proportions_root, parent_folder)
  dir.create(out_subdir, recursive = TRUE, showWarnings = FALSE)
  out_name <- sub("_binned\\.csv$", "_bin_proportions.csv", file_name)
  out_path <- file.path(out_subdir, out_name)
  fwrite(row_dt, out_path)

  master_rows[[i]] <- row_dt

  if (i %% 25L == 0L || i == length(files)) {
    log_message(sprintf("Processed %d/%d files", i, length(files)))
  }
}

master_dt <- rbindlist(master_rows, use.names = TRUE, fill = TRUE)
setorder(master_dt, folder, source_file)
fwrite(master_dt, master_path)

log_message(sprintf("Saved master proportion table to %s", master_path))
log_message("Completed proportion extraction.")
