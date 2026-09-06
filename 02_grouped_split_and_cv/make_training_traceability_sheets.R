#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(data.table))

base_dir <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)
split_dir <- file.path(base_dir, "jani_stuff", "grouped_holdout_90_10_microplotsafe_split")

make_traceability_sheet <- function(input_file, dataset_tag) {
  dt <- fread(input_file)

  keep_cols <- intersect(
    c(
      "unit_norm", "unit", "timestamp_chr", "timestamp", "folder",
      "source_file", "processed_file", "BB_rating", "cv_fold"
    ),
    names(dt)
  )

  out <- unique(copy(dt)[, ..keep_cols])
  setnames(out, old = "timestamp_chr", new = "scan_date_yyyymmdd", skip_absent = TRUE)
  setnames(out, old = "timestamp", new = "scan_date_numeric", skip_absent = TRUE)
  out[, dataset := dataset_tag]
  setcolorder(
    out,
    c(
      "dataset", "unit_norm", "unit", "scan_date_yyyymmdd", "scan_date_numeric",
      "folder", "BB_rating", "source_file", "processed_file", "cv_fold"
    )[c(
      "dataset", "unit_norm", "unit", "scan_date_yyyymmdd", "scan_date_numeric",
      "folder", "BB_rating", "source_file", "processed_file", "cv_fold"
    ) %in% names(out)]
  )

  for (f in 1:5) {
    out[, (sprintf("fold%d_role", f)) := ifelse(cv_fold == f, "validation", "training")]
  }

  setorder(out, unit_norm, scan_date_yyyymmdd, source_file)

  out_file <- file.path(split_dir, sprintf("%s_grouped_train_90_traceability_master.csv", dataset_tag))
  fwrite(out, out_file)

  unit_manifest <- unique(out[, .(dataset, unit_norm, unit, cv_fold)])
  for (f in 1:5) {
    unit_manifest[, (sprintf("fold%d_role", f)) := ifelse(cv_fold == f, "validation", "training")]
  }
  setorder(unit_manifest, unit_norm)

  unit_file <- file.path(split_dir, sprintf("%s_grouped_train_90_unit_manifest.csv", dataset_tag))
  fwrite(unit_manifest, unit_file)

  list(rows = nrow(out), units = uniqueN(out$unit_norm), out_file = out_file, unit_file = unit_file)
}

six_res <- make_traceability_sheet(
  file.path(split_dir, "sixclass_grouped_train_90_with_cv_fold.csv"),
  "sixclass"
)

four_res <- make_traceability_sheet(
  file.path(split_dir, "fourclass_grouped_train_90_with_cv_fold.csv"),
  "fourclass"
)

summary_dt <- rbindlist(list(
  data.table(dataset = "sixclass", rows = six_res$rows, units = six_res$units),
  data.table(dataset = "fourclass", rows = four_res$rows, units = four_res$units)
))

fwrite(summary_dt, file.path(split_dir, "grouped_train_90_traceability_summary.csv"))

cat("Saved traceability sheets in:\n")
cat(split_dir, "\n")
