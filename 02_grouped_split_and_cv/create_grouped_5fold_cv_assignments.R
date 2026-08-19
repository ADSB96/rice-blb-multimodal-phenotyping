#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(data.table))

set.seed(123)
setDTthreads(0)

base_dir <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)
split_dir <- file.path(base_dir, "jani_stuff", "grouped_holdout_80_20_microplotsafe_split")

paths <- list(
  six_train = file.path(split_dir, "sixclass_grouped_train_80_with_structural.csv"),
  four_train = file.path(split_dir, "fourclass_grouped_train_80_with_structural.csv")
)

for (p in paths) {
  if (!file.exists(p)) stop("Missing required input: ", p)
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
  grp_wide[, .(unit_norm = group, fold, total_obs)]
}

evaluate_assignment <- function(dt, assign_dt, label_col = "BB_rating") {
  work <- merge(copy(dt), assign_dt[, .(unit_norm, fold)], by = "unit_norm", all.x = TRUE, sort = FALSE)
  levels_all <- sort(unique(work[[label_col]]))

  overall <- work[, .(overall_n = .N), by = .(class_label = get(label_col))]
  overall[, overall_prop := overall_n / sum(overall_n)]

  fold_counts <- work[, .N, by = .(fold, class_label = get(label_col))]
  template <- CJ(fold = sort(unique(assign_dt$fold)), class_label = levels_all)
  fold_counts <- merge(template, fold_counts, by = c("fold", "class_label"), all.x = TRUE, sort = FALSE)
  fold_counts[is.na(N), N := 0L]
  fold_counts[, fold_total := sum(N), by = fold]
  fold_counts[, fold_prop := fifelse(fold_total == 0, NA_real_, N / fold_total)]
  fold_counts <- merge(fold_counts, overall, by = "class_label", all.x = TRUE, sort = FALSE)
  fold_counts[, abs_prop_diff := abs(fold_prop - overall_prop)]

  unit_counts <- assign_dt[, .N, by = fold]
  setnames(unit_counts, "N", "n_units")
  unit_counts[, target_units := nrow(assign_dt) / uniqueN(assign_dt$fold)]
  unit_counts[, abs_unit_gap := abs(n_units - target_units)]

  row_counts <- work[, .N, by = fold]
  setnames(row_counts, "N", "n_rows")
  row_counts[, target_rows := nrow(work) / uniqueN(assign_dt$fold)]
  row_counts[, abs_row_gap := abs(n_rows - target_rows)]

  overlap_rows <- list()
  for (f in sort(unique(assign_dt$fold))) {
    train_units <- unique(assign_dt[fold != f, unit_norm])
    test_units <- unique(assign_dt[fold == f, unit_norm])
    overlap_units <- length(intersect(train_units, test_units))

    key_cols <- intersect(c("unit_norm", "timestamp_chr", "folder", "source_file", "row_id"), names(work))
    train_keys <- unique(work[fold != f, ..key_cols])
    test_keys <- unique(work[fold == f, ..key_cols])
    overlap_row_keys <- nrow(merge(train_keys, test_keys, by = key_cols))

    overlap_rows[[length(overlap_rows) + 1L]] <- data.table(
      heldout_fold = f,
      train_units = length(train_units),
      test_units = length(test_units),
      unit_overlap = overlap_units,
      row_key_overlap = overlap_row_keys
    )
  }
  overlap_dt <- rbindlist(overlap_rows)

  list(
    fold_class_balance = fold_counts,
    unit_counts = unit_counts,
    row_counts = row_counts,
    overlap_checks = overlap_dt,
    mean_abs_prop_diff = mean(fold_counts$abs_prop_diff),
    max_abs_prop_diff = max(fold_counts$abs_prop_diff),
    mean_abs_unit_gap = mean(unit_counts$abs_unit_gap),
    max_abs_unit_gap = max(unit_counts$abs_unit_gap),
    mean_abs_row_gap = mean(row_counts$abs_row_gap),
    max_abs_row_gap = max(row_counts$abs_row_gap),
    missing_cells = sum(fold_counts$N == 0L)
  )
}

search_best_assignment <- function(dt, dataset_tag, label_col = "BB_rating", seeds = 1:500) {
  candidate_rows <- vector("list", length(seeds))
  best <- NULL

  for (i in seq_along(seeds)) {
    seed <- seeds[i]
    assign_dt <- make_group_folds(dt$unit_norm, dt[[label_col]], k = 5L, seed = seed)
    met <- evaluate_assignment(dt, assign_dt, label_col = label_col)

    score <- 0
    score <- score + 10 * met$mean_abs_prop_diff + 20 * met$max_abs_prop_diff
    score <- score + 0.25 * met$mean_abs_unit_gap + 0.5 * met$max_abs_unit_gap
    score <- score + 0.01 * met$mean_abs_row_gap + 0.02 * met$max_abs_row_gap
    score <- score + 100 * met$missing_cells
    score <- score + 1000 * sum(met$overlap_checks$unit_overlap)
    score <- score + 1000 * sum(met$overlap_checks$row_key_overlap)

    candidate_rows[[i]] <- data.table(
      dataset = dataset_tag,
      seed = seed,
      score = score,
      mean_abs_prop_diff = met$mean_abs_prop_diff,
      max_abs_prop_diff = met$max_abs_prop_diff,
      mean_abs_unit_gap = met$mean_abs_unit_gap,
      max_abs_unit_gap = met$max_abs_unit_gap,
      mean_abs_row_gap = met$mean_abs_row_gap,
      max_abs_row_gap = met$max_abs_row_gap,
      missing_cells = met$missing_cells,
      total_unit_overlap = sum(met$overlap_checks$unit_overlap),
      total_row_key_overlap = sum(met$overlap_checks$row_key_overlap)
    )

    if (is.null(best) || score < best$score) {
      best <- list(score = score, seed = seed, assign_dt = assign_dt, metrics = met)
    }

    if (seed %% 25L == 0L || i == length(seeds)) {
      cat(sprintf("%s: scored seeds %d/%d\n", dataset_tag, i, length(seeds)))
    }
  }

  list(best = best, candidates = rbindlist(candidate_rows, use.names = TRUE, fill = TRUE))
}

write_cv_outputs <- function(dt, dataset_tag) {
  dt <- copy(dt)
  dt[, `:=`(
    unit_norm = as.character(unit_norm),
    timestamp_chr = as.character(timestamp_chr),
    BB_rating = as.integer(BB_rating)
  )]

  res <- search_best_assignment(dt, dataset_tag = dataset_tag, label_col = "BB_rating", seeds = 1:500)
  best <- res$best
  cand <- res$candidates
  setorder(cand, score, max_abs_prop_diff, max_abs_unit_gap, max_abs_row_gap, seed)

  assign_dt <- copy(best$assign_dt)
  assign_dt[, split := "train"]
  fwrite(assign_dt, file.path(split_dir, sprintf("%s_grouped5fold_assignments.csv", dataset_tag)))
  fwrite(cand, file.path(split_dir, sprintf("%s_grouped5fold_candidate_scores.csv", dataset_tag)))

  dt_with_fold <- merge(dt, assign_dt[, .(unit_norm, cv_fold = fold)], by = "unit_norm", all.x = TRUE, sort = FALSE)
  fwrite(dt_with_fold, file.path(split_dir, sprintf("%s_grouped_train_80_with_cv_fold.csv", dataset_tag)))

  fwrite(best$metrics$fold_class_balance, file.path(split_dir, sprintf("%s_grouped5fold_class_balance.csv", dataset_tag)))
  fwrite(best$metrics$unit_counts, file.path(split_dir, sprintf("%s_grouped5fold_unit_counts.csv", dataset_tag)))
  fwrite(best$metrics$row_counts, file.path(split_dir, sprintf("%s_grouped5fold_row_counts.csv", dataset_tag)))
  fwrite(best$metrics$overlap_checks, file.path(split_dir, sprintf("%s_grouped5fold_overlap_checks.csv", dataset_tag)))

  summary_dt <- data.table(
    dataset = dataset_tag,
    selected_seed = best$seed,
    total_rows = nrow(dt),
    total_units = uniqueN(dt$unit_norm),
    folds = uniqueN(assign_dt$fold),
    mean_abs_prop_diff = best$metrics$mean_abs_prop_diff,
    max_abs_prop_diff = best$metrics$max_abs_prop_diff,
    mean_abs_unit_gap = best$metrics$mean_abs_unit_gap,
    max_abs_unit_gap = best$metrics$max_abs_unit_gap,
    mean_abs_row_gap = best$metrics$mean_abs_row_gap,
    max_abs_row_gap = best$metrics$max_abs_row_gap,
    missing_class_cells = best$metrics$missing_cells,
    total_unit_overlap = sum(best$metrics$overlap_checks$unit_overlap),
    total_row_key_overlap = sum(best$metrics$overlap_checks$row_key_overlap),
    best_score = best$score
  )
  fwrite(summary_dt, file.path(split_dir, sprintf("%s_grouped5fold_summary.csv", dataset_tag)))

  summary_lines <- c(
    sprintf("%s grouped 5-fold CV assignments on training-only set", dataset_tag),
    "",
    sprintf("Selected seed: %d", best$seed),
    sprintf("Total rows: %d", nrow(dt)),
    sprintf("Total units: %d", uniqueN(dt$unit_norm)),
    sprintf("Folds: %d", uniqueN(assign_dt$fold)),
    "",
    sprintf("Mean abs class-proportion difference across folds: %.6f", best$metrics$mean_abs_prop_diff),
    sprintf("Max abs class-proportion difference across folds: %.6f", best$metrics$max_abs_prop_diff),
    sprintf("Total unit overlap across held-out folds: %d", sum(best$metrics$overlap_checks$unit_overlap)),
    sprintf("Total row-key overlap across held-out folds: %d", sum(best$metrics$overlap_checks$row_key_overlap))
  )
  writeLines(summary_lines, file.path(split_dir, sprintf("%s_grouped5fold_README.txt", dataset_tag)))

  summary_dt
}

six_dt <- fread(paths$six_train)
four_dt <- fread(paths$four_train)

six_summary <- write_cv_outputs(six_dt, "sixclass")
four_summary <- write_cv_outputs(four_dt, "fourclass")

combined_summary <- rbindlist(list(six_summary, four_summary), use.names = TRUE, fill = TRUE)
fwrite(combined_summary, file.path(split_dir, "grouped5fold_assignment_summary.csv"))

cat("Saved grouped 5-fold CV assignments in:\n")
cat(split_dir, "\n")
