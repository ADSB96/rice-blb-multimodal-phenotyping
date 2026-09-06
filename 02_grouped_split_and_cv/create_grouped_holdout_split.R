#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(data.table))

set.seed(123)
setDTthreads(0)

base_dir <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)
out_dir <- file.path(base_dir, "jani_stuff", "grouped_holdout_90_10_microplotsafe_split")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# Add the shared de-identified complete dataset here before running the workflow.
# This input file should contain all observations together before train/test partitioning,
# and should retain the original 6-class BB_rating labels.
paths <- list(
  combined = file.path(base_dir, "input_data", "complete_dataset.csv")
)

for (p in paths) {
  if (!file.exists(p)) {
    stop("Missing required input: ", p, "\nAdd `complete_dataset.csv` to `input_data/` before running the workflow.")
  }
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

split_balance_dt <- function(dt, split_col, label_col, all_levels) {
  out <- dt[, .N, by = c(split_col, label_col)]
  setnames(out, c("split", "class_label", "n_rows"))
  template <- CJ(split = c("train", "test"), class_label = all_levels)
  out <- merge(template, out, by = c("split", "class_label"), all.x = TRUE, sort = FALSE)
  out[is.na(n_rows), n_rows := 0L]
  out[, split_total := sum(n_rows), by = split]
  out[, proportion := fifelse(split_total == 0, NA_real_, n_rows / split_total)]
  out
}

calc_split_metrics <- function(dt, test_groups, label_col, all_levels) {
  work <- copy(dt)
  work[, split := fifelse(unit_norm %in% test_groups, "test", "train")]
  bal <- split_balance_dt(work, "split", label_col, all_levels)

  train_props <- bal[split == "train", .(class_label, train_prop = proportion)]
  test_props <- bal[split == "test", .(class_label, test_prop = proportion)]
  comp <- merge(train_props, test_props, by = "class_label", all = TRUE, sort = FALSE)
  comp[is.na(train_prop), train_prop := 0]
  comp[is.na(test_prop), test_prop := 0]
  comp[, abs_diff := abs(train_prop - test_prop)]

  list(
    class_compare = comp,
    class_balance = bal,
    row_frac_test = sum(work$split == "test") / nrow(work),
    n_test_rows = sum(work$split == "test"),
    n_train_rows = sum(work$split == "train"),
    missing_train = sum(comp$train_prop == 0),
    missing_test = sum(comp$test_prop == 0),
    mean_abs_diff = mean(comp$abs_diff),
    max_abs_diff = max(comp$abs_diff)
  )
}

score_candidate <- function(six_dt, four_dt, groups_dt, fold_id) {
  test_groups <- groups_dt[fold == fold_id, group]
  six_met <- calc_split_metrics(six_dt, test_groups, "BB_rating", sort(unique(six_dt$BB_rating)))
  four_met <- calc_split_metrics(four_dt, test_groups, "BB_rating", sort(unique(four_dt$BB_rating)))

  group_frac_test <- length(test_groups) / uniqueN(six_dt$unit_norm)
  score <- 0
  score <- score + 50 * abs(six_met$row_frac_test - 0.10)
  score <- score + 25 * abs(group_frac_test - 0.10)
  score <- score + 10 * six_met$mean_abs_diff + 10 * six_met$max_abs_diff
  score <- score + 8 * four_met$mean_abs_diff + 8 * four_met$max_abs_diff
  score <- score + 100 * six_met$missing_test + 100 * six_met$missing_train
  score <- score + 50 * four_met$missing_test + 50 * four_met$missing_train

  list(
    score = score,
    test_groups = test_groups,
    six = six_met,
    four = four_met,
    group_frac_test = group_frac_test
  )
}

six_dt <- safe_unique(fread(paths$combined))
six_dt[, `:=`(
  unit_norm = as.character(unit_norm),
  timestamp_chr = as.character(timestamp_chr),
  BB_rating = as.integer(BB_rating)
)]

required_cols <- c("unit_norm", "timestamp_chr", "folder", "source_file", "BB_rating")
missing_cols <- setdiff(required_cols, names(six_dt))
if (length(missing_cols) > 0L) {
  stop("The shared complete dataset is missing required columns: ", paste(missing_cols, collapse = ", "))
}

four_dt <- copy(six_dt)
four_dt[, BB_rating := collapse_to_fourclass(BB_rating)]

fwrite(four_dt, file.path(out_dir, "fourclass_complete_dataset_with_structural.csv"))
fwrite(six_dt, file.path(out_dir, "sixclass_complete_dataset_with_structural.csv"))

candidate_rows <- list()
best <- NULL
best_seed <- NA_integer_
best_fold <- NA_integer_

search_seeds <- 1:500
for (seed in search_seeds) {
  fold_assign <- make_group_folds(six_dt$unit_norm, six_dt$BB_rating, k = 5L, seed = seed)
  for (fold_id in sort(unique(fold_assign$fold))) {
    cand <- score_candidate(six_dt, four_dt, fold_assign, fold_id)
    candidate_rows[[length(candidate_rows) + 1L]] <- data.table(
      seed = seed,
      fold = fold_id,
      score = cand$score,
      n_test_groups = length(cand$test_groups),
      group_frac_test = cand$group_frac_test,
      six_test_row_frac = cand$six$row_frac_test,
      six_mean_abs_diff = cand$six$mean_abs_diff,
      six_max_abs_diff = cand$six$max_abs_diff,
      six_missing_test = cand$six$missing_test,
      four_mean_abs_diff = cand$four$mean_abs_diff,
      four_max_abs_diff = cand$four$max_abs_diff,
      four_missing_test = cand$four$missing_test
    )
    if (is.null(best) || cand$score < best$score) {
      best <- cand
      best_seed <- seed
      best_fold <- fold_id
    }
  }
  if (seed %% 25L == 0L || seed == max(search_seeds)) {
    cat(sprintf("Scored seeds %d/%d\n", seed, max(search_seeds)))
  }
}

candidate_dt <- rbindlist(candidate_rows, use.names = TRUE, fill = TRUE)
candidate_dt[, test_row_gap := abs(six_test_row_frac - 0.10)]
candidate_dt[, test_group_gap := abs(group_frac_test - 0.10)]
setorder(candidate_dt, score, test_row_gap, test_group_gap, six_max_abs_diff, four_max_abs_diff)
fwrite(candidate_dt, file.path(out_dir, "candidate_split_scores.csv"))

best_assign <- make_group_folds(six_dt$unit_norm, six_dt$BB_rating, k = 5L, seed = best_seed)
best_assign[, split := fifelse(fold == best_fold, "test", "train")]
test_groups <- best_assign[split == "test", group]

six_split <- copy(six_dt)
six_split[, split := fifelse(unit_norm %in% test_groups, "test", "train")]
four_split <- copy(four_dt)
four_split[, split := fifelse(unit_norm %in% test_groups, "test", "train")]

if (length(intersect(unique(six_split[split == "train", unit_norm]), unique(six_split[split == "test", unit_norm]))) != 0L) {
  stop("Unit overlap detected in 6-class split.")
}
if (length(intersect(unique(four_split[split == "train", unit_norm]), unique(four_split[split == "test", unit_norm]))) != 0L) {
  stop("Unit overlap detected in 4-class split.")
}

six_train <- six_split[split == "train"][, split := NULL]
six_test <- six_split[split == "test"][, split := NULL]
four_train <- four_split[split == "train"][, split := NULL]
four_test <- four_split[split == "test"][, split := NULL]

fwrite(six_train, file.path(out_dir, "sixclass_grouped_train_90_with_structural.csv"))
fwrite(six_test, file.path(out_dir, "sixclass_grouped_test_10_with_structural.csv"))
fwrite(four_train, file.path(out_dir, "fourclass_grouped_train_90_with_structural.csv"))
fwrite(four_test, file.path(out_dir, "fourclass_grouped_test_10_with_structural.csv"))

manifest_dt <- best_assign[, .(unit_norm = group, holdout_fold = fold, split)]
fwrite(manifest_dt, file.path(out_dir, "grouped_holdout_unit_manifest.csv"))

six_balance <- split_balance_dt(six_split, "split", "BB_rating", sort(unique(six_split$BB_rating)))
six_balance[, severity_scheme := "6class"]
four_balance <- split_balance_dt(four_split, "split", "BB_rating", sort(unique(four_split$BB_rating)))
four_balance[, severity_scheme := "4class"]
balance_dt <- rbindlist(list(six_balance, four_balance), use.names = TRUE, fill = TRUE)
fwrite(balance_dt, file.path(out_dir, "grouped_holdout_class_balance.csv"))

selection_summary <- data.table(
  selected_seed = best_seed,
  selected_fold = best_fold,
  total_units = uniqueN(six_dt$unit_norm),
  test_units = length(test_groups),
  train_units = uniqueN(six_dt$unit_norm) - length(test_groups),
  six_total_rows = nrow(six_dt),
  six_train_rows = nrow(six_train),
  six_test_rows = nrow(six_test),
  six_test_row_fraction = nrow(six_test) / nrow(six_dt),
  four_total_rows = nrow(four_dt),
  four_train_rows = nrow(four_train),
  four_test_rows = nrow(four_test),
  four_test_row_fraction = nrow(four_test) / nrow(four_dt),
  six_mean_abs_prop_diff = best$six$mean_abs_diff,
  six_max_abs_prop_diff = best$six$max_abs_diff,
  four_mean_abs_prop_diff = best$four$mean_abs_diff,
  four_max_abs_prop_diff = best$four$max_abs_diff,
  combined_score = best$score
)
fwrite(selection_summary, file.path(out_dir, "grouped_holdout_selection_summary.csv"))

summary_lines <- c(
  "Grouped 90/10 microplot-safe holdout split",
  "",
  sprintf("Selected seed: %d", best_seed),
  sprintf("Selected holdout fold: %d", best_fold),
  sprintf("Total units: %d", uniqueN(six_dt$unit_norm)),
  sprintf("Test units: %d", length(test_groups)),
  sprintf("Train units: %d", uniqueN(six_dt$unit_norm) - length(test_groups)),
  "",
  sprintf("6-class rows: train=%d, test=%d (test fraction=%.4f)", nrow(six_train), nrow(six_test), nrow(six_test) / nrow(six_dt)),
  sprintf("4-class rows: train=%d, test=%d (test fraction=%.4f)", nrow(four_train), nrow(four_test), nrow(four_test) / nrow(four_dt)),
  "",
  sprintf("6-class mean abs proportion difference: %.6f", best$six$mean_abs_diff),
  sprintf("6-class max abs proportion difference: %.6f", best$six$max_abs_diff),
  sprintf("4-class mean abs proportion difference: %.6f", best$four$mean_abs_diff),
  sprintf("4-class max abs proportion difference: %.6f", best$four$max_abs_diff),
  "",
  "Files written:",
  "- sixclass_grouped_train_90_with_structural.csv",
  "- sixclass_grouped_test_10_with_structural.csv",
  "- fourclass_grouped_train_90_with_structural.csv",
  "- fourclass_grouped_test_10_with_structural.csv",
  "- grouped_holdout_unit_manifest.csv",
  "- grouped_holdout_class_balance.csv",
  "- grouped_holdout_selection_summary.csv",
  "- candidate_split_scores.csv"
)
writeLines(summary_lines, file.path(out_dir, "README.txt"))

cat("Saved grouped 90/10 microplot-safe holdout split in:\n")
cat(out_dir, "\n")
