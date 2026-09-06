#!/usr/bin/env python3

import csv
import os
from pathlib import Path


def fmt_num(val, ndigits=3):
    if val in (None, "", "NA"):
        return ""
    return f"{float(val):.{ndigits}f}"


def fmt_cv(row, prefix):
    mean = row.get(f"{prefix}_mean", "")
    sd = row.get(f"{prefix}_sd", "")
    low = row.get(f"{prefix}_ci_low", "")
    high = row.get(f"{prefix}_ci_high", "")
    if mean in (None, "", "NA"):
        return ""
    return (
        f"{float(mean):.3f} +/- {float(sd):.3f} "
        f"(95% CI: {float(low):.3f} to {float(high):.3f})"
    )


script_dir = Path(__file__).resolve().parent
bundle_root = script_dir.parent
model_output_root = Path(os.getenv("MODEL_OUTPUT_ROOT", str(bundle_root / "outputs" / "publication_table2_grouped_cv_holdout_runs")))
summary_path = Path(os.getenv("TABLE2_SUMMARY_PATH", str(model_output_root / "table2_model_summary_detailed.csv")))
spec_path = Path(os.getenv("MODEL_SPEC_FILE", str(bundle_root / "04_models" / "revised_table2_rf_svm_model_specs.csv")))
out_dir = Path(os.getenv("TABLE2_OUTPUT_DIR", str(model_output_root)))
out_dir.mkdir(parents=True, exist_ok=True)

with summary_path.open(newline="") as f:
    summary_rows = list(csv.DictReader(f))
with spec_path.open(newline="") as f:
    spec_rows = [row for row in csv.DictReader(f) if row.get("enabled", "1") not in {"0", "false", "False"}]

by_model = {row["model_id"]: row for row in summary_rows}

expanded_rows = []
manuscript_rows = []
for spec in spec_rows:
    model_id = spec["model_id"]
    if model_id not in by_model:
        raise KeyError(f"Missing model summary row for {model_id}")
    row = by_model[model_id]

    expanded_rows.append(row)
    manuscript_rows.append({
        "Model features": spec["feature_label"],
        "Number of severity categories": spec["severity_categories"],
        "Algorithm": spec["algorithm"],
        "Holdout Accuracy": fmt_num(row.get("holdout_accuracy")),
        "Holdout Precision": fmt_num(row.get("holdout_precision_weighted")),
        "Holdout Recall": fmt_num(row.get("holdout_recall_weighted")),
        "Holdout F1": fmt_num(row.get("holdout_f1_weighted")),
        "Holdout QWK": fmt_num(row.get("holdout_qwk")),
        "Holdout MAE (class units)": fmt_num(row.get("holdout_mae_class")),
        "CV Accuracy mean +/- SD (95% CI)": fmt_cv(row, "cv_accuracy"),
        "CV Precision mean +/- SD (95% CI)": fmt_cv(row, "cv_precision_weighted"),
        "CV Recall mean +/- SD (95% CI)": fmt_cv(row, "cv_recall_weighted"),
        "CV F1 mean +/- SD (95% CI)": fmt_cv(row, "cv_f1_weighted"),
        "CV Macro F1 mean +/- SD (95% CI)": fmt_cv(row, "cv_macro_f1"),
        "CV QWK mean +/- SD (95% CI)": fmt_cv(row, "cv_qwk"),
        "CV MAE mean +/- SD (95% CI)": fmt_cv(row, "cv_mae_class"),
    })

expanded_path = out_dir / "table2_expanded.csv"
manuscript_path = out_dir / "table2_manuscript_ready.csv"
legacy_path = out_dir / "table2_revised_updated_12c_manuscript_ready.csv"

with expanded_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(expanded_rows[0].keys()))
    writer.writeheader()
    writer.writerows(expanded_rows)

with manuscript_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(manuscript_rows[0].keys()))
    writer.writeheader()
    writer.writerows(manuscript_rows)

with legacy_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(manuscript_rows[0].keys()))
    writer.writeheader()
    writer.writerows(manuscript_rows)

print(expanded_path)
print(manuscript_path)
