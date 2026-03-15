"""
Evaluation framework for TREC classification.

Automatically scans src/evaluation/results/ for prediction CSV files,
calls the metric functions defined in metrics.py for each file,
and prints a comparison table across models and training set sizes.

Usage:
    python evaluate.py

Expected filename format of prediction files:
    {model_name}_train_{train_size}.csv
    e.g. bert_train_100.csv, gpt_train_1000.csv, bert_train_N.csv

Prediction CSV format (required columns):
    text | true_label | true_label_name | predicted_label | predicted_label_name
"""

import os
import sys
import csv
import re

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
RESULTS_DIR = os.path.join(ROOT, "src", "evaluation", "results")

# Import metric implementations
sys.path.insert(0, os.path.dirname(__file__))
from metrics import (
    compute_accuracy,
    compute_f1_micro,
    compute_f1_macro,
    compute_f1_weighted,
)

# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

# Expected pattern: bert_train_100.csv or gpt_train_N.csv
_FILENAME_PATTERN = re.compile(r"^(?P<model>.+)_train_(?P<size>\w+)\.csv$")

# Canonical order for training size columns
_SIZE_ORDER = ["0", "1", "10", "100", "1000", "N"]


def _parse_filename(filename: str):
    """Return (model_name, train_size_str) or None if the filename doesn't match."""
    m = _FILENAME_PATTERN.match(filename)
    if m:
        return m.group("model"), m.group("size")
    return None


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_predictions(csv_path: str) -> tuple[list[int], list[int]]:
    """
    Load a predictions CSV and return (true_labels, predicted_labels).
    Both are lists of integers.
    """
    true_labels, predicted_labels = [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            true_labels.append(int(row["true_label"]))
            predicted_labels.append(int(row["predicted_label"]))
    return true_labels, predicted_labels


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def run_metrics(true_labels: list, predicted_labels: list) -> dict:
    """
    Call all four metric functions and return a dict of results.
    If a function raises NotImplementedError, reports 'N/A'.
    """
    metrics = {
        "accuracy": compute_accuracy,
        "f1_micro": compute_f1_micro,
        "f1_macro": compute_f1_macro,
        "f1_weighted": compute_f1_weighted,
    }
    results = {}
    for name, fn in metrics.items():
        try:
            results[name] = fn(true_labels, predicted_labels)
        except NotImplementedError:
            results[name] = None
    return results


# ---------------------------------------------------------------------------
# Table rendering
# ---------------------------------------------------------------------------

def _format_cell(value) -> str:
    if value is None:
        return "  N/A  "
    return f" {value:.4f} "


def print_results_table(all_results: dict):
    """
    Print a formatted comparison table.
    all_results: { (model, size): {metric: value} }
    """
    models = sorted(set(m for m, _ in all_results))
    metric_names = ["accuracy", "f1_micro", "f1_macro", "f1_weighted"]

    for model in models:
        print(f"\n{'='*65}")
        print(f"  Model: {model.upper()}")
        print(f"{'='*65}")

        # Header row
        header = f"{'Metric':<15}" + "".join(f"{s:>10}" for s in _SIZE_ORDER)
        print(header)
        print("-" * len(header))

        for metric in metric_names:
            row = f"{metric:<15}"
            for size in _SIZE_ORDER:
                key = (model, size)
                if key in all_results:
                    row += f"{_format_cell(all_results[key].get(metric)):>10}"
                else:
                    row += f"{'  ---  ':>10}"
            print(row)

    print(f"\n{'='*65}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not os.path.isdir(RESULTS_DIR):
        print(f"Results directory not found: {RESULTS_DIR}")
        print("Run some training experiments first.")
        return

    csv_files = [f for f in os.listdir(RESULTS_DIR) if f.endswith(".csv")]
    if not csv_files:
        print("No prediction CSV files found in results/. Run training first.")
        return

    print(f"Found {len(csv_files)} prediction file(s) in {RESULTS_DIR}:\n")
    all_results = {}
    errors = []

    for filename in sorted(csv_files):
        parsed = _parse_filename(filename)
        if parsed is None:
            print(f"  [SKIP] {filename} — filename does not match expected pattern")
            continue

        model_name, train_size = parsed
        csv_path = os.path.join(RESULTS_DIR, filename)
        print(f"  Processing: {filename}")

        try:
            true_labels, predicted_labels = load_predictions(csv_path)
            results = run_metrics(true_labels, predicted_labels)
            all_results[(model_name, train_size)] = results
        except Exception as e:
            errors.append((filename, str(e)))
            print(f"    ERROR: {e}")

    if all_results:
        print_results_table(all_results)

    if errors:
        print("\nErrors encountered:")
        for f, e in errors:
            print(f"  {f}: {e}")

    # Check if any metrics are still unimplemented
    unimplemented = set()
    for results in all_results.values():
        unimplemented.update(k for k, v in results.items() if v is None)
    if unimplemented:
        print(
            f"\n⚠ The following metrics are not yet available in metrics.py: "
            f"{', '.join(sorted(unimplemented))}"
            f"If you want to use them, please implement them in metrics.py."
        )


if __name__ == "__main__":
    main()
