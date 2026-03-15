"""
Shared utility for saving model predictions in a standardized CSV format.

Both BERT and GPT training scripts should use this function to ensure
compatible output files for the evaluation framework.

Output format:
    text | true_label | true_label_name | predicted_label | predicted_label_name
"""

import os
import csv

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
RESULTS_DIR = os.path.join(ROOT, "src", "evaluation", "results")


def save_predictions(predictions: list[dict], model_name: str, train_size) -> str:
    """
    Save predictions to a standardized CSV file.

    Parameters
    ----------
    predictions : list of dicts, each with keys:
        - text              : str
        - true_label        : int
        - true_label_name   : str
        - predicted_label   : int
        - predicted_label_name : str
    model_name  : str  — e.g. 'bert' or 'gpt'
    train_size  : int | 'N' — size of the training set used

    Returns
    -------
    str  — path to the saved CSV file
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(RESULTS_DIR, f"{model_name}_train_{train_size}.csv")

    fieldnames = [
        "text",
        "true_label",
        "true_label_name",
        "predicted_label",
        "predicted_label_name",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(predictions)

    print(f"Predictions saved → {output_path}")
    return output_path
