"""
Metrics module for TREC classification evaluation.

Each function receives:
    true_labels      : list[int]  — ground truth class indices
    predicted_labels : list[int]  — model predictions as class indices

Each function must return a single float.

Once implemented, run the evaluation framework with:
    python evaluate.py
"""

from sklearn.metrics import accuracy_score, f1_score


def compute_accuracy(true_labels: list, predicted_labels: list) -> float:
    """Return accuracy score."""
    return accuracy_score(true_labels, predicted_labels)


def compute_f1_micro(true_labels: list, predicted_labels: list) -> float:
    """Return F1 score with micro averaging."""
    return f1_score(true_labels, predicted_labels, average="micro")


def compute_f1_macro(true_labels: list, predicted_labels: list) -> float:
    """Return F1 score with macro averaging."""
    return f1_score(true_labels, predicted_labels, average="macro")


def compute_f1_weighted(true_labels: list, predicted_labels: list) -> float:
    """Return F1 score with weighted averaging."""
    return f1_score(true_labels, predicted_labels, average="weighted")
