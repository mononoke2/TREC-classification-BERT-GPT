"""
Metrics module for TREC classification evaluation.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ELE TASK: implement the four functions below
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Each function receives:
    true_labels      : list[int]  — ground truth class indices
    predicted_labels : list[int]  — model predictions as class indices

Each function must return a single float.

Once implemented, run the evaluation framework with:
    python evaluate.py
"""


def compute_accuracy(true_labels: list, predicted_labels: list) -> float:
    """Return accuracy score."""
    raise NotImplementedError("TODO: implement compute_accuracy()")


def compute_f1_micro(true_labels: list, predicted_labels: list) -> float:
    """Return F1 score with micro averaging."""
    raise NotImplementedError("TODO: implement compute_f1_micro()")


def compute_f1_macro(true_labels: list, predicted_labels: list) -> float:
    """Return F1 score with macro averaging."""
    raise NotImplementedError("TODO: implement compute_f1_macro()")


def compute_f1_weighted(true_labels: list, predicted_labels: list) -> float:
    """Return F1 score with weighted averaging."""
    raise NotImplementedError("TODO: implement compute_f1_weighted()")
