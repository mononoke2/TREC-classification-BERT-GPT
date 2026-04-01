"""
BERT Fine-tuning for TREC Question Classification.

Usage:
    python train_bert.py --train_size 100
    python train_bert.py --train_size 0   
    python train_bert.py --train_size N  

Training strategy:
    - train_size == 0         : zero-shot via NLI pipeline (facebook/bart-large-mnli)
                                No BERT fine-tuning. Uses natural language descriptions
                                of TREC classes as NLI hypothesis labels.
    - train_size < 50         : fixed epochs (MAX_EPOCHS), no early stopping
                                (too few samples to create a meaningful val split)
    - train_size >= 50        : early stopping on a 20% validation split,
                                with patience EARLY_STOPPING_PATIENCE

Output:
    Predictions saved to src/evaluation/results/bert_train_{size}.csv
    with columns: [text, true_label, true_label_name, predicted_label, predicted_label_name]
"""

import copy
import json
import os
import sys
import argparse

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_DIR = os.path.join(ROOT, "data")

SHARED_DIR = os.path.join(ROOT, "src", "shared")
sys.path.append(SHARED_DIR)
from trec import _COARSE_LABELS
from save_predictions import save_predictions

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_NAME = "bert-base-uncased"
NLI_MODEL_NAME = "facebook/bart-large-mnli"  # used only for zero-shot (train_size=0)
MAX_LEN = 64
BATCH_SIZE = 16
MAX_EPOCHS = 10
LEARNING_RATE = 2e-5
SEED = 42
NUM_LABELS = len(_COARSE_LABELS)   

MAX_EPOCHS_WITH_EARLY_STOPPING = 50 # Upper bound on epochs when early stopping is active (for large datasets)
EARLY_STOPPING_MIN_TRAIN_SIZE = 50  # Below this, skip early stopping
EARLY_STOPPING_PATIENCE = 3         # Stop if val loss doesn't improve for N epochs
VAL_SPLIT = 0.2                     # Fraction of training set used as validation

# The NLI model computes: P(text entails "This is a <description> question")
# More descriptive labels will have better NLI performance rather than raw codes like "ABBR".
NLI_LABEL_DESCRIPTIONS = {
    "ABBR": "abbreviation or acronym question",
    "ENTY": "entity question about a thing",
    "DESC": "description or definition question",
    "HUM":  "question about a person or group",
    "LOC":  "location question",
    "NUM":  "numeric or quantity question",
}
# Ordered list matching _COARSE_LABELS indices
NLI_CANDIDATES = [NLI_LABEL_DESCRIPTIONS[lbl] for lbl in _COARSE_LABELS]
NLI_DESC_TO_IDX = {desc: idx for idx, desc in enumerate(NLI_CANDIDATES)}


def set_seed(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TrecDataset(Dataset):
    def __init__(self, records, tokenizer):
        self.records = records
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        enc = self.tokenizer(
            r["text"],
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(r["label"], dtype=torch.long),
            "text": r["text"],
            "label_name": r["label_name"],
        }


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_split(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_train_path(size):
    if str(size) == "N":
        return os.path.join(DATA_DIR, "train_full.json")
    return os.path.join(DATA_DIR, f"train_{size}.json")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def eval_loss(model, loader, device):
    """Compute average loss on a validation loader."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()
    return total_loss / len(loader)


def train_with_early_stopping(model, train_loader, val_loader, optimizer, scheduler, device):
    """Train with early stopping based on validation loss."""
    best_val_loss = float("inf")
    best_weights = None
    patience_counter = 0

    for epoch in range(1, MAX_EPOCHS_WITH_EARLY_STOPPING + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_l = eval_loss(model, val_loader, device)
        print(f"  Epoch {epoch} — Train Loss: {train_loss:.4f} | Val Loss: {val_l:.4f}", end="")

        if val_l < best_val_loss:
            best_val_loss = val_l
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print("  ✓ best")
        else:
            patience_counter += 1
            print(f"  (patience {patience_counter}/{EARLY_STOPPING_PATIENCE})")

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"  Early stopping triggered at epoch {epoch}.")
            break

    # Restore best model
    model.load_state_dict(best_weights)
    return model


def train_fixed_epochs(model, train_loader, optimizer, scheduler, device):
    """Train for a fixed number of epochs (used for small datasets)."""
    for epoch in range(MAX_EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"  Epoch {epoch + 1}/{MAX_EPOCHS} — Loss: {train_loss:.4f}")
    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict(model, loader, device):
    model.eval()
    results = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()

            for i, pred in enumerate(preds):
                results.append({
                    "text": batch["text"][i],
                    "true_label": batch["label"][i].item(),
                    "true_label_name": batch["label_name"][i],
                    "predicted_label": int(pred),
                    "predicted_label_name": _COARSE_LABELS[int(pred)],
                })
    return results


# ---------------------------------------------------------------------------
# Zero-shot NLI predictor
# ---------------------------------------------------------------------------

def predict_zero_shot_nli(test_records):
    """
    True zero-shot classification using a pre-trained NLI model.
    No BERT fine-tuning is involved. The NLI model scores the probability
    that each text entails each class description.
    """
    from transformers import pipeline as hf_pipeline

    print(f"Loading NLI model: {NLI_MODEL_NAME} ...")
    nli_clf = hf_pipeline(
        "zero-shot-classification",
        model=NLI_MODEL_NAME,
        device=0 if torch.cuda.is_available() else -1,
    )

    results = []
    print(f"Running zero-shot NLI on {len(test_records)} examples...")
    for record in test_records:
        out = nli_clf(record["text"], candidate_labels=NLI_CANDIDATES)
        # out["labels"][0] is the highest-scoring description
        pred_desc = out["labels"][0]
        pred_idx = NLI_DESC_TO_IDX[pred_desc]
        results.append({
            "text": record["text"],
            "true_label": record["label"],
            "true_label_name": record["label_name"],
            "predicted_label": pred_idx,
            "predicted_label_name": _COARSE_LABELS[pred_idx],
        })
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="BERT TREC Classifier")
    parser.add_argument(
        "--train_size", required=True,
        help="Number of training samples (0, 1, 10, 100, 1000, N)"
    )
    args = parser.parse_args()

    train_size = args.train_size if args.train_size == "N" else int(args.train_size)

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Training size: {train_size}")

    # Test set 
    test_records = load_split(os.path.join(DATA_DIR, "test.json"))

    # -------------------------------------------------------------------
    # Zero-shot: use NLI pipeline, no BERT fine-tuning
    # -------------------------------------------------------------------
    if train_size == 0:
        print("\nZero-shot mode: using NLI pipeline (no fine-tuning).")
        predictions = predict_zero_shot_nli(test_records)
        save_predictions(predictions, model_name="bert", train_size=train_size)
        print("Done.")
        return

    # -------------------------------------------------------------------
    # Few-shot / full training: standard BERT fine-tuning
    # -------------------------------------------------------------------
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
    model.to(device)

    test_dataset = TrecDataset(test_records, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    train_records = load_split(get_train_path(train_size))
    if len(train_records) == 0:
        print("No training data found. Falling back to zero-shot NLI.")
        predictions = predict_zero_shot_nli(test_records)
    else:
        use_early_stopping = (
            train_size == "N" or train_size >= EARLY_STOPPING_MIN_TRAIN_SIZE
        )

        if use_early_stopping:
            # Split training set into train and validation
            labels_for_split = [r["label"] for r in train_records]
            train_sub, val_sub = train_test_split(
                train_records,
                test_size=VAL_SPLIT,
                stratify=labels_for_split,
                random_state=SEED,
            )
            train_loader = DataLoader(
                TrecDataset(train_sub, tokenizer), batch_size=BATCH_SIZE, shuffle=True
            )
            val_loader = DataLoader(
                TrecDataset(val_sub, tokenizer), batch_size=BATCH_SIZE
            )
            print(
                f"\nEarly stopping enabled — "
                f"train: {len(train_sub)} | val: {len(val_sub)} | "
                f"patience: {EARLY_STOPPING_PATIENCE}\n"
            )
            total_steps = len(train_loader) * MAX_EPOCHS_WITH_EARLY_STOPPING  # upper bound
        else:
            train_loader = DataLoader(
                TrecDataset(train_records, tokenizer), batch_size=BATCH_SIZE, shuffle=True
            )
            val_loader = None
            print(
                f"\nFixed epochs ({MAX_EPOCHS}) — "
                f"train_size={train_size} < {EARLY_STOPPING_MIN_TRAIN_SIZE} "
                f"(early stopping disabled)\n"
            )
            total_steps = len(train_loader) * MAX_EPOCHS

        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )

        if use_early_stopping:
            model = train_with_early_stopping(
                model, train_loader, val_loader, optimizer, scheduler, device
            )
        else:
            model = train_fixed_epochs(
                model, train_loader, optimizer, scheduler, device
            )

        # Inference on test set
        print("\nRunning inference on test set...")
        predictions = predict(model, test_loader, device)

    save_predictions(predictions, model_name="bert", train_size=train_size)
    print("Done.")


if __name__ == "__main__":
    main()
