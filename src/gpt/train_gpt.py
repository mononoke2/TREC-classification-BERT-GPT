"""
GPT-2 Supervised Fine-Tuning (SFT) for TREC Question Classification.

Usage:
    python train_gpt.py --train_size 100
    python train_gpt.py --train_size 0   # zero-shot (no fine-tuning)
    python train_gpt.py --train_size N   # full training set

Training strategy:
    - train_size == 0         : zero-shot — no fine-tuning, GPT-2 base generates
                                the label directly from the prompt.
    - train_size < 50         : fixed epochs (MAX_EPOCHS), no early stopping
                                (too few samples to create a meaningful val split)
    - train_size >= 50        : early stopping on a 20% validation split,
                                with patience EARLY_STOPPING_PATIENCE

Output:
    Predictions saved to src/evaluation/results/gpt_train_{size}.csv
    with columns: [text, true_label, true_label_name, predicted_label, predicted_label_name]
"""

import copy
import json
import os
import sys
import argparse
import time

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    GPT2TokenizerFast,
    GPT2LMHeadModel,
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

MODEL_NAME = "gpt2"
MAX_LEN = 128
BATCH_SIZE = 8
MAX_EPOCHS = 10
LEARNING_RATE = 5e-5
SEED = 42
NUM_LABELS = len(_COARSE_LABELS)  # 6

MAX_EPOCHS_WITH_EARLY_STOPPING = 30
EARLY_STOPPING_MIN_TRAIN_SIZE = 50
EARLY_STOPPING_PATIENCE = 3
VAL_SPLIT = 0.2

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "Classify the following question into one of these categories: "
    "ABBR, ENTY, DESC, HUM, LOC, NUM."
)


def make_prompt(text: str, label_name: str | None = None) -> str:
    """Build the SFT prompt. If label_name is None, omit it (inference mode)."""
    prompt = f"{SYSTEM_PROMPT}\nQuestion: {text}\nCategory:"
    if label_name is not None:
        prompt += f" {label_name}"
    return prompt


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------


def set_seed(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class SFTDataset(Dataset):
    """
    Tokenizes prompt+label for causal LM training.
    The loss is masked on prompt tokens (set to -100) so
    the model only learns to generate the label.
    """

    def __init__(self, records, tokenizer, max_len=MAX_LEN):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.input_ids_list = []
        self.labels_list = []
        self.attention_mask_list = []

        for r in records:
            full_text = make_prompt(r["text"], r["label_name"])
            prompt_text = make_prompt(r["text"], label_name=None)

            full_enc = tokenizer(
                full_text,
                max_length=max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            prompt_enc = tokenizer(
                prompt_text,
                max_length=max_len,
                truncation=True,
                return_tensors="pt",
            )

            input_ids = full_enc["input_ids"].squeeze(0)
            attention_mask = full_enc["attention_mask"].squeeze(0)

            # Build labels: -100 for prompt tokens + padding, real ids for label tokens
            labels = input_ids.clone()
            prompt_len = prompt_enc["input_ids"].shape[1]
            labels[:prompt_len] = -100
            # Also mask padding tokens
            labels[attention_mask == 0] = -100

            self.input_ids_list.append(input_ids)
            self.labels_list.append(labels)
            self.attention_mask_list.append(attention_mask)

    def __len__(self):
        return len(self.input_ids_list)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids_list[idx],
            "attention_mask": self.attention_mask_list[idx],
            "labels": self.labels_list[idx],
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
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
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
            labels = batch["labels"].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
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
        print(
            f"  Epoch {epoch} — Train Loss: {train_loss:.4f} | Val Loss: {val_l:.4f}",
            end="",
        )

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


def map_output_to_label(generated_text: str) -> tuple[int, str]:
    """
    Map the generated text to one of the 6 coarse labels.

    Strategy:
      1. Exact match (case-insensitive)
      2. Substring match (e.g. "Abbreviation" → ABBR)
      3. Fallback: return -1, "UNKNOWN"
    """
    text = generated_text.strip().upper()

    # 1. Exact match
    if text in _COARSE_LABELS:
        idx = _COARSE_LABELS.index(text)
        return idx, _COARSE_LABELS[idx]

    # 2. Check if any label is a prefix of the output
    for i, label in enumerate(_COARSE_LABELS):
        if text.startswith(label):
            return i, label

    # 3. Keyword-based fallback
    keyword_map = {
        "ABBREVIATION": 0,
        "ENTITY": 1,
        "DESCRIPTION": 2,
        "DEFINITION": 2,
        "HUMAN": 3,
        "PERSON": 3,
        "LOCATION": 4,
        "PLACE": 4,
        "NUMBER": 5,
        "NUMERIC": 5,
        "QUANTITY": 5,
    }
    for keyword, idx in keyword_map.items():
        if keyword in text:
            return idx, _COARSE_LABELS[idx]

    # 4. Fallback
    return -1, "UNKNOWN"


def predict_with_logits_fallback(
    model, tokenizer, text: str, device
) -> tuple[int, str]:
    """
    Generate the label for a single question.
    If generation-based mapping fails, fall back to comparing
    the logits of each label token directly.
    """
    prompt = make_prompt(text, label_name=None)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the generated tokens (strip the prompt)
    generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    idx, label = map_output_to_label(generated_text)

    if idx != -1:
        return idx, label

    # Fallback: check logits for each label token
    with torch.no_grad():
        logits = model(**inputs).logits[:, -1, :]  # logits for next token

    label_token_ids = [tokenizer.encode(f" {lbl}", add_special_tokens=False)[0] for lbl in _COARSE_LABELS]
    label_logits = logits[0, label_token_ids]
    best_idx = label_logits.argmax().item()
    return best_idx, _COARSE_LABELS[best_idx]


def predict_all(model, tokenizer, test_records, device):
    """Run inference on all test records."""
    model.eval()
    results = []
    total = len(test_records)

    for i, record in enumerate(test_records):
        pred_idx, pred_label = predict_with_logits_fallback(
            model, tokenizer, record["text"], device
        )
        results.append(
            {
                "text": record["text"],
                "true_label": record["label"],
                "true_label_name": record["label_name"],
                "predicted_label": pred_idx,
                "predicted_label_name": pred_label,
            }
        )
        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f"  Inference: {i + 1}/{total}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="GPT-2 SFT TREC Classifier")
    parser.add_argument(
        "--train_size",
        required=True,
        help="Number of training samples (0, 1, 10, 100, 1000, N)",
    )
    args = parser.parse_args()

    train_size = args.train_size if args.train_size == "N" else int(args.train_size)

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Training size: {train_size}")

    # Load tokenizer and model
    print(f"\nLoading model: {MODEL_NAME} ...")
    tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token  # GPT-2 has no pad token by default

    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    model.config.pad_token_id = tokenizer.eos_token_id
    model.to(device)

    # Test set (always the same)
    test_records = load_split(os.path.join(DATA_DIR, "test.json"))

    # -------------------------------------------------------------------
    # Zero-shot: no fine-tuning, use GPT-2 base directly
    # -------------------------------------------------------------------
    if train_size == 0:
        print("\nZero-shot mode: using GPT-2 base (no fine-tuning).")
        print(f"Running inference on {len(test_records)} test examples...\n")
        start = time.time()
        predictions = predict_all(model, tokenizer, test_records, device)
        elapsed = time.time() - start
        print(f"\nInference completed in {elapsed:.1f}s")
        save_predictions(predictions, model_name="gpt", train_size=train_size)
        print("Done.")
        return

    # -------------------------------------------------------------------
    # SFT: fine-tune GPT-2 on training data
    # -------------------------------------------------------------------
    train_records = load_split(get_train_path(train_size))
    if len(train_records) == 0:
        print("No training data found. Falling back to zero-shot.")
        predictions = predict_all(model, tokenizer, test_records, device)
        save_predictions(predictions, model_name="gpt", train_size=train_size)
        print("Done.")
        return

    use_early_stopping = train_size == "N" or (
        isinstance(train_size, int) and train_size >= EARLY_STOPPING_MIN_TRAIN_SIZE
    )

    if use_early_stopping:
        labels_for_split = [r["label"] for r in train_records]
        train_sub, val_sub = train_test_split(
            train_records,
            test_size=VAL_SPLIT,
            stratify=labels_for_split,
            random_state=SEED,
        )
        train_dataset = SFTDataset(train_sub, tokenizer)
        val_dataset = SFTDataset(val_sub, tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
        print(
            f"\nEarly stopping enabled — "
            f"train: {len(train_sub)} | val: {len(val_sub)} | "
            f"patience: {EARLY_STOPPING_PATIENCE}\n"
        )
        total_steps = len(train_loader) * MAX_EPOCHS_WITH_EARLY_STOPPING
    else:
        train_dataset = SFTDataset(train_records, tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
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

    print("Starting SFT training...")
    start = time.time()

    if use_early_stopping:
        model = train_with_early_stopping(
            model, train_loader, val_loader, optimizer, scheduler, device
        )
    else:
        model = train_fixed_epochs(model, train_loader, optimizer, scheduler, device)

    train_elapsed = time.time() - start
    print(f"\nTraining completed in {train_elapsed:.1f}s")

    # Inference on test set
    print(f"\nRunning inference on {len(test_records)} test examples...")
    start = time.time()
    predictions = predict_all(model, tokenizer, test_records, device)
    infer_elapsed = time.time() - start
    print(f"Inference completed in {infer_elapsed:.1f}s")

    save_predictions(predictions, model_name="gpt", train_size=train_size)
    print("Done.")


if __name__ == "__main__":
    main()
