# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
# EDITED for TREC Classification task.

"""The Text REtrieval Conference (TREC) Question Classification dataset."""

import os
import json
import random
import requests
import numpy as np
import datasets
from sklearn.model_selection import train_test_split

_DESCRIPTION = """\
The Text REtrieval Conference (TREC) Question Classification dataset contains 5500 labeled questions in training set and another 500 for test set.
The dataset has 6 coarse class labels and 50 fine class labels.
"""

_URLs = {
    "train": "https://cogcomp.seas.upenn.edu/Data/QA/QC/train_5500.label",
    "test": "https://cogcomp.seas.upenn.edu/Data/QA/QC/TREC_10.label",
}

_COARSE_LABELS = ["ABBR", "ENTY", "DESC", "HUM", "LOC", "NUM"]

_FINE_LABELS = [
    "ABBR:abb", "ABBR:exp", "ENTY:animal", "ENTY:body", "ENTY:color", "ENTY:cremat", 
    "ENTY:currency", "ENTY:dismed", "ENTY:event", "ENTY:food", "ENTY:instru", 
    "ENTY:lang", "ENTY:letter", "ENTY:other", "ENTY:plant", "ENTY:product", 
    "ENTY:religion", "ENTY:sport", "ENTY:substance", "ENTY:symbol", "ENTY:techmeth", 
    "ENTY:termeq", "ENTY:veh", "ENTY:word", "DESC:def", "DESC:desc", "DESC:manner", 
    "DESC:reason", "HUM:gr", "HUM:ind", "HUM:title", "HUM:desc", "LOC:city", 
    "LOC:country", "LOC:mount", "LOC:other", "LOC:state", "NUM:code", "NUM:count", 
    "NUM:date", "NUM:dist", "NUM:money", "NUM:ord", "NUM:other", "NUM:period", 
    "NUM:perc", "NUM:speed", "NUM:temp", "NUM:volsize", "NUM:weight",
]

class Trec(datasets.GeneratorBasedBuilder):
    """The Text REtrieval Conference (TREC) Question Classification dataset."""

    VERSION = datasets.Version("2.0.0")

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "text": datasets.Value("string"),
                    "coarse_label": datasets.ClassLabel(names=_COARSE_LABELS),
                    "fine_label": datasets.ClassLabel(names=_FINE_LABELS),
                }
            ),
        )

    def _split_generators(self, dl_manager):
        dl_files = dl_manager.download(_URLs)
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": dl_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": dl_files["test"]}),
        ]

    def _generate_examples(self, filepath):
        with open(filepath, "rb") as f:
            for id_, row in enumerate(f):
                # Handle special byte and parsing
                fine_label, _, text = row.replace(b"\xf0", b" ").strip().decode("latin-1").partition(" ")
                coarse_label = fine_label.split(":")[0]
                yield id_, {
                    "text": text,
                    "coarse_label": coarse_label,
                    "fine_label": fine_label,
                }

# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data"))
SEED = 42

def download_raw_data():
    raw_dir = os.path.join(DATA_DIR, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    paths = {}
    for split, url in _URLs.items():
        path = os.path.join(raw_dir, f"{split}.label")
        if not os.path.exists(path):
            print(f"Downloading {url}...")
            # senza user agent non scarica (simula il browser)
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.37 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            try:
                r = requests.get(url, headers=headers, timeout=10)
                r.raise_for_status()
                with open(path, "wb") as f:
                    f.write(r.content)
            except Exception as e:
                print(f"\nERROR: Cannot download {url}")
                print(f"Error details: {e}")
                raise e
        paths[split] = path
    return paths

def prepare_data():
    """Download, parse, and generate all required subsets."""
    random.seed(SEED)
    np.random.seed(SEED)
    
    paths = download_raw_data()
    builder = Trec()
    
    # Setting up test set
    test_records = []
    for _, ex in builder._generate_examples(paths["test"]):
        test_records.append({
            "text": ex["text"],
            "label": _COARSE_LABELS.index(ex["coarse_label"]),
            "label_name": ex["coarse_label"]
        })
    
    with open(os.path.join(DATA_DIR, "test.json"), "w", encoding="utf-8") as f:
        json.dump(test_records, f, indent=4)
    print(f"Test set salvato ({len(test_records)} campioni)")

    # Mapping classes
    with open(os.path.join(DATA_DIR, "class_names.json"), "w", encoding="utf-8") as f:
        json.dump(_COARSE_LABELS, f, indent=4)

    # Setting up training subsets
    train_records = []
    for _, ex in builder._generate_examples(paths["train"]):
        train_records.append({
            "text": ex["text"],
            "label": _COARSE_LABELS.index(ex["coarse_label"]),
            "label_name": ex["coarse_label"]
        })
    
    y = [x["label"] for x in train_records]
    
    # Saving train full (N)
    with open(os.path.join(DATA_DIR, "train_full.json"), "w", encoding="utf-8") as f:
        json.dump(train_records, f, indent=4)
    print(f"Train Full saved ({len(train_records)} campioni)")

    # Subsets 0, 1, 10, 100, 1000
    for size in [0, 1, 10, 100, 1000]:
        if size == 0:
            subset = []
        elif size == 1:
            subset = [random.choice(train_records)]
        else:
            try:
                indices, _ = train_test_split(
                    range(len(train_records)),
                    train_size=size,
                    stratify=y,
                    random_state=SEED
                )
                subset = [train_records[i] for i in indices]
            except ValueError:
                print(f"Stratificazione in errore per size={size}, uso random sampling.")
                subset = random.sample(train_records, size)
        
        subset_path = os.path.join(DATA_DIR, f"train_{size}.json")
        with open(subset_path, "w", encoding="utf-8") as f:
            json.dump(subset, f, indent=4)
        print(f"Subset size {size} salvato in {subset_path}")

if __name__ == "__main__":
    prepare_data()
