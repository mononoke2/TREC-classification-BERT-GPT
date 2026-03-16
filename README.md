# TREC Question Classification — BERT & GPT

Progetto per la classificazione delle domande presenti nel dataset **TREC** in 6 categorie
(coarse labels) utilizzando **BERT** (fine-tuning) e **GPT** (Supervised Fine-Tuning).

---

## Struttura del Progetto

```
.
├── data/                          # Dataset generato da src/shared/trec.py
│   ├── raw/                       # File .label originali scaricati dai server CogComp
│   ├── class_names.json           # ["ABBR", "ENTY", "DESC", "HUM", "LOC", "NUM"]
│   ├── test.json                  # Test set ufficiale (500 esempi, invariato)
│   ├── train_0.json               # Subset 0 esempi (vuoto)
│   ├── train_1.json               # Subset  1 esempio  (stratificato)
│   ├── train_10.json              # Subset 10 esempi   (stratificato)
│   ├── train_100.json             # Subset 100 esempi  (stratificato, seed=42)
│   ├── train_1000.json            # Subset 1000 esempi (stratificato, seed=42)
│   └── train_full.json            # Training set completo (N = 5452 esempi)
│
├── src/
│   ├── shared/
│   │   ├── trec.py                # Download, parsing e generazione subset
│   │   └── save_predictions.py   # Utility condivisa: salva CSV di predizioni
│   │
│   ├── bert/
│   │   └── train_bert.py         # Training BERT + inferenza su test set
│   │
│   ├── gpt/                      # GPT SFT Pipeline
│   │   └── train_gpt.py          # SFT del modello GPT (Completato)
│   │
│   └── evaluation/
│       ├── metrics.py             # Metriche (Completato)
│       ├── evaluate.py            # Framework: genera la tabella comparativa
│       └── results/               # CSV di predizioni (BERT + GPT)
│           ├── bert_train_0.csv
│           ├── bert_train_10.csv
│           └── ...
│
├── requirements.txt
└── README.md
```

---

## Dataset: TREC

Il dataset è caricato tramite lo script ufficiale `src/shared/trec.py` (basato su
[CogComp/trec](https://huggingface.co/datasets/CogComp/trec) su Hugging Face).

**6 coarse labels:**

| ID  | Label | Descrizione               |
| --- | ----- | ------------------------- |
| 0   | ABBR  | Abbreviazione / acronimo  |
| 1   | ENTY  | Entità (cosa, oggetto)    |
| 2   | DESC  | Descrizione / definizione |
| 3   | HUM   | Persona o gruppo          |
| 4   | LOC   | Luogo                     |
| 5   | NUM   | Numero o quantità         |

**Split:** 5452 esempi di training — 500 esempi di test (fissi, mai usati per training/validation).

---

## Setup

```bash
# Crea e attiva il virtual environment
python -m venv .venv
source .venv/bin/activate

# Installa le dipendenze
pip install -r requirements.txt

# Scarica il dataset e genera i subset (una volta sola)
python src/shared/trec.py
```

---

## Flusso Completo

```
src/shared/trec.py
        │
        ▼ genera data/train_*.json e data/test.json
        │
        ├──► src/bert/train_bert.py --train_size {0,1,10,100,1000,N}
        │             │
        │             ▼ salva src/evaluation/results/bert_train_{size}.csv
        │
        └──► src/gpt/train_gpt.py --train_size {0,1,10,100,1000,N}   [Completato]
                      │
                      ▼ salva src/evaluation/results/gpt_train_{size}.csv
                      │
                      ▼
             src/evaluation/metrics.py   [Completato]
                      │
                      ▼
             src/evaluation/evaluate.py  → tabella comparativa finale
```

---

## BERT — Istruzioni

```bash
cd src/bert

python train_bert.py --train_size 0     # Zero-shot via NLI (facebook/bart-large-mnli)
python train_bert.py --train_size 1
python train_bert.py --train_size 10
python train_bert.py --train_size 100
python train_bert.py --train_size 1000
python train_bert.py --train_size N     # Training set completo
```

**Strategia di training:**

| Train size         | Modalità                                                                                                                                       |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `0`                | Zero-shot NLI — nessun fine-tuning, usa `facebook/bart-large-mnli`                                                                             |
| `1`, `10`          | Fine-tuning con **epoche fisse** (`MAX_EPOCHS=10`) — dataset troppo piccolo per validation split, epoche più grandi porterebbero a overfitting |
| `100`, `1000`, `N` | Fine-tuning con **early stopping** (patience=3, val split 20%)                                                                                 |

**Output:** `src/evaluation/results/bert_train_{size}.csv`

---

## Output CSV — Formato Standard

Ogni file di predizioni (BERT e GPT) ha questo formato:

| text                          | true_label | true_label_name | predicted_label | predicted_label_name |
| ----------------------------- | ---------- | --------------- | --------------- | -------------------- |
| What is the capital of Italy? | 4          | LOC             | 4               | LOC                  |
| Who invented the telephone?   | 3          | HUM             | 3               | HUM                  |

- `true_label` / `predicted_label`: indice intero (0–5)
- `true_label_name` / `predicted_label_name`: stringa leggibile (`ABBR`, `ENTY`, ...)

---

## Evaluation — Istruzioni

Dopo aver generato i CSV di predizioni (BERT + GPT), lancia:

```bash
python src/evaluation/evaluate.py
```

Il framework legge automaticamente tutti i file in `results/` e stampa una tabella
comparativa. Le metriche devono essere implementate in `src/evaluation/metrics.py`.

---

---

## Completato

### 1. GPT Zero-shot (train_size = 0)

Per una comparazione valida con BERT zero-shot, il GPT deve essere usato in modalità
**zero-shot**: nessun fine-tuning, solo il modello base con un prompt che descrive il task.

Esempio di prompt:
```
Classify the following question into one of these categories:
ABBR (abbreviation), ENTY (entity), DESC (description), HUM (human), LOC (location), NUM (numeric).

Question: Who invented the telephone?
Category:
```

Il modello deve rispondere con una singola label (`HUM`, `LOC`, ecc.).

---

### 2. GPT con SFT (Supervised Fine-Tuning) — train_size = 1, 10, 100, 1000, N

Sviluppato `src/gpt/train_gpt.py` che:

- [X] Carica il subset di training da `data/train_{size}.json`
- [X] Formatta i dati come coppie instruction/completion per SFT:
  ```json
  {"prompt": "Classify: Who invented the telephone?\nCategory:", "completion": " HUM"}
  ```
  Nota: "completion": si prende da "label_name"
- [X] Esegue il fine-tuning del modello GPT per i diversi train_size
- [X] Lancia l'inferenza sul test set `data/test.json`
    Durante l'inferenza GPT, il modello genera una stringa (es. `"HUM"`).
    Prima di salvare il CSV, è necessario convertirla in indice intero, perché il
    framework si aspetta `predicted_label` come intero (0–5).

    ```python
    import sys
    sys.path.append("src/shared")
    from trec import _COARSE_LABELS
    # _COARSE_LABELS = ["ABBR", "ENTY", "DESC", "HUM", "LOC", "NUM"]

    generated_text = "HUM"                               # output grezzo del modello
    predicted_label = _COARSE_LABELS.index(generated_text)  # → 3
    predicted_label_name = generated_text                # → "HUM"
    ```

    > **Nota:** se il modello genera un output inatteso (es. `"Human"` invece di `"HUM"`),
    > occorre un meccanismo di fallback (es. mappare al label più simile o assegnare -1).
- [X] Salva le predizioni usando la funzione condivisa:
  ```python
  import sys; sys.path.append("src/shared")
  from save_predictions import save_predictions
  save_predictions(predictions, model_name="gpt", train_size=size)
  ```

**Formato delle predizioni** (deve essere identico a quello di BERT):

```python
predictions = [
    {
        "text": "Who invented the telephone?",
        "true_label": 3,
        "true_label_name": "HUM",
        "predicted_label": 3,
        "predicted_label_name": "HUM",
    },
    ...
]
```

---

### 3. Metriche — `src/evaluation/metrics.py`

Implementate le 4 funzioni:

```python
from sklearn.metrics import accuracy_score, f1_score

def compute_accuracy(true_labels, predicted_labels) -> float:
    return accuracy_score(true_labels, predicted_labels)

def compute_f1_micro(true_labels, predicted_labels) -> float:
    return f1_score(true_labels, predicted_labels, average="micro")

def compute_f1_macro(true_labels, predicted_labels) -> float:
    return f1_score(true_labels, predicted_labels, average="macro")

def compute_f1_weighted(true_labels, predicted_labels) -> float:
    return f1_score(true_labels, predicted_labels, average="weighted")
```

Generata la tabella comparativa finale BERT vs GPT.

---
