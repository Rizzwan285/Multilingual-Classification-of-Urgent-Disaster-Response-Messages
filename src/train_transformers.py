"""
Transformer Fine-Tuning for HumAID Disaster Classification
-----------------------------------------------------------
Fine-tunes ONE multilingual transformer (mBERT, XLM-RoBERTa, MuRIL, or
IndicBERT) on the augmented HumAID training set. Evaluates on dev during
training and on test at the end. Designed to be invoked once per model,
which matches a SLURM-per-job cluster workflow.

Usage:
    python src/train_transformer.py --model_name muril
    python src/train_transformer.py --model_name xlm_roberta --run_id my_exp_1

Available model_name values:
    mbert         -> bert-base-multilingual-cased
    xlm_roberta   -> xlm-roberta-base
    muril         -> google/muril-base-cased
    indic_bert    -> ai4bharat/indic-bert

Outputs (all under results/<model_name>_<run_id>/):
    checkpoints/            HuggingFace intermediate checkpoints
    final_model/            Best model + tokenizer for downstream eval
    logs/                   TensorBoard / Trainer logs
    test_metrics.json       Final test-set metrics (for reporting)
    dev_metrics.json        Best-epoch dev-set metrics
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)


# ── Config ────────────────────────────────────────────────────────────────

# Keep in sync with CLASS_ORDER in baselines + LLM classification scripts
LABEL_NAMES = [
    "Critical Rescue",
    "Resource Requests",
    "Situational Awareness",
    "Volunteering and Donations",
    "Irrelevant",
]
label2id = {label: i for i, label in enumerate(LABEL_NAMES)}
id2label = {i: label for label, i in label2id.items()}

# model_name (CLI) → HuggingFace model path
MODELS_REGISTRY = {
    "mbert": "bert-base-multilingual-cased",
    "xlm_roberta": "xlm-roberta-base",
    "muril": "google/muril-base-cased",
    "indic_bert": "ai4bharat/indic-bert",
}

RANDOM_SEED = 42

# Training hyperparameters (shared across all four models for comparability)
LEARNING_RATE = 2e-5
BATCH_SIZE = 32
EPOCHS = 3
MAX_SEQ_LENGTH = 128
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1


# ── Paths ─────────────────────────────────────────────────────────────────

try:
    BASE_DIR = Path(__file__).resolve().parent.parent
except NameError:
    BASE_DIR = Path.cwd().parent

REAL_CSV = BASE_DIR / "datasets" / "processed" / "humaid_processed.csv"
AUGMENTED_CSV = BASE_DIR / "datasets" / "processed" / "humaid_train_augmented.csv"
RESULTS_DIR = BASE_DIR / "results"
OFFLINE_MODELS_DIR = BASE_DIR / "offline_models"


# ── Dataset class ─────────────────────────────────────────────────────────

class DisasterDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# ── Weighted loss trainer ─────────────────────────────────────────────────

def make_weighted_trainer_class(class_weights_tensor):
    """Factory so the tensor is captured cleanly without global state."""
    class WeightedLossTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fct = nn.CrossEntropyLoss(weight=class_weights_tensor)
            loss = loss_fct(
                logits.view(-1, self.model.config.num_labels),
                labels.view(-1),
            )
            return (loss, outputs) if return_outputs else loss
    return WeightedLossTrainer


# ── Metrics ───────────────────────────────────────────────────────────────

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    acc = accuracy_score(labels, preds)
    per_class_f1 = f1_score(labels, preds, average=None, zero_division=0,
                            labels=list(range(len(LABEL_NAMES))))
    metrics = {
        "accuracy": acc,
        "macro_f1": f1,
        "macro_precision": precision,
        "macro_recall": recall,
    }
    for name, score in zip(LABEL_NAMES, per_class_f1):
        metrics[f"f1_{name.replace(' ', '_')}"] = float(score)
    return metrics


# ── Data loading ──────────────────────────────────────────────────────────

def load_and_prepare_data():
    """Load augmented train + real dev/test. Returns three DataFrames."""
    if not REAL_CSV.exists():
        raise FileNotFoundError(
            f"{REAL_CSV} not found. Run data_preprocessing.py first."
        )
    if not AUGMENTED_CSV.exists():
        raise FileNotFoundError(
            f"{AUGMENTED_CSV} not found. "
            f"Run generate_synthetic_data.py then merge_synthetic_data.py."
        )

    real_df = pd.read_csv(REAL_CSV)
    dev_df = real_df[real_df["split"] == "dev"].copy()
    test_df = real_df[real_df["split"] == "test"].copy()

    train_df = pd.read_csv(AUGMENTED_CSV)

    for name, df in [("train", train_df), ("dev", dev_df), ("test", test_df)]:
        df["clean_text"] = df["clean_text"].astype(str).fillna("")
        df["label_id"] = df["target_label"].map(label2id)
        missing = df["label_id"].isna().sum()
        if missing > 0:
            bad_labels = df[df["label_id"].isna()]["target_label"].unique()
            raise ValueError(
                f"{missing} rows in {name} split have unknown labels: "
                f"{bad_labels}. Expected one of: {LABEL_NAMES}"
            )
        df["label_id"] = df["label_id"].astype(int)

    return train_df, dev_df, test_df


# ── Main training pipeline ────────────────────────────────────────────────

def train_one_model(model_name, run_id):
    # Resolve HuggingFace path
    if model_name not in MODELS_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available: {list(MODELS_REGISTRY.keys())}"
        )
    hf_path = MODELS_REGISTRY[model_name]

    # Prefer local cached copy if available under offline_models/
    local_cache = OFFLINE_MODELS_DIR / model_name
    if local_cache.exists() and any(local_cache.iterdir()):
        print(f"Using local cached model from: {local_cache}")
        model_path_to_load = str(local_cache)
    else:
        print(f"Local cache not found, downloading from HuggingFace: {hf_path}")
        model_path_to_load = hf_path

    # Run output directory
    run_dir = RESULTS_DIR / f"{model_name}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = run_dir / "checkpoints"
    final_model_dir = run_dir / "final_model"
    log_dir = run_dir / "logs"

    print("=" * 70)
    print(f"MODEL:    {model_name}  ({hf_path})")
    print(f"RUN ID:   {run_id}")
    print(f"OUTPUT:   {run_dir}")
    print("=" * 70)

    # Load data
    train_df, dev_df, test_df = load_and_prepare_data()
    print(f"\nDataset sizes:")
    print(f"  Train (augmented): {len(train_df):,}")
    print(f"  Dev:               {len(dev_df):,}")
    print(f"  Test:              {len(test_df):,}")

    # Class weights from training set only
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(LABEL_NAMES)),
        y=train_df["label_id"].values,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"\nClass weights:")
    for label, w in zip(LABEL_NAMES, class_weights):
        print(f"  {label:<30s} {w:.4f}")
    print(f"Device: {device}")

    # Tokenizer (IndicBERT needs slow tokenizer)
    use_fast = "indic-bert" not in hf_path.lower()
    tokenizer = AutoTokenizer.from_pretrained(model_path_to_load, use_fast=use_fast)

    def encode(texts):
        return tokenizer(
            texts, truncation=True, padding=True, max_length=MAX_SEQ_LENGTH
        )

    train_enc = encode(train_df["clean_text"].tolist())
    dev_enc = encode(dev_df["clean_text"].tolist())
    test_enc = encode(test_df["clean_text"].tolist())

    train_dataset = DisasterDataset(train_enc, train_df["label_id"].tolist())
    dev_dataset = DisasterDataset(dev_enc, dev_df["label_id"].tolist())
    test_dataset = DisasterDataset(test_enc, test_df["label_id"].tolist())

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path_to_load,
        num_labels=len(LABEL_NAMES),
        id2label=id2label,
        label2id=label2id,
    )

    # Training args
    training_args = TrainingArguments(
        output_dir=str(checkpoints_dir),
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        logging_dir=str(log_dir),
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        seed=RANDOM_SEED,
        report_to="none",
    )

    # Trainer
    TrainerCls = make_weighted_trainer_class(class_weights_tensor)
    trainer = TrainerCls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    # Save final (best) model + tokenizer
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))
    print(f"\nSaved best model → {final_model_dir}")

    # Final dev metrics (using best checkpoint since load_best_model_at_end=True)
    print("\nFinal dev evaluation...")
    dev_metrics = trainer.evaluate(dev_dataset, metric_key_prefix="dev")
    for k, v in dev_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")

    # Test evaluation
    print("\nFinal test evaluation...")
    test_metrics = trainer.evaluate(test_dataset, metric_key_prefix="test")
    for k, v in test_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")

    # Save metrics JSON
    with open(run_dir / "dev_metrics.json", "w") as f:
        json.dump({k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v))
                   for k, v in dev_metrics.items()}, f, indent=2)
    with open(run_dir / "test_metrics.json", "w") as f:
        json.dump({k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v))
                   for k, v in test_metrics.items()}, f, indent=2)

    print(f"\nDone. All artifacts under: {run_dir}")


# ── CLI entry point ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=list(MODELS_REGISTRY.keys()),
        help="Which transformer to fine-tune.",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Unique identifier for this run (defaults to timestamp). "
             "Use your SLURM job id here to keep runs separate.",
    )
    args = parser.parse_args()

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    train_one_model(args.model_name, run_id)


if __name__ == "__main__":
    main()