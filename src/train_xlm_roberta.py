"""
Transformer Fine-Tuning — Cluster/SLURM Version
-------------------------------------------------
Loads a pre-downloaded model from offline_models/<model_name>/ and fine-tunes
it on the augmented HumAID training set. Designed for GPU cluster environments
where outbound internet access may not be available.

Usage:
    python src/train_transformer_cluster.py --model_name muril
    python src/train_transformer_cluster.py --model_name xlm_roberta --run_id $SLURM_JOB_ID

Expected offline_models/ structure:
    offline_models/
    ├── muril/           (downloaded via download_models.py)
    ├── xlm_roberta/
    ├── mbert/
    └── indic_bert/

Available --model_name values:
    mbert, xlm_roberta, muril, indic_bert

Outputs (all under results/<model_name>_<run_id>/):
    checkpoints/         HuggingFace intermediate checkpoints
    final_model/         Best model + tokenizer
    logs/                Trainer logs
    test_metrics.json    Final test-set metrics
    dev_metrics.json     Best dev-set metrics
"""

import os
import json
import argparse
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)


# ── Config ────────────────────────────────────────────────────────────────

# Keep this order in sync with CLASS_ORDER in baselines + LLM scripts
LABEL_NAMES = [
    "Critical Rescue",
    "Resource Requests",
    "Situational Awareness",
    "Volunteering and Donations",
    "Irrelevant",
]
label2id = {label: i for i, label in enumerate(LABEL_NAMES)}
id2label = {i: label for label, i in label2id.items()}

MODELS_REGISTRY = {
    "mbert": "bert-base-multilingual-cased",
    "xlm_roberta": "xlm-roberta-base",
    "muril": "google/muril-base-cased",
    "indic_bert": "ai4bharat/indic-bert",
}

# Hyperparameters (match what is reported in the paper)
LEARNING_RATE = 2e-5
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 64
EPOCHS = 3
MAX_SEQ_LENGTH = 128
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
RANDOM_SEED = 42


# ── Paths ─────────────────────────────────────────────────────────────────

try:
    BASE_DIR = Path(__file__).resolve().parent.parent
except NameError:
    BASE_DIR = Path.cwd().parent

REAL_CSV = BASE_DIR / "datasets" / "processed" / "humaid_processed.csv"
AUGMENTED_CSV = BASE_DIR / "datasets" / "processed" / "humaid_train_augmented.csv"
OFFLINE_MODELS_DIR = BASE_DIR / "offline_models"
RESULTS_DIR = BASE_DIR / "results"


# ── Dataset ───────────────────────────────────────────────────────────────

def tokenize_hf_dataset(hf_dataset, tokenizer):
    """Tokenize a HuggingFace Dataset using batched map."""
    def _tokenize(batch):
        return tokenizer(
            batch["clean_text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
        )
    return hf_dataset.map(_tokenize, batched=True)


# ── Metrics ───────────────────────────────────────────────────────────────

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    acc = accuracy_score(labels, preds)
    per_class_f1 = f1_score(
        labels, preds, average=None, zero_division=0,
        labels=list(range(len(LABEL_NAMES))),
    )
    metrics = {
        "accuracy": acc,
        "macro_f1": f1,
        "macro_precision": precision,
        "macro_recall": recall,
    }
    for name, score in zip(LABEL_NAMES, per_class_f1):
        metrics[f"f1_{name.replace(' ', '_')}"] = float(score)
    return metrics


# ── Weighted trainer factory ──────────────────────────────────────────────

def make_weighted_trainer_class(weights_tensor):
    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fct = nn.CrossEntropyLoss(weight=weights_tensor)
            loss = loss_fct(
                logits.view(-1, self.model.config.num_labels),
                labels.view(-1),
            )
            return (loss, outputs) if return_outputs else loss
    return WeightedTrainer


# ── Data loading ──────────────────────────────────────────────────────────

def load_splits():
    """Load augmented train + real dev and test. Returns three DataFrames."""
    if not REAL_CSV.exists():
        raise FileNotFoundError(f"Not found: {REAL_CSV}. Run data_preprocessing.py first.")
    if not AUGMENTED_CSV.exists():
        raise FileNotFoundError(
            f"Not found: {AUGMENTED_CSV}. "
            "Run generate_synthetic_data.py then merge_synthetic_data.py."
        )

    real_df = pd.read_csv(REAL_CSV)
    dev_df = real_df[real_df["split"] == "dev"].copy()
    test_df = real_df[real_df["split"] == "test"].copy()
    train_df = pd.read_csv(AUGMENTED_CSV)

    for split_name, df in [("train", train_df), ("dev", dev_df), ("test", test_df)]:
        df["clean_text"] = df["clean_text"].astype(str).fillna("")
        df["label"] = df["target_label"].map(label2id)
        missing = df["label"].isna().sum()
        if missing > 0:
            bad = df[df["label"].isna()]["target_label"].unique()
            raise ValueError(
                f"{missing} rows in {split_name} have unknown labels: {bad}. "
                f"Expected: {LABEL_NAMES}"
            )
        df["label"] = df["label"].astype(int)

    return train_df, dev_df, test_df


# ── Main ──────────────────────────────────────────────────────────────────

def train_model(model_name, run_id):
    # Validate model name
    if model_name not in MODELS_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Choose from: {list(MODELS_REGISTRY.keys())}"
        )

    # Use pre-downloaded local copy
    model_load_path = OFFLINE_MODELS_DIR / model_name
    if not model_load_path.exists() or not any(model_load_path.iterdir()):
        raise FileNotFoundError(
            f"Offline model not found at {model_load_path}. "
            f"Download it first with: download_models.py --model_name {model_name}"
        )

    # Output directories
    run_dir = RESULTS_DIR / f"{model_name}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = run_dir / "checkpoints"
    final_model_dir = run_dir / "final_model"
    log_dir = run_dir / "logs"

    print("=" * 70)
    print(f"MODEL:   {model_name}  ({MODELS_REGISTRY[model_name]})")
    print(f"RUN ID:  {run_id}")
    print(f"OUTPUT:  {run_dir}")
    print("=" * 70)

    # Load data
    train_df, dev_df, test_df = load_splits()
    print(f"\nDataset sizes:")
    print(f"  Train (augmented): {len(train_df):,}")
    print(f"  Dev:               {len(dev_df):,}")
    print(f"  Test:              {len(test_df):,}")

    # Class weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(LABEL_NAMES)),
        y=train_df["label"].values,
    )
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"\nDevice: {device}")
    print("Class weights:")
    for name, w in zip(LABEL_NAMES, class_weights):
        print(f"  {name:<30s} {w:.4f}")

    # Tokenizer (IndicBERT requires slow tokenizer)
    use_fast = "indic-bert" not in str(model_load_path).lower()
    print(f"\nLoading tokenizer from {model_load_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_load_path), use_fast=use_fast)

    # Build HuggingFace Datasets and tokenize
    print("Tokenizing datasets...")
    train_hf = tokenize_hf_dataset(
        Dataset.from_pandas(train_df[["clean_text", "label"]]), tokenizer
    )
    dev_hf = tokenize_hf_dataset(
        Dataset.from_pandas(dev_df[["clean_text", "label"]]), tokenizer
    )
    test_hf = tokenize_hf_dataset(
        Dataset.from_pandas(test_df[["clean_text", "label"]]), tokenizer
    )

    # Rename 'label' to 'labels' so HuggingFace Trainer finds them
    train_hf = train_hf.rename_column("label", "labels")
    dev_hf = dev_hf.rename_column("label", "labels")
    test_hf = test_hf.rename_column("label", "labels")

    # Set format so Trainer gets tensors
    cols = ["input_ids", "attention_mask", "labels"]
    if "token_type_ids" in train_hf.column_names:
        cols.append("token_type_ids")
    train_hf.set_format("torch", columns=cols)
    dev_hf.set_format("torch", columns=cols)
    test_hf.set_format("torch", columns=cols)

    # Model
    print(f"Loading model from {model_load_path}")
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_load_path),
        num_labels=len(LABEL_NAMES),
        id2label=id2label,
        label2id=label2id,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(checkpoints_dir),
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
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
        dataloader_num_workers=4,
    )

    # Trainer
    TrainerCls = make_weighted_trainer_class(weights_tensor)
    trainer = TrainerCls(
        model=model,
        args=training_args,
        train_dataset=train_hf,
        eval_dataset=dev_hf,
        compute_metrics=compute_metrics,
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    # Save best model
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))
    print(f"\nSaved best model → {final_model_dir}")

    # Evaluate on dev (best checkpoint, since load_best_model_at_end=True)
    print("\nFinal dev evaluation...")
    dev_results = trainer.evaluate(dev_hf, metric_key_prefix="dev")
    for k, v in dev_results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")

    # Evaluate on test
    print("\nFinal test evaluation...")
    test_results = trainer.evaluate(test_hf, metric_key_prefix="test")
    for k, v in test_results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")

    # Save metrics to JSON
    def to_serializable(d):
        return {k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v))
                for k, v in d.items()}

    with open(run_dir / "dev_metrics.json", "w") as f:
        json.dump(to_serializable(dev_results), f, indent=2)
    with open(run_dir / "test_metrics.json", "w") as f:
        json.dump(to_serializable(test_results), f, indent=2)

    print(f"\nAll artifacts saved under: {run_dir}")


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune a transformer on the HumAID dataset (cluster version)."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=list(MODELS_REGISTRY.keys()),
        help="Which offline model to train.",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Unique run identifier. Defaults to timestamp. "
             "Recommend passing $SLURM_JOB_ID on cluster.",
    )
    args = parser.parse_args()

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    train_model(args.model_name, run_id)


if __name__ == "__main__":
    main()