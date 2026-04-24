import os
import argparse
import pandas as pd
import numpy as np
import torch
from torch import nn
from datasets import Dataset
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--run_id", type=str, default=None, help="Unique identifier to prevent overwriting")
args = parser.parse_args()

# Generate a timestamp if run manually, otherwise use the SLURM RUN_ID
run_identifier = args.run_id if args.run_id else datetime.now().strftime("%Y%m%d_%H%M%S")

# ==========================================
# 1. Setup Paths (Updated with Timestamps)
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(ROOT_DIR, "datasets", "processed", "humaid_processed.csv")
MODEL_LOAD_PATH = os.path.join(ROOT_DIR, "offline_models", args.model_name)

# Nesting all outputs inside a timestamped folder
MODEL_RESULTS_DIR = os.path.join(ROOT_DIR, "results", f"{args.model_name}_{run_identifier}")
OUTPUT_DIR = os.path.join(MODEL_RESULTS_DIR, "checkpoints")
FINAL_MODEL_DIR = os.path.join(MODEL_RESULTS_DIR, "final_model")
LOG_DIR = os.path.join(MODEL_RESULTS_DIR, "logs")

LABEL_NAMES = ['Situational Awareness', 'Critical Rescue', 'Volunteering and Donations', 'Irrelevant', 'Resource Requests']
label2id = {label: i for i, label in enumerate(LABEL_NAMES)}
id2label = {i: label for i, label in enumerate(LABEL_NAMES)}

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train_model():
    df = pd.read_csv(DATA_PATH)
    train_df = df[df['split'] == 'train'].copy()
    val_df = df[df['split'] == 'dev'].copy()
    
    train_df['label'] = train_df['target_label'].map(label2id)
    val_df['label'] = val_df['target_label'].map(label2id)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_LOAD_PATH, use_fast=False)

    def tokenize_function(examples):
        return tokenizer(examples["clean_text"], padding="max_length", truncation=True, max_length=128)

    train_dataset = Dataset.from_pandas(train_df[['clean_text', 'label']])
    val_dataset = Dataset.from_pandas(val_df[['clean_text', 'label']])
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_val = val_dataset.map(tokenize_function, batched=True)

    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_df['label']), y=train_df['label'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fct = nn.CrossEntropyLoss(weight=weights_tensor)
            return (loss_fct(logits, labels), outputs) if return_outputs else loss_fct(logits, labels)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_LOAD_PATH, num_labels=5, id2label=id2label, label2id=label2id)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",      
        logging_dir=LOG_DIR,
        fp16=True, 
        report_to="none"
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        processing_class=tokenizer,
        compute_metrics=compute_metrics  
    )

    trainer.train()
    trainer.save_model(FINAL_MODEL_DIR)
    tokenizer.save_pretrained(FINAL_MODEL_DIR)

if __name__ == "__main__":
    train_model()