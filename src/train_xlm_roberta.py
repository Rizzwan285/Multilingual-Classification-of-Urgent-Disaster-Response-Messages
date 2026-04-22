import os
# --- MANDATORY STABILITY OVERRIDES FOR KERNEL 4.18 ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
import pandas as pd
import numpy as np
from datasets import Dataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

# ==========================================
# 1. Configuration & Setup
# ==========================================
print("\n" + "="*50)
print("DEBUG: STARTING SCRIPT VERSION 2.0 (STABILITY FIX)")
print("="*50)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

MODEL_NAME = os.path.join(ROOT_DIR, "offline_models", "local_xlm_roberta")
DATA_PATH = os.path.join(ROOT_DIR, "datasets", "processed", "humaid_processed.csv")
OUTPUT_DIR = os.path.join(ROOT_DIR, "results")
FINAL_MODEL_DIR = os.path.join(ROOT_DIR, "xlm_roberta_humaid")

LABEL_NAMES = ['Situational Awareness', 'Critical Rescue', 'Volunteering and Donations', 'Irrelevant', 'Resource Requests']
label2id = {label: i for i, label in enumerate(LABEL_NAMES)}
id2label = {i: label for i, label in enumerate(LABEL_NAMES)}

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"DEBUG: Using Device: {device.upper()}")

    # ==========================================
    # 2. Load and Prepare Data
    # ==========================================
    df = pd.read_csv(DATA_PATH)
    df['label'] = df['target_label'].map(label2id)
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    
    train_df = df[df['split'] == 'train'].reset_index(drop=True)
    val_df = df[df['split'] == 'dev'].reset_index(drop=True)
    
    train_dataset = Dataset.from_pandas(train_df[['clean_text', 'label']])
    val_dataset = Dataset.from_pandas(val_df[['clean_text', 'label']])

    # ==========================================
    # 3. Tokenization
    # ==========================================
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, fix_mistral_regex=True)

    def tokenize_function(examples):
        texts = [str(x) if x is not None else "" for x in examples["clean_text"]]
        return tokenizer(texts, truncation=True, padding=False, max_length=128)

    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_val = val_dataset.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # ==========================================
    # 4. Model Setup
    # ==========================================
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=5, id2label=id2label, label2id=label2id
    )

    # ==========================================
    # 5. Training Arguments
    # ==========================================
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",        
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=32, # Setting to 32
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir='./logs',
        logging_steps=50,
        fp16=True,
        report_to="none",
        # CRITICAL FIX FOR KERNEL 4.18:
        dataloader_num_workers=0
    )

    print(f"DEBUG: Total training steps should be 4872.")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # ==========================================
    # 6. Train and Save
    # ==========================================
    print("Starting training...")
    trainer.train()
    trainer.save_model(FINAL_MODEL_DIR)
    tokenizer.save_pretrained(FINAL_MODEL_DIR)

if __name__ == "__main__":
    main()