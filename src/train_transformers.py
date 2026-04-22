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

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, required=True)
args = parser.parse_args()

print(f"\n{'='*50}\nSTARTING UNIFIED TRAINING SCRIPT FOR: {args.model_name}\n{'='*50}")

# ==========================================
# 1. Setup Paths
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

# Using the new augmented dataset with synthetic data!
DATA_PATH = os.path.join(ROOT_DIR, "datasets", "processed", "humaid_augmented.csv")
MODEL_LOAD_PATH = os.path.join(ROOT_DIR, "offline_models", args.model_name)

# Nesting all outputs inside the specific model's results folder
MODEL_RESULTS_DIR = os.path.join(ROOT_DIR, "results", args.model_name)
OUTPUT_DIR = os.path.join(MODEL_RESULTS_DIR, "checkpoints")
FINAL_MODEL_DIR = os.path.join(MODEL_RESULTS_DIR, "final_model")
LOG_DIR = os.path.join(MODEL_RESULTS_DIR, "logs")

LABEL_NAMES = ['Situational Awareness', 'Critical Rescue', 'Volunteering and Donations', 'Irrelevant', 'Resource Requests']
label2id = {label: i for i, label in enumerate(LABEL_NAMES)}
id2label = {i: label for i, label in enumerate(LABEL_NAMES)}

# ==========================================
# 1.5. Define Metrics Function (ADDED FIX)
# ==========================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro', zero_division=0)
    acc = accuracy_score(labels, predictions)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train_model():
    # ==========================================
    # 2. Load and Tokenize Data
    # ==========================================
    print(f"Loading the data from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    
    train_df = df[df['split'] == 'train'].copy()
    val_df = df[df['split'] == 'dev'].copy()
    
    train_df['label'] = train_df['target_label'].map(label2id)
    val_df['label'] = val_df['target_label'].map(label2id)

    print(f"Loading the tokenizer from {MODEL_LOAD_PATH}")
    
    # ADDED FIX: Only apply use_fast=False to IndicBERT to prevent mBERT from crashing
    if "indic_bert" in args.model_name:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_LOAD_PATH, use_fast=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_LOAD_PATH)

    def tokenize_function(examples):
        return tokenizer(examples["clean_text"], padding="max_length", truncation=True, max_length=128)

    train_dataset = Dataset.from_pandas(train_df[['clean_text', 'label']])
    val_dataset = Dataset.from_pandas(val_df[['clean_text', 'label']])

    print("Applying the tokenizer to the datasets...")
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_val = val_dataset.map(tokenize_function, batched=True)

    # ==========================================
    # 3. Handle Imbalance with Custom Trainer
    # ==========================================
    print("Calculating Class Weights to handle dataset imbalance...")
    class_weights = compute_class_weight(
        class_weight='balanced', 
        classes=np.unique(train_df['label']), 
        y=train_df['label']
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            
            loss_fct = nn.CrossEntropyLoss(weight=weights_tensor)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            
            return (loss, outputs) if return_outputs else loss

    # ==========================================
    # 4. Model Setup and Training
    # ==========================================
    print(f"Loading the model architecture from {MODEL_LOAD_PATH}")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_LOAD_PATH, 
        num_labels=5,
        id2label=id2label,
        label2id=label2id
    )

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
        metric_for_best_model="f1",      # Correctly mapped back to 'f1'
        logging_dir=LOG_DIR,
        logging_steps=50,
        fp16=True, 
        report_to="none",
        dataloader_num_workers=0
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        processing_class=tokenizer,
        compute_metrics=compute_metrics, # ADDED FIX: Trainer now knows how to calculate eval_f1
    )

    print(f"Beginning the fine-tuning process for {args.model_name}...")
    trainer.train()

    print(f"Saving the final fine-tuned model to {FINAL_MODEL_DIR}")
    trainer.save_model(FINAL_MODEL_DIR)
    tokenizer.save_pretrained(FINAL_MODEL_DIR)
    
    print(f"Successfully finished training {args.model_name}!\n")

if __name__ == "__main__":
    train_model()