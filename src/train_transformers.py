import os
import argparse

#Setting mandatory stability overrides before loading torch
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
    DataCollatorWithPadding,
    set_seed
)

#Locking the random seed for reproducibility
set_seed(42)

#Parsing the command line arguments to selecting the model
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, required=True, help="local_xlm_roberta, local_muril, local_indic_bert, or local_mbert")
args = parser.parse_args()

print("\n" + "="*50)
print(f"STARTING UNIFIED TRAINING SCRIPT FOR: {args.model_name}")
print("="*50)

#Setting up the directory structure dynamically based on the model name
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

MODEL_PATH = os.path.join(ROOT_DIR, "offline_models", args.model_name)
DATA_PATH = os.path.join(ROOT_DIR, "datasets", "processed", "humaid_processed.csv")

# --- UPDATED PATHS ---
# Defining the main results folder
MAIN_RESULTS_DIR = os.path.join(ROOT_DIR, "results")
# Creating a dedicated subfolder for the specific model being trained
MODEL_RESULTS_DIR = os.path.join(MAIN_RESULTS_DIR, args.model_name)

# Nesting the outputs inside the model's specific folder
OUTPUT_DIR = os.path.join(MODEL_RESULTS_DIR, "checkpoints")
FINAL_MODEL_DIR = os.path.join(MODEL_RESULTS_DIR, "final_model")
LOG_DIR = os.path.join(MODEL_RESULTS_DIR, "logs")


#Mapping the disaster categories to numerical IDs
LABEL_NAMES = ['Situational Awareness', 'Critical Rescue', 'Volunteering and Donations', 'Irrelevant', 'Resource Requests']
label2id = {label: i for i, label in enumerate(LABEL_NAMES)}
id2label = {i: label for i, label in enumerate(LABEL_NAMES)}

def compute_metrics(eval_pred):
    #Extracting the predictions and true labels
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    #Calculating the macro scores across all 5 classes
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro', zero_division=0)
    acc = accuracy_score(labels, predictions)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train_model():
    #Reading the processed dataset
    print(f"Loading the data from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    
    train_df = df[df['split'] == 'train'].copy()
    val_df = df[df['split'] == 'dev'].copy()
    
    #Converting the text labels to integers
    train_df['label'] = train_df['target_label'].map(label2id)
    val_df['label'] = val_df['target_label'].map(label2id)

    #Loading the correct tokenizer dynamically
    print(f"Loading the tokenizer from {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    #Defining the tokenization function
    #We are only truncating here; the data collator is handling the padding dynamically
    def tokenize_function(examples):
        return tokenizer(examples["clean_text"], truncation=True, max_length=128)

    #Converting to Hugging Face dataset format
    train_dataset = Dataset.from_pandas(train_df[['clean_text', 'label']])
    val_dataset = Dataset.from_pandas(val_df[['clean_text', 'label']])

    #Tokenizing the datasets
    print("Applying the tokenizer to the datasets")
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_val = val_dataset.map(tokenize_function, batched=True)

    #Initializing the dynamic data collator
    #This is ensuring efficient batching regardless of SentencePiece or WordPiece differences
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    #Loading the classification model
    print("Loading the model architecture")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH, 
        num_labels=len(LABEL_NAMES), 
        id2label=id2label, 
        label2id=label2id
    )

    #Configuring the training arguments
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
        logging_dir=LOG_DIR,              # <--- CHANGED THIS LINE
        logging_steps=50,
        fp16=True, 
        report_to="none",
        dataloader_num_workers=0
    )

    #Building the trainer object
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    #Starting the training loop
    print(f"Beginning the fine-tuning process for {args.model_name}")
    trainer.train()

    #Saving the final optimized model to your output directory
    print(f"Saving the model to {FINAL_MODEL_DIR}")
    trainer.save_model(FINAL_MODEL_DIR)
    tokenizer.save_pretrained(FINAL_MODEL_DIR)
    
    print("Training is completing successfully")

if __name__ == "__main__":
    train_model()