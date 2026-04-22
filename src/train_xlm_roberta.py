import os
import argparse
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, required=True)
args = parser.parse_args()

ROOT_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))
DATA_PATH = os.path.join(ROOT_DIR, "datasets", "processed", "humaid_processed.csv")
MODEL_LOAD_PATH = os.path.join(ROOT_DIR, "offline_models", args.model_name)
SAVE_PATH = os.path.join(ROOT_DIR, "trained_models", f"{args.model_name}_final")

RESULTS_DIR = os.path.join(ROOT_DIR, "results", args.model_name)
LOG_DIR = os.path.join(ROOT_DIR, "nlp_logs", args.model_name)

LABEL_NAMES = ['Situational Awareness', 'Critical Rescue', 'Volunteering and Donations', 'Irrelevant', 'Resource Requests']
label2id = {label: i for i, label in enumerate(LABEL_NAMES)}
id2label = {i: label for i, label in enumerate(LABEL_NAMES)}

def train_model():
    print(f"Loading the dataset for {args.model_name}")
    df = pd.read_csv(DATA_PATH)
    
    train_df = df[df['split'] == 'train'].copy()
    val_df = df[df['split'] == 'dev'].copy()
    
    train_df['label'] = train_df['target_label'].map(label2id)
    val_df['label'] = val_df['target_label'].map(label2id)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_LOAD_PATH)

    def tokenize_function(examples):
        return tokenizer(examples["clean_text"], padding="max_length", truncation=True, max_length=128)

    train_dataset = Dataset.from_pandas(train_df[['clean_text', 'label']])
    val_dataset = Dataset.from_pandas(val_df[['clean_text', 'label']])

    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_val = val_dataset.map(tokenize_function, batched=True)

    def model_init():
        return AutoModelForSequenceClassification.from_pretrained(
            MODEL_LOAD_PATH, 
            num_labels=5,
            id2label=id2label,
            label2id=label2id
        )

    training_args = TrainingArguments(
        output_dir=RESULTS_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        num_train_epochs=3,
        weight_decay=0.01,
        fp16=True,
        logging_dir=LOG_DIR,
        report_to="none",
        ddp_find_unused_parameters=False
    )

    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        processing_class=tokenizer
    )

    print("Starting the Optuna hyperparameter search")
    best_run = trainer.hyperparameter_search(
        direction="maximize", 
        backend="optuna", 
        n_trials=5 
    )
    
    print(f"Discovering the best hyperparameters: {best_run.hyperparameters}")
    
    for param_name, param_value in best_run.hyperparameters.items():
        setattr(trainer.args, param_name, param_value)
        
    print("Training the final model version")
    trainer.train()

    print(f"Saving the optimal model to {SAVE_PATH}")
    trainer.save_model(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)

if __name__ == "__main__":
    train_model()