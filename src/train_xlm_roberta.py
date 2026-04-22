import os
import numpy as np
import argparse
import pandas as pd
import torch
from torch import nn
from sklearn.utils.class_weight import compute_class_weight
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, required=True)
args = parser.parse_args()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
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

    print("Applying tokenizer to the datasets...")
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_val = val_dataset.map(tokenize_function, batched=True)

    # Load the model directly instead of using a model_init function
    print(f"Loading model architecture from {MODEL_LOAD_PATH}")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_LOAD_PATH, 
        num_labels=5,
        id2label=id2label,
        label2id=label2id
    )

    # --- ADD THIS BLOCK TO HANDLE CLASS IMBALANCE ---
    print("Calculating Class Weights to handle dataset imbalance...")
    
    # Automatically calculates the exact penalty needed for each class
    class_weights = compute_class_weight(
        class_weight='balanced', 
        classes=np.unique(train_df['label']), 
        y=train_df['label']
    )
    
    # Convert it to a PyTorch tensor and move it to the GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    # Create a custom trainer that uses these weights during training
    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Apply the penalties here!
            loss_fct = nn.CrossEntropyLoss(weight=weights_tensor)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            
            return (loss, outputs) if return_outputs else loss
    # ------------------------------------------------

    training_args = TrainingArguments(
        output_dir=RESULTS_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,              # Added a safe default learning rate
        per_device_train_batch_size=64,  # Note: Drop to 32 or 16 if you hit CUDA Out of Memory
        per_device_eval_batch_size=64,
        num_train_epochs=3,
        weight_decay=0.01,
        fp16=True,
        logging_dir=LOG_DIR,
        report_to="none",
        ddp_find_unused_parameters=False
    )

    # Pass the instantiated model directly to the Trainer
    # Change 'Trainer' to 'WeightedTrainer'
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        processing_class=tokenizer
    )

    print("Starting the direct training process...")
    trainer.train()

    print(f"Saving the final model to {SAVE_PATH}")
    trainer.save_model(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)

if __name__ == "__main__":
    train_model()