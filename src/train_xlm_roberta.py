import os
import argparse
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

#Setting up the command line arguments to choosing which model to training
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, required=True, help="local_xlm_roberta, local_muril, local_indic_bert, or local_mbert")
args = parser.parse_args()

#Defining the root and local paths for your project
ROOT_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))
DATA_PATH = os.path.join(ROOT_DIR, "datasets", "processed", "humaid_processed.csv")
MODEL_LOAD_PATH = os.path.join(ROOT_DIR, "offline_models", args.model_name)
SAVE_PATH = os.path.join(ROOT_DIR, "trained_models", f"{args.model_name}_final")

#Mapping your disaster categories to numerical IDs
LABEL_NAMES = ['Situational Awareness', 'Critical Rescue', 'Volunteering and Donations', 'Irrelevant', 'Resource Requests']
label2id = {label: i for i, label in enumerate(LABEL_NAMES)}
id2label = {i: label for i, label in enumerate(LABEL_NAMES)}

def train_model():
    #Loading the processed humaid dataset into a pandas dataframe
    print(f"Loading the dataset for training {args.model_name}")
    df = pd.read_csv(DATA_PATH)
    
    #Filtering for the training and validation splits
    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'dev']
    
    #Mapping the text labels to the integer IDs we are needing for the loss function
    train_df['label'] = train_df['target_label'].map(label2id)
    val_df['label'] = val_df['target_label'].map(label2id)

    #Loading the local tokenizer from your offline folder
    print(f"Loading the tokenizer from {MODEL_LOAD_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_LOAD_PATH)

    #Defining the tokenization function for the dataset mapping
    def tokenize_function(examples):
        return tokenizer(examples["clean_text"], padding="max_length", truncation=True, max_length=128)

    #Converting the dataframes into Hugging Face dataset objects
    train_dataset = Dataset.from_pandas(train_df[['clean_text', 'label']])
    val_dataset = Dataset.from_pandas(val_df[['clean_text', 'label']])

    #Applying the tokenization across the entire dataset in batches
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_val = val_dataset.map(tokenize_function, batched=True)

    #Loading the model and configuring it for our 5 specific classes
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_LOAD_PATH, 
        num_labels=5,
        id2label=id2label,
        label2id=label2id
    )

    #Setting up the training arguments for the A30 GPU
    training_args = TrainingArguments(
        output_dir=f"./results_{args.model_name}",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        fp16=True, #Utilizing the GPU half-precision to making training faster
        logging_dir="./logs",
        report_to="none"
    )

    #Initializing the trainer with the model and the datasets
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer
    )

    #Starting the actual fine-tuning process
    print(f"Starting the training loop for {args.model_name}")
    trainer.train()

    #Saving the final fine-tuned model and tokenizer to your trained_models folder
    print(f"Saving the finished model to {SAVE_PATH}")
    model.save_pretrained(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)

if __name__ == "__main__":
    train_model()