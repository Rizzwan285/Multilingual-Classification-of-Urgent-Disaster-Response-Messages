import os
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments
)
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. Setup Dynamic Paths & Data Loading
# ==========================================
try:
    BASE_DIR = Path(__file__).resolve().parent.parent
except NameError:
    BASE_DIR = Path.cwd().parent

REAL_CSV = BASE_DIR / "datasets" / "processed" / "humaid_processed.csv"
AUGMENTED_CSV = BASE_DIR / "datasets" / "processed" / "humaid_train_augmented.csv"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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

print("Loading datasets...")
# Load REAL data ONLY to extract the Dev and Test splits
real_df = pd.read_csv(REAL_CSV)
dev_df = real_df[real_df['split'] == 'dev'].copy()
test_df = real_df[real_df['split'] == 'test'].copy()

# Load AUGMENTED data for the Train split
train_df = pd.read_csv(AUGMENTED_CSV)

# Standardize text and label columns
for df in [train_df, dev_df, test_df]:
    df['clean_text'] = df['clean_text'].astype(str).fillna('')
    df['label_id'] = df['target_label'].map(label2id)

print(f"Train (Augmented) size: {len(train_df)}")
print(f"Dev size: {len(dev_df)}")
print(f"Test size: {len(test_df)}\n")

# Calculate Class Weights exclusively on the Augmented Train set
train_labels = train_df['label_id'].values
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
print(f"Calculated Class Weights: {class_weights_tensor.cpu().numpy()}\n")

# ==========================================
# 2. Dataset & Trainer Classes
# ==========================================
class DisasterDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class WeightedLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'macro_f1': f1, 'macro_precision': precision, 'macro_recall': recall}

# ==========================================
# 3. Main Training Loop
# ==========================================
def main():
    models_to_train = {
        "local_muril": "google/muril-base-cased",
        "local_indic_bert": "ai4bharat/indic-bert",
        "local_mbert": "bert-base-multilingual-cased"
    }

    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

    for local_name, hf_path in models_to_train.items():
        print("="*60)
        print(f"Initializing {local_name} ({hf_path})...")
        print("="*60)

        use_fast = False if "indic-bert" in hf_path.lower() else True
        tokenizer = AutoTokenizer.from_pretrained(hf_path, use_fast=use_fast)
        model = AutoModelForSequenceClassification.from_pretrained(
            hf_path, 
            num_labels=len(LABEL_NAMES),
            id2label=id2label,
            label2id=label2id
        )

        train_encodings = tokenizer(train_df['clean_text'].tolist(), truncation=True, padding=True, max_length=128)
        dev_encodings = tokenizer(dev_df['clean_text'].tolist(), truncation=True, padding=True, max_length=128)
        test_encodings = tokenizer(test_df['clean_text'].tolist(), truncation=True, padding=True, max_length=128)

        train_dataset = DisasterDataset(train_encodings, train_df['label_id'].tolist())
        dev_dataset = DisasterDataset(dev_encodings, dev_df['label_id'].tolist())
        test_dataset = DisasterDataset(test_encodings, test_df['label_id'].tolist())

        output_dir = RESULTS_DIR / f"{local_name}_{TIMESTAMP}"
        
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=3,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=str(output_dir / "logs"),
            logging_steps=100,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            fp16=torch.cuda.is_available()
        )

        trainer = WeightedLossTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            compute_metrics=compute_metrics
        )

        print(f"Training {local_name}...")
        trainer.train()

        print(f"Evaluating {local_name} on Real Test Set...")
        test_results = trainer.evaluate(test_dataset)
        print(f"Test Results for {local_name}: {test_results}")

        final_model_path = output_dir / "final_model"
        trainer.save_model(str(final_model_path))
        tokenizer.save_pretrained(str(final_model_path))
        print(f"Saved {local_name} to {final_model_path}\n")
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
    main()
