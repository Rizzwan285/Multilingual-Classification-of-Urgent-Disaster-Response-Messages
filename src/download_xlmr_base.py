import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

# Create a dedicated folder for raw, offline models
OFFLINE_DIR = os.path.join(ROOT_DIR, "offline_models")
SAVE_DIR = os.path.join(OFFLINE_DIR, "local_xlm_roberta")

# Ensure the directory exists before saving
os.makedirs(SAVE_DIR, exist_ok=True)

print("Downloading XLM-RoBERTa from Hugging Face...")

# 2. Download from the internet
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
model = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=5)

# 3. Save locally to your nested folder
print(f"Saving model to: {SAVE_DIR}")
tokenizer.save_pretrained(SAVE_DIR)
model.save_pretrained(SAVE_DIR)

print("Download complete! Ready for offline training.")