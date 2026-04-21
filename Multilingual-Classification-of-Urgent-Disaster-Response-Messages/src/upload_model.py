import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
MODEL_DIR = os.path.join(ROOT_DIR, "model")

# Choose your Hugging Face username and a name for the model
HF_REPO_NAME = "your-username/xlm-roberta-disaster-response"

print(f"Loading trained model from {MODEL_DIR}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

print(f"Pushing to Hugging Face Hub as '{HF_REPO_NAME}'...")
tokenizer.push_to_hub(HF_REPO_NAME)
model.push_to_hub(HF_REPO_NAME)

print("Upload complete!")