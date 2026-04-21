import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

# Create a dedicated folder for raw, offline models
OFFLINE_DIR = os.path.join(ROOT_DIR, "offline_models")
SAVE_DIR = os.path.join(OFFLINE_DIR, "local_muril")

# Ensure the directory exists before saving
os.makedirs(SAVE_DIR, exist_ok=True)

print("Downloading MuRIL from Hugging Face...")
model_checkpoint = "google/muril-base-cased"

# 2. Download from the internet
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=5)

# 3. Save locally to your nested folder
print(f"Saving MuRIL to: {SAVE_DIR}")
tokenizer.save_pretrained(SAVE_DIR)
model.save_pretrained(SAVE_DIR)

print("Download complete! Ready for offline training.")