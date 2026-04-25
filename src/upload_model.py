import os
import warnings
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import logging as hf_logging

warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()

# 1. Exact same path logic as your training script
current_dir = Path.cwd()
BASE_DIR    = current_dir.parent if current_dir.name in ("src", "notebooks") else current_dir
TRAINED_MODELS_DIR = BASE_DIR / "trained_models"

# 2. DECIDE WHICH RUN TO UPLOAD ("aug" or "no_aug")
TARGET_SUFFIX = "aug" 

models_to_upload = {
    "local_muril":       "Rizwan285/muril-disaster-response",
    "local_indic_bert":  "Rizwan285/indicbert-disaster-response",
    "local_mbert":       "Rizwan285/mbert-disaster-response",
    "local_xlm_roberta": "Rizwan285/xlm-roberta-disaster-response"
}

def upload_models():
    print(f"Looking for models in: {TRAINED_MODELS_DIR}")
    print(f"Targeting models with suffix: _{TARGET_SUFFIX}\n")

    for local_name, hf_repo in models_to_upload.items():
        # 3. CRITICAL FIX: Append the suffix to match the training output
        folder_name = f"{local_name}_{TARGET_SUFFIX}"
        model_path = TRAINED_MODELS_DIR / folder_name
        
        if not model_path.exists():
            print(f"Skipping {local_name}: Could not find folder {model_path}")
            continue
            
        print(f"Loading {folder_name}...")
        try:
            # Enforce local_files_only so it fails cleanly if the folder is empty
            tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(str(model_path), local_files_only=True)
            
            print(f"Pushing to Hugging Face as {hf_repo}...")
            tokenizer.push_to_hub(hf_repo)
            model.push_to_hub(hf_repo)
            print(f"Successfully uploaded {folder_name}!\n")
            
        except Exception as e:
            print(f"Failed to upload {folder_name}: {e}\n")

    print("All uploads finished!")

if __name__ == "__main__":
    upload_models()