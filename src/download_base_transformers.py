import os
import warnings
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import logging as hf_logging

warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()

# 1. Exact same path logic as your training script
current_dir = Path.cwd()
BASE_DIR    = current_dir.parent if current_dir.name in ("src", "notebooks") else current_dir
OFFLINE_DIR = BASE_DIR / "offline_models"

OFFLINE_DIR.mkdir(parents=True, exist_ok=True)

MODELS_TO_DOWNLOAD = {
    "local_xlm_roberta": "xlm-roberta-base",
    "local_muril":       "google/muril-base-cased",
    "local_indic_bert":  "ai4bharat/indic-bert",
    "local_mbert":       "bert-base-multilingual-cased"
}

def download_all_models():
    print(f"Starting the download to: {OFFLINE_DIR}")
    
    for local_name, hf_id in MODELS_TO_DOWNLOAD.items():
        save_path = OFFLINE_DIR / local_name
        
        # Check if config exists to skip
        if save_path.exists() and (save_path / "config.json").exists():
            print(f"Skipping {hf_id} because it is already downloaded.")
            continue
            
        print(f"Fetching {hf_id} now...")
        save_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Tokenizer
            tokenizer = AutoTokenizer.from_pretrained(hf_id)
            tokenizer.save_pretrained(str(save_path))
            
            # Model (Hardcoded to 5 labels for HumAID)
            model = AutoModelForSequenceClassification.from_pretrained(hf_id, num_labels=5)
            model.save_pretrained(str(save_path))
            
            print(f"Successfully finished downloading {local_name}.")
            
        except Exception as e:
            print(f"Issue downloading {hf_id}: {e}")

    print("\nAll models are now sitting in your offline folder and are ready for the cluster.")

if __name__ == "__main__":
    download_all_models()