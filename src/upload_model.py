import os
import warnings
from pathlib import Path
from dotenv import load_dotenv

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import logging as hf_logging

warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN:
    print("HF_TOKEN detected. Will use it for gated models.\n")
else:
    print("No HF_TOKEN found. Public models will still download.\n")

# -----------------------------
# Path setup (same as before)
# -----------------------------
current_dir = Path.cwd()
BASE_DIR = current_dir.parent if current_dir.name in ("src", "notebooks") else current_dir
OFFLINE_DIR = BASE_DIR / "offline_models"

OFFLINE_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Models to download
# -----------------------------
MODELS_TO_DOWNLOAD = {
    "local_xlm_roberta": "xlm-roberta-base",
    "local_muril":       "google/muril-base-cased",
    "local_indic_bert":  "ai4bharat/indic-bert",  # gated model
    "local_mbert":       "bert-base-multilingual-cased"
}

# -----------------------------
# Download function
# -----------------------------
def download_all_models():
    print(f"Starting the download to: {OFFLINE_DIR}\n")
    
    for local_name, hf_id in MODELS_TO_DOWNLOAD.items():
        save_path = OFFLINE_DIR / local_name

        # Skip if already downloaded
        if save_path.exists() and (save_path / "config.json").exists():
            print(f"Skipping {hf_id} (already exists)")
            continue

        print(f"\n--- Downloading {hf_id} ---")
        save_path.mkdir(parents=True, exist_ok=True)

        try:
            # -----------------------------
            # Load tokenizer
            # -----------------------------
            tokenizer = AutoTokenizer.from_pretrained(
                hf_id,
                token=HF_TOKEN  # works for both public & gated
            )
            tokenizer.save_pretrained(str(save_path))

            # -----------------------------
            # Load model
            # -----------------------------
            model = AutoModelForSequenceClassification.from_pretrained(
                hf_id,
                num_labels=5,
                token=HF_TOKEN
            )
            model.save_pretrained(str(save_path))

            print(f"SUCCESS: {local_name} downloaded")

        except Exception as e:
            print(f"FAILED: {hf_id}")
            
            # Helpful debugging hints
            if "401" in str(e) or "gated" in str(e).lower():
                print("→ This is likely a gated model.")
                print("→ Go to Hugging Face and ACCEPT TERMS:")
                print(f"   https://huggingface.co/{hf_id}")
                print("→ Then rerun the script.\n")
            else:
                print(f"→ Error: {str(e)}\n")

    print("\nAll downloads attempted!")

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    download_all_models()