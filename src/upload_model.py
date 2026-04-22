import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
TRAINED_MODELS_DIR = os.path.join(ROOT_DIR, "trained_models")

models_to_upload = {
    "local_muril": "Rizwan285/muril-disaster-response",
    "local_indic_bert": "Rizwan285/indicbert-disaster-response",
    "local_mbert": "Rizwan285/mbert-disaster-response",
    "local_xlm_roberta": "Rizwan285/xlm-roberta-disaster-response"
}

for local_name, hf_repo in models_to_upload.items():
    model_path = os.path.join(TRAINED_MODELS_DIR, local_name)
    
    print(f"Loading {local_name} from {model_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        print(f"Pushing to Hugging Face as {hf_repo}...")
        tokenizer.push_to_hub(hf_repo)
        model.push_to_hub(hf_repo)
        print(f"Successfully uploaded {local_name}!\n")
    except Exception as e:
        print(f"Failed to upload {local_name}: {e}\n")

print("All uploads finished!")