import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

#Defining the paths and setting up the folder structure
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
OFFLINE_DIR = os.path.join(ROOT_DIR, "offline_models")

#Creating the directory if it is not already there
os.makedirs(OFFLINE_DIR, exist_ok=True)

#Listing out the models we are needing to pull from Hugging Face
MODELS_TO_DOWNLOAD = {
    "local_xlm_roberta": "xlm-roberta-base",
    "local_muril": "google/muril-base-cased",
    "local_indic_bert": "ai4bharat/indic-bert",
    "local_mbert": "bert-base-multilingual-cased"
}

def download_all_models():
    #Starting the download process for the entire list
    print(f"Starting the download to: {OFFLINE_DIR}")
    
    for local_name, hf_id in MODELS_TO_DOWNLOAD.items():
        save_path = os.path.join(OFFLINE_DIR, local_name)
        
        #Checking if the files are already sitting in the folder to save us some time
        if os.path.exists(save_path) and "config.json" in os.listdir(save_path):
            print(f"Skipping {hf_id} because it is already downloaded")
            continue
            
        print(f"Fetching {hf_id} now")
        os.makedirs(save_path, exist_ok=True)
        
        try:
            #Downloading the tokenizer files for the specific model
            tokenizer = AutoTokenizer.from_pretrained(hf_id)
            tokenizer.save_pretrained(save_path)
            
            #Downloading the actual model weights and setting it up for our 5 classes
            model = AutoModelForSequenceClassification.from_pretrained(hf_id, num_labels=5)
            model.save_pretrained(save_path)
            
            print(f"Successfully finishing the download for {local_name}")
            
        except Exception as e:
            #Handling any errors that are popping up during the connection
            print(f"Running into an issue with {hf_id}: {e}")

    #Wrapping everything up once the loop is finishing
    print("All models are now sitting in your offline folder and are ready for the cluster")

if __name__ == "__main__":
    download_all_models()