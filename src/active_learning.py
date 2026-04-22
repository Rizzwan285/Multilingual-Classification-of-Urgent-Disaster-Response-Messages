import os
import torch
import pandas as pd
import numpy as np
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.join(os.getcwd(), "."))
INPUT_PATH = os.path.join(ROOT_DIR, "datasets", "malayalam_unlabeled.csv")
MODEL_PATH = os.path.join(ROOT_DIR, "trained_models", "xlm_roberta_final")
OUTPUT_PATH = os.path.join(ROOT_DIR, "datasets", "active_learning_to_annotate.csv")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def clean_text_for_active_learning(text):
    text = str(text)
    text = re.sub(r'\s+\d+(\s+\d+)*$', '', text)
    text = re.sub(r'http\S+', '[URL]', text)
    return text.strip()

def run_uncertainty_extraction():
    print("Reading the unlabeled dataset now")
    df = pd.read_csv(INPUT_PATH)
    
    print("Cleaning the malayalam and mixed text")
    df['clean_text'] = df['text'].apply(clean_text_for_active_learning)
    
    print("Loading the trained model and moving it to the GPU")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()

    texts = df['clean_text'].tolist()
    confidences = []
    
    print("Analyzing the model uncertainty for each tweet")
    batch_size = 32
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i:i + batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            max_probs, _ = torch.max(probs, dim=-1)
            confidences.extend(max_probs.cpu().numpy())

    df['model_confidence'] = confidences
    df_sorted = df.sort_values(by='model_confidence', ascending=False)
    
    print(f"Saving the 200 most uncertain samples to {OUTPUT_PATH}")
    df_sorted.head(200).to_csv(OUTPUT_PATH, index=False)
    
    print("Finished identifying the samples for your active learning phase")

if __name__ == "__main__":
    run_uncertainty_extraction()