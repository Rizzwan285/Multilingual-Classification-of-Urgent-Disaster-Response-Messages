import os
import time
import pandas as pd
from tqdm import tqdm
from deep_translator import GoogleTranslator

# ==========================================
# 1. Setup Paths
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

# Load your current augmented/processed dataset
INPUT_DATA_PATH = os.path.join(ROOT_DIR, "datasets", "processed", "humaid_augmented.csv")
OUTPUT_DATA_PATH = os.path.join(ROOT_DIR, "datasets", "processed", "humaid_mixed_en_ml.csv")

def create_mixed_dataset():
    print(f"Loading English dataset from {INPUT_DATA_PATH}")
    df = pd.read_csv(INPUT_DATA_PATH)
    
    # Drop any nulls just in case
    df = df.dropna(subset=['clean_text']).reset_index(drop=True)
    
    # ==========================================
    # 2. Stratified 80/20 Split
    # ==========================================
    print("Splitting dataset: 80% English, 20% Malayalam...")
    # Group by split and target_label to ensure perfectly balanced sampling
    df_ml = df.groupby(['split', 'target_label'], group_keys=False).apply(lambda x: x.sample(frac=0.2, random_state=42))
    df_en = df.drop(df_ml.index)
    
    print(f"  -> English Tweets: {len(df_en)}")
    print(f"  -> Tweets to Translate: {len(df_ml)}")

    # ==========================================
    # 3. Translation Setup
    # ==========================================
    translator = GoogleTranslator(source='en', target='ml')
    tqdm.pandas(desc="Translating to Malayalam")

    # We use a safe translation function to prevent the script from crashing 
    # if Google Translate temporarily blocks the connection
    def safe_translate(text):
        try:
            # Sleep for 0.1 seconds to avoid Google API rate limits
            time.sleep(0.1)
            return translator.translate(str(text))
        except Exception as e:
            # If the API fails, fall back to the original English text
            return text

    # Apply the translation
    print("\nStarting Google Translate API (This will take a while...)")
    df_ml['clean_text'] = df_ml['clean_text'].progress_apply(safe_translate)
    
    # Add a language tracking column just for your records
    df_ml['language'] = 'ml'
    df_en['language'] = 'en'

    # ==========================================
    # 4. Merge, Shuffle, and Save
    # ==========================================
    print("\nMerging and shuffling datasets...")
    mixed_df = pd.concat([df_en, df_ml])
    
    # Shuffle the dataset thoroughly so batches contain a mix of both languages
    mixed_df = mixed_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    mixed_df.to_csv(OUTPUT_DATA_PATH, index=False)
    print(f"Successfully saved mixed dataset to {OUTPUT_DATA_PATH}")

if __name__ == "__main__":
    create_mixed_dataset()