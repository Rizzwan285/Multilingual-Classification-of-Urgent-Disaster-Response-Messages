import os
import pandas as pd
from deep_translator import GoogleTranslator
from tqdm import tqdm

# 1. Setup Paths
ROOT_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))
INPUT_PATH = os.path.join(ROOT_DIR, "datasets", "processed", "humaid_processed.csv")
# Renamed to malayalam_test_100.csv
OUTPUT_PATH = os.path.join(ROOT_DIR, "datasets", "processed", "malayalam_test_100.csv")

# 2. Load and Filter for the test set
print("Loading dataset...")
df = pd.read_csv(INPUT_PATH)
test_df = df[df['split'] == 'test'].copy()

# 3. Uniform Class Balancing (20 per category = 100 total)
print("Balancing classes to 20 samples per category...")
# Ensure we have at least 20 in each; if not, it will take what is available
balanced_df = test_df.groupby('target_label').apply(lambda x: x.sample(n=min(len(x), 20), random_state=42)).reset_index(drop=True)

# 4. Translation Logic
translator = GoogleTranslator(source='en', target='ml')

print(f"Translating {len(balanced_df)} tweets to Malayalam. This will take ~2 minutes...")
tqdm.pandas() 

def translate_text(text):
    try:
        return translator.translate(text)
    except Exception:
        return None

balanced_df['malayalam_text'] = balanced_df['clean_text'].progress_apply(translate_text)

# 5. Clean up and Save
balanced_df = balanced_df.dropna(subset=['malayalam_text'])
balanced_df.to_csv(OUTPUT_PATH, index=False)

print(f"\nSuccess! Malayalam test set saved to: {OUTPUT_PATH}")