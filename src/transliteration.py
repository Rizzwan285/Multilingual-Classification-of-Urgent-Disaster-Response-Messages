import os
import pandas as pd
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
from tqdm import tqdm

# ==========================================
# 1. Setup Dynamic Paths
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

# Points to your existing dataset
INPUT_PATH = os.path.join(ROOT_DIR, "datasets", "processed", "malayalam_test.csv")
# Where the cleaned/transliterated dataset will be saved
OUTPUT_PATH = os.path.join(ROOT_DIR, "datasets", "processed", "malayalam_manglish.csv")

# ==========================================
# 2. Transliteration Function
# ==========================================
def malayalam_to_manglish(text):
    """
    Converts native Malayalam characters into Latin/Manglish script (ITRANS).
    """
    if pd.isna(text): 
        return ""
    # Flipped the order: From MALAYALAM -> ITRANS (English characters)
    return transliterate(str(text), sanscript.MALAYALAM, sanscript.ITRANS)

# ==========================================
# 3. Main Execution
# ==========================================
def main():
    print(f"Loading original dataset from {INPUT_PATH}...")
    df = pd.read_csv(INPUT_PATH)
    
    # Initialize tqdm for Pandas so we get a nice progress bar
    tqdm.pandas(desc="Transliterating Malayalam -> Manglish")
    
    print("Applying transliteration to the 'text' column...")
    # Apply the function to your text column
    df['text'] = df['text'].progress_apply(malayalam_to_manglish)
    
    print(f"Saving cleaned dataset to {OUTPUT_PATH}...")
    df.to_csv(OUTPUT_PATH, index=False)
    
    print("\nSuccess! Your dataset is now fully converted to Manglish (Latin script).")

if __name__ == "__main__":
    main()