import os
import pandas as pd

# ==========================================
# 1. Setup Paths
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

ORIGINAL_DATA_PATH = os.path.join(ROOT_DIR, "datasets", "processed", "humaid_processed.csv")
SYNTHETIC_DATA_PATH = os.path.join(ROOT_DIR, "datasets", "processed", "synthetic_resource_requests.csv")
OUTPUT_PATH = os.path.join(ROOT_DIR, "datasets", "processed", "humaid_augmented.csv")

def augment_data():
    print(f"Loading original data from {ORIGINAL_DATA_PATH}")
    df_orig = pd.read_csv(ORIGINAL_DATA_PATH)
    
    print(f"Loading synthetic data from {SYNTHETIC_DATA_PATH}")
    df_synth = pd.read_csv(SYNTHETIC_DATA_PATH)
    
    # Ensure the synthetic data is STRICTLY for training
    df_synth['split'] = 'train'
    
    # ==========================================
    # 2. ENFORCE COLUMN PARITY
    # ==========================================
    # We strip both datasets down to only the essential text classification columns
    columns_to_keep = ['clean_text', 'target_label', 'split']
    
    print(f"Stripping original dataset down to: {columns_to_keep}")
    df_orig = df_orig[columns_to_keep]
    
    print(f"Stripping synthetic dataset down to: {columns_to_keep}")
    df_synth = df_synth[columns_to_keep]
    
    # ==========================================
    # 3. Merge and Shuffle
    # ==========================================
    print("Merging datasets...")
    df_combined = pd.concat([df_orig, df_synth], ignore_index=True)
    
    # Shuffle the dataset thoroughly using a fixed random state for reproducibility
    print("Shuffling the combined dataset...")
    df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # ==========================================
    # 4. Print Stats and Save
    # ==========================================
    print("\n--- Final Dataset Class Distribution (Train Split Only) ---")
    train_only = df_combined[df_combined['split'] == 'train']
    print(train_only['target_label'].value_counts())
    
    df_combined.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSuccessfully saved augmented dataset to: {OUTPUT_PATH}")

if __name__ == "__main__":
    augment_data()