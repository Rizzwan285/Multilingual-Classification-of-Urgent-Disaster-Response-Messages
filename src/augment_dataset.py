# -*- coding: utf-8 -*-
"""
Merge Synthetic Data into HumAID Training Set

Loads:
  - datasets/processed/humaid_processed.csv              (real HumAID, all splits)
  - datasets/processed/synthetic_resource_requests.csv   (LLM-generated)

Produces:
  - datasets/processed/humaid_train_augmented.csv
      Contains: all real train rows + all synthetic rows, shuffled.
      Does NOT touch dev/test — those stay in humaid_processed.csv untouched.

Design notes:
- Synthetic rows ONLY augment the train split. Dev and test must stay pristine
  for honest evaluation.
- Synthetic rows carry `is_synthetic=True`, real rows carry `is_synthetic=False`,
  so downstream scripts can run ablations (train with/without synthetic).
- Shuffling uses a fixed seed for reproducibility.
- Schema is harmonized: both sources end up with the same columns, with NaN
  filling any metadata missing on the synthetic side (tweet_id, event, etc.).
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

# --- Paths ---
# Resolve repo root from script location (assumes script lives in <repo>/src/
# or <repo>/notebooks/). Adjust `.parent.parent` if your layout differs.
try:
    BASE_DIR = Path(__file__).resolve().parent.parent
except NameError:
    BASE_DIR = Path.cwd().parent

PROCESSED_DIR = BASE_DIR / "datasets" / "processed"
REAL_CSV = PROCESSED_DIR / "humaid_processed.csv"
SYNTH_CSV = PROCESSED_DIR / "synthetic_resource_requests.csv"
AUGMENTED_CSV = PROCESSED_DIR / "humaid_train_augmented.csv"

RANDOM_SEED = 42


def merge():
    # --- Load real data, keep only train split ---
    if not REAL_CSV.exists():
        raise FileNotFoundError(f"Real processed data not found at {REAL_CSV}")
    real_df = pd.read_csv(REAL_CSV)
    real_train = real_df[real_df["split"] == "train"].copy()
    real_train["is_synthetic"] = False

    print(f"Loaded real train rows: {len(real_train):,}")
    print("Real train class distribution:")
    print(real_train["target_label"].value_counts().to_string())
    print()

    # --- Load synthetic data ---
    if not SYNTH_CSV.exists():
        raise FileNotFoundError(
            f"Synthetic data not found at {SYNTH_CSV}. "
            f"Run generate_synthetic_data.py first."
        )
    synth_df = pd.read_csv(SYNTH_CSV)
    if "is_synthetic" not in synth_df.columns:
        synth_df["is_synthetic"] = True

    print(f"Loaded synthetic rows: {len(synth_df):,}")
    print()

    # --- Harmonize schemas ---
    # Real has: tweet_id, tweet_text, class_label, event, split, event_set,
    #          target_label, clean_text, text_length, word_count, is_synthetic
    # Synth has: clean_text, target_label, split, is_synthetic
    # Fill any missing columns on the synthetic side so concat is clean.
    for col in real_train.columns:
        if col not in synth_df.columns:
            synth_df[col] = np.nan

    # Reorder synth_df columns to match real_train
    synth_df = synth_df[real_train.columns]

    # Recompute text_length / word_count for synthetic rows if those cols exist
    if "text_length" in synth_df.columns:
        synth_df["text_length"] = synth_df["clean_text"].astype(str).apply(len)
    if "word_count" in synth_df.columns:
        synth_df["word_count"] = synth_df["clean_text"].astype(str).apply(
            lambda x: len(x.split())
        )

    # --- Concatenate and shuffle ---
    combined = pd.concat([real_train, synth_df], ignore_index=True)
    combined = combined.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    # --- Save ---
    combined.to_csv(AUGMENTED_CSV, index=False)

    # --- Report ---
    print("=" * 60)
    print(f"Saved augmented training set → {AUGMENTED_CSV}")
    print(f"Total rows: {len(combined):,}")
    print(f"  Real:      {(~combined['is_synthetic']).sum():,}")
    print(f"  Synthetic: {combined['is_synthetic'].sum():,}")
    print()
    print("Augmented train class distribution:")
    dist = combined["target_label"].value_counts()
    for cls, cnt in dist.items():
        pct = cnt / len(combined) * 100
        print(f"  {cls:30s}: {cnt:6,}  ({pct:.1f}%)")
    print()
    print("First 5 rows of shuffled output (spot-check that synthetic is mixed in):")
    print(combined[["clean_text", "target_label", "is_synthetic"]].head().to_string())


if __name__ == "__main__":
    merge()