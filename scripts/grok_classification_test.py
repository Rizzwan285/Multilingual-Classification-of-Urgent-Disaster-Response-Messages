"""
LLM Classification Test using Groq API (LLaMA 3.3 70B)
-------------------------------------------------------
Samples 50 data points (10 per label) from the HumAID dataset,
asks the LLM to classify each tweet into one of 5 disaster-response
categories, and compares with the ground-truth labels.
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os
import json
import time
import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ── Configuration ──────────────────────────────────────────────────────────
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'processed', 'humaid_processed.csv')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'llm_classification')
MODEL_NAME = "llama-3.3-70b-versatile"   # Groq-hosted LLaMA 3.3 model
SAMPLES_PER_LABEL = 10                   # 10 per label × 5 labels = 50
LABELS = [
    "Situational Awareness",
    "Volunteering and Donations",
    "Critical Rescue",
    "Irrelevant",
    "Resource Requests",
]

os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Step 1: Prepare balanced 50-sample test set ───────────────────────────
print("=" * 70)
print("STEP 1: Preparing balanced test set (50 samples, 10 per label)")
print("=" * 70)

df = pd.read_csv(DATA_PATH)
print(f"Total dataset size: {len(df)}")

# Filter to ONLY test split data
df_test_split = df[df['split'] == 'test'].copy()
print(f"Test split size: {len(df_test_split)}")
print(f"\nLabel distribution in test split:")
print(df_test_split['target_label'].value_counts())

# Stratified sampling: 10 per label from TEST split only
sampled_frames = []
for label in LABELS:
    label_df = df_test_split[df_test_split['target_label'] == label]
    sampled = label_df.sample(n=SAMPLES_PER_LABEL, random_state=42)
    sampled_frames.append(sampled)

test_df = pd.concat(sampled_frames).reset_index(drop=True)
test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle

print(f"\nTest set size: {len(test_df)}")
print(f"\nTest set label distribution:")
print(test_df['target_label'].value_counts())

# Save the test set for reference
test_csv_path = os.path.join(RESULTS_DIR, 'test_samples_50.csv')
test_df.to_csv(test_csv_path, index=False)
print(f"\nTest set saved to: {test_csv_path}")

# ── Step 2: Classify using Groq API ──────────────────────────────────────
print("\n" + "=" * 70)
print(f"STEP 2: Classifying with {MODEL_NAME} via Groq API")
print("=" * 70)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are a disaster response message classifier. You will be given a tweet related to a disaster event. Your task is to classify the tweet into EXACTLY ONE of the following 5 categories:

1. Situational Awareness - Information about the disaster situation, damage reports, weather updates, news coverage, sympathy/support messages, caution/advice, or general observations about the disaster.
2. Volunteering and Donations - Messages about donations, fundraising, volunteering, relief efforts, rescue operations organized by volunteers, or offers of help.
3. Critical Rescue - Messages about evacuations, displaced people, rescue operations, urgent evacuation orders, or people directly affected needing immediate help.
4. Irrelevant - Messages not directly related to humanitarian disaster response, off-topic remarks, political commentary unrelated to relief, or general unrelated observations.
5. Resource Requests - Explicit requests or urgent needs for resources, supplies, help, or assistance.

IMPORTANT: Respond with ONLY the label name, nothing else. No explanation, no punctuation, no extra text. Just the exact label from the 5 options above."""

predictions = []
raw_responses = []

for idx, row in test_df.iterrows():
    tweet_text = row['clean_text']
    true_label = row['target_label']
    
    safe_text = tweet_text[:80].encode('ascii', errors='replace').decode('ascii')
    print(f"\n[{idx + 1}/50] Classifying tweet...")
    print(f"  Text: {safe_text}...")
    print(f"  True Label: {true_label}")
    
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Classify this tweet:\n\n\"{tweet_text}\""}
            ],
            temperature=0.0,  # Deterministic output
            max_tokens=50,
        )
        
        response = completion.choices[0].message.content.strip()
        raw_responses.append(response)
        
        # Try to match response to a known label
        matched_label = None
        for label in LABELS:
            if label.lower() in response.lower():
                matched_label = label
                break
        
        if matched_label is None:
            print(f"  ⚠ Could not match response: '{response}' — defaulting to 'Irrelevant'")
            matched_label = "Irrelevant"
        
        predictions.append(matched_label)
        print(f"  Predicted: {matched_label}")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        predictions.append("ERROR")
        raw_responses.append(str(e))
        time.sleep(2)  # Back off on error
    
    # Rate limit: Groq allows 30 requests/min → 1 request every 2 seconds
    time.sleep(2.1)

# ── Step 3: Compare predictions with ground truth ────────────────────────
print("\n" + "=" * 70)
print("STEP 3: Evaluation Results")
print("=" * 70)

test_df['predicted_label'] = predictions
test_df['raw_response'] = raw_responses
test_df['correct'] = test_df['target_label'] == test_df['predicted_label']

# Filter out errors
valid_mask = test_df['predicted_label'] != "ERROR"
valid_df = test_df[valid_mask]

if len(valid_df) == 0:
    print("No valid predictions to evaluate!")
else:
    true_labels = valid_df['target_label'].tolist()
    pred_labels = valid_df['predicted_label'].tolist()
    
    # Overall accuracy
    accuracy = accuracy_score(true_labels, pred_labels)
    print(f"\nOverall Accuracy: {accuracy:.2%} ({int(accuracy * len(valid_df))}/{len(valid_df)} correct)")
    
    # Per-label accuracy
    print(f"\nPer-Label Accuracy:")
    for label in LABELS:
        label_mask = valid_df['target_label'] == label
        if label_mask.sum() > 0:
            label_acc = (valid_df.loc[label_mask, 'correct']).mean()
            correct_count = (valid_df.loc[label_mask, 'correct']).sum()
            total_count = label_mask.sum()
            print(f"  {label:<30s}: {label_acc:.0%} ({correct_count}/{total_count})")
    
    # Classification report
    print(f"\nDetailed Classification Report:")
    print(classification_report(true_labels, pred_labels, labels=LABELS, zero_division=0))
    
    # Confusion matrix
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(true_labels, pred_labels, labels=LABELS)
    cm_df = pd.DataFrame(cm, index=LABELS, columns=LABELS)
    print(cm_df.to_string())
    
    # Misclassified examples
    misclassified = valid_df[~valid_df['correct']]
    print(f"\n{'=' * 70}")
    print(f"Misclassified Examples ({len(misclassified)}):")
    print(f"{'=' * 70}")
    for _, row in misclassified.iterrows():
        safe_mis_text = str(row['clean_text'][:100]).encode('ascii', errors='replace').decode('ascii')
        print(f"\n  Text:      {safe_mis_text}...")
        print(f"  True:      {row['target_label']}")
        print(f"  Predicted: {row['predicted_label']}")
        print(f"  Raw LLM:   {row['raw_response']}")

# ── Step 4: Save full results ─────────────────────────────────────────────
results_csv_path = os.path.join(RESULTS_DIR, 'classification_results.csv')
test_df[['clean_text', 'target_label', 'predicted_label', 'raw_response', 'correct']].to_csv(
    results_csv_path, index=False
)
print(f"\nFull results saved to: {results_csv_path}")

# Save summary report
summary = {
    "model": MODEL_NAME,
    "total_samples": len(test_df),
    "valid_predictions": len(valid_df),
    "accuracy": float(accuracy) if len(valid_df) > 0 else 0,
    "per_label_accuracy": {},
    "labels": LABELS,
}
for label in LABELS:
    label_mask = valid_df['target_label'] == label
    if label_mask.sum() > 0:
        summary["per_label_accuracy"][label] = float((valid_df.loc[label_mask, 'correct']).mean())

summary_path = os.path.join(RESULTS_DIR, 'classification_summary.json')
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"Summary saved to: {summary_path}")

print(f"\n{'=' * 70}")
print(f"DONE! Model: {MODEL_NAME} | Accuracy: {accuracy:.2%}")
print(f"{'=' * 70}")
