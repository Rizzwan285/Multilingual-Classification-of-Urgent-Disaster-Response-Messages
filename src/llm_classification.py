"""
LLM Classification — Unified Runner
------------------------------------
Runs multiple LLM classification experiments (different prompting strategies
and seeds) in a single script, saving all outputs under one results subfolder.

Replaces the 4 separate scripts (llm_classification, _v2, _v3_fewshot, _v4).

Each experiment is defined by:
  - name:     short identifier used in folder/file names
  - strategy: 'zero_shot_detailed', 'zero_shot_minimal', or 'one_shot'
  - seed:     random seed for sample selection

All outputs land under results/llm_classification/, with per-experiment
subfolders and a combined comparison table at the top level.
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os
import json
import time
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ── Paths ──────────────────────────────────────────────────────────────────
try:
    BASE_DIR = Path(__file__).resolve().parent.parent
except NameError:
    BASE_DIR = Path.cwd().parent

load_dotenv(BASE_DIR / ".env")

DATA_PATH = BASE_DIR / "datasets" / "processed" / "humaid_processed.csv"
RESULTS_ROOT = BASE_DIR / "results" / "llm_classification"
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

# ── Global config ─────────────────────────────────────────────────────────
MODEL_NAME = "llama-3.3-70b-versatile"
SAMPLES_PER_LABEL = 10   # bump to 20-30 for more statistical power
LABELS = [
    "Situational Awareness",
    "Volunteering and Donations",
    "Critical Rescue",
    "Irrelevant",
    "Resource Requests",
]
RATE_LIMIT_SLEEP = 2.1   # Groq free tier: 30 req/min

# ── Prompts ───────────────────────────────────────────────────────────────
# Zero-shot with detailed category descriptions (original v1/v4 prompt)
PROMPT_ZERO_SHOT_DETAILED = """You are a disaster response message classifier. You will be given a tweet related to a disaster event. Your task is to classify the tweet into EXACTLY ONE of the following 5 categories:

1. Situational Awareness - Information about the disaster situation, damage reports, weather updates, news coverage, sympathy/support messages, caution/advice, or general observations about the disaster.
2. Volunteering and Donations - Messages about donations, fundraising, volunteering, relief efforts, rescue operations organized by volunteers, or offers of help.
3. Critical Rescue - Messages about evacuations, displaced people, rescue operations, urgent evacuation orders, or people directly affected needing immediate help.
4. Irrelevant - Messages not directly related to humanitarian disaster response, off-topic remarks, political commentary unrelated to relief, or general unrelated observations.
5. Resource Requests - Explicit requests or urgent needs for resources, supplies, help, or assistance.

IMPORTANT: Respond with ONLY the label name, nothing else. No explanation, no punctuation, no extra text. Just the exact label from the 5 options above."""

# Zero-shot with minimal category info (original v2 prompt — just the names)
PROMPT_ZERO_SHOT_MINIMAL = """You are a disaster response message classifier. You will be given a tweet related to a disaster event. Your task is to classify the tweet into EXACTLY ONE of the following 5 categories:

1. Situational Awareness
2. Volunteering and Donations
3. Critical Rescue
4. Irrelevant
5. Resource Requests

IMPORTANT: Respond with ONLY the label name, nothing else. No explanation, no punctuation, no extra text. Just the exact label from the 5 options above."""

# One-shot with one example per category (original v3 prompt)
PROMPT_ONE_SHOT = PROMPT_ZERO_SHOT_DETAILED + """

Here is one correct example for each category:

Example 1:
Tweet: "stay safe everyone.. aftershocks alert earthquake azadkashmir rawalpindi islamabaad"
Label: Situational Awareness

Example 2:
Tweet: "by attending 100pipersplayforacause music event you can donate for kerala flood relief"
Label: Volunteering and Donations

Example 3:
Tweet: "death toll continues to rise in california wildfire, as does the number missing"
Label: Critical Rescue

Example 4:
Tweet: "why would a verified sports journalism organization be so petty?"
Label: Irrelevant

Example 5:
Tweet: "hey guys please help me help my family in puerto rico from the devastation of hurricane maria. anything helps!"
Label: Resource Requests"""

STRATEGY_PROMPTS = {
    "zero_shot_detailed": PROMPT_ZERO_SHOT_DETAILED,
    "zero_shot_minimal": PROMPT_ZERO_SHOT_MINIMAL,
    "one_shot": PROMPT_ONE_SHOT,
}

# ── Experiment list ───────────────────────────────────────────────────────
# Each entry reproduces one of the original v1-v4 runs.
# Add/remove entries to run more or fewer experiments.
EXPERIMENTS = [
    {"name": "zero_shot_detailed_seed42",  "strategy": "zero_shot_detailed", "seed": 42},
    {"name": "zero_shot_minimal_seed99",   "strategy": "zero_shot_minimal",  "seed": 99},
    {"name": "one_shot_seed123",           "strategy": "one_shot",           "seed": 123},
    {"name": "zero_shot_detailed_seed200", "strategy": "zero_shot_detailed", "seed": 200},
]


# ── Core functions ────────────────────────────────────────────────────────

def sample_balanced_test_set(df_test_split, seed):
    """Stratified sample of SAMPLES_PER_LABEL tweets per label, then shuffled."""
    sampled_frames = []
    for label in LABELS:
        label_df = df_test_split[df_test_split["target_label"] == label]
        sampled = label_df.sample(n=SAMPLES_PER_LABEL, random_state=seed)
        sampled_frames.append(sampled)
    test_df = pd.concat(sampled_frames).reset_index(drop=True)
    return test_df.sample(frac=1, random_state=seed).reset_index(drop=True)


def classify_tweets(client, test_df, system_prompt, total):
    """Send each tweet to the LLM and collect predictions + raw responses."""
    predictions = []
    raw_responses = []

    for idx, row in test_df.iterrows():
        tweet_text = str(row["clean_text"])
        true_label = row["target_label"]

        safe_text = tweet_text[:80].encode("ascii", errors="replace").decode("ascii")
        print(f"\n[{idx + 1}/{total}] Classifying tweet...")
        print(f"  Text: {safe_text}...")
        print(f"  True Label: {true_label}")

        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f'Classify this tweet:\n\n"{tweet_text}"'},
                ],
                temperature=0.0,
                max_tokens=50,
            )
            response = completion.choices[0].message.content.strip()
            raw_responses.append(response)

            matched_label = None
            for label in LABELS:
                if label.lower() in response.lower():
                    matched_label = label
                    break
            if matched_label is None:
                print(f"  WARNING: Could not match response: '{response}' -- defaulting to 'Irrelevant'")
                matched_label = "Irrelevant"

            predictions.append(matched_label)
            print(f"  Predicted: {matched_label}")

        except Exception as e:
            print(f"  ERROR: {e}")
            predictions.append("ERROR")
            raw_responses.append(str(e))
            time.sleep(5)

        time.sleep(RATE_LIMIT_SLEEP)

    return predictions, raw_responses


def evaluate_and_save(test_df, predictions, raw_responses, experiment, out_dir):
    """Compute metrics, print them, save CSV + JSON. Returns summary dict."""
    test_df = test_df.copy()
    test_df["predicted_label"] = predictions
    test_df["raw_response"] = raw_responses
    test_df["correct"] = test_df["target_label"] == test_df["predicted_label"]

    valid_mask = test_df["predicted_label"] != "ERROR"
    valid_df = test_df[valid_mask]

    summary = {
        "experiment_name": experiment["name"],
        "model": MODEL_NAME,
        "strategy": experiment["strategy"],
        "random_seed": experiment["seed"],
        "total_samples": len(test_df),
        "valid_predictions": int(valid_mask.sum()),
        "accuracy": 0.0,
        "per_label_accuracy": {},
        "labels": LABELS,
    }

    if len(valid_df) == 0:
        print("\nNo valid predictions to evaluate!")
    else:
        true_labels = valid_df["target_label"].tolist()
        pred_labels = valid_df["predicted_label"].tolist()

        accuracy = accuracy_score(true_labels, pred_labels)
        summary["accuracy"] = float(accuracy)
        print(f"\nOverall Accuracy: {accuracy:.2%} "
              f"({int(accuracy * len(valid_df))}/{len(valid_df)} correct)")

        print(f"\nPer-Label Accuracy:")
        for label in LABELS:
            label_mask = valid_df["target_label"] == label
            if label_mask.sum() > 0:
                label_acc = valid_df.loc[label_mask, "correct"].mean()
                correct_count = valid_df.loc[label_mask, "correct"].sum()
                total_count = label_mask.sum()
                summary["per_label_accuracy"][label] = float(label_acc)
                print(f"  {label:<30s}: {label_acc:.0%} ({correct_count}/{total_count})")

        print(f"\nDetailed Classification Report:")
        print(classification_report(true_labels, pred_labels, labels=LABELS, zero_division=0))

        print(f"\nConfusion Matrix:")
        cm = confusion_matrix(true_labels, pred_labels, labels=LABELS)
        cm_df = pd.DataFrame(cm, index=LABELS, columns=LABELS)
        print(cm_df.to_string())

        misclassified = valid_df[~valid_df["correct"]]
        print(f"\n{'=' * 70}")
        print(f"Misclassified Examples ({len(misclassified)}):")
        print(f"{'=' * 70}")
        for _, row in misclassified.iterrows():
            safe = str(row["clean_text"][:100]).encode("ascii", errors="replace").decode("ascii")
            print(f"\n  Text:      {safe}...")
            print(f"  True:      {row['target_label']}")
            print(f"  Predicted: {row['predicted_label']}")
            print(f"  Raw LLM:   {row['raw_response']}")

    # Save per-experiment artifacts
    test_df.to_csv(out_dir / "test_samples.csv", index=False)
    test_df[["clean_text", "target_label", "predicted_label", "raw_response", "correct"]].to_csv(
        out_dir / "classification_results.csv", index=False
    )
    with open(out_dir / "classification_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved artifacts to: {out_dir}")
    return summary


def run_experiment(client, df_test_split, experiment):
    """Run a single experiment end-to-end."""
    name = experiment["name"]
    strategy = experiment["strategy"]
    seed = experiment["seed"]

    print("\n" + "#" * 70)
    print(f"# EXPERIMENT: {name}")
    print(f"# Strategy: {strategy} | Seed: {seed}")
    print("#" * 70)

    out_dir = RESULTS_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: sample
    print("\nSTEP 1: Preparing balanced test set")
    print("-" * 70)
    test_df = sample_balanced_test_set(df_test_split, seed)
    print(f"Test set size: {len(test_df)}")
    print(f"Label distribution:\n{test_df['target_label'].value_counts()}")

    # Step 2: classify
    print(f"\nSTEP 2: Classifying with {MODEL_NAME} ({strategy})")
    print("-" * 70)
    prompt = STRATEGY_PROMPTS[strategy]
    predictions, raw_responses = classify_tweets(client, test_df, prompt, len(test_df))

    # Step 3: evaluate + save
    print(f"\nSTEP 3: Evaluation Results")
    print("-" * 70)
    return evaluate_and_save(test_df, predictions, raw_responses, experiment, out_dir)


def save_comparison_table(summaries):
    """Write a side-by-side comparison of all experiments."""
    rows = []
    for s in summaries:
        row = {
            "experiment": s["experiment_name"],
            "strategy": s["strategy"],
            "seed": s["random_seed"],
            "overall_accuracy": s["accuracy"],
        }
        for label in LABELS:
            row[label] = s["per_label_accuracy"].get(label, None)
        rows.append(row)

    comparison_df = pd.DataFrame(rows)
    comparison_csv = RESULTS_ROOT / "comparison_across_experiments.csv"
    comparison_df.to_csv(comparison_csv, index=False)

    print("\n" + "=" * 70)
    print("FINAL COMPARISON ACROSS ALL EXPERIMENTS")
    print("=" * 70)
    print(comparison_df.to_string(index=False))
    print(f"\nComparison saved to: {comparison_csv}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("LLM CLASSIFICATION — UNIFIED RUNNER")
    print("=" * 70)
    print(f"Experiments to run: {len(EXPERIMENTS)}")
    for exp in EXPERIMENTS:
        print(f"  - {exp['name']}  ({exp['strategy']}, seed={exp['seed']})")
    print(f"Results root: {RESULTS_ROOT}")
    print()

    # Load data once
    df = pd.read_csv(DATA_PATH)
    df_test_split = df[df["split"] == "test"].copy()
    print(f"Loaded {len(df)} rows total, {len(df_test_split)} in test split")

    # One Groq client reused across experiments
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    # Run all experiments
    summaries = []
    for experiment in EXPERIMENTS:
        summary = run_experiment(client, df_test_split, experiment)
        summaries.append(summary)

    # Combined comparison
    save_comparison_table(summaries)

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()