import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import os
import json
import time
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#resolving base directory
try:
    BASE_DIR = Path(__file__).resolve().parent.parent
except NameError:
    BASE_DIR = Path.cwd().parent

load_dotenv(BASE_DIR / ".env")

DATA_PATH    = BASE_DIR / "datasets" / "processed" / "humaid_processed.csv"
RESULTS_ROOT = BASE_DIR / "results" / "llm_classification"
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

MODEL_NAME        = "llama-3.3-70b-versatile"
SAMPLES_PER_LABEL = 10
RATE_LIMIT_SLEEP  = 2.1
MAX_RETRIES       = 3

LABELS = [
    "Situational Awareness",
    "Volunteering and Donations",
    "Critical Rescue",
    "Irrelevant",
    "Resource Requests",
]

PROMPT_ZERO_SHOT_DETAILED = """You are a disaster response message classifier. You will be given a tweet related to a disaster event. Your task is to classify the tweet into EXACTLY ONE of the following 5 categories:

1. Situational Awareness - Information about the disaster situation, damage reports, weather updates, news coverage, sympathy/support messages, caution/advice, or general observations about the disaster.
2. Volunteering and Donations - Messages about donations, fundraising, volunteering, relief efforts, rescue operations organized by volunteers, or offers of help.
3. Critical Rescue - Messages about evacuations, displaced people, rescue operations, urgent evacuation orders, or people directly affected needing immediate help.
4. Irrelevant - Messages not directly related to humanitarian disaster response, off-topic remarks, political commentary unrelated to relief, or general unrelated observations.
5. Resource Requests - Explicit requests or urgent needs for resources, supplies, help, or assistance.

IMPORTANT: Respond with ONLY the label name, nothing else. No explanation, no punctuation, no extra text. Just the exact label from the 5 options above."""

PROMPT_ZERO_SHOT_MINIMAL = """You are a disaster response message classifier. You will be given a tweet related to a disaster event. Your task is to classify the tweet into EXACTLY ONE of the following 5 categories:

1. Situational Awareness
2. Volunteering and Donations
3. Critical Rescue
4. Irrelevant
5. Resource Requests

IMPORTANT: Respond with ONLY the label name, nothing else. No explanation, no punctuation, no extra text. Just the exact label from the 5 options above."""

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
    "zero_shot_minimal":  PROMPT_ZERO_SHOT_MINIMAL,
    "one_shot":           PROMPT_ONE_SHOT,
}

EXPERIMENTS = [
    {"name": "zero_shot_detailed_seed42",  "strategy": "zero_shot_detailed", "seed": 42},
    {"name": "zero_shot_minimal_seed99",   "strategy": "zero_shot_minimal",  "seed": 99},
    {"name": "one_shot_seed123",           "strategy": "one_shot",           "seed": 123},
    {"name": "zero_shot_detailed_seed200", "strategy": "zero_shot_detailed", "seed": 200},
]


def match_label(response):
    """
    Tries exact match first, then partial match.
    Returns the matched label string, or 'UNMATCHED' if nothing fits.
    Using UNMATCHED instead of defaulting to a real label keeps metrics honest.
    """
    cleaned = response.strip()
    #checking for exact match first
    if cleaned in LABELS:
        return cleaned
    #falling back to case-insensitive partial match
    cleaned_lower = cleaned.lower()
    for label in LABELS:
        if label.lower() in cleaned_lower:
            return label
    return "UNMATCHED"


def call_with_retry(client, system_prompt, tweet_text):
    """Calling the API and retrying with exponential backoff on failure."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": f'Classify this tweet:\n\n"{tweet_text}"'},
                ],
                temperature=0.0,
                max_tokens=50,
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            wait = 2 ** attempt
            print(f"  [attempt {attempt}/{MAX_RETRIES}] API error: {e}. Retrying in {wait}s...")
            time.sleep(wait)
    print(f"  [error] all {MAX_RETRIES} attempts failed. marking as ERROR.")
    return "ERROR"


def sample_balanced_test_set(df_test_split, seed):
    """Sampling SAMPLES_PER_LABEL tweets per label, then shuffling."""
    missing = [l for l in LABELS if l not in df_test_split["target_label"].values]
    if missing:
        raise ValueError(f"Labels not found in test split: {missing}")

    frames = []
    for label in LABELS:
        label_df = df_test_split[df_test_split["target_label"] == label]
        if len(label_df) < SAMPLES_PER_LABEL:
            raise ValueError(
                f"Label '{label}' has only {len(label_df)} samples, need {SAMPLES_PER_LABEL}."
            )
        frames.append(label_df.sample(n=SAMPLES_PER_LABEL, random_state=seed))

    return pd.concat(frames).sample(frac=1, random_state=seed).reset_index(drop=True)


def classify_tweets(client, test_df, system_prompt):
    """Sending each tweet to the LLM and collecting predictions."""
    predictions   = []
    raw_responses = []
    total         = len(test_df)

    for idx, row in test_df.iterrows():
        tweet_text = str(row["clean_text"])
        safe_text  = tweet_text[:80].encode("ascii", errors="replace").decode("ascii")
        print(f"\n[{idx + 1}/{total}] classifying...")
        print(f"  text : {safe_text}...")
        print(f"  true : {row['target_label']}")

        raw = call_with_retry(client, system_prompt, tweet_text)
        raw_responses.append(raw)

        if raw == "ERROR":
            predicted = "ERROR"
        else:
            predicted = match_label(raw)
            if predicted == "UNMATCHED":
                print(f"  [warn] could not match response: '{raw}' — marking as UNMATCHED")

        predictions.append(predicted)
        print(f"  pred : {predicted}")
        time.sleep(RATE_LIMIT_SLEEP)

    return predictions, raw_responses


def evaluate_and_save(test_df, predictions, raw_responses, experiment, out_dir):
    """Computing metrics and saving all results for one experiment."""
    test_df = test_df.copy()
    test_df["predicted_label"] = predictions
    test_df["raw_response"]    = raw_responses
    test_df["correct"]         = test_df["target_label"] == test_df["predicted_label"]

    #only scoring rows where the model actually returned a valid label
    valid_mask = ~test_df["predicted_label"].isin(["ERROR", "UNMATCHED"])
    valid_df   = test_df[valid_mask]
    skipped    = len(test_df) - len(valid_df)
    if skipped:
        print(f"\n[info] skipping {skipped} row(s) with ERROR or UNMATCHED predictions in metrics")

    summary = {
        "experiment_name":    experiment["name"],
        "model":              MODEL_NAME,
        "strategy":           experiment["strategy"],
        "random_seed":        experiment["seed"],
        "total_samples":      len(test_df),
        "valid_predictions":  int(valid_mask.sum()),
        "skipped":            skipped,
        "accuracy":           0.0,
        "per_label_accuracy": {},
        "labels":             LABELS,
    }

    if len(valid_df) == 0:
        print("\n[warn] no valid predictions — skipping metric computation")
    else:
        true_labels = valid_df["target_label"].tolist()
        pred_labels = valid_df["predicted_label"].tolist()

        accuracy = accuracy_score(true_labels, pred_labels)
        summary["accuracy"] = float(accuracy)
        print(f"\noverall accuracy: {accuracy:.2%} ({int(accuracy * len(valid_df))}/{len(valid_df)})")

        print("\nper-label accuracy:")
        for label in LABELS:
            mask = valid_df["target_label"] == label
            if mask.sum() > 0:
                label_acc = valid_df.loc[mask, "correct"].mean()
                summary["per_label_accuracy"][label] = float(label_acc)
                print(f"  {label:<30s}: {label_acc:.0%} ({int(valid_df.loc[mask,'correct'].sum())}/{mask.sum()})")

        print("\nclassification report:")
        print(classification_report(true_labels, pred_labels, labels=LABELS, zero_division=0))

        print("confusion matrix:")
        cm    = confusion_matrix(true_labels, pred_labels, labels=LABELS)
        cm_df = pd.DataFrame(cm, index=LABELS, columns=LABELS)
        print(cm_df.to_string())

        misclassified = valid_df[~valid_df["correct"]]
        print(f"\nmisclassified examples ({len(misclassified)}):")
        print("=" * 70)
        for _, row in misclassified.iterrows():
            safe = str(row["clean_text"][:100]).encode("ascii", errors="replace").decode("ascii")
            print(f"  text : {safe}...")
            print(f"  true : {row['target_label']}")
            print(f"  pred : {row['predicted_label']}")
            print(f"  raw  : {row['raw_response']}\n")

    test_df.to_csv(out_dir / "test_samples.csv", index=False)
    test_df[["clean_text", "target_label", "predicted_label", "raw_response", "correct"]].to_csv(
        out_dir / "classification_results.csv", index=False
    )
    with open(out_dir / "classification_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nsaved results to: {out_dir}")
    return summary


def run_experiment(client, df_test_split, experiment):
    """Running one full experiment — sampling, classifying, and evaluating."""
    print("\n" + "#" * 70)
    print(f"# experiment : {experiment['name']}")
    print(f"# strategy   : {experiment['strategy']}  seed: {experiment['seed']}")
    print("#" * 70)

    out_dir = RESULTS_ROOT / experiment["name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    test_df = sample_balanced_test_set(df_test_split, experiment["seed"])
    print(f"\ntest set size: {len(test_df)}")
    print(test_df["target_label"].value_counts().to_string())

    prompt = STRATEGY_PROMPTS[experiment["strategy"]]
    predictions, raw_responses = classify_tweets(client, test_df, prompt)

    return evaluate_and_save(test_df, predictions, raw_responses, experiment, out_dir)


def save_comparison_table(summaries):
    """Writing a side-by-side comparison table for all experiments."""
    rows = []
    for s in summaries:
        row = {
            "experiment": s["experiment_name"],
            "strategy":   s["strategy"],
            "seed":       s["random_seed"],
            "accuracy":   s["accuracy"],
            "skipped":    s["skipped"],
        }
        for label in LABELS:
            row[label] = s["per_label_accuracy"].get(label)
        rows.append(row)

    comparison_df  = pd.DataFrame(rows)
    comparison_csv = RESULTS_ROOT / "comparison_across_experiments.csv"
    comparison_df.to_csv(comparison_csv, index=False)

    print("\n" + "=" * 70)
    print("comparison across all experiments")
    print("=" * 70)
    print(comparison_df.to_string(index=False))
    print(f"\nsaved to: {comparison_csv}")


def main():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not set — check your .env file")

    print("=" * 70)
    print("llm classification — unified runner")
    print("=" * 70)
    for exp in EXPERIMENTS:
        print(f"  {exp['name']}  ({exp['strategy']}, seed={exp['seed']})")
    print(f"\nresults root: {RESULTS_ROOT}\n")

    df            = pd.read_csv(DATA_PATH)
    df_test_split = df[df["split"] == "test"].copy()
    print(f"loaded {len(df)} rows, {len(df_test_split)} in test split")

    client    = Groq(api_key=api_key)
    summaries = []
    for experiment in EXPERIMENTS:
        summary = run_experiment(client, df_test_split, experiment)
        summaries.append(summary)

    save_comparison_table(summaries)
    print("\n" + "=" * 70)
    print("all experiments complete")
    print("=" * 70)


if __name__ == "__main__":
    main()