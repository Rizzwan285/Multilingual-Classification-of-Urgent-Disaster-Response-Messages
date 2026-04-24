"""
Synthetic Data Generation for "Resource Requests" class.

Improvements over v1:
1. Rotating seeds     — resamples 5 new examples per API call, train-split only,
                         to avoid dev/test leakage and boost diversity.
2. Social media style — prompts the LLM for typos, abbreviations, hashtags,
                         urgency, broken grammar.
3. Quality filtering  — drops rows that are too short, too long, empty after
                         cleaning, or clearly off-label.
4. Deduplication      — removes exact duplicates and near-duplicates across
                         the whole generated set.
5. Safe path handling — Colab/Jupyter compatible, same as preprocessing.
"""

import os
import re
import time
import random
from pathlib import Path

import pandas as pd
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
# Resolve repo root from script location (assumes script lives in <repo>/src/
# or <repo>/notebooks/). Adjust `.parent.parent` if your layout differs.
try:
    BASE_DIR = Path(__file__).resolve().parent.parent
except NameError:
    BASE_DIR = Path.cwd().parent

DATA_PATH = BASE_DIR / "datasets" / "processed" / "humaid_processed.csv"
OUTPUT_PATH = BASE_DIR / "datasets" / "processed" / "synthetic_resource_requests.csv"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# --- Config ---
TARGET_COUNT = 2500          # How many final, filtered synthetic rows we want
TWEETS_PER_CALL = 10         # Number requested from LLM per call
SEEDS_PER_CALL = 5           # Number of real examples shown to LLM per call
MIN_WORDS = 4                # Quality filter: min word count
MAX_WORDS = 60               # Quality filter: max word count (tweets are short)
RATE_LIMIT_SLEEP = 7.5       # Seconds between calls (Groq free tier safety)
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def build_prompt(seed_examples):
    """Build a fresh prompt with rotating seeds and social-media style guidance."""
    seed_block = "\n".join(f"- {ex}" for ex in seed_examples)

    return f"""You are generating realistic social media posts (tweets) that people
post DURING a natural disaster when they urgently NEED resources.

Generate {TWEETS_PER_CALL} unique tweets requesting specific resources such as
food, water, medical aid, shelter, power, transport, baby supplies, or rescue
contact info. Each tweet should be DIFFERENT from the others — vary the location,
the resource, the phrasing, and the urgency level.

STYLE REQUIREMENTS (this is social media, not formal writing):
- Keep tweets short (under 280 characters, often 10-25 words).
- Use real Twitter patterns: hashtags (#KeralaFloods, #HurricaneHelp), @mentions
  to authorities (@RedCross, @NDRF), ALL CAPS for urgency, "plz" "pls" "ppl",
  ampersands, missing apostrophes, fragmented sentences.
- Occasionally include minor typos or abbreviations (its/it's, u/you, 4/for).
- Include concrete details: neighborhoods, street names, numbers of people,
  ages, medical conditions. Fake but plausible.
- Vary urgency: some desperate ("PLEASE HELP"), some calm requests, some
  coordinated asks ("group of 12 need...").
- Do NOT use perfect formal English. Do NOT write like a news report.

REAL EXAMPLES FROM DISASTER TWEETS (match this tone, not perfect grammar):
{seed_block}

Return ONLY the tweets, one per line. No numbering, no bullets, no quotes,
no preamble, no commentary. Just {TWEETS_PER_CALL} lines of tweet text."""


def is_good_tweet(text, seen_set):
    """Filter out low-quality generations. Returns True if the tweet passes."""
    text = text.strip()

    # Empty or too short
    if not text:
        return False
    words = text.split()
    if len(words) < MIN_WORDS or len(words) > MAX_WORDS:
        return False

    # Obvious LLM artifacts / non-tweets
    lower = text.lower()
    bad_prefixes = (
        "here are", "here is", "sure,", "sure!", "note:", "example:",
        "tweet:", "certainly", "of course", "i'll", "i will", "as an ai",
    )
    if lower.startswith(bad_prefixes):
        return False

    # Starts with numbering the model sometimes adds despite instructions
    if re.match(r"^\s*\d+[\.\)]\s", text):
        # Strip the number, re-check
        text_stripped = re.sub(r"^\s*\d+[\.\)]\s+", "", text)
        if len(text_stripped.split()) < MIN_WORDS:
            return False

    # Deduplication: exact match, or first 8 words already seen (near-dup)
    fingerprint = " ".join(words[:8]).lower()
    if fingerprint in seen_set:
        return False

    seen_set.add(fingerprint)
    return True


def clean_tweet(text):
    """Strip any stray numbering/bullets the LLM may have added."""
    text = text.strip()
    text = re.sub(r"^\s*[-•*]\s+", "", text)
    text = re.sub(r"^\s*\d+[\.\)]\s+", "", text)
    text = text.strip('"\'')
    return text.strip()


def generate_tweets():
    # --- Load only TRAIN split to avoid dev/test leakage in seeds ---
    df = pd.read_csv(DATA_PATH)
    train_mask = (df["split"] == "train") & (df["target_label"] == "Resource Requests")
    seed_pool = df[train_mask]["clean_text"].dropna().tolist()

    if len(seed_pool) < SEEDS_PER_CALL:
        raise ValueError(
            f"Not enough Resource Requests seeds in train split "
            f"({len(seed_pool)} found, need at least {SEEDS_PER_CALL})."
        )

    print(f"Seed pool size (train-only, Resource Requests): {len(seed_pool)}")
    print(f"Target: {TARGET_COUNT} filtered synthetic tweets")
    print(f"Seeds rotated per call: {SEEDS_PER_CALL}")
    print("-" * 60)

    kept_tweets = []
    seen_fingerprints = set()
    call_count = 0
    max_calls = (TARGET_COUNT // TWEETS_PER_CALL) * 3  # Allow ~3x buffer for filtering

    while len(kept_tweets) < TARGET_COUNT and call_count < max_calls:
        call_count += 1

        # Rotate seeds every call
        seeds = random.sample(seed_pool, SEEDS_PER_CALL)
        prompt = build_prompt(seeds)

        try:
            completion = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.9,
                max_tokens=600,
            )
            response = completion.choices[0].message.content

            batch_kept = 0
            for line in response.split("\n"):
                tweet = clean_tweet(line)
                if is_good_tweet(tweet, seen_fingerprints):
                    kept_tweets.append(tweet)
                    batch_kept += 1
                    if len(kept_tweets) >= TARGET_COUNT:
                        break

            print(
                f"[Call {call_count:3d}] kept {batch_kept}/{TWEETS_PER_CALL} "
                f"| total {len(kept_tweets)}/{TARGET_COUNT}"
            )

            time.sleep(RATE_LIMIT_SLEEP)

        except Exception as e:
            print(f"[Call {call_count}] ERROR: {e}")
            time.sleep(20)

    # --- Save ---
    new_df = pd.DataFrame({
        "clean_text": kept_tweets,
        "target_label": "Resource Requests",
        "split": "train",
        "is_synthetic": True,
    })

    new_df.to_csv(OUTPUT_PATH, index=False)

    print("-" * 60)
    print(f"Done. Saved {len(new_df)} rows → {OUTPUT_PATH}")
    print(f"Total API calls made: {call_count}")
    print(f"Effective keep rate: {len(new_df) / (call_count * TWEETS_PER_CALL):.1%}")

    print("\nSample generations:")
    for s in new_df["clean_text"].sample(min(5, len(new_df)), random_state=42):
        print(f"  > {s}")


if __name__ == "__main__":
    generate_tweets()