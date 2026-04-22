import os
import pandas as pd
from groq import Groq
from dotenv import load_dotenv
import time

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

ROOT_DIR = os.path.abspath(os.path.join(os.getcwd(), "."))
DATA_PATH = os.path.join(ROOT_DIR, "datasets", "processed", "humaid_processed.csv")
OUTPUT_PATH = os.path.join(ROOT_DIR, "datasets", "synthetic_resource_requests.csv")

def generate_tweets():
    df = pd.read_csv(DATA_PATH)
    resource_df = df[df['target_label'] == 'Resource Requests'].sample(3)
    examples = resource_df['clean_text'].tolist()

    prompt = f"""
    You are an expert at simulating urgent disaster response social media posts.
    Generate 10 unique, realistic tweets requesting specific resources (food, water, medical aid, shelter).
    Tone: Urgent, concise, often includes locations.
    Format: Return ONLY a list of tweets, one per line, no numbering.

    Examples:
    1. {examples[0]}
    2. {examples[1]}
    3. {examples[2]}
    """

    all_synthetic_tweets = []
    total_needed = 2500 # Adjust this based on how much you want
    
    print(f"Starting generation for {total_needed} tweets...")

    for i in range(total_needed // 10):
        try:
            completion = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                max_tokens=500
            )
            
            response = completion.choices[0].message.content
            tweets = [t.strip() for t in response.split('\n') if len(t.strip()) > 5]
            all_synthetic_tweets.extend(tweets)
            
            print(f"Generated {len(all_synthetic_tweets)} tweets so far...")
            
            # Rate limit safety: 8 requests per minute
            time.sleep(7.5) 

        except Exception as e:
            print(f"Error at step {i}: {e}")
            time.sleep(20)

    new_df = pd.DataFrame({
        'clean_text': all_synthetic_tweets,
        'target_label': 'Resource Requests',
        'split': 'train'
    })
    
    new_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Success! Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    generate_tweets()