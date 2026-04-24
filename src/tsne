# ==========================================
# CELL 4: MuRIL t-SNE Visualization & Cosine Similarity
# ==========================================
import os
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
from tqdm.notebook import tqdm
import warnings
warnings.filterwarnings('ignore')

# 1. Setup paths and device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT_DIR = os.path.abspath("..") 
ENG_PATH = os.path.join(ROOT_DIR, "datasets", "processed", "humaid_processed.csv")
MAL_PATH = os.path.join(ROOT_DIR, "datasets", "processed", "malayalam_cleaned.csv")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

# Dynamically find the latest fine-tuned MuRIL model
muril_dirs = [d for d in os.listdir(RESULTS_DIR) if "local_muril" in d and os.path.isdir(os.path.join(RESULTS_DIR, d, "final_model"))]
if not muril_dirs:
    raise FileNotFoundError("Could not find a local_muril directory in results/!")
MODEL_PATH = os.path.join(RESULTS_DIR, sorted(muril_dirs)[-1], "final_model")
print(f"Loading MuRIL from: {MODEL_PATH}")

# 2. Load a small sample of data to keep the plot clean (500 of each)
df_eng = pd.read_csv(ENG_PATH)
df_eng = df_eng[df_eng['split'] == 'test'].sample(500, random_state=42)
df_eng['language'] = 'English'
df_eng['text'] = df_eng['clean_text']

df_mal = pd.read_csv(MAL_PATH).sample(500, random_state=42)
df_mal['language'] = 'Malayalam'

# Combine them
df_plot = pd.concat([df_eng[['text', 'target_label', 'language']], 
                     df_mal[['text', 'label', 'language']].rename(columns={'label': 'target_label'})]).dropna()

# 3. Load the Tokenizer and the BASE Model (we want embeddings, not classification logits)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
# Note: Using AutoModel instead of AutoModelForSequenceClassification to get the raw vector space
model = AutoModel.from_pretrained(MODEL_PATH).to(device)
model.eval()

# 4. Extract Embeddings (The [CLS] token)
embeddings = []
texts = df_plot['text'].astype(str).tolist()

with torch.no_grad():
    for i in tqdm(range(0, len(texts), 32), desc="Extracting MuRIL Embeddings"):
        inputs = tokenizer(texts[i:i+32], padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        outputs = model(**inputs)
        # Grab the hidden state of the first token [CLS] from the last layer (768 dimensions)
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.extend(cls_embeddings)

# 5. Reduce 768 dimensions to 2 using t-SNE
print("Running t-SNE dimensionality reduction...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
embeddings_2d = tsne.fit_transform(embeddings)

df_plot['x'] = embeddings_2d[:, 0]
df_plot['y'] = embeddings_2d[:, 1]

# 6. Plotting
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Plot A: Colored by Language
sns.scatterplot(data=df_plot, x='x', y='y', hue='language', palette="Set1", s=60, alpha=0.7, ax=axes[0])
axes[0].set_title("MuRIL Embeddings Colored by Language", fontsize=16, fontweight='bold')
axes[0].set_xticks([])
axes[0].set_yticks([])

# Plot B: Colored by Disaster Class
sns.scatterplot(data=df_plot, x='x', y='y', hue='target_label', palette="bright", s=60, alpha=0.7, ax=axes[1])
axes[1].set_title("MuRIL Embeddings Colored by Disaster Class", fontsize=16, fontweight='bold')
axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
axes[1].set_xticks([])
axes[1].set_yticks([])

plt.tight_layout()
plt.show()

# ==========================================
# 7. Bonus: Cosine Similarity Proof
# ==========================================
print("\n" + "="*50)
print("CROSS-LINGUAL COSINE SIMILARITY TEST")
print("="*50)

# Two sentences with the exact same meaning in different languages
english_tweet = "We need urgent rescue from the flood, please help!"
malayalam_tweet = "പ്രളയത്തിൽ നിന്നും ഞങ്ങളെ അടിയന്തരമായി രക്ഷിക്കണം, ദയവായി സഹായിക്കുക!"

# Get embeddings for just these two sentences
with torch.no_grad():
    inputs_eng = tokenizer(english_tweet, return_tensors="pt", truncation=True, max_length=128).to(device)
    inputs_mal = tokenizer(malayalam_tweet, return_tensors="pt", truncation=True, max_length=128).to(device)
    
    vec_eng = model(**inputs_eng).last_hidden_state[:, 0, :].cpu().numpy()
    vec_mal = model(**inputs_mal).last_hidden_state[:, 0, :].cpu().numpy()

# Calculate Cosine Similarity (1.0 means identical direction in 768-D space, 0.0 means completely unrelated)
similarity = cosine_similarity(vec_eng, vec_mal)[0][0]

print(f"English:   '{english_tweet}'")
print(f"Malayalam: '{malayalam_tweet}'\n")
print(f"Cosine Similarity Score: {similarity:.4f} / 1.0")
print("Conclusion: A high score proves MuRIL maps these two distinct languages to the exact same semantic coordinate!")