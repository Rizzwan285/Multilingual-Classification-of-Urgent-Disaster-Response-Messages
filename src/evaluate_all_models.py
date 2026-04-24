import os
import torch
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. Setup Paths and Unique Timestamps
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

DATA_PATH = os.path.join(ROOT_DIR, "datasets", "processed", "humaid_processed.csv")
ML_MODELS_DIR = os.path.join(ROOT_DIR, "offline_models", "ml_baselines")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
PLOT_DIR = os.path.join(RESULTS_DIR, "evaluation_plots_v2")

os.makedirs(PLOT_DIR, exist_ok=True)

# Generate a unique timestamp for this specific run
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

LABEL_NAMES = ['Situational Awareness', 'Critical Rescue', 'Volunteering and Donations', 'Irrelevant', 'Resource Requests']
label2id = {label: i for i, label in enumerate(LABEL_NAMES)}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Evaluating using device: {device}")

# ==========================================
# 2. Evaluation Helper Functions
# ==========================================
def evaluate_ml_models(df):
    results = []
    print("\n--- Evaluating Traditional ML Baselines ---")
    
    # Load the vectorizer
    vectorizer_path = os.path.join(ML_MODELS_DIR, "tfidf_vectorizer.joblib")
    if not os.path.exists(vectorizer_path):
        print(f"TF-IDF Vectorizer not found at {vectorizer_path}. Skipping ML baselines.")
        return results
        
    vectorizer = joblib.load(vectorizer_path)
    X_test = vectorizer.transform(df['clean_text'].astype(str).fillna(''))
    
    # Use 'target_label' (strings) because the ML models output string predictions
    y_test = df['target_label'].values

    ml_models = ["Logistic_Regression", "Linear_SVM", "Random_Forest", "Naive_Bayes"]
    
    for model_name in ml_models:
        model_path = os.path.join(ML_MODELS_DIR, f"{model_name}.joblib")
        if os.path.exists(model_path):
            print(f"Evaluating {model_name}...")
            model = joblib.load(model_path)
            predictions = model.predict(X_test)
            
            p, r, f1, _ = precision_recall_fscore_support(y_test, predictions, average='macro', zero_division=0)
            acc = accuracy_score(y_test, predictions)
            
            results.append({"Model": model_name, "Type": "ML Baseline", "Accuracy": acc, "Macro F1": f1})
        else:
            print(f"Warning: {model_name}.joblib not found in offline_models/ml_baselines!")
            
    return results

def evaluate_transformer(model_name, model_path, df, model_type="Fine-Tuned"):
    print(f"\n--- Evaluating {model_type} Transformer: {model_name} ---")
    if not os.path.exists(model_path):
        print(f"Model path {model_path} not found. Skipping.")
        return None

    # THE FIX: Only force the slow tokenizer for IndicBERT to prevent the SentencePiece crash for XLMR
    if "indic_bert" in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=5)
    model.to(device)
    model.eval()

    texts = df['clean_text'].astype(str).tolist()
    y_true = df['label'].values
    y_pred = []

    batch_size = 32
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc=f"Evaluating {model_name}"):
            batch_texts = texts[i:i+batch_size]
            inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            outputs = model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1).cpu().numpy()
            y_pred.extend(predictions)

    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    
    # Prevent Out Of Memory errors on the cluster
    del model
    del tokenizer
    torch.cuda.empty_cache()
    
    return {"Model": model_name, "Type": model_type, "Accuracy": acc, "Macro F1": f1}

# ==========================================
# 3. Main Execution Block
# ==========================================
def main():
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Dataset not found at {DATA_PATH}.")
        return

    print(f"Loading data from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    
    # Prioritize 'test' split if available, otherwise fallback to 'dev' (validation)
    if 'test' in df['split'].values:
        eval_df = df[df['split'] == 'test'].copy()
    else:
        eval_df = df[df['split'] == 'dev'].copy()
        
    eval_df['label'] = eval_df['target_label'].map(label2id)
    all_results = []

    # 1. Run ML Baselines
    all_results.extend(evaluate_ml_models(eval_df))

    # 2. Run Fine-Tuned Transformers
    transformers = ["local_muril", "local_indic_bert", "local_mbert", "local_xlm_roberta"]
    for t_name in transformers:
        t_path = os.path.join(RESULTS_DIR, t_name, "final_model")
        display_name = f"Fine-Tuned_{t_name.replace('local_', '')}"
        res = evaluate_transformer(display_name, t_path, eval_df, model_type="Fine-Tuned Transformer")
        if res: all_results.append(res)

    # 3. Run the Untrained BASE XLM-RoBERTa for Reference
    base_xlm_path = os.path.join(ROOT_DIR, "offline_models", "local_xlm_roberta")
    res_base = evaluate_transformer("Base_XLM_RoBERTa", base_xlm_path, eval_df, model_type="Untrained Base")
    if res_base: all_results.append(res_base)

    # ==========================================
    # 4. Save and Plot Results
    # ==========================================
    if all_results:
        results_df = pd.DataFrame(all_results)
        print("\n--- Final Results ---")
        print(results_df.sort_values("Macro F1", ascending=False).to_string(index=False))
        
        # Save raw numbers to CSV for your report tables
        csv_out = os.path.join(PLOT_DIR, f"final_model_comparisons_{TIMESTAMP}.csv")
        results_df.to_csv(csv_out, index=False)
        print(f"\nSaved numerical results to {csv_out}")

        # Plotting: Macro F1 Score Comparison
        plt.figure(figsize=(12, 7))
        sns.set_theme(style="whitegrid")
        
        # Create the barplot
        ax = sns.barplot(data=results_df.sort_values("Macro F1", ascending=False), 
                         x="Macro F1", y="Model", hue="Type", dodge=False, palette="viridis")
        
        plt.title("Final Model Comparison: Macro F1 Score on HumAID", fontsize=16, fontweight='bold')
        plt.xlabel("Macro F1 Score", fontsize=12)
        plt.ylabel("Model", fontsize=12)
        plt.legend(title="Model Category", loc='lower right')
        
        # Add the text labels to the bars
        for p in ax.patches:
            width = p.get_width()
            if not np.isnan(width):
                ax.annotate(f"{width:.3f}", 
                            (width, p.get_y() + p.get_height() / 2.), 
                            ha='left', va='center', xytext=(5, 0), textcoords='offset points')

        plt.tight_layout()
        plot_out = os.path.join(PLOT_DIR, f"f1_score_comparison_{TIMESTAMP}.png")
        plt.savefig(plot_out, dpi=300)
        
        print(f"Evaluation Complete!")
        print(f"Plot saved to: {plot_out}")

if __name__ == "__main__":
    main()