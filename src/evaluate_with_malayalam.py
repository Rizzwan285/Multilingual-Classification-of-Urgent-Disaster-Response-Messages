import os
import torch
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. Setup Paths and Unique Timestamps
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

# Pointing directly to your newly uploaded Malayalam dataset
DATA_PATH = os.path.join(ROOT_DIR, "datasets", "processed", "malayalam_test.csv")
ML_MODELS_DIR = os.path.join(ROOT_DIR, "offline_models", "ml_baselines")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

# Generate a unique timestamp to keep these plots grouped together
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
PLOT_DIR = os.path.join(RESULTS_DIR, f"malayalam_eval_run_{TIMESTAMP}")
os.makedirs(PLOT_DIR, exist_ok=True)

LABEL_NAMES = ['Situational Awareness', 'Critical Rescue', 'Volunteering and Donations', 'Irrelevant', 'Resource Requests']
label2id = {label: i for i, label in enumerate(LABEL_NAMES)}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Evaluating using device: {device}")
print(f"All plots and metrics will be saved to: {PLOT_DIR}\n")

# ==========================================
# 2. Evaluation Helper Functions
# ==========================================
def evaluate_ml_models(df):
    predictions_dict = {}
    print("--- Evaluating Traditional ML Baselines ---")
    
    vectorizer_path = os.path.join(ML_MODELS_DIR, "tfidf_vectorizer.joblib")
    if not os.path.exists(vectorizer_path):
        print(f"TF-IDF Vectorizer not found. Skipping ML baselines.")
        return predictions_dict
        
    vectorizer = joblib.load(vectorizer_path)
    
    # CHANGED: Using 'text' column from the new dataset
    X_test = vectorizer.transform(df['text'].astype(str).fillna(''))
    
    # CHANGED: Using 'label' column
    y_test = df['label'].values

    ml_models = ["Logistic_Regression", "Linear_SVM", "Random_Forest", "Naive_Bayes"]
    
    for model_name in ml_models:
        model_path = os.path.join(ML_MODELS_DIR, f"{model_name}.joblib")
        if os.path.exists(model_path):
            print(f"Evaluating {model_name}...")
            model = joblib.load(model_path)
            y_pred = model.predict(X_test)
            
            # Map string predictions back to integer IDs for uniform metric calculation
            # Using .get() with a default fallback (-1) just in case of unexpected labels
            y_pred_ids = [label2id.get(label, -1) for label in y_pred]
            y_test_ids = [label2id.get(label, -1) for label in y_test]
            
            predictions_dict[model_name] = {"y_true": y_test_ids, "y_pred": y_pred_ids, "type": "ML Baseline"}
            
    return predictions_dict

def evaluate_transformer(model_name, model_path, df, model_type="Fine-Tuned"):
    print(f"--- Evaluating {model_type} Transformer: {model_name} ---")
    if not os.path.exists(model_path):
        print(f"Model path {model_path} not found. Skipping.")
        return None

    # Safe tokenizer loading for IndicBERT
    if "indic_bert" in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=5)
    model.to(device)
    model.eval()

    # CHANGED: Using 'text' instead of 'clean_text'
    texts = df['text'].astype(str).tolist()
    
    # CHANGED: Using the newly mapped 'label_id' column
    y_true = df['label_id'].values
    y_pred = []

    batch_size = 32
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc=f"Predicting"):
            batch_texts = texts[i:i+batch_size]
            inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            outputs = model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1).cpu().numpy()
            y_pred.extend(predictions)

    del model
    del tokenizer
    torch.cuda.empty_cache()
    
    return {"y_true": list(y_true), "y_pred": list(y_pred), "type": model_type}

# ==========================================
# 3. Main Execution Block
# ==========================================
def main():
    print(f"Loading Malayalam evaluation data from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    
    # CHANGED: Removed the 'split' logic because the entire file is used for evaluation
    eval_df = df.copy()
        
    # CHANGED: Map string labels directly to a new 'label_id' column
    eval_df['label_id'] = eval_df['label'].map(label2id)
    
    # Drop any rows where the label didn't match our exact 5 categories to prevent metric calculation crashes
    eval_df = eval_df.dropna(subset=['label_id'])
    eval_df['label_id'] = eval_df['label_id'].astype(int)
    
    model_predictions = {}

    # 1. Gather ML Baseline Predictions
    model_predictions.update(evaluate_ml_models(eval_df))

    # 2. Gather Fine-Tuned Transformer Predictions
    transformers = ["local_muril", "local_indic_bert", "local_mbert", "local_xlm_roberta"]
    
    for t_name in transformers:
        matching_dirs = [d for d in os.listdir(RESULTS_DIR) if t_name in d and os.path.isdir(os.path.join(RESULTS_DIR, d, "final_model"))]
        if matching_dirs:
            latest_dir = sorted(matching_dirs)[-1]
            t_path = os.path.join(RESULTS_DIR, latest_dir, "final_model")
            display_name = f"FT_{t_name.replace('local_', '')}"
            res = evaluate_transformer(display_name, t_path, eval_df, model_type="Fine-Tuned Transformer")
            if res: model_predictions[display_name] = res

    # 3. Gather Untrained Base XLM-R Predictions
    base_xlm_path = os.path.join(ROOT_DIR, "offline_models", "local_xlm_roberta")
    res_base = evaluate_transformer("Base_XLM_RoBERTa", base_xlm_path, eval_df, model_type="Untrained Base")
    if res_base: model_predictions["Base_XLM_RoBERTa"] = res_base

    # ==========================================
    # 4. Calculate Metrics and Plot Visualizations
    # ==========================================
    if not model_predictions:
        print("No models were evaluated. Exiting.")
        return

    print("\nGenerating Plots and Metrics...")
    
    overall_metrics = []
    class_f1_scores = []

    for model_name, data in model_predictions.items():
        y_true = data['y_true']
        y_pred = data['y_pred']
        m_type = data['type']
        
        # 4a. Overall Metrics
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        overall_metrics.append({"Model": model_name, "Type": m_type, "Accuracy": acc, "Precision": p, "Recall": r, "Macro F1": f1})
        
        # 4b. Per-Class F1 Scores
        _, _, class_f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=range(5), zero_division=0)
        for i, class_name in enumerate(LABEL_NAMES):
            class_f1_scores.append({"Model": model_name, "Class": class_name, "F1 Score": class_f1[i]})

        # 4c. Confusion Matrix Plotting
        cm = confusion_matrix(y_true, y_pred, labels=range(5))
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES)
        plt.title(f"Confusion Matrix: {model_name} (Malayalam Data)", fontsize=14)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f"CM_{model_name}.png"), dpi=300)
        plt.close()

    # Save Overall Metrics CSV
    results_df = pd.DataFrame(overall_metrics).sort_values("Macro F1", ascending=False)
    results_df.to_csv(os.path.join(PLOT_DIR, "overall_metrics.csv"), index=False)
    
    # 4d. Overall F1 Bar Chart
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")
    ax = sns.barplot(data=results_df, x="Macro F1", y="Model", hue="Type", dodge=False, palette="viridis")
    plt.title("Overall Macro F1 Score Comparison (Malayalam Data)", fontsize=16, fontweight='bold')
    plt.xlabel("Macro F1 Score", fontsize=12)
    plt.ylabel("Model", fontsize=12)
    plt.legend(title="Model Type", loc='lower right')
    for p in ax.patches:
        ax.annotate(f"{p.get_width():.3f}", (p.get_width(), p.get_y() + p.get_height() / 2.), ha='left', va='center', xytext=(5, 0), textcoords='offset points')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "overall_f1_comparison.png"), dpi=300)
    plt.close()

    # 4e. Class-Wise F1 Heatmap
    class_df = pd.DataFrame(class_f1_scores).pivot(index="Model", columns="Class", values="F1 Score")
    class_df = class_df.reindex(results_df["Model"].values)
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(class_df, annot=True, cmap='RdYlGn', fmt='.3f', linewidths=.5)
    plt.title("Class-Wise F1 Score Breakdown (Malayalam Data)", fontsize=16, fontweight='bold')
    plt.xlabel("Disaster Category", fontsize=12)
    plt.ylabel("Model", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "class_wise_f1_heatmap.png"), dpi=300)
    plt.close()

    print(f"\nAll Malayalam evaluations complete! Check {PLOT_DIR} for your new plots and CSV reports.")

if __name__ == "__main__":
    main()