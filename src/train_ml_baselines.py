import os
import json
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(ROOT_DIR, "datasets", "processed", "humaid_processed.csv")
OUTPUT_DIR = os.path.join(ROOT_DIR, "offline_models", "ml_baselines")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def train_baselines():
    df = pd.read_csv(DATA_PATH)
    df['clean_text'] = df['clean_text'].astype(str).fillna('')
    
    train_df = df[df['split'] == 'train']
    test_df = df[df['split'] == 'test']
    
    X_train_text = train_df['clean_text'].values
    y_train = train_df['target_label'].values
    
    X_test_text = test_df['clean_text'].values
    y_test = test_df['target_label'].values

    vectorizer = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), min_df=3, max_df=0.95, sublinear_tf=True)
    X_train_raw = vectorizer.fit_transform(X_train_text)
    X_test_raw = vectorizer.transform(X_test_text)
    
    scaler = MaxAbsScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    
    vectorizer_path = os.path.join(OUTPUT_DIR, "tfidf_vectorizer.joblib")
    joblib.dump(vectorizer, vectorizer_path)
    
    scaler_path = os.path.join(OUTPUT_DIR, "maxabs_scaler.joblib")
    joblib.dump(scaler, scaler_path)

    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    models = {
        "Logistic_Regression": LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42, n_jobs=-1),
        "Linear_SVM": LinearSVC(class_weight="balanced", C=1.0, max_iter=5000),
        "Naive_Bayes": MultinomialNB(alpha=0.1)
    }

    results = []
    
    for model_name, model in models.items():
        if model_name == "Naive_Bayes":
            model.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            model.fit(X_train, y_train)
            
        predictions = model.predict(X_test)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, predictions, average='macro', zero_division=0
        )
        acc = accuracy_score(y_test, predictions)
        
        report_dict = classification_report(y_test, predictions, output_dict=True, zero_division=0)
        report_path = os.path.join(OUTPUT_DIR, f"{model_name}_results.json")
        with open(report_path, "w") as f:
            json.dump(report_dict, f, indent=2)
            
        results.append({
            "Model": model_name,
            "Accuracy": acc,
            "Macro_F1": f1,
            "Macro_Precision": precision,
            "Macro_Recall": recall
        })
        
        model_save_path = os.path.join(OUTPUT_DIR, f"{model_name}.joblib")
        joblib.dump(model, model_save_path)

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(OUTPUT_DIR, "ml_baseline_metrics.csv"), index=False)

if __name__ == "__main__":
    train_baselines()