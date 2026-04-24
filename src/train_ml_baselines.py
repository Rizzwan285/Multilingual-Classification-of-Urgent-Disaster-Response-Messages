"""
ML Baselines Training & Evaluation — with Synthetic Data Ablation
------------------------------------------------------------------
Trains Logistic Regression, Linear SVM, and Naive Bayes twice:
  - Run A: on REAL training data only (humaid_processed.csv, split=='train')
  - Run B: on AUGMENTED training data (humaid_train_augmented.csv)

Both runs evaluate on the SAME dev and test sets (from humaid_processed.csv),
so numbers are directly comparable. Generates:

  Per model:
    - Confusion matrix plot (dev + test)
    - Per-class F1/precision/recall bar chart (test)
    - Saved model + vectorizer + scaler (.joblib)
    - Full classification report (.json)
    - Top-20 misclassified examples (.csv)

  Cross-model, per run:
    - Metrics summary CSV

  Ablation (the headline comparison):
    - Macro-F1 comparison plot: real vs. augmented across all models
    - Resource Requests F1 comparison (the key minority class)
    - Per-class F1 heatmap-style comparison
    - Ablation summary CSV
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score,
    f1_score,
)

# ── Paths ──────────────────────────────────────────────────────────────────
try:
    BASE_DIR = Path(__file__).resolve().parent.parent
except NameError:
    BASE_DIR = Path.cwd().parent

REAL_CSV = BASE_DIR / "datasets" / "processed" / "humaid_processed.csv"
AUGMENTED_CSV = BASE_DIR / "datasets" / "processed" / "humaid_train_augmented.csv"

MODELS_ROOT = BASE_DIR / "offline_models" / "ml_baselines"
PLOTS_ROOT = BASE_DIR / "results" / "plots" / "baselines"
ERRORS_ROOT = BASE_DIR / "results" / "errors" / "baselines"

for p in [MODELS_ROOT, PLOTS_ROOT, ERRORS_ROOT]:
    p.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────
CLASS_ORDER = [
    "Critical Rescue",
    "Resource Requests",
    "Situational Awareness",
    "Volunteering and Donations",
    "Irrelevant",
]

CLASS_COLORS = {
    "Critical Rescue": "#d32f2f",
    "Resource Requests": "#ff9800",
    "Situational Awareness": "#1976d2",
    "Volunteering and Donations": "#4caf50",
    "Irrelevant": "#9e9e9e",
}

MODEL_COLORS = {
    "Logistic_Regression": "#1f77b4",
    "Linear_SVM": "#ff7f0e",
    "Naive_Bayes": "#2ca02c",
}

RANDOM_SEED = 42


# ── Core training & evaluation ────────────────────────────────────────────

def build_models():
    """Factory returning a fresh set of models for each run."""
    return {
        "Logistic_Regression": LogisticRegression(
            class_weight="balanced", max_iter=1000,
            random_state=RANDOM_SEED, n_jobs=-1,
        ),
        "Linear_SVM": LinearSVC(
            class_weight="balanced", C=1.0, max_iter=5000,
            random_state=RANDOM_SEED,
        ),
        "Naive_Bayes": MultinomialNB(alpha=0.1),
    }


def vectorize(train_texts, other_text_sets):
    """Fit TF-IDF + scaler on train_texts, transform all. Returns vectorizer,
    scaler, and list of transformed matrices in the order given."""
    vectorizer = TfidfVectorizer(
        max_features=30000, ngram_range=(1, 2),
        min_df=3, max_df=0.95, sublinear_tf=True,
    )
    X_train_raw = vectorizer.fit_transform(train_texts)
    scaler = MaxAbsScaler()
    X_train = scaler.fit_transform(X_train_raw)

    transformed = []
    for texts in other_text_sets:
        X = scaler.transform(vectorizer.transform(texts))
        transformed.append(X)

    return vectorizer, scaler, X_train, transformed


def plot_confusion_matrix(y_true, y_pred, title, out_path):
    """Normalized + raw confusion matrix in one figure."""
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_ORDER)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_ORDER, yticklabels=CLASS_ORDER,
        ax=axes[0], cbar=True,
    )
    axes[0].set_title(f"{title} — Confusion Matrix (counts)", fontsize=12)
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].tick_params(axis="y", rotation=0)

    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1,
        xticklabels=CLASS_ORDER, yticklabels=CLASS_ORDER,
        ax=axes[1], cbar=True,
    )
    axes[1].set_title(f"{title} — Confusion Matrix (normalized by row)", fontsize=12)
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].tick_params(axis="y", rotation=0)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_per_class_metrics(y_true, y_pred, title, out_path):
    """Grouped bar chart: precision, recall, F1 per class."""
    report = classification_report(
        y_true, y_pred, labels=CLASS_ORDER, output_dict=True, zero_division=0
    )

    metrics = ["precision", "recall", "f1-score"]
    data = {m: [report[c][m] for c in CLASS_ORDER] for m in metrics}

    x = np.arange(len(CLASS_ORDER))
    width = 0.27

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, m in enumerate(metrics):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, data[m], width, label=m)
        for bar, val in zip(bars, data[m]):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                    f"{val:.2f}", ha="center", fontsize=8)

    ax.set_ylabel("Score")
    ax.set_title(f"{title} — Per-Class Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_ORDER, rotation=20, ha="right")
    ax.set_ylim(0, 1.1)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_errors(test_df, predictions, model_name, out_dir):
    """Save top-20 misclassified examples per true class."""
    errors_df = test_df.copy()
    errors_df["predicted_label"] = predictions
    errors_df = errors_df[errors_df["target_label"] != errors_df["predicted_label"]]

    # Take up to 20 per true class
    sampled = (
        errors_df.groupby("target_label", group_keys=False)
        .apply(lambda g: g.head(20))
        .reset_index(drop=True)
    )
    cols = ["clean_text", "target_label", "predicted_label"]
    if "event" in sampled.columns:
        cols.append("event")
    sampled[cols].to_csv(out_dir / f"{model_name}_errors.csv", index=False)


def evaluate_on_split(model, X, y_true, split_name):
    """Returns dict of metrics for a given split."""
    preds = model.predict(X)
    acc = accuracy_score(y_true, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, preds, average="macro", zero_division=0
    )
    per_class_f1 = f1_score(y_true, preds, labels=CLASS_ORDER, average=None, zero_division=0)

    print(f"  {split_name}: acc={acc:.4f}  macro-F1={f1:.4f}  "
          f"macro-P={precision:.4f}  macro-R={recall:.4f}")
    print(f"    Per-class F1: " + "  ".join(
        f"{c[:4]}={v:.3f}" for c, v in zip(CLASS_ORDER, per_class_f1)
    ))

    return {
        "split": split_name,
        "accuracy": acc,
        "macro_f1": f1,
        "macro_precision": precision,
        "macro_recall": recall,
        "per_class_f1": dict(zip(CLASS_ORDER, per_class_f1.tolist())),
        "predictions": preds,
    }


def run_training_pass(
    run_name,
    X_train_text, y_train,
    dev_df, test_df,
):
    """One full training + evaluation pass. Returns summary dict across models."""
    print("\n" + "=" * 70)
    print(f"RUN: {run_name}")
    print(f"Training size: {len(X_train_text):,}")
    print("=" * 70)

    # Set up output dirs for this run
    models_dir = MODELS_ROOT / run_name
    plots_dir = PLOTS_ROOT / run_name
    errors_dir = ERRORS_ROOT / run_name
    for p in [models_dir, plots_dir, errors_dir]:
        p.mkdir(parents=True, exist_ok=True)

    # Vectorize
    print("\nFitting TF-IDF + scaler on training set...")
    vectorizer, scaler, X_train, transformed = vectorize(
        X_train_text,
        [dev_df["clean_text"].astype(str).values,
         test_df["clean_text"].astype(str).values],
    )
    X_dev, X_test = transformed

    joblib.dump(vectorizer, models_dir / "tfidf_vectorizer.joblib")
    joblib.dump(scaler, models_dir / "maxabs_scaler.joblib")

    y_dev = dev_df["target_label"].values
    y_test = test_df["target_label"].values

    # Sample weights for Naive Bayes
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    # Train each model
    models = build_models()
    run_summary = {"run": run_name, "models": {}}
    metrics_rows = []

    for model_name, model in models.items():
        print(f"\n--- {model_name} ---")
        model_plot_dir = plots_dir / model_name.lower()
        model_plot_dir.mkdir(parents=True, exist_ok=True)

        # Fit
        if model_name == "Naive_Bayes":
            model.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            model.fit(X_train, y_train)

        # Evaluate on dev and test
        dev_metrics = evaluate_on_split(model, X_dev, y_dev, "dev")
        test_metrics = evaluate_on_split(model, X_test, y_test, "test")

        # Save model
        joblib.dump(model, models_dir / f"{model_name}.joblib")

        # Save full JSON report (test)
        report = classification_report(
            y_test, test_metrics["predictions"],
            labels=CLASS_ORDER, output_dict=True, zero_division=0,
        )
        with open(models_dir / f"{model_name}_results.json", "w") as f:
            json.dump({
                "run": run_name,
                "model": model_name,
                "dev": {k: v for k, v in dev_metrics.items() if k != "predictions"},
                "test": {k: v for k, v in test_metrics.items() if k != "predictions"},
                "test_classification_report": report,
            }, f, indent=2, default=str)

        # Plots
        plot_confusion_matrix(
            y_test, test_metrics["predictions"],
            f"{model_name} ({run_name}) — test",
            model_plot_dir / "confusion_matrix_test.png",
        )
        plot_confusion_matrix(
            y_dev, dev_metrics["predictions"],
            f"{model_name} ({run_name}) — dev",
            model_plot_dir / "confusion_matrix_dev.png",
        )
        plot_per_class_metrics(
            y_test, test_metrics["predictions"],
            f"{model_name} ({run_name}) — test",
            model_plot_dir / "per_class_metrics_test.png",
        )

        # Errors
        save_errors(test_df, test_metrics["predictions"], model_name, errors_dir)

        # Collect for metrics CSV
        metrics_rows.append({
            "Model": model_name,
            "Run": run_name,
            "Dev_Accuracy": dev_metrics["accuracy"],
            "Dev_Macro_F1": dev_metrics["macro_f1"],
            "Test_Accuracy": test_metrics["accuracy"],
            "Test_Macro_F1": test_metrics["macro_f1"],
            "Test_Macro_Precision": test_metrics["macro_precision"],
            "Test_Macro_Recall": test_metrics["macro_recall"],
            **{f"Test_F1_{c}": test_metrics["per_class_f1"][c] for c in CLASS_ORDER},
        })

        run_summary["models"][model_name] = {
            "dev": {k: v for k, v in dev_metrics.items() if k != "predictions"},
            "test": {k: v for k, v in test_metrics.items() if k != "predictions"},
        }

    # Save per-run metrics CSV
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(models_dir / "ml_baseline_metrics.csv", index=False)
    print(f"\nSaved run metrics → {models_dir / 'ml_baseline_metrics.csv'}")

    return run_summary, metrics_df


# ── Ablation plots ────────────────────────────────────────────────────────

def plot_ablation_macro_f1(metrics_combined, out_path):
    """Grouped bar: real vs. augmented macro-F1, per model."""
    pivot = metrics_combined.pivot(index="Model", columns="Run", values="Test_Macro_F1")
    pivot = pivot.reindex(["Logistic_Regression", "Linear_SVM", "Naive_Bayes"])

    fig, ax = plt.subplots(figsize=(10, 6))
    pivot.plot(kind="bar", ax=ax, color=["#1976d2", "#d32f2f"], edgecolor="white")

    ax.set_title("Macro-F1: Real-Only vs Augmented Training (Test Set)", fontsize=13)
    ax.set_ylabel("Macro F1")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Training Set")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_ablation_resource_requests(metrics_combined, out_path):
    """The key plot: does synthetic data actually help on Resource Requests?"""
    pivot = metrics_combined.pivot(
        index="Model", columns="Run", values="Test_F1_Resource Requests"
    )
    pivot = pivot.reindex(["Logistic_Regression", "Linear_SVM", "Naive_Bayes"])

    fig, ax = plt.subplots(figsize=(10, 6))
    pivot.plot(kind="bar", ax=ax, color=["#1976d2", "#ff9800"], edgecolor="white")

    ax.set_title(
        "Resource Requests F1: Real-Only vs Augmented Training\n"
        "(Key ablation — does synthetic data help the minority class?)",
        fontsize=12,
    )
    ax.set_ylabel("F1 Score — Resource Requests class")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Training Set")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_ablation_per_class_heatmap(metrics_combined, out_path):
    """Heatmap showing F1 delta (augmented - real) per model per class."""
    f1_cols = [f"Test_F1_{c}" for c in CLASS_ORDER]

    real = metrics_combined[metrics_combined["Run"] == "real_only"].set_index("Model")[f1_cols]
    aug = metrics_combined[metrics_combined["Run"] == "augmented"].set_index("Model")[f1_cols]
    delta = (aug - real).reindex(["Logistic_Regression", "Linear_SVM", "Naive_Bayes"])
    delta.columns = CLASS_ORDER

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(
        delta, annot=True, fmt=".3f", cmap="RdBu_r", center=0,
        cbar_kws={"label": "F1 delta (augmented − real)"},
        ax=ax,
    )
    ax.set_title(
        "Per-Class F1 Change from Adding Synthetic Data\n"
        "(Positive = synthetic helped; Negative = synthetic hurt)",
        fontsize=12,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("ML BASELINES — WITH SYNTHETIC DATA ABLATION")
    print("=" * 70)

    # Load real data (for dev/test — shared across both runs)
    if not REAL_CSV.exists():
        raise FileNotFoundError(f"{REAL_CSV} not found. Run data_preprocessing.py first.")
    real_df = pd.read_csv(REAL_CSV)
    real_df["clean_text"] = real_df["clean_text"].astype(str).fillna("")

    real_train_df = real_df[real_df["split"] == "train"]
    dev_df = real_df[real_df["split"] == "dev"]
    test_df = real_df[real_df["split"] == "test"]

    print(f"\nLoaded {REAL_CSV.name}")
    print(f"  Real train: {len(real_train_df):,}")
    print(f"  Dev:        {len(dev_df):,}")
    print(f"  Test:       {len(test_df):,}")

    # Load augmented data
    if not AUGMENTED_CSV.exists():
        raise FileNotFoundError(
            f"{AUGMENTED_CSV} not found. "
            f"Run generate_synthetic_data.py then merge_synthetic_data.py first."
        )
    augmented_df = pd.read_csv(AUGMENTED_CSV)
    augmented_df["clean_text"] = augmented_df["clean_text"].astype(str).fillna("")
    print(f"\nLoaded {AUGMENTED_CSV.name}")
    print(f"  Augmented train: {len(augmented_df):,} "
          f"(synthetic added: {len(augmented_df) - len(real_train_df):,})")

    # --- Run A: real only ---
    summary_real, metrics_real = run_training_pass(
        run_name="real_only",
        X_train_text=real_train_df["clean_text"].values,
        y_train=real_train_df["target_label"].values,
        dev_df=dev_df,
        test_df=test_df,
    )

    # --- Run B: augmented ---
    summary_aug, metrics_aug = run_training_pass(
        run_name="augmented",
        X_train_text=augmented_df["clean_text"].values,
        y_train=augmented_df["target_label"].values,
        dev_df=dev_df,
        test_df=test_df,
    )

    # --- Ablation comparison ---
    print("\n" + "=" * 70)
    print("ABLATION COMPARISON")
    print("=" * 70)

    metrics_combined = pd.concat([metrics_real, metrics_aug], ignore_index=True)
    ablation_dir = PLOTS_ROOT / "ablation"
    ablation_dir.mkdir(parents=True, exist_ok=True)

    metrics_combined.to_csv(ablation_dir / "ablation_summary.csv", index=False)

    plot_ablation_macro_f1(
        metrics_combined, ablation_dir / "macro_f1_real_vs_augmented.png"
    )
    plot_ablation_resource_requests(
        metrics_combined, ablation_dir / "resource_requests_f1_comparison.png"
    )
    plot_ablation_per_class_heatmap(
        metrics_combined, ablation_dir / "per_class_f1_delta_heatmap.png"
    )

    # Print quick ablation table to stdout
    print("\nQuick ablation table (Test macro-F1):")
    summary = metrics_combined.pivot(
        index="Model", columns="Run", values="Test_Macro_F1"
    )
    summary["Delta"] = summary["augmented"] - summary["real_only"]
    print(summary.round(4).to_string())

    print("\nResource Requests F1 (the minority class we augmented):")
    rr_summary = metrics_combined.pivot(
        index="Model", columns="Run", values="Test_F1_Resource Requests"
    )
    rr_summary["Delta"] = rr_summary["augmented"] - rr_summary["real_only"]
    print(rr_summary.round(4).to_string())

    print(f"\nAll plots saved to: {PLOTS_ROOT}")
    print(f"All models saved to: {MODELS_ROOT}")
    print(f"All errors saved to: {ERRORS_ROOT}")
    print("\nDone.")


if __name__ == "__main__":
    main()