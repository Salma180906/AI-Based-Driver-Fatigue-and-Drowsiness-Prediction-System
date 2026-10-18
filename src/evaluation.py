"""
================================================================================
AI-Based Driver Fatigue and Drowsiness Prediction System
Module: src/evaluation.py
Purpose: Evaluation metrics calculation (Accuracy, Precision, Recall, F1, AUC),
         confusion matrix generation, feature importance plotting, and comparison table.
         Member 2: Machine Learning & Application
================================================================================
"""

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless / script execution
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    RESULTS_DIR,
    METRICS_CSV_PATH,
    MODEL_COMPARISON_CSV_PATH,
    CONFUSION_MATRIX_PNG_PATH,
    ALL_CONFUSION_MATRICES_PNG_PATH,
    ROC_CURVES_PNG_PATH,
    FEATURE_IMPORTANCE_PNG_PATH
)


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model"
) -> Dict[str, Any]:
    """
    Evaluates a trained classifier on unseen test data.
    Calculates actual statistical metrics (NO hard-coding).
    """
    y_pred = model.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))

    roc_auc = 0.0
    if hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(X_test)[:, 1]
            roc_auc = float(roc_auc_score(y_test, y_prob))
        except Exception:
            roc_auc = 0.0

    cm = confusion_matrix(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    print(f"\n========================================================")
    print(f" Evaluation Results: {model_name}")
    print(f"========================================================")
    print(f" Accuracy:  {acc * 100:.2f}%")
    print(f" Precision: {prec * 100:.2f}%")
    print(f" Recall:    {rec * 100:.2f}%")
    print(f" F1-Score:  {f1 * 100:.2f}%")
    if roc_auc > 0:
        print(f" ROC-AUC:   {roc_auc:.4f}")
    print(f" Confusion Matrix:\n{cm}")

    return {
        "Model": model_name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-Score": round(f1, 4),
        "ROC-AUC": round(roc_auc, 4),
        "Confusion_Matrix": cm,
        "Classification_Report": report_dict,
        "y_pred": y_pred
    }


def plot_and_save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Random Forest",
    save_path: Path = CONFUSION_MATRIX_PNG_PATH
):
    """
    Renders and saves a clean, academic confusion matrix heatmap.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(6, 5), dpi=300)
    labels = ["Alert (0)", "Drowsy (1)"]

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
        annot_kws={"size": 14, "weight": "bold"}
    )

    plt.title(f"Confusion Matrix: {model_name}\n(Unseen Test Subjects)", fontsize=12, pad=12)
    plt.xlabel("Predicted Class", fontsize=11)
    plt.ylabel("Ground Truth Class", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Evaluation] Saved confusion matrix visualization to '{save_path}'.")


def plot_all_confusion_matrices(
    results_list: List[Dict[str, Any]],
    y_true: np.ndarray,
    save_path: Path = ALL_CONFUSION_MATRICES_PNG_PATH
):
    """
    Renders a unified side-by-side comparison figure of confusion matrices
    for all evaluated models (Random Forest, SVM, Logistic Regression).
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    n_models = len(results_list)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5), dpi=300)
    if n_models == 1:
        axes = [axes]

    labels = ["Alert (0)", "Drowsy (1)"]
    color_maps = ["Blues", "Greens", "Purples"]

    for idx, (res, ax) in enumerate(zip(results_list, axes)):
        cm = confusion_matrix(y_true, res["y_pred"])
        cmap = color_maps[idx % len(color_maps)]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap=cmap,
            xticklabels=labels,
            yticklabels=labels,
            cbar=False,
            annot_kws={"size": 13, "weight": "bold"},
            ax=ax
        )
        acc_pct = res["Accuracy"] * 100
        f1_pct = res["F1-Score"] * 100
        ax.set_title(f"{res['Model']}\nAcc: {acc_pct:.1f}% | F1: {f1_pct:.1f}%", fontsize=11, pad=10)
        ax.set_xlabel("Predicted Class", fontsize=10)
        if idx == 0:
            ax.set_ylabel("Ground Truth Class", fontsize=10)
        else:
            ax.set_ylabel("")

    plt.suptitle("Model Confusion Matrices on Held-Out Test Subjects (NTHU-DDD)", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[Evaluation] Saved all confusion matrices figure to '{save_path}'.")


def plot_and_save_roc_curves(
    models_dict: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_path: Path = ROC_CURVES_PNG_PATH
):
    """
    Computes and plots Receiver Operating Characteristic (ROC) curves
    comparing all models on the held-out test subjects.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 6), dpi=300)

    model_colors = {
        "Random Forest": "#1f77b4",
        "SVM (RBF Kernel)": "#2ca02c",
        "Logistic Regression": "#ff7f0e"
    }

    for name, model in models_dict.items():
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            df_vals = model.decision_function(X_test)
            y_prob = 1.0 / (1.0 + np.exp(-df_vals))
        else:
            continue

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        color = model_colors.get(name, None)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})", linewidth=2.2, color=color)

    # Reference random guess line
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Guess (AUC = 0.5000)")

    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    plt.title("Receiver Operating Characteristic (ROC) Comparison\nTest Subjects (Subject-Aware Partition)", fontsize=12, pad=12)
    plt.legend(loc="lower right", frameon=True, fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Evaluation] Saved ROC curves visualization to '{save_path}'.")


def plot_feature_importance(
    model: Any,
    feature_names: List[str],
    save_path: Path = FEATURE_IMPORTANCE_PNG_PATH
):
    """
    Extracts and plots feature importances for tree-based models (Random Forest).
    """
    if not hasattr(model, "feature_importances_"):
        return

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    sorted_names = [feature_names[i] for i in indices]
    sorted_vals = [importances[i] for i in indices]

    plt.figure(figsize=(9, 6), dpi=300)
    palette = sns.color_palette("Blues_r", len(sorted_names))
    sns.barplot(x=sorted_vals, y=sorted_names, hue=sorted_names, palette=palette, legend=False)

    plt.title("Random Forest Feature Importance Ranking", fontsize=13, pad=12)
    plt.xlabel("Gini Importance (Relative Contribution)", fontsize=11)
    plt.ylabel("Behavioral Feature", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Evaluation] Saved feature importance plot to '{save_path}'.")


def save_comparison_table(
    results_list: List[Dict[str, Any]],
    save_path: Path = MODEL_COMPARISON_CSV_PATH
) -> pd.DataFrame:
    """
    Formats model comparison table and saves to CSV.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for r in results_list:
        summary_rows.append({
            "Model": r["Model"],
            "Accuracy": f"{r['Accuracy'] * 100:.2f}%",
            "Precision": f"{r['Precision'] * 100:.2f}%",
            "Recall": f"{r['Recall'] * 100:.2f}%",
            "F1-Score": f"{r['F1-Score'] * 100:.2f}%",
            "ROC-AUC": f"{r['ROC-AUC']:.4f}"
        })

    df_comp = pd.DataFrame(summary_rows)
    df_comp.to_csv(save_path, index=False)
    print(f"\n[Evaluation] Model Comparison Table saved to '{save_path}':")
    print(df_comp.to_string(index=False))
    return df_comp


if __name__ == "__main__":
    print("[Self-Test] Testing Evaluation Module...")
    # Synthetic test data
    y_t = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    y_p = np.array([0, 0, 1, 0, 0, 1, 0, 1])

    class MockModel:
        def predict(self, X):
            return y_p
        def predict_proba(self, X):
            return np.array([[0.8, 0.2] if p == 0 else [0.2, 0.8] for p in y_p])

    res = evaluate_model(MockModel(), np.zeros((8, 2)), y_t, model_name="Mock Classifier")
    plot_and_save_confusion_matrix(y_t, y_p, model_name="Mock Classifier")
    assert res["Accuracy"] > 0.80
    print("[Self-Test] Evaluation module verified successfully.")
