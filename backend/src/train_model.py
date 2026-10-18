"""
================================================================================
AI-Based Driver Fatigue and Drowsiness Prediction System
Module: src/train_model.py
Purpose: Trains and compares Random Forest, Support Vector Machine (SVM),
         and Logistic Regression; evaluates models on unseen test subjects;
         persists primary model, scaler, and performance artifacts.
         Member 2: Machine Learning & Application
================================================================================
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    FEATURES_CSV_PATH,
    MODEL_PATH,
    RF_MODEL_PATH,
    SVM_MODEL_PATH,
    LR_MODEL_PATH,
    SCALER_PATH,
    FEATURE_NAMES_PATH,
    MODEL_DIR,
    RESULTS_DIR,
    RANDOM_STATE
)
from src.dataset_extractor import generate_benchmark_feature_dataset
from src.preprocessing import (
    load_dataset,
    clean_dataset,
    subject_aware_split,
    scale_features
)
from src.evaluation import (
    evaluate_model,
    plot_and_save_confusion_matrix,
    plot_all_confusion_matrices,
    plot_and_save_roc_curves,
    plot_feature_importance,
    save_comparison_table
)


def train_and_compare_models(csv_path: Path = FEATURES_CSV_PATH) -> Dict[str, Any]:
    """
    Complete ML training pipeline:
      1. Loads feature CSV (or generates benchmark dataset if not yet created).
      2. Cleans data and splits by Subject ID (preventing data leakage).
      3. Fits StandardScaler on training set only.
      4. Trains Random Forest, SVM (RBF), and Logistic Regression.
      5. Evaluates on unseen test subjects.
      6. Selects and persists the best model (Random Forest primary).
      7. Saves comparison table, confusion matrix, and feature importances.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"[Training] Feature dataset not found at '{csv_path}'. Generating benchmark dataset...")
        generate_benchmark_feature_dataset(output_path=csv_path, num_subjects=12, windows_per_subject=120)

    # 1. Preprocessing
    df_raw = load_dataset(csv_path)
    df_clean = clean_dataset(df_raw)

    X_train, X_test, y_train, y_test, train_subs, test_subs = subject_aware_split(df_clean)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    feature_names = list(X_train.columns)

    # 2. Model Definitions
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=120,
            max_depth=10,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            class_weight="balanced"
        ),
        "SVM (RBF Kernel)": SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            random_state=RANDOM_STATE,
            class_weight="balanced"
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            C=1.0,
            random_state=RANDOM_STATE,
            class_weight="balanced"
        )
    }

    results = []
    trained_instances = {}

    # 3. Train and Evaluate each candidate model
    for name, model in models.items():
        print(f"\n[Training] Fitting {name}...")
        model.fit(X_train_scaled, y_train)
        trained_instances[name] = model

        eval_res = evaluate_model(model, X_test_scaled, y_test.values, model_name=name)
        results.append(eval_res)

    # 4. Save Model Comparison Table
    save_comparison_table(results)

    # 5. Model Selection Strategy
    # Random Forest is the primary academic baseline due to superior handling
    # of non-linear interaction terms (e.g. EAR droop * blink duration).
    rf_eval = [r for r in results if r["Model"] == "Random Forest"][0]
    best_name = "Random Forest"
    best_model = trained_instances["Random Forest"]

    # If another model has strictly higher F1-score by > 0.05, consider it
    for r in results:
        if r["F1-Score"] > (rf_eval["F1-Score"] + 0.05):
            best_name = r["Model"]
            best_model = trained_instances[best_name]

    print(f"\n[Training] Selected Primary Model: '{best_name}'")
    print(f"[Training] Rationale: Non-linear temporal feature interactions, robust Gini importance, "
          f"and empirical test performance on held-out subjects.")

    # 6. Save Model Artifacts (Primary model and all individual candidate models)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(trained_instances["Random Forest"], RF_MODEL_PATH)
    joblib.dump(trained_instances["SVM (RBF Kernel)"], SVM_MODEL_PATH)
    joblib.dump(trained_instances["Logistic Regression"], LR_MODEL_PATH)
    print(f"[Training] Saved primary model to '{MODEL_PATH}'.")
    print(f"[Training] Saved individual models to '{MODEL_DIR}'.")

    # 7. Generate Visualizations for Models
    best_eval = [r for r in results if r["Model"] == best_name][0]
    plot_and_save_confusion_matrix(
        y_true=y_test.values,
        y_pred=best_eval["y_pred"],
        model_name=best_name
    )

    plot_all_confusion_matrices(
        results_list=results,
        y_true=y_test.values
    )

    plot_and_save_roc_curves(
        models_dict=trained_instances,
        X_test=X_test_scaled,
        y_test=y_test.values
    )

    if hasattr(best_model, "feature_importances_"):
        plot_feature_importance(best_model, feature_names)

    # Save detailed metrics CSV
    metrics_summary = pd.DataFrame([{
        "Primary_Model": best_name,
        "Accuracy": best_eval["Accuracy"],
        "Precision": best_eval["Precision"],
        "Recall": best_eval["Recall"],
        "F1_Score": best_eval["F1-Score"],
        "ROC_AUC": best_eval["ROC-AUC"],
        "Test_Subjects": ",".join(test_subs),
        "Train_Subjects": ",".join(train_subs)
    }])
    metrics_summary.to_csv(RESULTS_DIR / "metrics.csv", index=False)

    return {
        "best_model_name": best_name,
        "best_model": best_model,
        "results": results,
        "train_subjects": train_subs,
        "test_subjects": test_subs
    }


if __name__ == "__main__":
    train_and_compare_models()
