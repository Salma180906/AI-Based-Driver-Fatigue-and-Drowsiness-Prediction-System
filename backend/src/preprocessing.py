"""
================================================================================
AI-Based Driver Fatigue and Drowsiness Prediction System
Module: src/preprocessing.py
Purpose: Data cleaning, missing value handling, subject-aware train/test splitting
         (avoiding identity data leakage), and standard scaling.
         Member 2: Machine Learning & Application
================================================================================
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    FEATURES_CSV_PATH,
    SCALER_PATH,
    FEATURE_NAMES_PATH,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    TEST_SPLIT_RATIO,
    RANDOM_STATE
)


def load_dataset(csv_path: Path = FEATURES_CSV_PATH) -> pd.DataFrame:
    """
    Loads extracted feature CSV dataset. If missing, prompts user or falls back.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset file '{csv_path}' not found. Please extract features from NTHU videos "
            f"or run src/dataset_extractor.py to generate benchmark features."
        )

    df = pd.read_csv(csv_path)
    print(f"[Preprocessing] Loaded dataset with {len(df)} rows and {len(df.columns)} columns.")
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs data cleaning:
      1. Drops exact duplicate rows.
      2. Replaces +/- inf with NaN.
      3. Imputes missing feature values with column medians.
      4. Validates target column format.
    """
    cleaned = df.copy()

    # Drop duplicates
    initial_rows = len(cleaned)
    cleaned = cleaned.drop_duplicates()
    dropped_dups = initial_rows - len(cleaned)
    if dropped_dups > 0:
        print(f"[Preprocessing] Removed {dropped_dups} duplicate rows.")

    # Replace inf
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan)

    # Missing value handling
    missing_counts = cleaned[FEATURE_COLUMNS].isna().sum()
    total_missing = missing_counts.sum()
    if total_missing > 0:
        print(f"[Preprocessing] Handling {total_missing} missing values across features...")
        for col in FEATURE_COLUMNS:
            if cleaned[col].isna().any():
                col_median = cleaned[col].median()
                cleaned[col] = cleaned[col].fillna(col_median)

    # Validate target label
    cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].astype(int)
    unique_labels = sorted(cleaned[TARGET_COLUMN].unique())
    if unique_labels != [0, 1]:
        print(f"[Preprocessing] Note: Detected target labels {unique_labels}. Ensuring binary classification.")

    return cleaned


def subject_aware_split(
    df: pd.DataFrame,
    feature_cols: List[str] = FEATURE_COLUMNS,
    target_col: str = TARGET_COLUMN,
    subject_col: str = "subject_id",
    test_size: float = TEST_SPLIT_RATIO,
    random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str], List[str]]:
    """
    Splits dataset into train and test sets grouped by Subject ID.
    CRITICAL ACADEMIC REQUIREMENT: Prevents data leakage so frames from the
    same driver do not appear in both train and test partitions.

    Returns:
        X_train, X_test, y_train, y_test, train_subjects, test_subjects
    """
    if subject_col not in df.columns:
        print(f"[Preprocessing] Warning: '{subject_col}' not found. Falling back to index-based split.")
        subjects = np.arange(len(df))
    else:
        subjects = df[subject_col]

    unique_subjects = np.unique(subjects)
    print(f"[Preprocessing] Performing subject-aware partition across {len(unique_subjects)} unique subjects...")

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, groups=subjects))

    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    train_subs = sorted(list(train_df[subject_col].unique())) if subject_col in df.columns else []
    test_subs = sorted(list(test_df[subject_col].unique())) if subject_col in df.columns else []

    print(f"[Preprocessing] Training subjects ({len(train_subs)}): {train_subs}")
    print(f"[Preprocessing] Testing subjects  ({len(test_subs)}): {test_subs}")
    print(f"[Preprocessing] Train split: {len(train_df)} rows | Test split: {len(test_df)} rows")

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    return X_train, X_test, y_train, y_test, train_subs, test_subs


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    scaler_path: Path = SCALER_PATH,
    feature_names_path: Path = FEATURE_NAMES_PATH
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Fits StandardScaler on X_train only and transforms both X_train and X_test.
    Saves the fitted scaler for production webcam inference.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Persist scaler and feature list
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)
    joblib.dump(list(X_train.columns), feature_names_path)
    print(f"[Preprocessing] Saved fitted scaler to '{scaler_path}'.")

    return X_train_scaled, X_test_scaled, scaler


if __name__ == "__main__":
    print("[Self-Test] Testing Preprocessing Pipeline...")
    from src.dataset_extractor import generate_benchmark_feature_dataset

    # Ensure sample dataset exists
    if not FEATURES_CSV_PATH.exists():
        generate_benchmark_feature_dataset(num_subjects=6, windows_per_subject=30)

    df_raw = load_dataset()
    df_clean = clean_dataset(df_raw)
    X_train, X_test, y_train, y_test, tr_s, te_s = subject_aware_split(df_clean)

    # Check zero overlap between train and test subjects
    overlap = set(tr_s).intersection(set(te_s))
    assert len(overlap) == 0, f"Subject leakage detected: {overlap}"

    X_tr_sc, X_te_sc, sc = scale_features(X_train, X_test)
    print(f"Scaled train shape: {X_tr_sc.shape}, test shape: {X_te_sc.shape}")
    print("[Self-Test] Preprocessing pipeline verified successfully.")
