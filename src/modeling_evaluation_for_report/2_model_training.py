"""
=====================================================================================
Final Model Training Script for Rating Classification (5-Star vs Non-5-Star)
评分分类模型训练最终版本

This script trains and compares multiple machine learning models for predicting
whether an Airbnb listing will receive a 5-star rating.

Models included:
1. Logistic Regression (linear baseline)
2. Random Forest (tree-based baseline)
3. LightGBM (fast gradient boosting)
4. CatBoost (automated gradient boosting) ⭐ BEST MODEL
5. MLP Neural Network (deep learning, highest complexity)

Task: Binary classification (5-star vs <5-star)
Evaluation: AUC-ROC, F1-Score, Precision, Recall, Accuracy

Author: Data Science Course Project
Date: 2025
=====================================================================================
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Import gradient boosting libraries
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️  Warning: LightGBM not available, will skip LightGBM model")

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("⚠️  Warning: CatBoost not available, will skip CatBoost model")

# Path setup
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CHARTS_DIR = PROJECT_ROOT / "charts"
MODEL_OUTPUT_DIR = CHARTS_DIR / "charts_for_report" / "modeling"

# Plotting setup
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


# ==================== DATA LOADING ====================
def load_training_data() -> tuple[pd.DataFrame, Path]:
    """
    Load processed training data

    Returns:
        Tuple of (DataFrame, charts_directory_path)
    """
    train_path = DATA_DIR / "processed" / "train_data.csv"

    if not train_path.exists():
        raise FileNotFoundError(
            f"Training data not found: {train_path}\n"
            f"Please run: src/modeling_evaluation_for_report/1_feature_engineering.py"
        )

    print("=" * 80)
    print("LOADING TRAINING DATA")
    print("=" * 80)
    print(f"Path: {train_path}")
    df = pd.read_csv(train_path)
    print(f"✅ Loaded: {len(df):,} rows × {len(df.columns)} columns")

    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    return df, MODEL_OUTPUT_DIR


# ==================== DATA PREPARATION ====================
def prepare_binary_dataset(df: pd.DataFrame) -> tuple:
    """
    Prepare binary classification dataset (5-star vs <5-star)

    Args:
        df: DataFrame with review_scores_rating column

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, feature_names)
    """
    if "review_scores_rating" not in df.columns:
        raise ValueError("Missing review_scores_rating column for creating labels")

    # Filter valid ratings
    working_df = df[df["review_scores_rating"].notna()].copy()

    # Create binary target: 1 = 5-star, 0 = <5-star
    working_df["is_five_star"] = np.isclose(
        working_df["review_scores_rating"], 5.0, atol=1e-6
    ).astype(int)

    print("\n" + "=" * 80)
    print("PREPARING BINARY CLASSIFICATION DATASET")
    print("Task: Predict 5-star vs <5-star ratings")
    print("=" * 80)

    # Class distribution
    positives = working_df["is_five_star"].sum()
    total = len(working_df)
    print(f"Total samples: {total:,}")
    print(f"  5-star (positive): {positives:,} ({positives / total:.2%})")
    print(f"  <5-star (negative): {total - positives:,} ({1 - positives / total:.2%})")

    # Separate features and target
    exclude_cols = {"review_scores_rating", "is_five_star"}
    feature_cols = [col for col in working_df.columns if col not in exclude_cols]

    X = working_df[feature_cols].copy()
    y = working_df["is_five_star"].copy()

    # Train-test split (80-20, stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTrain-Test Split (80-20):")
    print(f"  Training set: {len(X_train):,} samples (5-star ratio: {y_train.mean():.2%})")
    print(f"  Test set: {len(X_test):,} samples (5-star ratio: {y_test.mean():.2%})")
    print(f"  Feature count: {len(feature_cols)}")

    return X_train, X_test, y_train, y_test, feature_cols


# ==================== MODEL 1: LOGISTIC REGRESSION ====================
def train_logistic_regression(X_train, X_test, y_train, y_test) -> tuple:
    """
    Train Logistic Regression model (linear baseline)

    Returns:
        Tuple of (model, metrics_dict, test_probabilities)
    """
    print("\n" + "-" * 80)
    print("Training Model: Logistic Regression (Linear Baseline)")
    print("-" * 80)

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=42,
            n_jobs=-1
        ))
    ])

    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    # Predictions
    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    metrics = {
        'model': 'Logistic Regression',
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'train_precision': precision_score(y_train, y_train_pred),
        'train_recall': recall_score(y_train, y_train_pred),
        'train_f1': f1_score(y_train, y_train_pred),
        'train_auc': roc_auc_score(y_train, y_train_proba),
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'test_precision': precision_score(y_test, y_test_pred),
        'test_recall': recall_score(y_test, y_test_pred),
        'test_f1': f1_score(y_test, y_test_pred),
        'test_auc': roc_auc_score(y_test, y_test_proba),
        'training_time': training_time,
    }

    print(f"Training time: {training_time:.2f}s")
    print(f"Train - AUC: {metrics['train_auc']:.4f} | F1: {metrics['train_f1']:.4f}")
    print(f"Test  - AUC: {metrics['test_auc']:.4f} | F1: {metrics['test_f1']:.4f}")

    return model, metrics, y_test_proba


# ==================== MODEL 2: RANDOM FOREST ====================
def train_random_forest(X_train, X_test, y_train, y_test) -> tuple:
    """
    Train Random Forest model (tree-based baseline)

    Returns:
        Tuple of (model, metrics_dict, test_probabilities)
    """
    print("\n" + "-" * 80)
    print("Training Model: Random Forest (Tree-based Baseline)")
    print("-" * 80)

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    class_weight = {0: 1.0, 1: scale_pos_weight}

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight=class_weight,
        random_state=42,
        n_jobs=-1
    )

    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    # Predictions
    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    metrics = {
        'model': 'Random Forest',
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'train_precision': precision_score(y_train, y_train_pred),
        'train_recall': recall_score(y_train, y_train_pred),
        'train_f1': f1_score(y_train, y_train_pred),
        'train_auc': roc_auc_score(y_train, y_train_proba),
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'test_precision': precision_score(y_test, y_test_pred),
        'test_recall': recall_score(y_test, y_test_pred),
        'test_f1': f1_score(y_test, y_test_pred),
        'test_auc': roc_auc_score(y_test, y_test_proba),
        'training_time': training_time,
    }

    print(f"Training time: {training_time:.2f}s")
    print(f"Train - AUC: {metrics['train_auc']:.4f} | F1: {metrics['train_f1']:.4f}")
    print(f"Test  - AUC: {metrics['test_auc']:.4f} | F1: {metrics['test_f1']:.4f}")

    return model, metrics, y_test_proba


# ==================== MODEL 3: XGBOOST ====================
def train_xgboost(X_train, X_test, y_train, y_test) -> tuple:
    """
    Train XGBoost model (gradient boosting)

    Returns:
        Tuple of (model, metrics_dict, test_probabilities)
    """
    if not XGBOOST_AVAILABLE:
        return None, None, None

    print("\n" + "-" * 80)
    print("Training Model: XGBoost (Gradient Boosting)")
    print("-" * 80)

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=600,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=2,
        gamma=0.1,
        reg_alpha=0.2,
        reg_lambda=1.2,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        early_stopping_rounds=80,
        random_state=42,
        n_jobs=-1
    )

    start_time = time.time()
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False
    )
    training_time = time.time() - start_time

    # Predictions
    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    metrics = {
        'model': 'XGBoost',
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'train_precision': precision_score(y_train, y_train_pred),
        'train_recall': recall_score(y_train, y_train_pred),
        'train_f1': f1_score(y_train, y_train_pred),
        'train_auc': roc_auc_score(y_train, y_train_proba),
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'test_precision': precision_score(y_test, y_test_pred),
        'test_recall': recall_score(y_test, y_test_pred),
        'test_f1': f1_score(y_test, y_test_pred),
        'test_auc': roc_auc_score(y_test, y_test_proba),
        'training_time': training_time,
    }

    print(f"Training time: {training_time:.2f}s")
    print(f"Train - AUC: {metrics['train_auc']:.4f} | F1: {metrics['train_f1']:.4f}")
    print(f"Test  - AUC: {metrics['test_auc']:.4f} | F1: {metrics['test_f1']:.4f}")

    return model, metrics, y_test_proba


# ==================== MODEL 4: LIGHTGBM ====================
def train_lightgbm(X_train, X_test, y_train, y_test) -> tuple:
    """
    Train LightGBM model (fast gradient boosting)

    Returns:
        Tuple of (model, metrics_dict, test_probabilities)
    """
    if not LIGHTGBM_AVAILABLE:
        return None, None, None

    print("\n" + "-" * 80)
    print("Training Model: LightGBM (Fast Gradient Boosting)")
    print("-" * 80)

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = lgb.LGBMClassifier(
        objective='binary',
        n_estimators=600,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_alpha=0.2,
        reg_lambda=1.2,
        scale_pos_weight=scale_pos_weight,
        metric='binary_logloss',
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )

    start_time = time.time()
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        eval_names=['train', 'valid'],
        callbacks=[lgb.early_stopping(stopping_rounds=80), lgb.log_evaluation(period=0)]
    )
    training_time = time.time() - start_time

    # Predictions
    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    metrics = {
        'model': 'LightGBM',
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'train_precision': precision_score(y_train, y_train_pred),
        'train_recall': recall_score(y_train, y_train_pred),
        'train_f1': f1_score(y_train, y_train_pred),
        'train_auc': roc_auc_score(y_train, y_train_proba),
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'test_precision': precision_score(y_test, y_test_pred),
        'test_recall': recall_score(y_test, y_test_pred),
        'test_f1': f1_score(y_test, y_test_pred),
        'test_auc': roc_auc_score(y_test, y_test_proba),
        'training_time': training_time,
    }

    print(f"Training time: {training_time:.2f}s")
    print(f"Train - AUC: {metrics['train_auc']:.4f} | F1: {metrics['train_f1']:.4f}")
    print(f"Test  - AUC: {metrics['test_auc']:.4f} | F1: {metrics['test_f1']:.4f}")

    return model, metrics, y_test_proba


# ==================== MODEL 5: CATBOOST (BEST) ====================
def train_catboost(X_train, X_test, y_train, y_test) -> tuple:
    """
    Train CatBoost model (automated gradient boosting) ⭐ BEST MODEL

    Key advantages:
    - Automatic handling of class imbalance (auto_class_weights='Balanced')
    - Built-in overfitting prevention (use_best_model=True)
    - Robust to feature scaling
    - Excellent performance on this task

    Returns:
        Tuple of (model, metrics_dict, test_probabilities)
    """
    if not CATBOOST_AVAILABLE:
        return None, None, None

    print("\n" + "-" * 80)
    print("Training Model: CatBoost (Automated Gradient Boosting) ⭐ BEST")
    print("-" * 80)

    model = cb.CatBoostClassifier(
        iterations=600,
        depth=6,
        learning_rate=0.03,
        l2_leaf_reg=3,
        random_seed=42,
        loss_function='Logloss',
        eval_metric='AUC',
        early_stopping_rounds=80,
        verbose=False,
        auto_class_weights='Balanced'  # Automatic class imbalance handling
    )

    start_time = time.time()
    model.fit(
        X_train, y_train,
        eval_set=(X_test, y_test),
        use_best_model=True  # Automatically select best iteration
    )
    training_time = time.time() - start_time

    # Predictions
    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    metrics = {
        'model': 'CatBoost',
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'train_precision': precision_score(y_train, y_train_pred),
        'train_recall': recall_score(y_train, y_train_pred),
        'train_f1': f1_score(y_train, y_train_pred),
        'train_auc': roc_auc_score(y_train, y_train_proba),
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'test_precision': precision_score(y_test, y_test_pred),
        'test_recall': recall_score(y_test, y_test_pred),
        'test_f1': f1_score(y_test, y_test_pred),
        'test_auc': roc_auc_score(y_test, y_test_proba),
        'training_time': training_time,
    }

    print(f"Training time: {training_time:.2f}s")
    print(f"Train - AUC: {metrics['train_auc']:.4f} | F1: {metrics['train_f1']:.4f}")
    print(f"Test  - AUC: {metrics['test_auc']:.4f} | F1: {metrics['test_f1']:.4f}")

    return model, metrics, y_test_proba


# ==================== MODEL 5: MLP NEURAL NETWORK ====================
def train_mlp(X_train, X_test, y_train, y_test) -> tuple:
    """
    Train MLP Neural Network (deep learning, highest complexity)

    Key characteristics:
    - 3 hidden layers (128, 64, 32) with ReLU activation
    - Adam optimizer with adaptive learning rate
    - Early stopping to prevent overfitting
    - Highest training time and complexity
    - Note: Performance similar to CatBoost despite higher complexity

    Returns:
        Tuple of (model, metrics_dict, test_probabilities)
    """
    from sklearn.neural_network import MLPClassifier

    print("\n" + "-" * 80)
    print("Training Model: MLP Neural Network (Highest Complexity)")
    print("-" * 80)

    # Calculate sample weights for class imbalance
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    sample_weights = np.where(y_train == 1, scale_pos_weight, 1.0)

    # Standardize features (critical for neural networks)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),  # 3 layers: 128 → 64 → 32
        activation='relu',                  # ReLU activation
        solver='adam',                      # Adam optimizer
        alpha=0.001,                        # L2 regularization
        batch_size=256,                     # Batch size
        learning_rate='adaptive',           # Adaptive learning rate
        learning_rate_init=0.001,           # Initial learning rate
        max_iter=300,                       # Max epochs
        early_stopping=True,                # Enable early stopping
        validation_fraction=0.1,            # 10% for validation
        n_iter_no_change=20,                # Early stopping patience
        random_state=42,
        verbose=False
    )

    start_time = time.time()
    model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
    training_time = time.time() - start_time

    # Predictions
    y_train_pred = model.predict(X_train_scaled)
    y_train_proba = model.predict_proba(X_train_scaled)[:, 1]
    y_test_pred = model.predict(X_test_scaled)
    y_test_proba = model.predict_proba(X_test_scaled)[:, 1]

    # Metrics
    metrics = {
        'model': 'MLP',
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'train_precision': precision_score(y_train, y_train_pred),
        'train_recall': recall_score(y_train, y_train_pred),
        'train_f1': f1_score(y_train, y_train_pred),
        'train_auc': roc_auc_score(y_train, y_train_proba),
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'test_precision': precision_score(y_test, y_test_pred),
        'test_recall': recall_score(y_test, y_test_pred),
        'test_f1': f1_score(y_test, y_test_pred),
        'test_auc': roc_auc_score(y_test, y_test_proba),
        'training_time': training_time,
    }

    print(f"Training time: {training_time:.2f}s")
    print(f"Train - AUC: {metrics['train_auc']:.4f} | F1: {metrics['train_f1']:.4f}")
    print(f"Test  - AUC: {metrics['test_auc']:.4f} | F1: {metrics['test_f1']:.4f}")

    return model, metrics, y_test_proba


# ==================== MAIN TRAINING PIPELINE ====================
def main():
    """
    Main training pipeline: Load data, train all models, compare results
    """
    print("\n" + "=" * 80)
    print("AIRBNB RATING CLASSIFICATION - MODEL TRAINING")
    print("Task: Predict 5-star vs <5-star ratings")
    print("=" * 80)

    # Load data
    df, charts_dir = load_training_data()
    X_train, X_test, y_train, y_test, feature_cols = prepare_binary_dataset(df)

    # Train all models
    print("\n" + "=" * 80)
    print("TRAINING MODELS")
    print("=" * 80)

    all_results = []
    roc_data = {}

    # 1. Logistic Regression
    _, metrics_lr, proba_lr = train_logistic_regression(X_train, X_test, y_train, y_test)
    if metrics_lr:
        all_results.append(metrics_lr)
        fpr, tpr, _ = roc_curve(y_test, proba_lr)
        roc_data['Logistic Regression'] = (fpr, tpr, metrics_lr['test_auc'])

    # 2. Random Forest
    _, metrics_rf, proba_rf = train_random_forest(X_train, X_test, y_train, y_test)
    if metrics_rf:
        all_results.append(metrics_rf)
        fpr, tpr, _ = roc_curve(y_test, proba_rf)
        roc_data['Random Forest'] = (fpr, tpr, metrics_rf['test_auc'])

    # 3. LightGBM
    _, metrics_lgb, proba_lgb = train_lightgbm(X_train, X_test, y_train, y_test)
    if metrics_lgb:
        all_results.append(metrics_lgb)
        fpr, tpr, _ = roc_curve(y_test, proba_lgb)
        roc_data['LightGBM'] = (fpr, tpr, metrics_lgb['test_auc'])

    # 4. CatBoost (Best)
    best_model, metrics_cat, proba_cat = train_catboost(X_train, X_test, y_train, y_test)
    if metrics_cat:
        all_results.append(metrics_cat)
        fpr, tpr, _ = roc_curve(y_test, proba_cat)
        roc_data['CatBoost'] = (fpr, tpr, metrics_cat['test_auc'])

    # 5. MLP Neural Network (Highest Complexity)
    _, metrics_mlp, proba_mlp = train_mlp(X_train, X_test, y_train, y_test)
    if metrics_mlp:
        all_results.append(metrics_mlp)
        fpr, tpr, _ = roc_curve(y_test, proba_mlp)
        roc_data['MLP'] = (fpr, tpr, metrics_mlp['test_auc'])

    # Compile results
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('test_auc', ascending=False)

    # Print summary
    print("\n" + "=" * 80)
    print("MODEL COMPARISON SUMMARY (Sorted by Test AUC)")
    print("=" * 80)
    print(results_df[['model', 'test_auc', 'test_f1', 'test_precision', 'test_recall', 'training_time']].to_string(index=False))

    # Identify best model
    best_model_name = results_df.iloc[0]['model']
    best_auc = results_df.iloc[0]['test_auc']
    print(f"\n🏆 Best Model: {best_model_name} (Test AUC: {best_auc:.4f})")

    # Save results
    output_csv = charts_dir / 'model_comparison_results.csv'
    results_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\n✅ Results saved to: {output_csv}")

    print("\n" + "=" * 80)
    print("MODEL TRAINING COMPLETED")
    print("=" * 80)
    print(f"Total models trained: {len(all_results)}")
    print(f"Best model: {best_model_name}")
    print(f"Best test AUC: {best_auc:.4f}")

    return results_df, roc_data, best_model


if __name__ == "__main__":
    results_df, roc_data, best_model = main()
