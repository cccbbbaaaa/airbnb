"""
Rating Classification Model Comparison (Rating = 5 vs < 5)
评分分类模型对比脚本：预测评分是否为 5 分

模型列表：
1. Logistic Regression（线性模型基线，带标准化 & 类别权重）
2. Linear SVM（线性支持向量机作为基线，带标准化 & 类别权重）
3. Random Forest（树模型基线）
4. XGBoost（梯度提升模型）

输出内容：
- 控制台打印每个模型的训练/测试指标
- `charts/model/rating_classifier_model_comparison.csv` 保存指标对比表
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

# -----------------------------------------------------------------------------
# 路径设置 / Add project paths
# -----------------------------------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "EDA"))
from utils import get_project_paths  # noqa: E402


# -----------------------------------------------------------------------------
# 数据加载与准备 / Data loading and preparation
# -----------------------------------------------------------------------------
def load_training_data() -> pd.DataFrame:
    project_root, data_dir, charts_eda_dir, charts_model_dir = get_project_paths()
    train_path = data_dir / "processed" / "train_data.csv"
    if not train_path.exists():
        raise FileNotFoundError(
            f"未找到 {train_path}。请先运行 src/feature_engineering/build_train_features.py"
        )
    print("=" * 80)
    print("Loading training dataset...")
    print(f"  路径: {train_path}")
    df = pd.read_csv(train_path)
    print(f"  ✅ 数据规模: {len(df):,} 行 × {len(df.columns)} 列")
    return df, charts_model_dir


def prepare_binary_dataset(df: pd.DataFrame):
    if "review_scores_rating" not in df.columns:
        raise ValueError("数据中缺少 review_scores_rating 字段，无法构建标签")

    working_df = df[df["review_scores_rating"].notna()].copy()
    working_df["is_five_star"] = np.isclose(working_df["review_scores_rating"], 5.0, atol=1e-6).astype(int)

    print("\n" + "=" * 80)
    print("Preparing dataset (Rating = 5 vs < 5)")
    positives = working_df["is_five_star"].sum()
    total = len(working_df)
    print(f"  总样本: {total:,}")
    print(f"  5 星样本: {positives:,} ({positives / total:.2%})")
    print(f"  <5 星样本: {total - positives:,} ({1 - positives / total:.2%})")

    exclude_cols = {"review_scores_rating", "is_five_star"}
    feature_cols = [col for col in working_df.columns if col not in exclude_cols]

    X = working_df[feature_cols].copy()
    y = working_df["is_five_star"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"  训练集: {len(X_train):,} (5 星占比: {y_train.mean():.2%})")
    print(f"  测试集: {len(X_test):,} (5 星占比: {y_test.mean():.2%})")

    return X_train, X_test, y_train, y_test, feature_cols


# -----------------------------------------------------------------------------
# 模型定义 / Model definitions
# -----------------------------------------------------------------------------
def build_models():
    scaler = StandardScaler()

    logistic = Pipeline(
        steps=[
            ("scaler", scaler),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=-1)),
        ]
    )

    linear_svm = Pipeline(
        steps=[
            ("scaler", scaler),
            ("clf", LinearSVC(class_weight="balanced", max_iter=5000)),
        ]
    )

    random_forest = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_split=4,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    xgb_model = xgb.XGBClassifier(
        objective="binary:logistic",
        n_estimators=600,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.2,
        reg_lambda=1.0,
        min_child_weight=3,
        gamma=0.15,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42,
    )

    return {
        "LogisticRegression": logistic,
        "LinearSVM": linear_svm,
        "RandomForest": random_forest,
        "XGBoost": xgb_model,
    }


# -----------------------------------------------------------------------------
# 评估函数 / Evaluation helpers
# -----------------------------------------------------------------------------
def get_probabilities(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        decision = model.decision_function(X)
        decision_min = decision.min()
        decision_max = decision.max()
        if decision_max - decision_min > 1e-8:
            return (decision - decision_min) / (decision_max - decision_min)
        return np.zeros_like(decision)
    return np.zeros(len(X))


def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    print(f"\n训练模型: {name}")
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_proba = get_probabilities(model, X_train)
    test_proba = get_probabilities(model, X_test)

    metrics = {
        "model": name,
        "train_accuracy": accuracy_score(y_train, train_pred),
        "train_precision": precision_score(y_train, train_pred, zero_division=0),
        "train_recall": recall_score(y_train, train_pred, zero_division=0),
        "train_f1": f1_score(y_train, train_pred, zero_division=0),
        "train_auc": roc_auc_score(y_train, train_proba) if len(np.unique(y_train)) > 1 else np.nan,
        "test_accuracy": accuracy_score(y_test, test_pred),
        "test_precision": precision_score(y_test, test_pred, zero_division=0),
        "test_recall": recall_score(y_test, test_pred, zero_division=0),
        "test_f1": f1_score(y_test, test_pred, zero_division=0),
        "test_auc": roc_auc_score(y_test, test_proba) if len(np.unique(y_test)) > 1 else np.nan,
    }

    print(
        "  训练集: Acc={train_accuracy:.3f}  Prec={train_precision:.3f}  "
        "Recall={train_recall:.3f}  F1={train_f1:.3f}  AUC={train_auc:.3f}".format(**metrics)
    )
    print(
        "  测试集: Acc={test_accuracy:.3f}  Prec={test_precision:.3f}  "
        "Recall={test_recall:.3f}  F1={test_f1:.3f}  AUC={test_auc:.3f}".format(**metrics)
    )
    return metrics


# -----------------------------------------------------------------------------
# 主流程 / Main routine
# -----------------------------------------------------------------------------
def main():
    df, charts_model_dir = load_training_data()
    X_train, X_test, y_train, y_test, feature_cols = prepare_binary_dataset(df)

    models = build_models()
    metrics_list = []
    for name, model in models.items():
        metrics = evaluate_model(name, model, X_train, y_train, X_test, y_test)
        metrics_list.append(metrics)

    metrics_df = pd.DataFrame(metrics_list)
    metrics_df = metrics_df[
        [
            "model",
            "train_accuracy",
            "train_precision",
            "train_recall",
            "train_f1",
            "train_auc",
            "test_accuracy",
            "test_precision",
            "test_recall",
            "test_f1",
            "test_auc",
        ]
    ]

    charts_model_dir.mkdir(parents=True, exist_ok=True)
    output_csv = charts_model_dir / "rating_classifier_model_comparison.csv"
    metrics_df.to_csv(output_csv, index=False)

    print("\n" + "=" * 80)
    print("Model Comparison Summary")
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("=" * 80)
    print(f"指标对比表已保存：{output_csv}")


if __name__ == "__main__":
    main()


