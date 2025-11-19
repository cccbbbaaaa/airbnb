"""
CatBoost Rating Classification Model v1
CatBoost 评分分类模型 v1

- 复用 v3 特征工程（分类任务）
- 使用 CatBoostClassifier 处理数值特征
- 输出 Accuracy / Precision / Recall / F1 / AUC
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)
from sklearn.model_selection import train_test_split

# 将项目根目录加入路径 / Add repo root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.EDA.utils import setup_plotting, get_project_paths
from src.feature_engineering.rating_feature_engineering_v3 import engineer_rating_features_v3

setup_plotting()


def load_classification_features() -> pd.DataFrame:
    """
    加载或生成分类特征数据集 / Load or build classification feature dataset.
    """
    project_root, data_dir, _ = get_project_paths()
    processed_path = data_dir / "processed" / "rating_features_classification_v1.csv"

    if processed_path.exists():
        print(f"  读取已处理特征 / Loading processed features from: {processed_path}")
        return pd.read_csv(processed_path)

    print("  未找到处理后数据，重新运行特征工程 / Processed data not found, rebuilding...")
    return engineer_rating_features_v3(save_processed=True)


def evaluate_classifier(model_name: str, y_true, y_pred, y_proba) -> dict:
    """
    计算并打印分类评估指标 / Compute and print evaluation metrics.
    """
    print(f"\n4. {model_name} 模型评估 / {model_name} Model Evaluation...")
    print("\n测试集分类报告 / Test Set Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Not High Rating", "High Rating"]))

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)

    print(f"测试集 Accuracy: {accuracy:.4f}")
    print(f"测试集 Precision: {precision:.4f}")
    print(f"测试集 Recall: {recall:.4f}")
    print(f"测试集 F1-Score: {f1:.4f}")
    print(f"测试集 AUC: {auc:.4f}")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
    }


def train_catboost_classifier_v1():
    """
    训练 CatBoost 评分分类模型 / Train CatBoost rating classification model.
    """
    print("=" * 80)
    print("CatBoost Rating Classification Model (v1)")
    print("CatBoost 评分分类模型（v1）")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. 加载或生成特征 / Load or Build Features
    # -------------------------------------------------------------------------
    print("\n1. 加载分类特征数据 / Loading classification features...")
    processed_df = load_classification_features()
    print(f"  [OK] 特征数据集: {processed_df.shape[0]} 行 × {processed_df.shape[1]} 列")

    # -------------------------------------------------------------------------
    # 2. 准备训练/测试集 / Prepare Train/Test Sets
    # -------------------------------------------------------------------------
    print("\n2. 准备训练/测试集 / Preparing Train/Test sets...")
    target_col = "is_high_rating"
    feature_cols = [col for col in processed_df.columns if col != target_col]

    X = processed_df[feature_cols].copy()
    y = processed_df[target_col].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    print(f"  训练集: {len(X_train)} 条 (高评分占比: {y_train.mean():.2%})")
    print(f"  测试集: {len(X_test)} 条 (高评分占比: {y_test.mean():.2%})")

    # -------------------------------------------------------------------------
    # 3. 训练 CatBoost 模型 / Train CatBoost
    # -------------------------------------------------------------------------
    print("\n3. 训练 CatBoost 模型 / Training CatBoost model...")
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    catboost_params = {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "iterations": 2000,
        "learning_rate": 0.03,
        "depth": 8,
        "l2_leaf_reg": 4.0,
        "random_seed": 42,
        "class_weights": [1.0, scale_pos_weight],
        "allow_writing_files": False,
        "thread_count": -1,
        "verbose": 200,
    }

    model = CatBoostClassifier(**catboost_params)
    model.fit(
        X_train,
        y_train,
        eval_set=(X_test, y_test),
        use_best_model=True,
    )
    print("  [OK] CatBoost 模型训练完成 / Training complete.")

    # -------------------------------------------------------------------------
    # 4. 评估 / Evaluate
    # -------------------------------------------------------------------------
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = evaluate_classifier("CatBoost", y_test, y_pred, y_proba)

    # -------------------------------------------------------------------------
    # 5. 特征重要性 / Feature Importance
    # -------------------------------------------------------------------------
    print("\n5. 保存特征重要性 / Saving feature importance...")
    project_root, _, charts_eda_dir, charts_model_dir = get_project_paths()
    charts_dir = charts_model_dir  # 使用模型目录 / Use model directory
    importance = model.get_feature_importance(type="FeatureImportance")
    importance_df = pd.DataFrame(
        {"feature": feature_cols, "importance": importance}
    ).sort_values("importance", ascending=False)
    importance_path = charts_dir / "catboost_rating_classifier_v1_importance.csv"
    importance_df.to_csv(importance_path, index=False)
    print(f"  [OK] 特征重要性已保存到: {importance_path}")

    # -------------------------------------------------------------------------
    # 6. 保存模型 / Save Model
    # -------------------------------------------------------------------------
    model_path = charts_dir / "catboost_rating_classifier_v1.cbm"
    model.save_model(str(model_path))
    print(f"  [OK] CatBoost 模型已保存到: {model_path}")

    print("=" * 80)
    print("CatBoost v1 分类模型完成 / CatBoost v1 training complete.")
    print("=" * 80)
    return metrics


def main():
    metrics = train_catboost_classifier_v1()
    print("\n最终指标 / Final Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()

