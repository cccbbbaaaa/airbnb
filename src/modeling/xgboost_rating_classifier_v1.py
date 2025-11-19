"""
XGBoost Rating Classification Model v1
XGBoost 评分分类模型 v1

- 将回归问题转换为二分类问题 (is_high_rating)
- 使用 v3 特征工程脚本
- 使用分类指标 (AUC, F1-Score) 进行评估

- Converts the regression problem into a binary classification problem (is_high_rating)
- Uses the v3 feature engineering script
- Evaluates using classification metrics (AUC, F1-Score)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)
from sklearn.model_selection import train_test_split

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.EDA.utils import setup_plotting, get_project_paths
from src.feature_engineering.rating_feature_engineering_v3 import engineer_rating_features_v3

setup_plotting()


def train_rating_classifier_v1():
    """训练 v1 评分分类模型 / Train v1 rating classification model."""
    print("=" * 80)
    print("XGBoost Rating Classification Model (v1)")
    print("XGBoost 评分分类模型（v1 - 分类任务）")
    print("=" * 80)

    # =========================================================================
    # 1. 构建特征 / Build Features
    # =========================================================================
    print("\n1. 构建 v3 (分类) 特征 / Building v3 (Classification) Features...")
    try:
        processed_df = engineer_rating_features_v3(save_processed=True)
        print(f"  [OK] 特征构建完成: {len(processed_df)} 行 × {processed_df.shape[1]} 列")
    except Exception as e:
        print(f"  [ERROR] 特征构建失败 / Feature engineering failed: {e}")
        return

    # =========================================================================
    # 2. 准备数据 / Prepare Data
    # =========================================================================
    print("\n2. 准备数据 / Preparing Data...")
    target_col = "is_high_rating"
    if target_col not in processed_df.columns:
        print(f"  [ERROR] 目标列 '{target_col}' 不在数据中！")
        return
        
    # 移除原始评分字段，避免信息泄露 / Remove original rating field to prevent data leakage
    feature_cols = [col for col in processed_df.columns if col not in [target_col, 'review_scores_rating']]

    X = processed_df[feature_cols].copy()
    y = processed_df[target_col].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y # 使用 stratify 保持类别比例
    )
    print(f"  训练集: {len(X_train)} 条 (高评分占比: {y_train.mean():.2%})")
    print(f"  测试集: {len(X_test)} 条 (高评分占比: {y_test.mean():.2%})")

    # =========================================================================
    # 3. 训练模型 / Train Model
    # =========================================================================
    print("\n3. 训练 XGBoost 分类器 / Training XGBoost Classifier...")
    
    # 处理类别不平衡问题 / Handle class imbalance
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"  计算 scale_pos_weight 用于处理类别不平衡: {scale_pos_weight:.2f}")

    params = {
        "objective": "binary:logistic",
        "n_estimators": 500,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "gamma": 0.2,
        "eval_metric": "logloss",
        "early_stopping_rounds": 50,
        "random_state": 42,
        "n_jobs": -1,
        "scale_pos_weight": scale_pos_weight, # 添加权重参数
    }

    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )
    print("  [OK] 模型训练完成")

    # =========================================================================
    # 4. 模型评估 / Evaluation
    # =========================================================================
    print("\n4. 模型评估 / Model Evaluation...")
    y_pred_test = model.predict(X_test)
    y_pred_proba_test = model.predict_proba(X_test)[:, 1]

    print("\n测试集分类报告 / Test Set Classification Report:")
    print(classification_report(y_test, y_pred_test, target_names=['Not High Rating', 'High Rating']))

    auc = roc_auc_score(y_test, y_pred_proba_test)
    print(f"测试集 AUC / Test Set AUC: {auc:.4f}")

    # =========================================================================
    # 5. 特征重要性 / Feature Importance
    # =========================================================================
    project_root, _, charts_eda_dir, charts_model_dir = get_project_paths()
    charts_dir = charts_model_dir  # 使用模型目录 / Use model directory
    importance_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    output_importance = charts_dir / "xgboost_rating_classifier_v1_importance.csv"
    importance_df.to_csv(output_importance, index=False)
    print(f"\n  [OK] 特征重要性已保存到: {output_importance}")

    # =========================================================================
    # 6. 保存模型 / Save Model
    # =========================================================================
    model_path = charts_dir / "xgboost_rating_classifier_v1.json"
    model.save_model(model_path)
    print(f"  [OK] 模型已保存到: {model_path}")

    return {"test_auc": auc, "test_f1_score": f1_score(y_test, y_pred_test)}


def main():
    metrics = train_rating_classifier_v1()
    print("\n最终指标 / Final Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    print("=" * 80)
    print("v1 评分分类模型完成 / v1 rating classification model training complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
