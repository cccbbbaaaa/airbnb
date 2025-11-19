"""
XGBoost Rating Classification Model (Merged Data)
XGBoost 评分分类模型（整合数据版本）

整合 listings_detailed 和 listings_detailed_2，预测高分（评分>4.8）
Merges listings_detailed and listings_detailed_2, predicts high rating (rating > 4.8)
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
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# 添加 EDA 目录到路径 / Add EDA directory to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "EDA"))

from utils import setup_plotting, get_project_paths

# 添加特征工程目录 / Add feature engineering directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "feature_engineering"))
from rating_feature_engineering_merged import engineer_rating_features_merged

setup_plotting()


def train_merged_rating_classifier():
    """训练整合数据版本的评分分类模型 / Train merged data rating classification model."""
    print("=" * 80)
    print("XGBoost Rating Classification Model (Merged Data)")
    print("XGBoost 评分分类模型（整合数据版本 - 预测高分评分>4.8）")
    print("=" * 80)

    # =========================================================================
    # 1. 构建特征 / Build Features
    # =========================================================================
    print("\n1. 构建整合特征 / Building Merged Features...")
    try:
        processed_df = engineer_rating_features_merged(save_processed=False)  # 不重复保存
        print(f"  [OK] 特征构建完成: {len(processed_df)} 行 × {processed_df.shape[1]} 列")
    except Exception as e:
        print(f"  [ERROR] 特征构建失败 / Feature engineering failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # =========================================================================
    # 2. 准备数据 / Prepare Data
    # =========================================================================
    print("\n2. 准备数据 / Preparing Data...")
    target_col = "review_scores_rating"
    if target_col not in processed_df.columns:
        print(f"  [ERROR] 目标列 '{target_col}' 不在数据中！")
        return
    
    # 创建二分类标签（评分>4.8为高分）
    processed_df['is_high_rating'] = (processed_df[target_col] > 4.8).astype(int)
    
    # 只保留有评分的记录
    mask = processed_df[target_col].notna() & (processed_df[target_col] >= 0) & (processed_df[target_col] <= 100)
    processed_df = processed_df[mask].copy()
    
    print(f"  有效数据: {len(processed_df)} 条记录")
    print(f"  高分样本数（评分>4.8）: {processed_df['is_high_rating'].sum()} ({processed_df['is_high_rating'].mean():.2%})")
    print(f"  低分样本数（评分≤4.8）: {(~processed_df['is_high_rating'].astype(bool)).sum()} ({(1-processed_df['is_high_rating'].mean()):.2%})")
    
    # 排除目标列和数据源列
    feature_cols = [col for col in processed_df.columns 
                   if col not in [target_col, 'is_high_rating', 'data_source']]

    X = processed_df[feature_cols].copy()
    y = processed_df['is_high_rating'].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y  # 使用 stratify 保持类别比例
    )
    print(f"  训练集: {len(X_train)} 条 (高分占比: {y_train.mean():.2%})")
    print(f"  测试集: {len(X_test)} 条 (高分占比: {y_test.mean():.2%})")

    # =========================================================================
    # 3. 训练模型 / Train Model
    # =========================================================================
    print("\n3. 训练 XGBoost 分类器 / Training XGBoost Classifier...")
    
    # 处理类别不平衡问题 / Handle class imbalance
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum() if (y_train == 1).sum() > 0 else 1.0
    print(f"  计算 scale_pos_weight 用于处理类别不平衡: {scale_pos_weight:.2f}")

    params = {
        "objective": "binary:logistic",
        "n_estimators": 600,
        "max_depth": 6,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "colsample_bylevel": 0.8,
        "min_child_weight": 2,
        "gamma": 0.1,
        "reg_alpha": 0.2,
        "reg_lambda": 1.2,
        "eval_metric": "logloss",
        "early_stopping_rounds": 80,
        "random_state": 42,
        "n_jobs": -1,
        "scale_pos_weight": scale_pos_weight,  # 添加权重参数处理类别不平衡
    }

    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=50,
    )
    print("  [OK] 模型训练完成")

    # =========================================================================
    # 4. 模型评估 / Evaluation
    # =========================================================================
    print("\n4. 模型评估 / Model Evaluation...")
    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]

    # 训练集指标
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred)
    train_recall = recall_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    train_auc = roc_auc_score(y_train, y_train_proba)

    # 测试集指标
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_proba)

    if hasattr(model, "best_iteration") and model.best_iteration is not None:
        print(f"  最佳迭代次数 / Best Iteration: {model.best_iteration}")

    print("\n训练集指标 / Training Metrics:")
    print(f"  Accuracy: {train_accuracy:.4f}")
    print(f"  Precision: {train_precision:.4f}")
    print(f"  Recall: {train_recall:.4f}")
    print(f"  F1-Score: {train_f1:.4f}")
    print(f"  AUC: {train_auc:.4f}")

    print("\n测试集指标 / Test Metrics:")
    print(f"  Accuracy: {test_accuracy:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall: {test_recall:.4f}")
    print(f"  F1-Score: {test_f1:.4f}")
    print(f"  AUC: {test_auc:.4f}")

    print("\n测试集分类报告 / Test Set Classification Report:")
    print(classification_report(y_test, y_test_pred, target_names=['Low Rating (≤4.8)', 'High Rating (>4.8)']))

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_test_pred)
    print("\n混淆矩阵 / Confusion Matrix:")
    print(f"  True Negatives (TN): {cm[0, 0]}")
    print(f"  False Positives (FP): {cm[0, 1]}")
    print(f"  False Negatives (FN): {cm[1, 0]}")
    print(f"  True Positives (TP): {cm[1, 1]}")

    # =========================================================================
    # 5. 特征重要性 / Feature Importance
    # =========================================================================
    print("\n5. 特征重要性分析 / Feature Importance Analysis...")
    project_root, _, charts_eda_dir, charts_model_dir = get_project_paths()
    charts_dir = charts_model_dir
    
    importance_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    
    print("\n前20个最重要特征 / Top 20 Most Important Features:")
    print(importance_df.head(20).to_string(index=False))
    
    output_importance = charts_dir / "xgboost_rating_classifier_merged_importance.csv"
    importance_df.to_csv(output_importance, index=False)
    print(f"\n  [OK] 特征重要性已保存到: {output_importance}")

    # =========================================================================
    # 6. 可视化结果 / Visualization
    # =========================================================================
    print("\n6. 生成可视化图表 / Generating Visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 6.1 特征重要性Top 20
    top_features = importance_df.head(20)
    axes[0, 0].barh(range(len(top_features)), top_features['importance'].values)
    axes[0, 0].set_yticks(range(len(top_features)))
    axes[0, 0].set_yticklabels(top_features['feature'].values)
    axes[0, 0].set_xlabel('Importance')
    axes[0, 0].set_title('Top 20 Feature Importance (Merged Data - Classification)')
    axes[0, 0].invert_yaxis()

    # 6.2 ROC曲线（使用测试集）
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
    axes[0, 1].plot(fpr, tpr, label=f'ROC Curve (AUC = {test_auc:.4f})')
    axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random')
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_title('ROC Curve')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # 6.3 混淆矩阵热力图
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
                xticklabels=['Low Rating (≤4.8)', 'High Rating (>4.8)'],
                yticklabels=['Low Rating (≤4.8)', 'High Rating (>4.8)'])
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('Actual')
    axes[1, 0].set_title('Confusion Matrix')

    # 6.4 预测概率分布
    axes[1, 1].hist(y_test_proba[y_test == 0], bins=50, alpha=0.5, label='Low Rating (≤4.8)', color='red')
    axes[1, 1].hist(y_test_proba[y_test == 1], bins=50, alpha=0.5, label='High Rating (>4.8)', color='green')
    axes[1, 1].set_xlabel('Predicted Probability')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Prediction Probability Distribution')
    axes[1, 1].legend()
    axes[1, 1].axvline(x=0.5, color='black', linestyle='--', lw=2, label='Threshold (0.5)')
    axes[1, 1].legend()

    plt.tight_layout()
    output_chart = charts_dir / 'xgboost_rating_classifier_merged_results.png'
    plt.savefig(output_chart, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  [OK] 可视化图表已保存到: {output_chart}")

    # =========================================================================
    # 7. 保存模型 / Save Model
    # =========================================================================
    print("\n7. 保存模型 / Saving Model...")
    model_path = charts_dir / "xgboost_rating_classifier_merged.json"
    try:
        model.save_model(str(model_path))
        print(f"  [OK] 模型已保存到: {model_path}")
    except Exception:
        import pickle
        pkl_path = charts_dir / "xgboost_rating_classifier_merged.pkl"
        with open(pkl_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"  [OK] 模型已保存到 (fallback): {pkl_path}")

    return {
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "train_precision": train_precision,
        "test_precision": test_precision,
        "train_recall": train_recall,
        "test_recall": test_recall,
        "train_f1": train_f1,
        "test_f1": test_f1,
        "train_auc": train_auc,
        "test_auc": test_auc,
    }


def main():
    metrics = train_merged_rating_classifier()
    print("\n" + "=" * 80)
    print("最终指标 / Final Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    print("=" * 80)
    print("整合数据评分分类模型完成 / Merged data rating classification model training complete")
    print("=" * 80)


if __name__ == "__main__":
    main()

