"""
XGBoost Rating Classification Model - Cleaned Dataset (5-star vs <5-star)
XGBoost 评分分类模型 - 使用清洗后的特征数据 (5 分 vs <5 分)

用法：
    1. 运行 `src/feature_engineering/build_train_features.py` 生成 `data/processed/train_data.csv`
    2. 运行本脚本训练二分类模型（阈值：rating == 5）
输出内容：
    - 性能指标（Accuracy / Precision / Recall / F1 / AUC）
    - 分类报告 & 混淆矩阵
    - 特征重要性 CSV 与可视化
    - ROC 曲线、概率分布图
    - 训练好的 XGBoost 模型（.json）
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

# 添加项目工具路径 / Add project utility paths
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "EDA"))
from utils import setup_plotting, get_project_paths  # noqa: E402

setup_plotting()


# =============================================================================
# 数据加载与准备
# =============================================================================
def load_train_dataset():
    """加载特征工程输出数据 / Load engineered training dataset."""
    project_root, data_dir, charts_eda_dir, charts_model_dir = get_project_paths()
    train_path = data_dir / "processed" / "train_data.csv"

    if not train_path.exists():
        raise FileNotFoundError(
            f"未找到训练数据 {train_path}，请先运行 src/feature_engineering/build_train_features.py"
        )

    print("=" * 80)
    print("Loading Training Dataset / 加载训练数据")
    print("=" * 80)
    df = pd.read_csv(train_path)
    print(f"  ✅ 载入成功: {len(df):,} 行 × {len(df.columns)} 列")
    return df, train_path


def prepare_dataset(df: pd.DataFrame):
    """构建 5-star vs <5-star 标签，并拆分特征/目标 / Prepare binary labels and feature matrix."""
    print("\n" + "=" * 80)
    print("Preparing Dataset / 准备数据集")
    print("=" * 80)

    if "review_scores_rating" not in df.columns:
        raise ValueError("数据中缺少 review_scores_rating 字段，无法构建标签")

    working_df = df.copy()
    working_df = working_df[working_df["review_scores_rating"].notna()].copy()

    # 评分==5 视为正类 / Label 1 if rating equals 5
    working_df["is_five_star"] = np.isclose(working_df["review_scores_rating"], 5.0, atol=1e-6).astype(int)

    positive_ratio = working_df["is_five_star"].mean()
    print(f"  样本总数: {len(working_df):,}")
    print(f"  5 星样本: {working_df['is_five_star'].sum():,} ({positive_ratio:.2%})")
    print(f"  <5 星样本: {(len(working_df) - working_df['is_five_star'].sum()):,} ({1 - positive_ratio:.2%})")

    # 构建特征矩阵
    exclude_cols = {"review_scores_rating", "is_five_star"}
    feature_cols = [col for col in working_df.columns if col not in exclude_cols]

    X = working_df[feature_cols].copy()
    y = working_df["is_five_star"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print(f"  训练集: {len(X_train):,} (5 星占比: {y_train.mean():.2%})")
    print(f"  测试集: {len(X_test):,} (5 星占比: {y_test.mean():.2%})")

    return X_train, X_test, y_train, y_test, feature_cols


# =============================================================================
# 模型训练与评估
# =============================================================================
def train_xgboost_classifier(X_train, X_test, y_train, y_test):
    """训练 XGBoost 二分类模型 / Train binary XGBoost classifier."""
    print("\n" + "=" * 80)
    print("Training XGBoost Classifier / 训练 XGBoost 分类器")
    print("=" * 80)

    positive_ratio = y_train.mean()
    if positive_ratio == 0:
        raise ValueError("训练集中没有 5 星样本，无法训练模型")

    scale_pos_weight = (1 - positive_ratio) / positive_ratio
    print(f"  类别不平衡权重 scale_pos_weight = {scale_pos_weight:.2f}")

    params = {
        "objective": "binary:logistic",
        "n_estimators": 600,
        "max_depth": 6,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "colsample_bylevel": 0.8,
        "gamma": 0.15,
        "reg_alpha": 0.2,
        "reg_lambda": 1.0,
        "min_child_weight": 3,
        "eval_metric": "logloss",
        "n_jobs": -1,
        "random_state": 42,
        "scale_pos_weight": scale_pos_weight,
        "early_stopping_rounds": 80,
    }

    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=50,
    )

    print("  ✅ 模型训练完成 / Model training finished")

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "train_accuracy": accuracy_score(y_train, y_train_pred),
        "train_precision": precision_score(y_train, y_train_pred, zero_division=0),
        "train_recall": recall_score(y_train, y_train_pred, zero_division=0),
        "train_f1": f1_score(y_train, y_train_pred, zero_division=0),
        "train_auc": roc_auc_score(y_train, y_train_proba),
        "test_accuracy": accuracy_score(y_test, y_test_pred),
        "test_precision": precision_score(y_test, y_test_pred, zero_division=0),
        "test_recall": recall_score(y_test, y_test_pred, zero_division=0),
        "test_f1": f1_score(y_test, y_test_pred, zero_division=0),
        "test_auc": roc_auc_score(y_test, y_test_proba),
    }

    print("\n训练集指标 / Training Metrics:")
    print(
        "  Accuracy={train_accuracy:.4f}  Precision={train_precision:.4f}  "
        "Recall={train_recall:.4f}  F1={train_f1:.4f}  AUC={train_auc:.4f}".format(**metrics)
    )

    print("\n测试集指标 / Test Metrics:")
    print(
        "  Accuracy={test_accuracy:.4f}  Precision={test_precision:.4f}  "
        "Recall={test_recall:.4f}  F1={test_f1:.4f}  AUC={test_auc:.4f}".format(**metrics)
    )

    print("\n测试集分类报告 / Test Classification Report:")
    print(
        classification_report(
            y_test,
            y_test_pred,
            target_names=["Rating < 5", "Rating = 5"],
            zero_division=0,
        )
    )

    cm = confusion_matrix(y_test, y_test_pred)
    print("混淆矩阵 / Confusion Matrix:")
    print(cm)

    return model, metrics, y_test, y_test_pred, y_test_proba, cm


# =============================================================================
# 可视化与导出
# =============================================================================
def export_results(model, feature_cols, metrics, y_test, y_test_proba, cm):
    """保存特征重要性、可视化结果与模型文件 / Export artifacts."""
    project_root, _, _, charts_model_dir = get_project_paths()
    charts_model_dir.mkdir(parents=True, exist_ok=True)

    importance_df = pd.DataFrame(
        {"feature": feature_cols, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    importance_path = charts_model_dir / "xgboost_rating_classifier_cleaned_5star_importance.csv"
    importance_df.to_csv(importance_path, index=False)
    print(f"\n  ✅ 特征重要性导出: {importance_path}")

    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Top 20 feature importance
    top20 = importance_df.head(20)
    axes[0, 0].barh(top20["feature"], top20["importance"])
    axes[0, 0].invert_yaxis()
    axes[0, 0].set_title("Top 20 Feature Importance")
    axes[0, 0].set_xlabel("Importance")

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    axes[0, 1].plot(fpr, tpr, label=f"AUC = {metrics['test_auc']:.4f}")
    axes[0, 1].plot([0, 1], [0, 1], "k--")
    axes[0, 1].set_xlabel("False Positive Rate")
    axes[0, 1].set_ylabel("True Positive Rate")
    axes[0, 1].set_title("ROC Curve")
    axes[0, 1].legend()

    # Confusion matrix heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=axes[1, 0],
        xticklabels=["Pred < 5", "Pred = 5"],
        yticklabels=["Actual < 5", "Actual = 5"],
    )
    axes[1, 0].set_title("Confusion Matrix")

    # Probability distribution
    axes[1, 1].hist(
        y_test_proba[y_test == 0], bins=50, alpha=0.6, label="Rating < 5", color="#e74c3c"
    )
    axes[1, 1].hist(
        y_test_proba[y_test == 1], bins=50, alpha=0.6, label="Rating = 5", color="#2ecc71"
    )
    axes[1, 1].axvline(0.5, color="black", linestyle="--", label="Threshold=0.5")
    axes[1, 1].set_xlabel("Predicted Probability")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].set_title("Prediction Probability Distribution")
    axes[1, 1].legend()

    plt.tight_layout()
    results_chart = charts_model_dir / "xgboost_rating_classifier_cleaned_5star_results.png"
    plt.savefig(results_chart, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✅ 可视化图表导出: {results_chart}")

    # Save model
    model_path = charts_model_dir / "xgboost_rating_classifier_cleaned_5star.json"
    model.save_model(str(model_path))
    print(f"  ✅ 模型已保存: {model_path}")


# =============================================================================
# 主流程
# =============================================================================
def main():
    df, train_path = load_train_dataset()
    (
        X_train,
        X_test,
        y_train,
        y_test,
        feature_cols,
    ) = prepare_dataset(df)

    model, metrics, y_test_true, y_test_pred, y_test_proba, cm = train_xgboost_classifier(
        X_train, X_test, y_train, y_test
    )

    export_results(model, feature_cols, metrics, y_test_true, y_test_proba, cm)

    print("\n" + "=" * 80)
    print("Final Metrics / 最终指标")
    print("=" * 80)
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    print("=" * 80)
    print("XGBoost Rating Classification (Rating=5 vs <5) Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()


