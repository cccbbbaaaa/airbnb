"""
XGBoost Rating Classification Model (listings_detailed_2 only)
XGBoost 评分分类模型（仅使用 listings_detailed_2 数据）

使用 listings_detailed_2 数据，分别测试阈值 4.8 和 4.9 的分类效果
Uses listings_detailed_2 data only, tests classification performance at thresholds 4.8 and 4.9
"""

import sys
import json
import ast
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
from rating_feature_engineering_merged import (
    clean_price_column,
    parse_percentage,
    parse_boolean,
    compute_amenity_features,
    compute_text_features,
    generate_text_embeddings,
    REFERENCE_DATE,
    CITY_CENTER_LAT,
    CITY_CENTER_LON,
    NLTK_AVAILABLE,
    SentimentIntensityAnalyzer,
)

setup_plotting()


def load_listings_detailed_2() -> pd.DataFrame:
    """
    仅加载 listings_detailed_2 数据文件
    Load only listings_detailed_2 data file
    """
    paths = get_project_paths()
    if len(paths) == 4:
        project_root, data_dir, _, _ = paths
    else:
        project_root, data_dir, _ = paths
    
    data_path = data_dir / "listings_detailed_2.xlsx"
    
    if not data_path.exists():
        raise FileNotFoundError(f"数据文件不存在 / Data file not found: {data_path}")
    
    print("  加载 listings_detailed_2.xlsx...")
    df = pd.read_excel(data_path)
    print(f"    [OK] 加载完成: {len(df)} 行 × {len(df.columns)} 列")
    
    return df


def engineer_features_listings2(df: pd.DataFrame) -> pd.DataFrame:
    """
    对 listings_detailed_2 数据进行特征工程
    Perform feature engineering on listings_detailed_2 data
    """
    print("\n开始特征工程 / Starting Feature Engineering...")
    
    # 基础清洗
    df = clean_price_column(df)
    
    # 处理百分比字段
    if "host_response_rate" in df.columns:
        df["host_response_rate"] = parse_percentage(df["host_response_rate"])
    if "host_acceptance_rate" in df.columns:
        df["host_acceptance_rate"] = parse_percentage(df["host_acceptance_rate"])
    
    # 处理布尔字段
    if "host_is_superhost" in df.columns:
        df["host_is_superhost_flag"] = parse_boolean(df["host_is_superhost"], ["t", "true"])
    else:
        df["host_is_superhost_flag"] = 0
    
    if "instant_bookable" in df.columns:
        df["instant_bookable_flag"] = parse_boolean(df["instant_bookable"], ["t", "true"])
    else:
        df["instant_bookable_flag"] = 0
    
    # 处理 license
    if "license" in df.columns:
        df["has_license_info"] = df["license"].notna().astype(int)
    else:
        df["has_license_info"] = 0
    
    # 处理价格相关特征
    if "accommodates" in df.columns:
        df["accommodates"] = pd.to_numeric(df["accommodates"], errors="coerce").fillna(1)
        df["price_per_person"] = df["price_clean"] / df["accommodates"].replace(0, 1)
        df["price_per_person"] = df["price_per_person"].replace([np.inf, -np.inf], 0).fillna(0)
    else:
        df["price_per_person"] = 0
    
    df["log_price"] = np.log1p(df["price_clean"])
    
    # 处理房间类型和属性类型编码
    if "room_type" in df.columns:
        df["room_type_encoded"] = pd.Categorical(df["room_type"]).codes
    else:
        df["room_type_encoded"] = 0
    
    if "property_type" in df.columns:
        df["property_type_encoded"] = pd.Categorical(df["property_type"]).codes
    else:
        df["property_type_encoded"] = 0
    
    if "neighbourhood_cleansed" in df.columns:
        df["neighbourhood_encoded"] = pd.Categorical(df["neighbourhood_cleansed"]).codes
    else:
        df["neighbourhood_encoded"] = 0
    
    # 处理数值字段
    numeric_cols = ["bedrooms", "beds", "number_of_reviews", "number_of_reviews_ltm", 
                     "reviews_per_month", "availability_30", "availability_60", 
                     "availability_90", "availability_365"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        else:
            df[col] = 0
    
    # 计算入住率和可用性比率
    if "availability_365" in df.columns and "number_of_reviews" in df.columns:
        df["occupancy_rate"] = 1 - (df["availability_365"] / 365).clip(0, 1)
        df["availability_ratio"] = df["availability_30"] / 30
    else:
        df["occupancy_rate"] = 0
        df["availability_ratio"] = 0
    
    # 处理 reviews_per_month
    if "reviews_per_month" in df.columns:
        df["log_reviews_per_month"] = np.log1p(df["reviews_per_month"])
    else:
        df["log_reviews_per_month"] = 0
    
    if "number_of_reviews" in df.columns:
        df["log_number_of_reviews"] = np.log1p(df["number_of_reviews"])
        if "number_of_reviews_ltm" in df.columns:
            df["reviews_growth_ratio"] = (
                df["number_of_reviews_ltm"] / (df["number_of_reviews"] + 1)
            ).fillna(0)
        else:
            df["reviews_growth_ratio"] = 0
    else:
        df["log_number_of_reviews"] = 0
        df["reviews_growth_ratio"] = 0
    
    # 处理主机相关特征
    host_cols = ["host_listings_count", "host_total_listings_count"]
    for col in host_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        else:
            df[col] = 0
    
    # 计算主机活动分数
    df["host_activity_score"] = (
        df["host_response_rate"] * 0.4 +
        df["host_acceptance_rate"] * 0.3 +
        df["host_is_superhost_flag"] * 0.3
    )
    
    # 处理日期特征
    date_cols = ["first_review", "last_review"]
    for col in date_cols:
        if col in df.columns:
            date_series = pd.to_datetime(df[col], errors="coerce")
            df[f"{col}_days_ago"] = (REFERENCE_DATE - date_series).dt.days.fillna(0)
            if col == "first_review":
                df["listing_age_days"] = df[f"{col}_days_ago"]
            if col == "last_review":
                df["days_since_last_review"] = df[f"{col}_days_ago"]
                df["recent_review_flag"] = (df[f"{col}_days_ago"] <= 90).astype(int)
                df["recent_review_score"] = np.exp(-df[f"{col}_days_ago"] / 365)
        else:
            if col == "first_review":
                df["listing_age_days"] = 0
            if col == "last_review":
                df["days_since_last_review"] = 0
                df["recent_review_flag"] = 0
                df["recent_review_score"] = 0
    
    # 处理地理位置特征
    if "latitude" in df.columns and "longitude" in df.columns:
        df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
        # 计算到市中心的距离（使用简单的欧氏距离近似）
        df["distance_to_center_km"] = np.sqrt(
            (df["latitude"] - CITY_CENTER_LAT) ** 2 +
            (df["longitude"] - CITY_CENTER_LON) ** 2
        ) * 111  # 粗略转换为公里
        df["is_central"] = (df["distance_to_center_km"] <= 5).astype(int)
    else:
        df["distance_to_center_km"] = 0
        df["is_central"] = 0
    
    # 计算 amenity 特征
    df = compute_amenity_features(df)
    df["amenity_comfort_score"] = (
        df["amenity_score_luxury"] * 0.3 +
        df["amenity_score_family"] * 0.2 +
        df["amenity_score_business"] * 0.2 +
        df["amenity_score_safety"] * 0.3
    )
    
    # 计算文本特征
    df = compute_text_features(df)
    
    # 生成文本嵌入
    text_embedding_cols = []
    for col, prefix in [("description", "desc"), ("neighborhood_overview", "neighborhood")]:
        cols = generate_text_embeddings(df, col, prefix, n_components=20)
        text_embedding_cols.extend(cols)
    
    # 情感分析
    if NLTK_AVAILABLE and SentimentIntensityAnalyzer is not None:
        print("  进行情感分析...")
        try:
            sia = SentimentIntensityAnalyzer()
            
            for col, prefix in [("description", "desc"), ("neighborhood_overview", "neighborhood"), ("host_about", "host_about")]:
                if col in df.columns:
                    texts = df[col].fillna("").astype(str).tolist()
                    sentiments = [sia.polarity_scores(text) for text in texts]
                    df[f"{prefix}_sentiment_compound"] = [s["compound"] for s in sentiments]
                else:
                    df[f"{prefix}_sentiment_compound"] = 0
            
            df["text_sentiment_score"] = (
                df["desc_sentiment_compound"] * 0.5 + 
                df["neighborhood_sentiment_compound"] * 0.3 + 
                df["host_about_sentiment_compound"] * 0.2
            )
        except Exception as e:
            print(f"  警告: 情感分析失败: {e}")
            df["desc_sentiment_compound"] = 0
            df["neighborhood_sentiment_compound"] = 0
            df["host_about_sentiment_compound"] = 0
            df["text_sentiment_score"] = 0
    else:
        print("  跳过情感分析（nltk 不可用）/ Skipping sentiment analysis (nltk not available)")
        df["desc_sentiment_compound"] = 0
        df["neighborhood_sentiment_compound"] = 0
        df["host_about_sentiment_compound"] = 0
        df["text_sentiment_score"] = 0
    
    # 处理 host_verifications
    if "host_verifications" in df.columns:
        def count_verifications(x):
            try:
                if pd.isna(x):
                    return 0
                if isinstance(x, str):
                    if x.startswith('['):
                        return len(ast.literal_eval(x))
                    elif x.startswith('{'):
                        parsed = json.loads(x)
                        return len(parsed) if isinstance(parsed, (list, dict)) else 0
                return 0
            except:
                return 0
        
        df["host_verifications_count"] = df["host_verifications"].apply(count_verifications)
    else:
        df["host_verifications_count"] = 0
    
    print(f"  [OK] 特征工程完成: {len(df)} 行 × {len(df.columns)} 列")
    
    return df


def train_rating_classifier_listings2(threshold: float = 4.8):
    """
    训练评分分类模型（仅使用 listings_detailed_2）
    Train rating classification model (listings_detailed_2 only)
    
    Args:
        threshold: 分类阈值，默认4.8
    """
    print("=" * 80)
    print(f"XGBoost Rating Classification Model (listings_detailed_2 only - Threshold {threshold})")
    print(f"XGBoost 评分分类模型（仅使用 listings_detailed_2 - 阈值 {threshold}）")
    print("=" * 80)

    # =========================================================================
    # 1. 加载数据 / Load Data
    # =========================================================================
    print("\n1. 加载数据 / Loading Data...")
    try:
        df = load_listings_detailed_2()
    except Exception as e:
        print(f"  [ERROR] 数据加载失败 / Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    # =========================================================================
    # 2. 特征工程 / Feature Engineering
    # =========================================================================
    print("\n2. 特征工程 / Feature Engineering...")
    try:
        processed_df = engineer_features_listings2(df)
    except Exception as e:
        print(f"  [ERROR] 特征工程失败 / Feature engineering failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    # =========================================================================
    # 3. 准备数据 / Prepare Data
    # =========================================================================
    print("\n3. 准备数据 / Preparing Data...")
    target_col = "review_scores_rating"
    if target_col not in processed_df.columns:
        print(f"  [ERROR] 目标列 '{target_col}' 不在数据中！")
        return None
    
    # 创建二分类标签（根据阈值）
    processed_df['is_high_rating'] = (processed_df[target_col] > threshold).astype(int)
    
    # 只保留有评分的记录
    mask = processed_df[target_col].notna() & (processed_df[target_col] >= 0) & (processed_df[target_col] <= 100)
    processed_df = processed_df[mask].copy()
    
    print(f"  有效数据: {len(processed_df)} 条记录")
    print(f"  高分样本数（评分>{threshold}）: {processed_df['is_high_rating'].sum()} ({processed_df['is_high_rating'].mean():.2%})")
    print(f"  低分样本数（评分≤{threshold}）: {(~processed_df['is_high_rating'].astype(bool)).sum()} ({(1-processed_df['is_high_rating'].mean()):.2%})")
    
    # 排除目标列
    feature_cols = [col for col in processed_df.columns 
                   if col not in [target_col, 'is_high_rating']]

    X = processed_df[feature_cols].copy()
    y = processed_df['is_high_rating'].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y  # 使用 stratify 保持类别比例
    )
    print(f"  训练集: {len(X_train)} 条 (高分占比: {y_train.mean():.2%})")
    print(f"  测试集: {len(X_test)} 条 (高分占比: {y_test.mean():.2%})")

    # =========================================================================
    # 4. 训练模型 / Train Model
    # =========================================================================
    print("\n4. 训练 XGBoost 分类器 / Training XGBoost Classifier...")
    
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
        "scale_pos_weight": scale_pos_weight,
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
    # 5. 模型评估 / Evaluation
    # =========================================================================
    print("\n5. 模型评估 / Model Evaluation...")
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

    print(f"\n测试集分类报告 / Test Set Classification Report (Threshold {threshold}):")
    print(classification_report(y_test, y_test_pred, 
                                target_names=[f'Low Rating (≤{threshold})', f'High Rating (>{threshold})']))

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_test_pred)
    print("\n混淆矩阵 / Confusion Matrix:")
    print(f"  True Negatives (TN): {cm[0, 0]}")
    print(f"  False Positives (FP): {cm[0, 1]}")
    print(f"  False Negatives (FN): {cm[1, 0]}")
    print(f"  True Positives (TP): {cm[1, 1]}")

    # =========================================================================
    # 6. 特征重要性 / Feature Importance
    # =========================================================================
    print("\n6. 特征重要性分析 / Feature Importance Analysis...")
    project_root, _, charts_eda_dir, charts_model_dir = get_project_paths()
    charts_dir = charts_model_dir
    
    importance_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    
    print(f"\n前20个最重要特征 / Top 20 Most Important Features (Threshold {threshold}):")
    print(importance_df.head(20).to_string(index=False))
    
    output_importance = charts_dir / f"xgboost_rating_classifier_listings2_{threshold}_importance.csv"
    importance_df.to_csv(output_importance, index=False)
    print(f"\n  [OK] 特征重要性已保存到: {output_importance}")

    # =========================================================================
    # 7. 可视化结果 / Visualization
    # =========================================================================
    print("\n7. 生成可视化图表 / Generating Visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 7.1 特征重要性Top 20
    top_features = importance_df.head(20)
    axes[0, 0].barh(range(len(top_features)), top_features['importance'].values)
    axes[0, 0].set_yticks(range(len(top_features)))
    axes[0, 0].set_yticklabels(top_features['feature'].values)
    axes[0, 0].set_xlabel('Importance')
    axes[0, 0].set_title(f'Top 20 Feature Importance (listings_detailed_2 - Threshold {threshold})')
    axes[0, 0].invert_yaxis()

    # 7.2 ROC曲线（使用测试集）
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
    axes[0, 1].plot(fpr, tpr, label=f'ROC Curve (AUC = {test_auc:.4f})')
    axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random')
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_title('ROC Curve')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # 7.3 混淆矩阵热力图
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
                xticklabels=[f'Low (≤{threshold})', f'High (>{threshold})'],
                yticklabels=[f'Low (≤{threshold})', f'High (>{threshold})'])
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('Actual')
    axes[1, 0].set_title('Confusion Matrix')

    # 7.4 预测概率分布
    axes[1, 1].hist(y_test_proba[y_test == 0], bins=50, alpha=0.5, 
                   label=f'Low Rating (≤{threshold})', color='red')
    axes[1, 1].hist(y_test_proba[y_test == 1], bins=50, alpha=0.5, 
                   label=f'High Rating (>{threshold})', color='green')
    axes[1, 1].set_xlabel('Predicted Probability')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Prediction Probability Distribution')
    axes[1, 1].legend()
    axes[1, 1].axvline(x=0.5, color='black', linestyle='--', lw=2, label='Threshold (0.5)')
    axes[1, 1].legend()

    plt.tight_layout()
    output_chart = charts_dir / f'xgboost_rating_classifier_listings2_{threshold}_results.png'
    plt.savefig(output_chart, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  [OK] 可视化图表已保存到: {output_chart}")

    # =========================================================================
    # 8. 保存模型 / Save Model
    # =========================================================================
    print("\n8. 保存模型 / Saving Model...")
    model_path = charts_dir / f"xgboost_rating_classifier_listings2_{threshold}.json"
    try:
        model.save_model(str(model_path))
        print(f"  [OK] 模型已保存到: {model_path}")
    except Exception:
        import pickle
        pkl_path = charts_dir / f"xgboost_rating_classifier_listings2_{threshold}.pkl"
        with open(pkl_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"  [OK] 模型已保存到 (fallback): {pkl_path}")

    return {
        "threshold": threshold,
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
    """主函数：分别测试阈值4.8和4.9 / Main function: test thresholds 4.8 and 4.9"""
    print("=" * 80)
    print("开始测试不同阈值的分类效果 / Starting classification tests with different thresholds")
    print("=" * 80)
    
    results = {}
    
    # 测试阈值 4.8
    print("\n" + "=" * 80)
    print("测试阈值 4.8 / Testing Threshold 4.8")
    print("=" * 80)
    results[4.8] = train_rating_classifier_listings2(threshold=4.8)
    
    # 测试阈值 4.9
    print("\n" + "=" * 80)
    print("测试阈值 4.9 / Testing Threshold 4.9")
    print("=" * 80)
    results[4.9] = train_rating_classifier_listings2(threshold=4.9)
    
    # 对比结果
    print("\n" + "=" * 80)
    print("阈值对比结果 / Threshold Comparison Results")
    print("=" * 80)
    
    comparison_data = []
    for threshold, metrics in results.items():
        if metrics:
            comparison_data.append({
                'Threshold': threshold,
                'Train_Accuracy': metrics['train_accuracy'],
                'Test_Accuracy': metrics['test_accuracy'],
                'Train_Precision': metrics['train_precision'],
                'Test_Precision': metrics['test_precision'],
                'Train_Recall': metrics['train_recall'],
                'Test_Recall': metrics['test_recall'],
                'Train_F1': metrics['train_f1'],
                'Test_F1': metrics['test_f1'],
                'Train_AUC': metrics['train_auc'],
                'Test_AUC': metrics['test_auc'],
            })
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        print("\n性能对比表 / Performance Comparison:")
        print(comparison_df.to_string(index=False))
        
        # 保存对比结果
        project_root, _, charts_eda_dir, charts_model_dir = get_project_paths()
        charts_dir = charts_model_dir
        comparison_path = charts_dir / "xgboost_rating_classifier_listings2_threshold_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)
        print(f"\n  [OK] 对比结果已保存到: {comparison_path}")
    
    print("\n" + "=" * 80)
    print("所有测试完成 / All tests complete")
    print("=" * 80)


if __name__ == "__main__":
    main()

