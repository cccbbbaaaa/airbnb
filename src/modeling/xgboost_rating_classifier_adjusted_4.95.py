"""
XGBoost Rating Classification Model (Adjusted 2021 Data) - Threshold 4.95
XGBoost 评分分类模型（使用调整后的2021年数据 - 阈值4.95）

参考 Rating_distribution.py 中的评分分布调整方法，调整2021年数据使其分布与2025年相似
然后整合调整后的2021年数据和2025年数据，使用XGBoost构建分类模型预测评分>4.95

References Rating_distribution.py for rating distribution adjustment methods, adjusts 2021 data 
to match 2025 distribution, then merges adjusted 2021 data with 2025 data to build XGBoost 
classification model predicting rating > 4.95
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
    roc_curve,
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 添加 EDA 目录到路径 / Add EDA directory to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "EDA"))

from utils import setup_plotting, get_project_paths

# 添加特征工程目录 / Add feature engineering directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "feature_engineering"))
from rating_feature_engineering_merged import engineer_rating_features_merged

setup_plotting()


def quantile_matching(source_data, target_data):
    """
    分位数匹配：将源数据的分位数映射到目标数据的分位数
    Quantile Matching: Map quantiles of source data to target data quantiles
    """
    source_sorted = np.sort(source_data)
    target_sorted = np.sort(target_data)
    
    n_source = len(source_data)
    n_target = len(target_data)
    
    adjusted = np.zeros_like(source_data)
    for i, val in enumerate(source_data):
        quantile = np.searchsorted(source_sorted, val, side='right') / n_source
        target_idx = int(quantile * (n_target - 1))
        target_idx = min(target_idx, n_target - 1)
        adjusted[i] = target_sorted[target_idx]
    
    return adjusted


def load_and_adjust_data():
    """
    加载数据并应用调整后的评分，返回合并后的DataFrame
    Load data and apply adjusted ratings, return merged DataFrame
    """
    print("=" * 80)
    print("Loading and Adjusting Data")
    print("加载并调整数据")
    print("=" * 80)
    
    project_root, data_dir, charts_eda_dir, charts_model_dir = get_project_paths()
    
    # 1. 加载原始数据文件
    print("\n1. 加载原始数据文件 / Loading Original Data Files...")
    listings_2021 = pd.read_excel(data_dir / 'listings_detailed.xlsx')
    listings_2025 = pd.read_excel(data_dir / 'listings_detailed_2.xlsx')
    
    print(f"  ✅ listings_detailed.xlsx (2021): {len(listings_2021):,} 行 × {len(listings_2021.columns)} 列")
    print(f"  ✅ listings_detailed_2.xlsx (2025): {len(listings_2025):,} 行 × {len(listings_2025.columns)} 列")
    
    # 2. 提取评分数据
    print("\n2. 提取评分数据 / Extracting Rating Data...")
    rating_2021 = listings_2021['review_scores_rating'].dropna()
    rating_2025 = listings_2025['review_scores_rating'].dropna()
    
    print(f"  - 2021年有效评分数量: {len(rating_2021):,}")
    print(f"  - 2025年有效评分数量: {len(rating_2025):,}")
    
    # 3. 调整2021年评分分布（使用分位数匹配方法）
    print("\n3. 调整2021年评分分布 / Adjusting 2021 Rating Distribution...")
    print("  使用方法: 分位数匹配 / Method: Quantile Matching")
    
    rating_2021_adjusted = quantile_matching(rating_2021.values, rating_2025.values)
    
    # 统计调整效果
    stats_2021_orig = rating_2021.describe()
    stats_2021_adj = pd.Series(rating_2021_adjusted).describe()
    stats_2025 = rating_2025.describe()
    
    print(f"  - 原始2021年评分均值: {stats_2021_orig['mean']:.2f}")
    print(f"  - 调整后2021年评分均值: {stats_2021_adj['mean']:.2f} (目标: {stats_2025['mean']:.2f})")
    print(f"  - 原始2021年评分中位数: {stats_2021_orig['50%']:.2f}")
    print(f"  - 调整后2021年评分中位数: {stats_2021_adj['50%']:.2f} (目标: {stats_2025['50%']:.2f})")
    
    # 4. 应用调整后的评分到2021年数据
    print("\n4. 应用调整后的评分 / Applying Adjusted Ratings...")
    mask_2021 = listings_2021['review_scores_rating'].notna()
    rating_indices = listings_2021[mask_2021].index
    
    # 按顺序匹配调整后的评分
    min_len = min(len(rating_indices), len(rating_2021_adjusted))
    listings_2021.loc[rating_indices[:min_len], 'review_scores_rating'] = \
        rating_2021_adjusted[:min_len]
    
    print(f"  ✅ 已调整 {min_len:,} 条记录的评分")
    
    # 5. 合并数据
    print("\n5. 合并数据 / Merging Data...")
    cols_2021 = set(listings_2021.columns)
    cols_2025 = set(listings_2025.columns)
    common_cols = cols_2021 & cols_2025
    
    # 添加数据源标识
    listings_2021['data_source'] = 'listings_detailed_adjusted'
    listings_2025['data_source'] = 'listings_detailed_2'
    
    # 合并
    all_cols = list(common_cols) + ['data_source']
    merged_df = pd.concat([
        listings_2021[all_cols],
        listings_2025[all_cols]
    ], ignore_index=True)
    
    print(f"  ✅ 数据合并完成: {len(merged_df):,} 行 × {len(merged_df.columns)} 列")
    print(f"  - 来源1 (调整后的2021年数据): {(merged_df['data_source'] == 'listings_detailed_adjusted').sum():,} 行")
    print(f"  - 来源2 (2025年数据): {(merged_df['data_source'] == 'listings_detailed_2').sum():,} 行")
    
    return merged_df


def engineer_features_with_adjusted_data(merged_df: pd.DataFrame):
    """
    使用调整后的数据进行特征工程
    直接复制特征工程函数的逻辑，但使用传入的DataFrame而不是从文件加载
    Perform feature engineering with adjusted data by copying feature engineering logic
    """
    # 导入特征工程函数中需要的函数
    from rating_feature_engineering_merged import (
        clean_price_column,
        parse_percentage,
        parse_boolean,
        compute_amenity_features,
        compute_text_features,
        generate_text_embeddings,
    )
    
    # 导入常量
    from rating_feature_engineering_merged import REFERENCE_DATE, CITY_CENTER_LAT, CITY_CENTER_LON, LUXURY_AMENITIES
    
    # 导入NLTK相关（如果可用）
    try:
        from rating_feature_engineering_merged import NLTK_AVAILABLE, SentimentIntensityAnalyzer
    except ImportError:
        NLTK_AVAILABLE = False
        SentimentIntensityAnalyzer = None
    
    import json
    import ast
    import re
    
    print("\n开始特征工程 / Starting Feature Engineering...")
    df = merged_df.copy()
    
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
        df["distance_to_center_km"] = np.sqrt(
            (df["latitude"] - CITY_CENTER_LAT) ** 2 +
            (df["longitude"] - CITY_CENTER_LON) ** 2
        ) * 111
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
                    if x.strip().startswith("["):
                        return len(ast.literal_eval(x))
                    elif x.strip().startswith("{"):
                        parsed = json.loads(x)
                        return len(parsed) if isinstance(parsed, (list, dict)) else 0
                return 0
            except:
                return 0
        df["host_verifications_count"] = df["host_verifications"].apply(count_verifications)
        df["host_has_gov_id"] = df["host_verifications"].fillna("").astype(str).str.contains(
            "government", case=False, na=False
        ).astype(int)
    else:
        df["host_verifications_count"] = 0
        df["host_has_gov_id"] = 0
    
    # 处理 bathrooms_text
    if 'bathrooms_text' in df.columns:
        df['bathrooms_numeric'] = df['bathrooms_text'].str.extract(r'(\d+\.?\d*)').astype(float)
        df['bathrooms_numeric'] = df['bathrooms_numeric'].fillna(0)
        df['is_shared_bath'] = df['bathrooms_text'].str.contains('shared', case=False, na=False).astype(int)
    else:
        df['bathrooms_numeric'] = 0
        df['is_shared_bath'] = 0
    
    # 选择用于建模的字段（参考特征工程函数）
    feature_columns = [
        "review_scores_rating",
        "price_clean", "log_price", "price_per_person",
        "occupancy_rate", "availability_ratio",
        "reviews_per_month", "log_reviews_per_month",
        "number_of_reviews", "log_number_of_reviews", "number_of_reviews_ltm", "reviews_growth_ratio",
        "host_response_rate", "host_acceptance_rate", "host_activity_score",
        "host_is_superhost_flag",
        "host_listings_count", "host_total_listings_count",
        "instant_bookable_flag", "has_license_info",
        "room_type_encoded", "property_type_encoded", "neighbourhood_encoded",
        "accommodates", "bedrooms", "beds", "bathrooms_numeric", "is_shared_bath",
        "amenities_count", "amenity_comfort_score",
        "amenity_score_luxury", "amenity_score_family", "amenity_score_business", "amenity_score_safety",
        "description_length", "neighborhood_desc_length", "host_about_length",
        "listing_age_days", "days_since_last_review", "recent_review_flag", "recent_review_score",
        "distance_to_center_km", "is_central",
        "host_verifications_count", "host_has_gov_id",
        "desc_sentiment_compound", "neighborhood_sentiment_compound", "host_about_sentiment_compound",
        "text_sentiment_score",
        "data_source",
    ]
    
    # 添加关键设施特征列
    amenity_flag_cols = [col for col in df.columns if col.startswith("amenity_has_")]
    feature_columns.extend(amenity_flag_cols)
    feature_columns.extend(text_embedding_cols)
    
    # 确保所有列都存在
    final_feature_columns = []
    for col in feature_columns:
        if col in df.columns:
            final_feature_columns.append(col)
    
    # 去重
    final_feature_columns = sorted(list(set(final_feature_columns)))
    
    processed_df = df[final_feature_columns].copy()
    
    return processed_df


def train_adjusted_rating_classifier_4_95():
    """训练使用调整后数据的评分分类模型（阈值4.95）/ Train rating classification model with adjusted data (threshold 4.95)."""
    print("=" * 80)
    print("XGBoost Rating Classification Model (Adjusted 2021 Data - Threshold 4.95)")
    print("XGBoost 评分分类模型（使用调整后的2021年数据 - 预测高分评分>4.95）")
    print("=" * 80)

    # =========================================================================
    # 1. 加载并调整数据 / Load and Adjust Data
    # =========================================================================
    merged_df = load_and_adjust_data()
    
    # =========================================================================
    # 2. 构建特征 / Build Features
    # =========================================================================
    print("\n" + "=" * 80)
    print("Building Features")
    print("构建特征")
    print("=" * 80)
    
    try:
        processed_df = engineer_features_with_adjusted_data(merged_df)
        print(f"  [OK] 特征构建完成: {len(processed_df)} 行 × {processed_df.shape[1]} 列")
    except Exception as e:
        print(f"  [ERROR] 特征构建失败 / Feature engineering failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # =========================================================================
    # 3. 准备数据 / Prepare Data
    # =========================================================================
    print("\n" + "=" * 80)
    print("Preparing Data")
    print("准备数据")
    print("=" * 80)
    
    target_col = "review_scores_rating"
    if target_col not in processed_df.columns:
        raise ValueError(f"目标列 '{target_col}' 不在数据中！")
    
    # 创建二分类标签（评分>4.95为高分）
    processed_df['is_high_rating'] = (processed_df[target_col] > 4.95).astype(int)
    
    # 只保留有评分的记录
    mask = processed_df[target_col].notna() & (processed_df[target_col] >= 0) & (processed_df[target_col] <= 100)
    processed_df = processed_df[mask].copy()
    
    print(f"  有效数据: {len(processed_df):,} 条记录")
    print(f"  高分样本数（评分>4.95）: {processed_df['is_high_rating'].sum():,} ({processed_df['is_high_rating'].mean():.2%})")
    print(f"  低分样本数（评分≤4.95）: {(~processed_df['is_high_rating'].astype(bool)).sum():,} ({(1-processed_df['is_high_rating'].mean()):.2%})")
    
    # 排除目标列、数据源列和其他不需要的特征
    exclude_from_features = [
        target_col, 'is_high_rating', 'data_source', 'scrape_id',
        'review_scores_accuracy', 'review_scores_cleanliness', 
        'review_scores_checkin', 'review_scores_communication',
        'review_scores_location', 'review_scores_value'
    ]
    feature_cols = [col for col in processed_df.columns 
                   if col not in exclude_from_features]

    X = processed_df[feature_cols].copy()
    y = processed_df['is_high_rating'].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  训练集: {len(X_train):,} 条 (高分占比: {y_train.mean():.2%})")
    print(f"  测试集: {len(X_test):,} 条 (高分占比: {y_test.mean():.2%})")

    # =========================================================================
    # 4. 训练模型 / Train Model
    # =========================================================================
    print("\n" + "=" * 80)
    print("Training XGBoost Classifier")
    print("训练 XGBoost 分类器")
    print("=" * 80)
    
    # 处理类别不平衡问题
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
    print("\n" + "=" * 80)
    print("Model Evaluation")
    print("模型评估")
    print("=" * 80)
    
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
    print(classification_report(y_test, y_test_pred, target_names=['Low Rating (≤4.95)', 'High Rating (>4.95)']))

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
    print("\n" + "=" * 80)
    print("Feature Importance Analysis")
    print("特征重要性分析")
    print("=" * 80)
    
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
    
    output_importance = charts_dir / "xgboost_rating_classifier_adjusted_4.95_importance.csv"
    importance_df.to_csv(output_importance, index=False)
    print(f"\n  [OK] 特征重要性已保存到: {output_importance}")

    # =========================================================================
    # 7. 可视化结果 / Visualization
    # =========================================================================
    print("\n" + "=" * 80)
    print("Generating Visualizations")
    print("生成可视化图表")
    print("=" * 80)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 7.1 特征重要性Top 20
    top_features = importance_df.head(20)
    axes[0, 0].barh(range(len(top_features)), top_features['importance'].values)
    axes[0, 0].set_yticks(range(len(top_features)))
    axes[0, 0].set_yticklabels(top_features['feature'].values)
    axes[0, 0].set_xlabel('Importance')
    axes[0, 0].set_title('Top 20 Feature Importance (Adjusted Data - Classification >4.95)')
    axes[0, 0].invert_yaxis()

    # 7.2 ROC曲线
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
                xticklabels=['Low Rating (≤4.95)', 'High Rating (>4.95)'],
                yticklabels=['Low Rating (≤4.95)', 'High Rating (>4.95)'])
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('Actual')
    axes[1, 0].set_title('Confusion Matrix')

    # 7.4 预测概率分布
    axes[1, 1].hist(y_test_proba[y_test == 0], bins=50, alpha=0.5, label='Low Rating (≤4.95)', color='red')
    axes[1, 1].hist(y_test_proba[y_test == 1], bins=50, alpha=0.5, label='High Rating (>4.95)', color='green')
    axes[1, 1].set_xlabel('Predicted Probability')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Prediction Probability Distribution')
    axes[1, 1].legend()
    axes[1, 1].axvline(x=0.5, color='black', linestyle='--', lw=2, label='Threshold (0.5)')
    axes[1, 1].legend()

    plt.tight_layout()
    output_chart = charts_dir / 'xgboost_rating_classifier_adjusted_4.95_results.png'
    plt.savefig(output_chart, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  [OK] 可视化图表已保存到: {output_chart}")

    # =========================================================================
    # 8. 保存模型 / Save Model
    # =========================================================================
    print("\n" + "=" * 80)
    print("Saving Model")
    print("保存模型")
    print("=" * 80)
    
    model_path = charts_dir / "xgboost_rating_classifier_adjusted_4.95.json"
    try:
        model.save_model(str(model_path))
        print(f"  [OK] 模型已保存到: {model_path}")
    except Exception:
        import pickle
        pkl_path = charts_dir / "xgboost_rating_classifier_adjusted_4.95.pkl"
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
    try:
        metrics = train_adjusted_rating_classifier_4_95()
        if metrics is None:
            print("\n[ERROR] 模型训练失败，未返回指标")
            return
        
        print("\n" + "=" * 80)
        print("Final Metrics")
        print("最终指标")
        print("=" * 80)
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
        print("=" * 80)
        print("使用调整后数据的评分分类模型完成（阈值4.95）/ Adjusted data rating classification model training complete (threshold 4.95)")
        print("=" * 80)
    except Exception as e:
        print(f"\n[ERROR] 程序执行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

