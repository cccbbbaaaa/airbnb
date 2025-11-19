"""
评分预测特征工程脚本 v2 / Rating Prediction Feature Engineering Script v2

在 v1 版本基础上增加以下特征：
- 主机验证信息 (host_verifications)：验证数量、是否包含政府ID等
- 文本情感分析 (sentiment analysis)：对 description, neighborhood_overview 进行情感打分

Adds the following features on top of v1:
- Host verification info (number of verifications, has government ID, etc.)
- Text sentiment analysis (sentiment scores for description, neighborhood_overview)
"""

import json
import ast
import re
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# 添加 EDA 目录到路径 / Add EDA directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "EDA"))

from utils import setup_plotting, get_project_paths
from feature_registry import write_feature_registry, register_feature
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# 下载 VADER 词典 / Download VADER lexicon
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    print("  下载 VADER 情感分析词典 / Downloading VADER sentiment analysis lexicon...")
    nltk.download('vader_lexicon')


# 启用绘图风格（如后续需要可视化）/ Setup plotting style (optional)
setup_plotting()

# 常量 / Constants
REFERENCE_DATE = pd.Timestamp("2021-09-07")
CITY_CENTER_LAT = 52.3676
CITY_CENTER_LON = 4.9041

LUXURY_AMENITIES = {
    "luxury": [
        "hot tub", "pool", "gym", "sauna", "spa", "game console", "fireplace",
        "rooftop", "sonos", "sound system", "cinema", "home theater"
    ],
    "family": [
        "crib", "high chair", "children", "babysitter", "baby monitor",
        "changing table", "children’s books", "playground"
    ],
    "business": [
        "workspace", "desk", "printer", "wifi", "ethernet", "monitor",
        "scanner", "office"
    ],
    "safety": [
        "smoke detector", "carbon monoxide detector", "first aid kit",
        "fire extinguisher"
    ],
}


def load_listings_data() -> pd.DataFrame:
    """加载 listings_detailed 数据 / Load listings_detailed data."""
    project_root, data_dir, _ = get_project_paths()
    data_path = data_dir / "listings_detailed.xlsx"
    if not data_path.exists():
        raise FileNotFoundError(f"数据文件不存在 / Data file not found: {data_path}")
    df = pd.read_excel(data_path)
    return df


def clean_price_column(df: pd.DataFrame) -> pd.DataFrame:
    """清洗价格字段 / Clean price column."""
    if "price" in df.columns:
        if df["price"].dtype == "object":
            df["price_clean"] = (
                df["price"].astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False)
            )
            df["price_clean"] = pd.to_numeric(df["price_clean"], errors="coerce")
        else:
            df["price_clean"] = pd.to_numeric(df["price"], errors="coerce")
        df["price_clean"] = df["price_clean"].fillna(0)
    else:
        df["price_clean"] = 0
    return df


def parse_percentage(series: pd.Series) -> pd.Series:
    """解析百分比字符串 / Parse percentage strings."""
    def _convert(x):
        if pd.isna(x):
            return 0.0
        if isinstance(x, str):
            x = x.strip()
            if x.endswith("%"):
                x = x[:-1]
        try:
            return float(x) / 100 if float(x) > 1 else float(x)
        except (ValueError, TypeError):
            return 0.0

    return series.apply(_convert)


def parse_boolean(series: pd.Series, true_values: List[str]) -> pd.Series:
    """解析布尔字段 / Parse boolean columns."""
    return series.fillna("").astype(str).str.lower().isin(true_values).astype(int)


def parse_amenities(amenities_value) -> List[str]:
    """解析 amenities 字段，返回小写列表 / Parse amenities into lowercase list."""
    if pd.isna(amenities_value) or amenities_value == "":
        return []
    try:
        if isinstance(amenities_value, str) and amenities_value.strip().startswith("["):
            parsed = ast.literal_eval(amenities_value)
        else:
            parsed = json.loads(amenities_value)
        if isinstance(parsed, list):
            return [str(item).strip().lower() for item in parsed]
        if isinstance(parsed, dict):
            return [str(item).strip().lower() for item in parsed.keys()]
    except (ValueError, SyntaxError, json.JSONDecodeError):
        cleaned = (
            str(amenities_value)
            .strip()
            .strip("[]")
            .replace("'", "")
            .replace('"', "")
        )
        if cleaned:
            return [item.strip().lower() for item in cleaned.split(",") if item.strip()]
    return []


def compute_amenity_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算 amenity 相关特征 / Compute amenity-related features."""
    df["amenities_parsed"] = df["amenities"].apply(parse_amenities)
    df["amenities_count"] = df["amenities_parsed"].apply(len)

    for category, keywords in LUXURY_AMENITIES.items():
        df[f"amenity_score_{category}"] = df["amenities_parsed"].apply(
            lambda ams, kws=keywords: sum(1 for a in ams if any(k in a for k in kws))
        )

    # 是否包含高频关键设施 / Flag for top amenities
    TOP_KEY_AMENITIES = [
        "wifi", "kitchen", "heating", "air conditioning", "washer", "dryer",
        "tv", "dishwasher", "parking", "balcony", "elevator", "coffee maker",
        "pets allowed", "long term stays allowed"
    ]
    for amenity in TOP_KEY_AMENITIES:
        col_name = f"amenity_has_{re.sub(r'[^a-z0-9]+', '_', amenity)}"
        df[col_name] = df["amenities_parsed"].apply(lambda ams, a=amenity: int(a in ams))

    return df


def compute_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """构建文本长度等特征 / Build text-based features."""
    text_fields = {
        "description": "description_length",
        "neighborhood_overview": "neighborhood_desc_length",
        "host_about": "host_about_length",
    }
    for col, new_col in text_fields.items():
        if col in df.columns:
            df[new_col] = df[col].fillna("").astype(str).str.len()
        else:
            df[new_col] = 0
    return df


def generate_text_embeddings(
    df: pd.DataFrame,
    column: str,
    prefix: str,
    n_components: int = 20,
) -> List[str]:
    """
    使用 TF-IDF 和 SVD 生成文本嵌入
    Generate text embeddings using TF-IDF and SVD
    
    Args:
        df: DataFrame
        column: 文本列名 / Text column name
        prefix: 特征前缀 / Feature prefix
        n_components: 降维后的维度 / Reduced dimensions
    """
    if column not in df.columns:
        return []

    texts = df[column].fillna("").astype(str).tolist()
    if all(not t.strip() for t in texts):
        return []

    try:
        print(f"  使用 TF-IDF 生成 {column} 的嵌入...")
        vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words="english",
        )
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        # 确保 n_components 不大于特征数量
        max_components = min(n_components, tfidf_matrix.shape[1])
        if max_components == 0:
            return []

        svd = TruncatedSVD(n_components=max_components, random_state=42)
        embeddings = svd.fit_transform(tfidf_matrix)

        col_names = []
        for idx in range(max_components):
            col_name = f"{prefix}_embed_{idx + 1}"
            df[col_name] = embeddings[:, idx]
            col_names.append(col_name)
            register_feature(
                {
                    "name": col_name,
                    "description": f"{column} 文本嵌入第 {idx + 1} 维（TF-IDF方法）",
                    "feature_type": "衍生字段",
                    "logic": f"对 {column} 使用 TF-IDF（max_features=5000）后经 TruncatedSVD（n_components={max_components}）得到的第 {idx + 1} 维",
                    "example": f"{embeddings[0, idx]:.4f}",
                    "source": "listings_detailed.xlsx",
                }
            )
        return col_names
    except Exception as e:
        print(f"  错误: 生成 TF-IDF 嵌入时出错: {e}")
        return []


def compute_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """构建时间相关特征 / Build time-related features."""
    if "last_review" in df.columns:
        df["last_review_dt"] = pd.to_datetime(df["last_review"], errors="coerce")
        df["days_since_last_review"] = (REFERENCE_DATE - df["last_review_dt"]).dt.days.fillna(9999)
        df["recent_review_flag"] = (df["days_since_last_review"] <= 30).astype(int)
    else:
        df["days_since_last_review"] = 9999
        df["recent_review_flag"] = 0

    if "first_review" in df.columns:
        df["first_review_dt"] = pd.to_datetime(df["first_review"], errors="coerce")
        df["listing_age_days"] = (REFERENCE_DATE - df["first_review_dt"]).dt.days.fillna(0)
    else:
        df["listing_age_days"] = 0

    if "host_since" in df.columns:
        df["host_since_dt"] = pd.to_datetime(df["host_since"], errors="coerce")
        df["host_experience_years"] = ((REFERENCE_DATE - df["host_since_dt"]).dt.days / 365).fillna(0)
    else:
        df["host_experience_years"] = 0

    return df


def compute_location_features(df: pd.DataFrame) -> pd.DataFrame:
    """位置特征 / Location features."""
    if {"latitude", "longitude"}.issubset(df.columns):
        lat_rad = np.radians(df["latitude"].astype(float))
        lon_rad = np.radians(df["longitude"].astype(float))
        center_lat = np.radians(CITY_CENTER_LAT)
        center_lon = np.radians(CITY_CENTER_LON)
        dlat = lat_rad - center_lat
        dlon = lon_rad - center_lon
        a = np.sin(dlat / 2) ** 2 + np.cos(center_lat) * np.cos(lat_rad) * np.sin(dlon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        earth_radius_km = 6371
        df["distance_to_center_km"] = earth_radius_km * c
        df["is_central"] = (df["distance_to_center_km"] <= 4).astype(int)
    else:
        df["distance_to_center_km"] = np.nan
        df["is_central"] = 0
    return df


def compute_numeric_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """构建数值比例特征 / Build numeric ratio features."""
    df = clean_price_column(df)
    df["accommodates"] = pd.to_numeric(df["accommodates"], errors="coerce").replace(0, np.nan)
    df["beds"] = pd.to_numeric(df["beds"], errors="coerce")
    df["bedrooms"] = pd.to_numeric(df["bedrooms"], errors="coerce")

    df["price_per_person"] = (df["price_clean"] / df["accommodates"]).replace([np.inf, -np.inf], np.nan)
    df["price_per_person"] = df["price_per_person"].fillna(df["price_clean"])
    df["log_price"] = np.log1p(df["price_clean"])

    df["availability_365"] = pd.to_numeric(df["availability_365"], errors="coerce").fillna(0)
    df["availability_ratio"] = df["availability_365"] / 365.0
    df["occupancy_rate"] = 1 - df["availability_ratio"]

    df["reviews_per_month"] = pd.to_numeric(df["reviews_per_month"], errors="coerce").fillna(0)
    df["log_reviews_per_month"] = np.log1p(df["reviews_per_month"])

    df["number_of_reviews"] = pd.to_numeric(df["number_of_reviews"], errors="coerce").fillna(0)
    df["log_number_of_reviews"] = np.log1p(df["number_of_reviews"])

    df["number_of_reviews_ltm"] = pd.to_numeric(df["number_of_reviews_ltm"], errors="coerce").fillna(0)
    df["reviews_growth_ratio"] = (
        (df["number_of_reviews_l30d"].fillna(0) + 1) / (df["number_of_reviews_ltm"] + 1)
    )

    return df


def compute_host_verification_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    构建主机验证相关特征
    Build host verification related features
    """
    if 'host_verifications' in df.columns:
        df['verifications_parsed'] = df['host_verifications'].apply(parse_amenities) # Re-use amenities parser
        df['host_verifications_count'] = df['verifications_parsed'].apply(len)
        df['host_has_gov_id'] = df['verifications_parsed'].apply(
            lambda x: 1 if 'government_id' in x else 0
        )
    else:
        df['host_verifications_count'] = 0
        df['host_has_gov_id'] = 0
    return df


def compute_sentiment_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    使用 VADER 对文本字段进行情感分析
    Use VADER for sentiment analysis on text fields
    """
    sid = SentimentIntensityAnalyzer()
    
    text_cols_for_sentiment = {
        'description': 'desc',
        'neighborhood_overview': 'neighborhood',
        'host_about': 'host_about'
    }
    
    for col, prefix in text_cols_for_sentiment.items():
        if col in df.columns:
            # VADER 性能尚可，不需要批量处理 / VADER performance is acceptable, no batching needed
            sentiments = df[col].fillna('').apply(sid.polarity_scores)
            df[f'{prefix}_sentiment_pos'] = sentiments.apply(lambda x: x['pos'])
            df[f'{prefix}_sentiment_neu'] = sentiments.apply(lambda x: x['neu'])
            df[f'{prefix}_sentiment_neg'] = sentiments.apply(lambda x: x['neg'])
            df[f'{prefix}_sentiment_compound'] = sentiments.apply(lambda x: x['compound'])
        else:
            df[f'{prefix}_sentiment_pos'] = 0
            df[f'{prefix}_sentiment_neu'] = 0
            df[f'{prefix}_sentiment_neg'] = 0
            df[f'{prefix}_sentiment_compound'] = 0
            
    return df


def engineer_rating_features_v2(save_processed: bool = True) -> pd.DataFrame:
    """
    构建评分预测特征 (v2)
    Build engineered features for rating prediction (v2).

    Args:
        save_processed (bool): 是否保存处理后的数据 / Whether to save processed data.

    Returns:
        pd.DataFrame: 包含特征和目标字段的数据框 / DataFrame with features and target.
    """
    df = load_listings_data()

    # --- DEBUG: Inspect bathrooms_text ---
    if 'bathrooms_text' in df.columns:
        print("--- Unique values in bathrooms_text ---")
        print(df['bathrooms_text'].unique())
        print("------------------------------------")
    # --- END DEBUG ---

    # 目标变量准备 / Target preparation
    df["review_scores_rating"] = pd.to_numeric(df["review_scores_rating"], errors="coerce")
    df = df[df["review_scores_rating"].notna()].copy()
    # 在 XGBoost v2 版本中，目标变量评分范围是 0-100，这里统一为 0-5
    # In XGBoost v2, the target variable rating range was 0-100. Standardizing to 0-5.
    if df["review_scores_rating"].max() > 5:
        df["review_scores_rating"] = df["review_scores_rating"] / 20.0
    df = df[(df["review_scores_rating"] >= 0) & (df["review_scores_rating"] <= 5)].copy()

    # 基础清洗 / Basic cleaning
    df["host_response_rate"] = parse_percentage(df["host_response_rate"])
    df["host_acceptance_rate"] = parse_percentage(df["host_acceptance_rate"])
    df["host_is_superhost_flag"] = parse_boolean(df["host_is_superhost"], ["t", "true", "yes", "1"])
    df["instant_bookable_flag"] = parse_boolean(df["instant_bookable"], ["t", "true", "yes", "1"])
    df["has_license_info"] = df["license"].notna().astype(int)
    df["room_type_encoded"] = df["room_type"].astype("category").cat.codes
    df["property_type_encoded"] = df["property_type"].astype("category").cat.codes
    df["neighbourhood_encoded"] = df["neighbourhood_cleansed"].astype("category").cat.codes

    # --- v2 新增特征 / New features in v2 ---
    print("  计算主机验证和情感分析特征 / Computing host verification and sentiment features...")
    df = compute_host_verification_features(df)
    df = compute_sentiment_features(df)
    # --- 结束 / End ---

    # 高级特征工程 / Advanced feature engineering
    df = compute_numeric_ratios(df)
    df = compute_amenity_features(df)
    df = compute_text_features(df)
    text_embedding_cols: List[str] = []
    # 使用 TF-IDF 生成文本嵌入
    text_embedding_cols += generate_text_embeddings(
        df, column="description", prefix="description", n_components=20
    )
    text_embedding_cols += generate_text_embeddings(
        df, column="neighborhood_overview", prefix="neighborhood", n_components=15
    )
    text_embedding_cols += generate_text_embeddings(
        df, column="host_about", prefix="host_about", n_components=10
    )
    df = compute_time_features(df)
    df = compute_location_features(df)

    # 复合指标 / Composite metrics
    df["host_activity_score"] = (
        df["host_response_rate"] * 0.5
        + df["host_acceptance_rate"] * 0.3
        + df["instant_bookable_flag"] * 0.2
    )
    df["amenity_comfort_score"] = (
        df["amenity_score_luxury"] * 1.5
        + df["amenity_score_family"] * 1.2
        + df["amenity_score_business"] * 1.0
        + df["amenity_score_safety"] * 1.3
        + df["amenities_count"] * 0.05
    )
    df["host_experience_score"] = (
        df["host_is_superhost_flag"] * 1.5
        + np.clip(df["host_experience_years"], 0, 15) * 0.1
        + np.log1p(df["host_listings_count"].fillna(0)) * 0.2
        + df["host_verifications_count"] * 0.1 # v2 新增
        + df["host_has_gov_id"] * 0.5 # v2 新增
    )
    df["recent_review_score"] = (
        (30 - np.clip(df["days_since_last_review"], 0, 30)) / 30
        + df["reviews_growth_ratio"] * 0.5
    )
    df["text_sentiment_score"] = ( # v2 新增
        df["desc_sentiment_compound"] * 0.5 + 
        df["neighborhood_sentiment_compound"] * 0.3 + 
        df["host_about_sentiment_compound"] * 0.2
    )

    # 修正 bathrooms_text 列
    # Fix bathrooms_text column
    if 'bathrooms_text' in df.columns:
        df['bathrooms_numeric'] = df['bathrooms_text'].str.extract(r'(\d+\.?\d*)').astype(float)
        df['bathrooms_numeric'] = df['bathrooms_numeric'].fillna(0)
        df['is_shared_bath'] = df['bathrooms_text'].str.contains('shared', case=False, na=False).astype(int)
    else:
        df['bathrooms_numeric'] = 0
        df['is_shared_bath'] = 0


    # 选择用于建模的字段 / Select modeling fields
    feature_columns = [
        "review_scores_rating",
        "price_clean", "log_price", "price_per_person",
        "occupancy_rate", "availability_ratio",
        "reviews_per_month", "log_reviews_per_month",
        "number_of_reviews", "log_number_of_reviews", "number_of_reviews_ltm", "reviews_growth_ratio",
        "host_response_rate", "host_acceptance_rate", "host_activity_score",
        "host_is_superhost_flag", "host_experience_years", "host_experience_score",
        "host_listings_count", "host_total_listings_count",
        "instant_bookable_flag", "has_license_info",
        "room_type_encoded", "property_type_encoded", "neighbourhood_encoded",
        "accommodates", "bedrooms", "beds", "bathrooms_numeric", "is_shared_bath",
        "amenities_count", "amenity_comfort_score",
        "amenity_score_luxury", "amenity_score_family", "amenity_score_business", "amenity_score_safety",
        "description_length", "neighborhood_desc_length", "host_about_length",
        "listing_age_days", "days_since_last_review", "recent_review_flag", "recent_review_score",
        "distance_to_center_km", "is_central",
        # --- v2 新增特征 ---
        "host_verifications_count", "host_has_gov_id",
        "desc_sentiment_compound", "neighborhood_sentiment_compound", "host_about_sentiment_compound",
        "text_sentiment_score",
    ]

    # 添加关键设施特征列 / Add key amenity columns
    amenity_flag_cols = [col for col in df.columns if col.startswith("amenity_has_")]
    feature_columns.extend(amenity_flag_cols)
    feature_columns.extend(text_embedding_cols)
    
    # 确保所有列都存在 / Ensure all columns exist, handling potential misses
    final_feature_columns = []
    for col in feature_columns:
        if col in df.columns:
            final_feature_columns.append(col)
        elif col == 'bathrooms_text': # 替换旧列名 / Replace old column name
             if 'bathrooms_numeric' in df.columns: final_feature_columns.append('bathrooms_numeric')
             if 'is_shared_bath' in df.columns: final_feature_columns.append('is_shared_bath')
        else:
            print(f"  警告: 特征列 '{col}' 不在DataFrame中，将被跳过。")
    
    # 去重 / Deduplicate
    final_feature_columns = sorted(list(set(final_feature_columns)))

    processed_df = df[final_feature_columns].copy()
    
    # 最终检查 / Final check
    missing_features = [col for col in final_feature_columns if col not in df.columns]
    if missing_features:
        raise ValueError(f"缺少必要特征列 / Missing required feature columns: {missing_features}")


    # 保存处理后的数据 / Save processed data
    if save_processed:
        project_root, data_dir, _ = get_project_paths()
        processed_dir = data_dir / "processed"
        processed_dir.mkdir(exist_ok=True)
        parquet_path = processed_dir / "rating_features_v2.parquet"
        csv_path = processed_dir / "rating_features_v2.csv"
        try:
            processed_df.to_parquet(parquet_path, index=False)
            print(f"  [OK] 处理后的特征已保存到 / Processed features saved to: {parquet_path}")
        except Exception as exc:
            processed_df.to_csv(csv_path, index=False)
            print(
                f"  [WARNING] 保存 Parquet 失败（{exc}），已自动保存为 CSV: {csv_path}"
            )

    # 同步更新特征登记表 / Update feature registry excel
    # registry_path = write_feature_registry()
    # print(f"  [OK] 特征登记表已更新: {registry_path}")

    return processed_df


def main():
    """运行特征工程 / Run feature engineering."""
    print("=" * 80)
    print("构建评分预测特征 (v2) / Building Rating Prediction Features (v2)")
    print("=" * 80)
    processed_df = engineer_rating_features_v2(save_processed=True)
    print(f"数据规模 / Dataset size: {len(processed_df)} 行 × {processed_df.shape[1]} 列")
    print("=" * 80)


if __name__ == "__main__":
    main()
