"""
评分预测特征工程脚本 - 适用于已合并+已清洗的数据集
Rating Prediction Feature Engineering Script - For Merged and Cleaned Dataset

从清洗后的数据构建训练特征
Build training features from cleaned data
"""

import json
import ast
import re
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# 尝试导入 nltk，如果失败则设置为 None / Try to import nltk, set to None if failed
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    import nltk
    NLTK_AVAILABLE = True
    # 下载 VADER 词典 / Download VADER lexicon
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        print("  下载 VADER 情感分析词典 / Downloading VADER sentiment analysis lexicon...")
        nltk.download('vader_lexicon', quiet=True)
except (ImportError, ModuleNotFoundError) as e:
    print(f"  警告: nltk 不可用，将跳过情感分析 / Warning: nltk not available, skipping sentiment analysis: {e}")
    SentimentIntensityAnalyzer = None
    NLTK_AVAILABLE = False

# 添加 EDA 目录到路径 / Add EDA directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "EDA"))

from utils import get_project_paths
try:
    from feature_registry import write_feature_registry, register_feature
except ImportError:
    def write_feature_registry():
        return None
    def register_feature(*args, **kwargs):
        pass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

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
        "changing table", "children's books", "playground"
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


def load_cleaned_data() -> pd.DataFrame:
    """
    从清洗后的数据文件加载数据
    Load data from cleaned data file
    """
    project_root, data_dir, _, _ = get_project_paths()
    
    cleaned_data_path = project_root / "data" / "cleaned" / "listings_cleaned.csv"
    
    if not cleaned_data_path.exists():
        raise FileNotFoundError(f"清洗后的数据文件不存在 / Cleaned data file not found: {cleaned_data_path}")
    
    print(f"  加载清洗后的数据 / Loading cleaned data: {cleaned_data_path.name}...")
    df = pd.read_csv(cleaned_data_path)
    print(f"    [OK] 加载完成: {len(df):,} 行 × {len(df.columns)} 列")
    
    # 检查关键字段
    required_fields = ['id', 'data_year']
    missing_fields = [f for f in required_fields if f not in df.columns]
    if missing_fields:
        raise ValueError(f"缺少必需字段 / Missing required fields: {missing_fields}")
    
    # 检查年份分布
    if 'data_year' in df.columns:
        year_dist = df['data_year'].value_counts().sort_index()
        print(f"\n  年份分布 / Year Distribution:")
        for year, count in year_dist.items():
            print(f"    {year}年: {count:,} 条记录")
    
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
        
        max_components = min(n_components, tfidf_matrix.shape[1])
        if max_components == 0:
            return []

        svd = TruncatedSVD(n_components=max_components, random_state=42)
        embeddings = svd.fit_transform(tfidf_matrix)
        
        embedding_cols = [f"{prefix}_embed_{i}" for i in range(max_components)]
        for i, col in enumerate(embedding_cols):
            df[col] = embeddings[:, i]
        
        return embedding_cols
    except Exception as e:
        print(f"  警告: {column} 嵌入生成失败: {e}")
        return []


def build_features() -> pd.DataFrame:
    """
    从清洗后的数据构建训练特征
    Build training features from cleaned data
    """
    print("\n开始特征工程 / Starting Feature Engineering...")
    
    # 加载清洗后的数据
    df = load_cleaned_data()
    
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
        df["has_license_info"] = (df["license"].notna() & (df["license"] != "Unknown")).astype(int)
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
    
    # 选择用于建模的字段
    feature_columns = [
        "review_scores_rating",  # 目标变量
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
    ]
    
    # 添加缺失指示变量作为特征
    indicator_cols = [
        col for col in df.columns
        if col.endswith('_is_missing') and col != 'license_is_missing'
    ]
    feature_columns.extend(indicator_cols)
    
    # 添加关键设施特征列
    amenity_flag_cols = [col for col in df.columns if col.startswith("amenity_has_")]
    feature_columns.extend(amenity_flag_cols)
    feature_columns.extend(text_embedding_cols)
    
    # 确保所有列都存在
    final_feature_columns = []
    for col in feature_columns:
        if col in df.columns:
            final_feature_columns.append(col)
        else:
            print(f"  警告: 特征列 '{col}' 不在DataFrame中，将被跳过。")
    
    # 去重
    final_feature_columns = sorted(list(set(final_feature_columns)))
    
    processed_df = df[final_feature_columns].copy()
    
    return processed_df


def main():
    """运行特征工程 / Run feature engineering."""
    print("=" * 80)
    print("构建训练特征 / Building Training Features")
    print("=" * 80)
    
    # 构建特征
    train_data = build_features()
    
    # 保存训练数据
    project_root, data_dir, _, _ = get_project_paths()
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(exist_ok=True)
    
    train_data_path = processed_dir / "train_data.csv"
    train_data.to_csv(train_data_path, index=False, encoding='utf-8-sig')
    print(f"\n  [OK] 训练数据已保存到 / Training data saved to: {train_data_path}")
    print(f"  数据规模 / Dataset size: {len(train_data):,} 行 × {train_data.shape[1]} 列")
    print(f"  特征数量 / Feature count: {train_data.shape[1]} 个")
    
    # 显示特征统计
    print(f"\n  特征统计 / Feature Statistics:")
    print(f"    - 目标变量: review_scores_rating")
    print(f"    - 数值特征: {len([c for c in train_data.columns if pd.api.types.is_numeric_dtype(train_data[c])])} 个")
    print(f"    - 缺失指示变量: {len([c for c in train_data.columns if c.endswith('_is_missing')])} 个")
    
    print("=" * 80)


if __name__ == "__main__":
    main()

