"""
评分预测特征工程脚本 - 整合版本 / Rating Prediction Feature Engineering Script - Merged Version

整合 listings_detailed 和 listings_detailed_2 两个数据文件，然后进行特征工程
Merges listings_detailed and listings_detailed_2, then performs feature engineering
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

from utils import setup_plotting, get_project_paths
try:
    from feature_registry import write_feature_registry, register_feature
except ImportError:
    # 如果 feature_registry 不可用，定义空函数 / If feature_registry not available, define empty functions
    def write_feature_registry():
        return None
    def register_feature(*args, **kwargs):
        pass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

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


def load_and_merge_listings_data() -> pd.DataFrame:
    """
    加载并整合两个 listings_detailed 数据文件
    Load and merge two listings_detailed data files
    """
    paths = get_project_paths()
    if len(paths) == 4:
        project_root, data_dir, _, _ = paths
    else:
        project_root, data_dir, _ = paths
    
    # 加载两个文件
    data_path1 = data_dir / "listings_detailed.xlsx"
    data_path2 = data_dir / "listings_detailed_2.xlsx"
    
    if not data_path1.exists():
        raise FileNotFoundError(f"数据文件不存在 / Data file not found: {data_path1}")
    if not data_path2.exists():
        raise FileNotFoundError(f"数据文件不存在 / Data file not found: {data_path2}")
    
    print("  加载 listings_detailed.xlsx...")
    df1 = pd.read_excel(data_path1)
    print(f"    [OK] 加载完成: {len(df1)} 行 × {len(df1.columns)} 列")
    
    print("  加载 listings_detailed_2.xlsx...")
    df2 = pd.read_excel(data_path2)
    print(f"    [OK] 加载完成: {len(df2)} 行 × {len(df2.columns)} 列")
    
    # 统一列名（确保两个数据集的列名一致）
    # 获取两个数据集的列名
    cols1 = set(df1.columns)
    cols2 = set(df2.columns)
    
    # 找出共同的列和独有的列
    common_cols = cols1 & cols2
    only_in_df1 = cols1 - cols2
    only_in_df2 = cols2 - cols1
    
    print(f"\n  列名分析 / Column Analysis:")
    print(f"    共同列数 / Common columns: {len(common_cols)}")
    print(f"    仅在文件1中的列 / Only in file 1: {len(only_in_df1)}")
    print(f"    仅在文件2中的列 / Only in file 2: {len(only_in_df2)}")
    
    # 使用共同列进行合并
    # 添加数据源标识
    df1['data_source'] = 'listings_detailed'
    df2['data_source'] = 'listings_detailed_2'
    
    # 合并数据（使用所有列，缺失的列用NaN填充）
    all_cols = list(common_cols) + ['data_source']
    df1_selected = df1[all_cols].copy()
    df2_selected = df2[all_cols].copy()
    
    # 合并
    merged_df = pd.concat([df1_selected, df2_selected], ignore_index=True)
    
    print(f"\n  [OK] 数据整合完成: {len(merged_df)} 行 × {len(merged_df.columns)} 列")
    print(f"    来源1 (listings_detailed): {len(df1_selected)} 行")
    print(f"    来源2 (listings_detailed_2): {len(df2_selected)} 行")
    
    return merged_df


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


def engineer_rating_features_merged(save_processed: bool = True) -> pd.DataFrame:
    """
    整合数据并构建评分预测特征
    Merge data and build rating prediction features
    """
    print("\n开始特征工程 / Starting Feature Engineering...")
    
    # 加载并整合数据
    df = load_and_merge_listings_data()
    
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
        "data_source",  # 保留数据源标识
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
        else:
            print(f"  警告: 特征列 '{col}' 不在DataFrame中，将被跳过。")
    
    # 去重
    final_feature_columns = sorted(list(set(final_feature_columns)))
    
    processed_df = df[final_feature_columns].copy()
    
    # 保存处理后的数据
    if save_processed:
        paths = get_project_paths()
        if len(paths) == 4:
            project_root, data_dir, _, _ = paths
        else:
            project_root, data_dir, _ = paths
        processed_dir = data_dir / "processed"
        processed_dir.mkdir(exist_ok=True)
        parquet_path = processed_dir / "rating_features_merged.parquet"
        csv_path = processed_dir / "rating_features_merged.csv"
        try:
            processed_df.to_parquet(parquet_path, index=False)
            print(f"  [OK] 处理后的特征已保存到 / Processed features saved to: {parquet_path}")
        except Exception as exc:
            processed_df.to_csv(csv_path, index=False)
            print(
                f"  [WARNING] 保存 Parquet 失败（{exc}），已自动保存为 CSV: {csv_path}"
            )
    
    return processed_df


def main():
    """运行特征工程 / Run feature engineering."""
    print("=" * 80)
    print("构建评分预测特征 (整合版本) / Building Rating Prediction Features (Merged Version)")
    print("=" * 80)
    processed_df = engineer_rating_features_merged(save_processed=True)
    print(f"\n数据规模 / Dataset size: {len(processed_df)} 行 × {processed_df.shape[1]} 列")
    print("=" * 80)


if __name__ == "__main__":
    main()

