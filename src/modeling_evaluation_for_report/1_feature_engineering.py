"""
=====================================================================================
Final Feature Engineering Script for Rating Classification (5-Star vs Non-5-Star)
评分分类特征工程最终版本（5星 vs 非5星预测）

This script performs comprehensive feature engineering for the Airbnb rating
classification task, transforming raw listing data into ML-ready features.

Task: Binary classification to predict whether a listing will receive a 5-star
      rating (review_scores_rating = 5.0) or not (<5.0)

Data: Merged 2021 and 2025 Amsterdam Airbnb listings data
Output: data/processed/train_data.csv with 140+ engineered features

Author: Data Science Course Project
Date: 2025
=====================================================================================
"""

import json
import ast
import re
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# Import NLTK for sentiment analysis
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    import nltk
    NLTK_AVAILABLE = True
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        print("  Downloading VADER sentiment lexicon...")
        nltk.download('vader_lexicon', quiet=True)
except (ImportError, ModuleNotFoundError) as e:
    print(f"  Warning: NLTK not available, skipping sentiment analysis: {e}")
    SentimentIntensityAnalyzer = None
    NLTK_AVAILABLE = False

# Project paths setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
CHARTS_DIR = PROJECT_ROOT / 'charts'

# Text embeddings - using Sentence Transformers instead of TF-IDF+SVD
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    print("Warning: sentence-transformers not installed. Falling back to TF-IDF+SVD.")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    SENTENCE_TRANSFORMER_AVAILABLE = False

# ==================== CONSTANTS ====================
REFERENCE_DATE = pd.Timestamp("2021-09-07")
CITY_CENTER_LAT = 52.3676  # Amsterdam city center coordinates
CITY_CENTER_LON = 4.9041

# Amenity categorization for feature engineering
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


# ==================== DATA LOADING ====================
def load_cleaned_listings_data() -> pd.DataFrame:
    """
    Load cleaned and merged Airbnb listings data

    Returns:
        DataFrame with cleaned listings from 2021 and 2025
    """
    project_root, data_dir = PROJECT_ROOT, DATA_DIR
    cleaned_data_path = project_root / "data" / "cleaned" / "listings_cleaned.csv"

    if not cleaned_data_path.exists():
        raise FileNotFoundError(f"Cleaned data file not found: {cleaned_data_path}")

    print(f"Loading cleaned data: {cleaned_data_path.name}...")
    df = pd.read_csv(cleaned_data_path)
    print(f"  ✅ Loaded: {len(df):,} rows × {len(df.columns)} columns")

    # Verify required fields
    required_fields = ['id', 'data_year']
    missing_fields = [f for f in required_fields if f not in df.columns]
    if missing_fields:
        raise ValueError(f"Missing required fields: {missing_fields}")

    # Show year distribution
    if 'data_year' in df.columns:
        year_dist = df['data_year'].value_counts().sort_index()
        print(f"\nYear Distribution:")
        for year, count in year_dist.items():
            print(f"  {year}: {count:,} records")

    return df


# ==================== BASIC CLEANING ====================
def clean_price_column(df: pd.DataFrame) -> pd.DataFrame:
    """Clean price column by removing currency symbols and converting to numeric"""
    if "price" in df.columns:
        if df["price"].dtype == "object":
            df["price_clean"] = (
                df["price"].astype(str)
                .str.replace("$", "", regex=False)
                .str.replace(",", "", regex=False)
            )
            df["price_clean"] = pd.to_numeric(df["price_clean"], errors="coerce")
        else:
            df["price_clean"] = pd.to_numeric(df["price"], errors="coerce")
        df["price_clean"] = df["price_clean"].fillna(0)
    else:
        df["price_clean"] = 0
    return df


def parse_percentage(series: pd.Series) -> pd.Series:
    """Parse percentage strings (e.g., '95%') to float (0.95)"""
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
    """Parse boolean columns to 0/1"""
    return series.fillna("").astype(str).str.lower().isin(true_values).astype(int)


# ==================== AMENITY FEATURES ====================
def parse_amenities(amenities_value) -> List[str]:
    """Parse amenities field into lowercase list"""
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
    """
    Compute amenity-related features

    Features created:
    - amenities_count: Total number of amenities
    - amenity_score_luxury/family/business/safety: Category-specific scores
    - amenity_has_*: Binary flags for key amenities
    """
    df["amenities_parsed"] = df["amenities"].apply(parse_amenities)
    df["amenities_count"] = df["amenities_parsed"].apply(len)

    # Category scores
    for category, keywords in LUXURY_AMENITIES.items():
        df[f"amenity_score_{category}"] = df["amenities_parsed"].apply(
            lambda ams, kws=keywords: sum(1 for a in ams if any(k in a for k in kws))
        )

    # Key amenity flags
    TOP_KEY_AMENITIES = [
        "wifi", "kitchen", "heating", "air conditioning", "washer", "dryer",
        "tv", "dishwasher", "parking", "balcony", "elevator", "coffee maker",
        "pets allowed", "long term stays allowed"
    ]
    for amenity in TOP_KEY_AMENITIES:
        col_name = f"amenity_has_{re.sub(r'[^a-z0-9]+', '_', amenity)}"
        df[col_name] = df["amenities_parsed"].apply(lambda ams, a=amenity: int(a in ams))

    return df


# ==================== TEXT FEATURES ====================
def compute_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute text length features"""
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
    n_components: int = 40,
    model_name: str = "all-MiniLM-L6-v2",
) -> List[str]:
    """
    Generate text embeddings using Sentence Transformers (or fallback to TF-IDF + SVD)

    Args:
        df: DataFrame containing text data
        column: Column name to generate embeddings for
        prefix: Prefix for embedding column names
        n_components: Number of embedding dimensions (default 40 for Sentence Transformers)
        model_name: Sentence Transformer model name (default: all-MiniLM-L6-v2, 384-dim)

    Returns:
        List of embedding column names created

    Note:
        - Sentence Transformers produces 384-dim embeddings, reduced to 40 using PCA
        - Falls back to TF-IDF + SVD if sentence-transformers not available
    """
    if column not in df.columns:
        return []

    texts = df[column].fillna("").astype(str).tolist()
    if all(not t.strip() for t in texts):
        return []

    try:
        print(f"  Generating embeddings for {column}...")

        if SENTENCE_TRANSFORMER_AVAILABLE:
            # Use Sentence Transformers (semantic embeddings)
            print(f"    Using Sentence Transformer: {model_name}")
            model = SentenceTransformer(model_name)

            # Generate embeddings (384-dimensional for all-MiniLM-L6-v2)
            embeddings_full = model.encode(texts, show_progress_bar=True, batch_size=64)

            # Reduce dimensionality to n_components using PCA
            from sklearn.decomposition import PCA
            pca = PCA(n_components=n_components, random_state=42)
            embeddings = pca.fit_transform(embeddings_full)

            print(f"    Generated {embeddings.shape[1]}-dimensional embeddings")
            print(f"    Explained variance: {pca.explained_variance_ratio_.sum():.2%}")
        else:
            # Fallback to TF-IDF + SVD
            print(f"    Using TF-IDF + SVD (fallback)")
            vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
            tfidf_matrix = vectorizer.fit_transform(texts)

            max_components = min(n_components, tfidf_matrix.shape[1])
            if max_components == 0:
                return []

            from sklearn.decomposition import TruncatedSVD
            svd = TruncatedSVD(n_components=max_components, random_state=42)
            embeddings = svd.fit_transform(tfidf_matrix)

        # Create embedding columns
        embedding_cols = [f"{prefix}_embed_{i}" for i in range(embeddings.shape[1])]
        for i, col in enumerate(embedding_cols):
            df[col] = embeddings[:, i]

        return embedding_cols
    except Exception as e:
        print(f"  Warning: Embedding generation failed for {column}: {e}")
        import traceback
        traceback.print_exc()
        return []


# ==================== MAIN FEATURE ENGINEERING ====================
def engineer_rating_features(save_processed: bool = True) -> pd.DataFrame:
    """
    Main feature engineering pipeline for rating classification

    Steps:
    1. Load cleaned data
    2. Generate price features
    3. Generate host behavior features
    4. Generate review features
    5. Generate amenity features
    6. Generate text features (embeddings + sentiment)
    7. Generate location features
    8. Generate time-based features
    9. Save processed data

    Returns:
        DataFrame with ~140 engineered features ready for modeling
    """
    print("\n" + "=" * 80)
    print("FEATURE ENGINEERING FOR RATING CLASSIFICATION")
    print("=" * 80)

    # Load data
    df = load_cleaned_listings_data()

    # ========== 1. PRICE FEATURES ==========
    print("\n[1/8] Engineering price features...")
    df = clean_price_column(df)
    df["log_price"] = np.log1p(df["price_clean"])

    if "accommodates" in df.columns:
        df["accommodates"] = pd.to_numeric(df["accommodates"], errors="coerce").fillna(1)
        df["price_per_person"] = df["price_clean"] / df["accommodates"].replace(0, 1)
        df["price_per_person"] = df["price_per_person"].replace([np.inf, -np.inf], 0).fillna(0)
    else:
        df["price_per_person"] = 0

    # ========== 2. HOST BEHAVIOR FEATURES ==========
    print("[2/8] Engineering host behavior features...")
    if "host_response_rate" in df.columns:
        df["host_response_rate"] = parse_percentage(df["host_response_rate"])
    else:
        df["host_response_rate"] = 0

    if "host_acceptance_rate" in df.columns:
        df["host_acceptance_rate"] = parse_percentage(df["host_acceptance_rate"])
    else:
        df["host_acceptance_rate"] = 0

    if "host_is_superhost" in df.columns:
        df["host_is_superhost_flag"] = parse_boolean(df["host_is_superhost"], ["t", "true"])
    else:
        df["host_is_superhost_flag"] = 0

    # Host activity score (composite feature)
    df["host_activity_score"] = (
        df["host_response_rate"] * 0.4 +
        df["host_acceptance_rate"] * 0.3 +
        df["host_is_superhost_flag"] * 0.3
    )

    # Host verification features
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

    # ========== 3. REVIEW FEATURES ==========
    print("[3/8] Engineering review features...")
    numeric_cols = ["number_of_reviews", "number_of_reviews_ltm", "reviews_per_month"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        else:
            df[col] = 0

    df["log_reviews_per_month"] = np.log1p(df["reviews_per_month"])
    df["log_number_of_reviews"] = np.log1p(df["number_of_reviews"])

    if "number_of_reviews_ltm" in df.columns and "number_of_reviews" in df.columns:
        df["reviews_growth_ratio"] = (
            df["number_of_reviews_ltm"] / (df["number_of_reviews"] + 1)
        ).fillna(0)
    else:
        df["reviews_growth_ratio"] = 0

    # ========== 4. AVAILABILITY & OCCUPANCY FEATURES ==========
    print("[4/8] Engineering availability features...")
    avail_cols = ["availability_30", "availability_60", "availability_90", "availability_365"]
    for col in avail_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        else:
            df[col] = 0

    if "availability_365" in df.columns:
        df["occupancy_rate"] = 1 - (df["availability_365"] / 365).clip(0, 1)
        df["availability_ratio"] = df["availability_30"] / 30
    else:
        df["occupancy_rate"] = 0
        df["availability_ratio"] = 0

    # ========== 5. PROPERTY FEATURES ==========
    print("[5/8] Engineering property features...")
    # Room and property type encoding
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

    # Capacity features
    capacity_cols = ["bedrooms", "beds"]
    for col in capacity_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        else:
            df[col] = 0

    # Bathroom features
    if 'bathrooms_text' in df.columns:
        df['bathrooms_numeric'] = df['bathrooms_text'].str.extract(r'(\d+\.?\d*)').astype(float)
        df['bathrooms_numeric'] = df['bathrooms_numeric'].fillna(0)
        df['is_shared_bath'] = df['bathrooms_text'].str.contains('shared', case=False, na=False).astype(int)
    else:
        df['bathrooms_numeric'] = 0
        df['is_shared_bath'] = 0

    # Booking preferences
    if "instant_bookable" in df.columns:
        df["instant_bookable_flag"] = parse_boolean(df["instant_bookable"], ["t", "true"])
    else:
        df["instant_bookable_flag"] = 0

    if "license" in df.columns:
        df["has_license_info"] = df["license"].notna().astype(int)
    else:
        df["has_license_info"] = 0

    # ========== 6. AMENITY FEATURES ==========
    print("[6/8] Engineering amenity features...")
    df = compute_amenity_features(df)
    df["amenity_comfort_score"] = (
        df["amenity_score_luxury"] * 0.3 +
        df["amenity_score_family"] * 0.2 +
        df["amenity_score_business"] * 0.2 +
        df["amenity_score_safety"] * 0.3
    )

    # ========== 7. TEXT FEATURES ==========
    print("[7/8] Engineering text features...")
    df = compute_text_features(df)

    # Generate text embeddings
    text_embedding_cols = []
    for col, prefix in [("description", "desc"), ("neighborhood_overview", "neighborhood")]:
        cols = generate_text_embeddings(df, col, prefix, n_components=40)
        text_embedding_cols.extend(cols)

    # Sentiment analysis
    if NLTK_AVAILABLE and SentimentIntensityAnalyzer is not None:
        print("  Performing sentiment analysis...")
        try:
            sia = SentimentIntensityAnalyzer()

            for col, prefix in [("description", "desc"),
                               ("neighborhood_overview", "neighborhood"),
                               ("host_about", "host_about")]:
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
            print(f"  Warning: Sentiment analysis failed: {e}")
            df["desc_sentiment_compound"] = 0
            df["neighborhood_sentiment_compound"] = 0
            df["host_about_sentiment_compound"] = 0
            df["text_sentiment_score"] = 0
    else:
        print("  Skipping sentiment analysis (NLTK not available)")
        df["desc_sentiment_compound"] = 0
        df["neighborhood_sentiment_compound"] = 0
        df["host_about_sentiment_compound"] = 0
        df["text_sentiment_score"] = 0

    # ========== 8. LOCATION & TIME FEATURES ==========
    print("[8/8] Engineering location and time features...")
    # Location features
    if "latitude" in df.columns and "longitude" in df.columns:
        df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
        df["distance_to_center_km"] = np.sqrt(
            (df["latitude"] - CITY_CENTER_LAT) ** 2 +
            (df["longitude"] - CITY_CENTER_LON) ** 2
        ) * 111  # Approximate km conversion
        df["is_central"] = (df["distance_to_center_km"] <= 5).astype(int)
    else:
        df["distance_to_center_km"] = 0
        df["is_central"] = 0

    # Time features
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

    # Host listing count features
    host_cols = ["host_listings_count", "host_total_listings_count"]
    for col in host_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        else:
            df[col] = 0

    # ========== SELECT FEATURES FOR MODELING ==========
    print("\nSelecting features for modeling...")
    feature_columns = [
        # Target variable
        "review_scores_rating",
        # Price features
        "price_clean", "log_price", "price_per_person",
        # Availability features
        "occupancy_rate", "availability_ratio",
        # Review features
        "reviews_per_month", "log_reviews_per_month",
        "number_of_reviews", "log_number_of_reviews", "number_of_reviews_ltm", "reviews_growth_ratio",
        # Host features
        "host_response_rate", "host_acceptance_rate", "host_activity_score",
        "host_is_superhost_flag", "host_listings_count", "host_total_listings_count",
        "host_verifications_count", "host_has_gov_id",
        # Booking features
        "instant_bookable_flag", "has_license_info",
        # Property features
        "room_type_encoded", "property_type_encoded", "neighbourhood_encoded",
        "accommodates", "bedrooms", "beds", "bathrooms_numeric", "is_shared_bath",
        # Amenity features
        "amenities_count", "amenity_comfort_score",
        "amenity_score_luxury", "amenity_score_family", "amenity_score_business", "amenity_score_safety",
        # Text features
        "description_length", "neighborhood_desc_length", "host_about_length",
        "desc_sentiment_compound", "neighborhood_sentiment_compound", "host_about_sentiment_compound",
        "text_sentiment_score",
        # Time features
        "listing_age_days", "days_since_last_review", "recent_review_flag", "recent_review_score",
        # Location features
        "distance_to_center_km", "is_central",
        # Data source
        "data_year",
    ]

    # Add missing indicator variables
    indicator_cols = [col for col in df.columns if col.endswith('_is_missing')]
    feature_columns.extend(indicator_cols)

    # Add amenity flags
    amenity_flag_cols = [col for col in df.columns if col.startswith("amenity_has_")]
    feature_columns.extend(amenity_flag_cols)

    # Add text embeddings
    feature_columns.extend(text_embedding_cols)

    # Ensure all columns exist
    final_feature_columns = []
    for col in feature_columns:
        if col in df.columns:
            final_feature_columns.append(col)
        else:
            print(f"  Warning: Feature '{col}' not in DataFrame, skipping.")

    # Remove duplicates and sort
    final_feature_columns = sorted(list(set(final_feature_columns)))

    processed_df = df[final_feature_columns].copy()

    print(f"\n✅ Feature engineering complete!")
    print(f"   Total features: {len(final_feature_columns)} (including target variable)")
    print(f"   Total samples: {len(processed_df):,}")

    # ========== SAVE PROCESSED DATA ==========
    if save_processed:
        project_root, data_dir = PROJECT_ROOT, DATA_DIR
        processed_dir = data_dir / "processed"
        processed_dir.mkdir(exist_ok=True)

        csv_path = processed_dir / "train_data.csv"
        processed_df.to_csv(csv_path, index=False)
        print(f"\n✅ Processed features saved to: {csv_path}")

    return processed_df


# ==================== MAIN ====================
def main():
    """Run complete feature engineering pipeline"""
    print("\n" + "=" * 80)
    print("AIRBNB RATING CLASSIFICATION - FEATURE ENGINEERING")
    print("Task: Predict 5-star vs Non-5-star ratings")
    print("=" * 80)

    processed_df = engineer_rating_features(save_processed=True)

    # Summary statistics
    print("\n" + "=" * 80)
    print("FEATURE ENGINEERING SUMMARY")
    print("=" * 80)
    print(f"Dataset shape: {processed_df.shape[0]:,} rows × {processed_df.shape[1]} columns")
    print(f"\nTarget variable distribution:")
    if "review_scores_rating" in processed_df.columns:
        rating_dist = processed_df["review_scores_rating"].value_counts().sort_index()
        for rating, count in rating_dist.items():
            print(f"  Rating {rating}: {count:,} ({count/len(processed_df)*100:.1f}%)")

        # Binary classification target
        is_five_star = (processed_df["review_scores_rating"] == 5.0).sum()
        print(f"\nBinary classification target:")
        print(f"  5-star: {is_five_star:,} ({is_five_star/len(processed_df)*100:.1f}%)")
        print(f"  <5-star: {len(processed_df)-is_five_star:,} ({(len(processed_df)-is_five_star)/len(processed_df)*100:.1f}%)")

    print("\n" + "=" * 80)
    print("Feature engineering pipeline completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
