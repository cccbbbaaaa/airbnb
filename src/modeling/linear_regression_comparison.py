"""
Linear Regression Model Comparison for Income and Rating Prediction
线性回归模型对比：收入预测 vs 评分预测

本脚本分别训练收入预测和评分预测的线性回归模型，并对比结果
This script trains linear regression models for both income and rating prediction and compares results
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time
import re
import json
import ast
from collections import Counter

# 添加 EDA 目录到路径 / Add EDA directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'EDA'))
from utils import setup_plotting, get_project_paths

# 设置中文字体支持 / Set Chinese font support
setup_plotting()

# 获取项目根目录路径 / Get project root directory path
project_root, data_dir, charts_eda_dir, charts_model_dir = get_project_paths()
charts_dir = charts_model_dir  # 使用模型目录 / Use model directory

print("=" * 80)
print("Linear Regression Model Comparison")
print("线性回归模型对比：收入预测 vs 评分预测")
print("=" * 80)

# ============================================================================
# 通用数据清洗函数 / Common Data Cleaning Functions
# ============================================================================

def convert_to_flat(x):
    """转换百分比到浮点数 / Convert percentage to float"""
    if pd.isna(x):
        return 0
    if isinstance(x, str):
        return float(x.rstrip('%'))/100
    else:
        return float(x)/100

def extract_bathrooms_number(x):
    """提取浴室数量 / Extract bathroom number"""
    if pd.isna(x):
        return "missing"
    if 'half-bath' in str(x).lower():
        return 0.5
    match = re.search(r'(\d+\.?\d*)', str(x))
    if match:
        return float(match.group(1))
    else:
        return "missing"

def parse_amenities(x):
    """解析 amenities / Parse amenities"""
    if pd.isna(x) or x == '':
        return []
    try:
        if isinstance(x, str) and x.strip().startswith('['):
            return ast.literal_eval(x)
        elif isinstance(x, str) and x.strip().startswith('{'):
            amenities_dict = json.loads(x)
            if isinstance(amenities_dict, list):
                return amenities_dict
            elif isinstance(amenities_dict, dict):
                return list(amenities_dict.keys())
        else:
            parsed = json.loads(x)
            if isinstance(parsed, list):
                return parsed
            elif isinstance(parsed, dict):
                return list(parsed.keys())
    except (json.JSONDecodeError, ValueError, SyntaxError):
        if isinstance(x, str):
            cleaned = x.strip().strip('[]').replace("'", "").replace('"', '')
            if cleaned:
                return [item.strip() for item in cleaned.split(',') if item.strip()]
    return []

def clean_column_name(amenity_name):
    """清理 amenity 列名 / Clean amenity column name"""
    cleaned = re.sub(r'[^\w\s-]', '_', str(amenity_name))
    cleaned = re.sub(r'[\s-]+', '_', cleaned)
    cleaned = re.sub(r'_+', '_', cleaned)
    cleaned = cleaned.strip('_')
    return cleaned.lower()

def prepare_features(listings_detailed):
    """准备特征 / Prepare features"""
    # 数据清洗（简化版，复用之前逻辑）
    order = ['no_response', 'a few day or more', 'within a day', 'within a few hours', 'within an hour']
    cat_type = pd.CategoricalDtype(categories=order, ordered=True)
    listings_detailed['host_response_time'] = listings_detailed['host_response_time'].fillna('no_response')
    listings_detailed['host_response_time'] = listings_detailed['host_response_time'].astype(cat_type)
    
    listings_detailed['host_response_rate'] = listings_detailed['host_response_rate'].fillna(0)
    listings_detailed['host_response_rate'] = listings_detailed['host_response_rate'].apply(convert_to_flat)
    
    if 'host_acceptance_rate' in listings_detailed.columns:
        listings_detailed['host_acceptance_rate'] = listings_detailed['host_acceptance_rate'].fillna(0)
        listings_detailed['host_acceptance_rate'] = listings_detailed['host_acceptance_rate'].apply(convert_to_flat)
    
    listings_detailed['host_is_superhost'] = listings_detailed['host_is_superhost'].fillna(False)
    listings_detailed['host_listings_count'] = listings_detailed['host_listings_count'].fillna(0)
    listings_detailed['host_total_listings_count'] = listings_detailed['host_total_listings_count'].fillna(0)
    
    listings_detailed['bathrooms'] = listings_detailed['bathrooms_text'].apply(extract_bathrooms_number)
    listings_detailed['bedrooms'] = listings_detailed['bedrooms'].fillna("missing")
    listings_detailed['beds'] = listings_detailed['beds'].fillna("missing")
    
    # Amenities 处理（简化：只处理前100个最常见的）
    print("  解析 amenities 列...")
    listings_detailed['amenities_parsed'] = listings_detailed['amenities'].apply(parse_amenities)
    all_amenities = []
    for amenity_list in listings_detailed['amenities_parsed']:
        all_amenities.extend(amenity_list)
    amenity_counts = Counter(all_amenities)
    min_frequency = 5
    frequent_amenities = {amenity: count for amenity, count in amenity_counts.items() if count >= min_frequency}
    
    # 只使用前100个最常见的amenities以加速
    top_amenities = sorted(frequent_amenities.items(), key=lambda x: x[1], reverse=True)[:100]
    amenity_columns = [amenity for amenity, _ in top_amenities]
    
    amenity_data = {}
    for amenity in amenity_columns:
        col_name = f'amenity_{clean_column_name(amenity)}'
        amenity_data[col_name] = listings_detailed['amenities_parsed'].apply(
            lambda x, a=amenity: 1 if a in x else 0
        )
    amenity_df = pd.DataFrame(amenity_data, index=listings_detailed.index)
    listings_detailed = pd.concat([listings_detailed, amenity_df], axis=1)
    print(f"  创建了 {len(amenity_columns)} 个amenity特征列")
    
    listings_detailed['minimum_nights'] = listings_detailed['minimum_nights'].clip(upper=365)
    listings_detailed['maximum_nights'] = listings_detailed['maximum_nights'].clip(upper=365)
    listings_detailed['minimum_nights_avg_ntm'] = listings_detailed['minimum_nights_avg_ntm'].fillna("missing")
    listings_detailed['maximum_nights_avg_ntm'] = listings_detailed['maximum_nights_avg_ntm'].fillna("missing")
    
    review_scores_cols = [
        'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness',
        'review_scores_checkin', 'review_scores_communication', 'review_scores_location',
        'review_scores_value'
    ]
    for col in review_scores_cols:
        if col in listings_detailed.columns:
            listings_detailed[col] = listings_detailed[col].fillna("missing")
    
    listings_detailed['reviews_per_month'] = listings_detailed['reviews_per_month'].fillna('missing')
    
    if 'license' in listings_detailed.columns:
        listings_detailed['license'] = listings_detailed['license'].notna()
    
    return listings_detailed

def feature_engineering(X, y, exclude_cols=None):
    """特征工程 / Feature engineering"""
    if exclude_cols is None:
        exclude_cols = []
    
    # 移除排除的列
    X = X.drop(columns=[col for col in exclude_cols if col in X.columns], errors='ignore')
    
    numeric_features = []
    categorical_features = []
    
    for col in X.columns:
        if X[col].dtype in ['int64', 'float64']:
            numeric_features.append(col)
        else:
            categorical_features.append(col)
    
    # 分类特征编码
    label_encoders = {}
    for col in categorical_features:
        if isinstance(X[col].dtype, pd.CategoricalDtype):
            X[col] = X[col].astype(str)
        X[col] = X[col].astype(str).fillna('missing')
        le = LabelEncoder()
        try:
            X[col] = le.fit_transform(X[col])
            label_encoders[col] = le
        except Exception as e:
            print(f"  警告: {col} 编码失败: {e}")
            X[col] = 0
    
    # 数值特征缺失值处理
    for col in numeric_features:
        X[col] = pd.to_numeric(X[col], errors='coerce')
        median_val = X[col].median()
        if pd.isna(median_val):
            X[col] = X[col].fillna(0)
        else:
            X[col] = X[col].fillna(median_val)
    
    # 日期特征处理
    date_features = ['first_review', 'last_review']
    date_new_cols = {}
    for col in date_features:
        if col in X.columns:
            date_series = pd.to_datetime(X[col], errors='coerce')
            date_new_cols[f'{col}_year'] = date_series.dt.year.fillna(0).astype(int)
            date_new_cols[f'{col}_month'] = date_series.dt.month.fillna(0).astype(int)
            date_new_cols[f'{col}_day'] = date_series.dt.day.fillna(0).astype(int)
            reference_date = pd.Timestamp('2021-09-07')
            date_new_cols[f'{col}_days_ago'] = (reference_date - date_series).dt.days.fillna(0).astype(int)
            X = X.drop(col, axis=1)
    
    if date_new_cols:
        date_df = pd.DataFrame(date_new_cols, index=X.index)
        X = pd.concat([X, date_df], axis=1)
    
    # 处理 host_verifications
    if 'host_verifications' in X.columns:
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
        X['host_verifications_count'] = X['host_verifications'].apply(count_verifications)
        X = X.drop('host_verifications', axis=1)
    
    # 布尔特征
    bool_features = ['host_is_superhost', 'host_has_profile_pic', 'host_identity_verified', 
                     'has_availability', 'license']
    for col in bool_features:
        if col in X.columns:
            X[col] = X[col].astype(int)
    
    # 移除 inf
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    
    return X

# ============================================================================
# 任务1: 收入预测 / Task 1: Income Prediction
# ============================================================================

print("\n" + "=" * 80)
print("任务1: 收入预测模型 / Task 1: Income Prediction Model")
print("=" * 80)

print("\n1. 加载和准备数据 / Loading and Preparing Data...")
listings_detailed = pd.read_excel(data_dir / 'listings_detailed.xlsx')
print(f"  [OK] 数据加载完成: {len(listings_detailed)} 行 × {len(listings_detailed.columns)} 列")

print("\n2. 数据清洗 / Data Cleaning...")
listings_detailed = prepare_features(listings_detailed)

print("\n3. 创建目标变量 / Creating Target Variable...")
if 'price' in listings_detailed.columns:
    if listings_detailed['price'].dtype == 'object':
        listings_detailed['price_clean'] = listings_detailed['price'].astype(str).str.replace('$', '').str.replace(',', '')
        listings_detailed['price_clean'] = pd.to_numeric(listings_detailed['price_clean'], errors='coerce')
    else:
        listings_detailed['price_clean'] = pd.to_numeric(listings_detailed['price'], errors='coerce')
    listings_detailed['price_clean'] = listings_detailed['price_clean'].fillna(0)

if 'availability_365' in listings_detailed.columns:
    listings_detailed['availability_365'] = pd.to_numeric(listings_detailed['availability_365'], errors='coerce').fillna(0)

listings_detailed['income'] = listings_detailed['price_clean'] * (365 - listings_detailed['availability_365'])
initial_count = len(listings_detailed)
listings_detailed_income = listings_detailed[listings_detailed['income'] > 0].copy()
print(f"  移除了 {initial_count - len(listings_detailed_income)} 条income <= 0的记录")
print(f"  剩余: {len(listings_detailed_income)} 条记录")

# 准备特征
feature_columns = [
    'host_response_time', 'host_response_rate', 'host_acceptance_rate', 
    'host_is_superhost', 'host_listings_count', 'host_total_listings_count',
    'host_verifications', 'host_has_profile_pic', 'host_identity_verified',
    'neighbourhood_cleansed', 'room_type', 'accommodates', 'bathrooms_text', 
    'bedrooms', 'beds', 'amenities', 'price', 'minimum_nights', 'maximum_nights',
    'minimum_nights_avg_ntm', 'maximum_nights_avg_ntm', 'has_availability',
    'number_of_reviews', 'number_of_reviews_ltm', 'number_of_reviews_l30d',
    'first_review', 'last_review', 'license',
    'calculated_host_listings_count', 'calculated_host_listings_count_entire_homes',
    'calculated_host_listings_count_private_rooms',
    'calculated_host_listings_count_shared_rooms', 'reviews_per_month'
]

amenity_cols = [col for col in listings_detailed_income.columns if col.startswith('amenity_')]
feature_columns.extend(amenity_cols)

if 'bathrooms' in listings_detailed_income.columns:
    feature_columns.append('bathrooms')

available_features = [col for col in feature_columns if col in listings_detailed_income.columns]
X_income = listings_detailed_income[available_features].copy()
y_income = listings_detailed_income['income'].copy()

print("\n4. 特征工程 / Feature Engineering...")
X_income = feature_engineering(X_income, y_income)
print(f"  最终特征数: {X_income.shape[1]}")

print("\n5. 数据分割 / Data Splitting...")
# 异常值处理
Q1 = y_income.quantile(0.25)
Q3 = y_income.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
mask = (y_income >= lower_bound) & (y_income <= upper_bound)
X_income = X_income[mask].copy()
y_income = y_income[mask].copy()

X_train_income, X_test_income, y_train_income, y_test_income = train_test_split(
    X_income, y_income, test_size=0.2, random_state=42
)
print(f"  训练集: {len(X_train_income)} 条")
print(f"  测试集: {len(X_test_income)} 条")

print("\n6. 特征标准化 / Feature Standardization...")
scaler_income = StandardScaler()
X_train_income_scaled = scaler_income.fit_transform(X_train_income)
X_test_income_scaled = scaler_income.transform(X_test_income)
X_train_income_scaled = pd.DataFrame(X_train_income_scaled, columns=X_train_income.columns, index=X_train_income.index)
X_test_income_scaled = pd.DataFrame(X_test_income_scaled, columns=X_test_income.columns, index=X_test_income.index)

print("\n7. 训练线性回归模型 / Training Linear Regression Model...")
start_time = time.time()
lr_model_income = LinearRegression()
lr_model_income.fit(X_train_income_scaled, y_train_income)
elapsed_time = time.time() - start_time
print(f"  [OK] 模型训练完成（耗时: {elapsed_time:.2f}秒）")

# 预测和评估
y_train_pred_income = lr_model_income.predict(X_train_income_scaled)
y_test_pred_income = lr_model_income.predict(X_test_income_scaled)

train_rmse_income = np.sqrt(mean_squared_error(y_train_income, y_train_pred_income))
test_rmse_income = np.sqrt(mean_squared_error(y_test_income, y_test_pred_income))
train_mae_income = mean_absolute_error(y_train_income, y_train_pred_income)
test_mae_income = mean_absolute_error(y_test_income, y_test_pred_income)
train_r2_income = r2_score(y_train_income, y_train_pred_income)
test_r2_income = r2_score(y_test_income, y_test_pred_income)

print(f"\n收入预测模型 - 训练集指标 / Income Model - Training Metrics:")
print(f"  RMSE: {train_rmse_income:,.2f}")
print(f"  MAE: {train_mae_income:,.2f}")
print(f"  R²: {train_r2_income:.4f}")

print(f"\n收入预测模型 - 测试集指标 / Income Model - Test Metrics:")
print(f"  RMSE: {test_rmse_income:,.2f}")
print(f"  MAE: {test_mae_income:,.2f}")
print(f"  R²: {test_r2_income:.4f}")

# ============================================================================
# 任务2: 评分预测 / Task 2: Rating Prediction
# ============================================================================

print("\n" + "=" * 80)
print("任务2: 评分预测模型 / Task 2: Rating Prediction Model")
print("=" * 80)

print("\n1. 准备数据 / Preparing Data...")
listings_detailed_rating = listings_detailed[listings_detailed['review_scores_rating'].notna()].copy()
listings_detailed_rating['review_scores_rating'] = pd.to_numeric(listings_detailed_rating['review_scores_rating'], errors='coerce')
listings_detailed_rating = listings_detailed_rating[
    (listings_detailed_rating['review_scores_rating'] >= 0) & 
    (listings_detailed_rating['review_scores_rating'] <= 100)
].copy()
print(f"  有效评分记录: {len(listings_detailed_rating)} 条")

# 准备特征（排除 review_scores 相关列）
feature_columns_rating = [
    'host_response_time', 'host_response_rate', 'host_acceptance_rate', 
    'host_is_superhost', 'host_listings_count', 'host_total_listings_count',
    'host_verifications', 'host_has_profile_pic', 'host_identity_verified',
    'neighbourhood_cleansed', 'room_type', 'accommodates', 'bathrooms_text', 
    'bedrooms', 'beds', 'amenities', 'price', 'minimum_nights', 'maximum_nights',
    'minimum_nights_avg_ntm', 'maximum_nights_avg_ntm', 'has_availability',
    'availability_30', 'availability_60', 'availability_90', 'availability_365',
    'number_of_reviews', 'number_of_reviews_ltm', 'number_of_reviews_l30d',
    'first_review', 'last_review', 'license',
    'calculated_host_listings_count', 'calculated_host_listings_count_entire_homes',
    'calculated_host_listings_count_private_rooms',
    'calculated_host_listings_count_shared_rooms', 'reviews_per_month'
]

amenity_cols_rating = [col for col in listings_detailed_rating.columns if col.startswith('amenity_')]
feature_columns_rating.extend(amenity_cols_rating)

if 'bathrooms' in listings_detailed_rating.columns:
    feature_columns_rating.append('bathrooms')

if 'price_clean' in listings_detailed_rating.columns and 'accommodates' in listings_detailed_rating.columns:
    listings_detailed_rating['accommodates'] = pd.to_numeric(listings_detailed_rating['accommodates'], errors='coerce')
    listings_detailed_rating['accommodates'] = listings_detailed_rating['accommodates'].replace(0, np.nan).fillna(1)
    listings_detailed_rating['Average_price_per_person'] = listings_detailed_rating['price_clean'] / listings_detailed_rating['accommodates']
    listings_detailed_rating['Average_price_per_person'] = listings_detailed_rating['Average_price_per_person'].fillna(0).replace([np.inf, -np.inf], 0)
    feature_columns_rating.append('Average_price_per_person')

exclude_cols_rating = [
    'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness',
    'review_scores_checkin', 'review_scores_communication', 'review_scores_location',
    'review_scores_value'
]
feature_columns_rating = [col for col in feature_columns_rating if col not in exclude_cols_rating]

available_features_rating = [col for col in feature_columns_rating if col in listings_detailed_rating.columns]
X_rating = listings_detailed_rating[available_features_rating].copy()
y_rating = listings_detailed_rating['review_scores_rating'].copy()

print("\n2. 特征工程 / Feature Engineering...")
X_rating = feature_engineering(X_rating, y_rating, exclude_cols=exclude_cols_rating)
print(f"  最终特征数: {X_rating.shape[1]}")

print("\n3. 数据分割 / Data Splitting...")
# 异常值处理
Q1 = y_rating.quantile(0.25)
Q3 = y_rating.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
mask = (y_rating >= lower_bound) & (y_rating <= upper_bound)
X_rating = X_rating[mask].copy()
y_rating = y_rating[mask].copy()

X_train_rating, X_test_rating, y_train_rating, y_test_rating = train_test_split(
    X_rating, y_rating, test_size=0.2, random_state=42
)
print(f"  训练集: {len(X_train_rating)} 条")
print(f"  测试集: {len(X_test_rating)} 条")

print("\n4. 特征标准化 / Feature Standardization...")
scaler_rating = StandardScaler()
X_train_rating_scaled = scaler_rating.fit_transform(X_train_rating)
X_test_rating_scaled = scaler_rating.transform(X_test_rating)
X_train_rating_scaled = pd.DataFrame(X_train_rating_scaled, columns=X_train_rating.columns, index=X_train_rating.index)
X_test_rating_scaled = pd.DataFrame(X_test_rating_scaled, columns=X_test_rating.columns, index=X_test_rating.index)

print("\n5. 训练线性回归模型 / Training Linear Regression Model...")
start_time = time.time()
lr_model_rating = LinearRegression()
lr_model_rating.fit(X_train_rating_scaled, y_train_rating)
elapsed_time = time.time() - start_time
print(f"  [OK] 模型训练完成（耗时: {elapsed_time:.2f}秒）")

# 预测和评估
y_train_pred_rating = lr_model_rating.predict(X_train_rating_scaled)
y_test_pred_rating = lr_model_rating.predict(X_test_rating_scaled)

train_rmse_rating = np.sqrt(mean_squared_error(y_train_rating, y_train_pred_rating))
test_rmse_rating = np.sqrt(mean_squared_error(y_test_rating, y_test_pred_rating))
train_mae_rating = mean_absolute_error(y_train_rating, y_train_pred_rating)
test_mae_rating = mean_absolute_error(y_test_rating, y_test_pred_rating)
train_r2_rating = r2_score(y_train_rating, y_train_pred_rating)
test_r2_rating = r2_score(y_test_rating, y_test_pred_rating)

print(f"\n评分预测模型 - 训练集指标 / Rating Model - Training Metrics:")
print(f"  RMSE: {train_rmse_rating:.4f}")
print(f"  MAE: {train_mae_rating:.4f}")
print(f"  R²: {train_r2_rating:.4f}")

print(f"\n评分预测模型 - 测试集指标 / Rating Model - Test Metrics:")
print(f"  RMSE: {test_rmse_rating:.4f}")
print(f"  MAE: {test_mae_rating:.4f}")
print(f"  R²: {test_r2_rating:.4f}")

# ============================================================================
# 模型对比 / Model Comparison
# ============================================================================

print("\n" + "=" * 80)
print("模型对比总结 / Model Comparison Summary")
print("=" * 80)

comparison_data = {
    'Task': ['Income Prediction', 'Rating Prediction'],
    'Train_RMSE': [train_rmse_income, train_rmse_rating],
    'Test_RMSE': [test_rmse_income, test_rmse_rating],
    'Train_MAE': [train_mae_income, train_mae_rating],
    'Test_MAE': [test_mae_income, test_mae_rating],
    'Train_R²': [train_r2_income, train_r2_rating],
    'Test_R²': [test_r2_income, test_r2_rating],
    'Train_Size': [len(X_train_income), len(X_train_rating)],
    'Test_Size': [len(X_test_income), len(X_test_rating)],
    'Features': [X_train_income.shape[1], X_train_rating.shape[1]]
}

comparison_df = pd.DataFrame(comparison_data)
print("\n模型性能对比 / Model Performance Comparison:")
print(comparison_df.to_string(index=False))

# 保存对比结果
comparison_df.to_csv(charts_dir / 'linear_regression_comparison.csv', index=False)
print(f"\n  [OK] 对比结果已保存到: {charts_dir / 'linear_regression_comparison.csv'}")

# 可视化对比
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. RMSE对比
axes[0, 0].bar(['Income\nPrediction', 'Rating\nPrediction'], 
               [test_rmse_income, test_rmse_rating], 
               color=['blue', 'green'], alpha=0.7)
axes[0, 0].set_ylabel('RMSE')
axes[0, 0].set_title('Test RMSE Comparison')
axes[0, 0].grid(True, alpha=0.3, axis='y')

# 2. MAE对比
axes[0, 1].bar(['Income\nPrediction', 'Rating\nPrediction'], 
               [test_mae_income, test_mae_rating], 
               color=['blue', 'green'], alpha=0.7)
axes[0, 1].set_ylabel('MAE')
axes[0, 1].set_title('Test MAE Comparison')
axes[0, 1].grid(True, alpha=0.3, axis='y')

# 3. R²对比
axes[0, 2].bar(['Income\nPrediction', 'Rating\nPrediction'], 
               [test_r2_income, test_r2_rating], 
               color=['blue', 'green'], alpha=0.7)
axes[0, 2].set_ylabel('R²')
axes[0, 2].set_title('Test R² Comparison')
axes[0, 2].set_ylim([0, 1])
axes[0, 2].grid(True, alpha=0.3, axis='y')

# 4. 收入预测 - 预测值 vs 真实值
axes[1, 0].scatter(y_test_income, y_test_pred_income, alpha=0.5, s=10, color='blue')
axes[1, 0].plot([y_test_income.min(), y_test_income.max()], 
                [y_test_income.min(), y_test_income.max()], 'r--', lw=2)
axes[1, 0].set_xlabel('True Income')
axes[1, 0].set_ylabel('Predicted Income')
axes[1, 0].set_title(f'Income Prediction (R² = {test_r2_income:.4f})')
axes[1, 0].grid(True, alpha=0.3)

# 5. 评分预测 - 预测值 vs 真实值
axes[1, 1].scatter(y_test_rating, y_test_pred_rating, alpha=0.5, s=10, color='green')
axes[1, 1].plot([y_test_rating.min(), y_test_rating.max()], 
                [y_test_rating.min(), y_test_rating.max()], 'r--', lw=2)
axes[1, 1].set_xlabel('True Rating')
axes[1, 1].set_ylabel('Predicted Rating')
axes[1, 1].set_title(f'Rating Prediction (R² = {test_r2_rating:.4f})')
axes[1, 1].grid(True, alpha=0.3)

# 6. 过拟合分析
overfitting_income = ((test_rmse_income - train_rmse_income) / train_rmse_income) * 100
overfitting_rating = ((test_rmse_rating - train_rmse_rating) / train_rmse_rating) * 100
axes[1, 2].bar(['Income\nPrediction', 'Rating\nPrediction'], 
               [overfitting_income, overfitting_rating], 
               color=['blue', 'green'], alpha=0.7)
axes[1, 2].set_ylabel('Overfitting Ratio (%)')
axes[1, 2].set_title('Overfitting Analysis')
axes[1, 2].axhline(y=15, color='r', linestyle='--', label='Warning Threshold (15%)')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(charts_dir / 'linear_regression_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"\n  [OK] 可视化图表已保存到: {charts_dir / 'linear_regression_comparison.png'}")

print("\n" + "=" * 80)
print("线性回归模型对比完成！/ Linear Regression Comparison Complete!")
print("=" * 80)

