"""
Linear Regression & SVM Income Prediction Model
线性回归和支持向量机收入预测模型

本脚本使用线性回归和支持向量机预测房源收入（income = price × (365 - availability_365)）
用于与XGBoost模型进行对比
This script uses Linear Regression and SVM to predict listing income
for comparison with XGBoost model
"""

import os
import re
import json
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
import sys
from pathlib import Path

# 添加 EDA 目录到路径，以便导入 utils 模块 / Add EDA directory to path for importing utils module
sys.path.insert(0, str(Path(__file__).parent.parent / 'EDA'))
from utils import setup_plotting, get_project_paths

# 设置中文字体支持 / Set Chinese font support
setup_plotting()

# 获取项目根目录路径 / Get project root directory path
project_root, data_dir, charts_eda_dir, charts_model_dir = get_project_paths()
charts_dir = charts_model_dir  # 使用模型目录 / Use model directory

# 设置控制台编码（Windows兼容）
import sys
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

print("=" * 80)
print("Linear Regression & SVM Income Prediction Model")
print("线性回归和支持向量机收入预测模型")
print("=" * 80)

# ============================================================================
# 1. 加载数据 / Load Data
# ============================================================================

print("\n1. 加载数据 / Loading Data...")
listings_detailed = pd.read_excel(data_dir / 'listings_detailed.xlsx')
print(f"  [OK] 数据加载完成: {len(listings_detailed)} 行 × {len(listings_detailed.columns)} 列")

# ============================================================================
# 2. 数据清洗 / Data Cleaning (基于 Feature_engine.py)
# ============================================================================

print("\n2. 数据清洗 / Data Cleaning...")

# 2.1 host_response_time 处理
order = ['no_response', 'a few day or more', 'within a day', 'within a few hours', 'within an hour']
cat_type = pd.CategoricalDtype(categories=order, ordered=True)
listings_detailed['host_response_time'] = listings_detailed['host_response_time'].fillna('no_response')
listings_detailed['host_response_time'] = listings_detailed['host_response_time'].astype(cat_type)

# 2.2 host_response_rate 处理
def convert_to_flat(x):
    if pd.isna(x):
        return 0
    if isinstance(x, str):
        return float(x.rstrip('%'))/100
    else:
        return float(x)/100

listings_detailed['host_response_rate'] = listings_detailed['host_response_rate'].fillna(0)
listings_detailed['host_response_rate'] = listings_detailed['host_response_rate'].apply(convert_to_flat)

# 2.3 host_acceptance_rate 处理
if 'host_acceptance_rate' in listings_detailed.columns:
    listings_detailed['host_acceptance_rate'] = listings_detailed['host_acceptance_rate'].fillna(0)
    listings_detailed['host_acceptance_rate'] = listings_detailed['host_acceptance_rate'].apply(convert_to_flat)

# 2.4 host_is_superhost 处理
listings_detailed['host_is_superhost'] = listings_detailed['host_is_superhost'].fillna(False)

# 2.5 host_listings_count 和 host_total_listings_count 处理
listings_detailed['host_listings_count'] = listings_detailed['host_listings_count'].fillna(0)
listings_detailed['host_total_listings_count'] = listings_detailed['host_total_listings_count'].fillna(0)

# 2.6 bathrooms_text 处理
def extract_bathrooms_number(x):
    if pd.isna(x):
        return "missing"
    if 'half-bath' in str(x).lower():
        return 0.5
    match = re.search(r'(\d+\.?\d*)', str(x))
    if match:
        return float(match.group(1))
    else:
        return "missing"

listings_detailed['bathrooms'] = listings_detailed['bathrooms_text'].apply(extract_bathrooms_number)

# 2.7 bedrooms 和 beds 处理
listings_detailed['bedrooms'] = listings_detailed['bedrooms'].fillna("missing")
listings_detailed['beds'] = listings_detailed['beds'].fillna("missing")

# 2.8 amenities Multi-hot Encoding
def parse_amenities(x):
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

print("  解析 amenities 列...")
listings_detailed['amenities_parsed'] = listings_detailed['amenities'].apply(parse_amenities)

all_amenities = []
for amenity_list in listings_detailed['amenities_parsed']:
    all_amenities.extend(amenity_list)

amenity_counts = Counter(all_amenities)
min_frequency = 5
frequent_amenities = {amenity: count for amenity, count in amenity_counts.items() if count >= min_frequency}

def clean_column_name(amenity_name):
    cleaned = re.sub(r'[^\w\s-]', '_', str(amenity_name))
    cleaned = re.sub(r'[\s-]+', '_', cleaned)
    cleaned = re.sub(r'_+', '_', cleaned)
    cleaned = cleaned.strip('_')
    return cleaned.lower()

amenity_columns = sorted(frequent_amenities.keys())
# 使用pd.concat一次性创建所有amenity列，避免DataFrame碎片化警告
amenity_data = {}
for amenity in amenity_columns:
    col_name = f'amenity_{clean_column_name(amenity)}'
    amenity_data[col_name] = listings_detailed['amenities_parsed'].apply(
        lambda x, a=amenity: 1 if a in x else 0
    )
# 一次性添加所有amenity列
amenity_df = pd.DataFrame(amenity_data, index=listings_detailed.index)
listings_detailed = pd.concat([listings_detailed, amenity_df], axis=1)

print(f"  创建了 {len(amenity_columns)} 个amenity特征列")

# 2.9 minimum_nights 和 maximum_nights 处理
listings_detailed['minimum_nights'] = listings_detailed['minimum_nights'].clip(upper=365)
listings_detailed['maximum_nights'] = listings_detailed['maximum_nights'].clip(upper=365)

# 2.10 minimum_nights_avg_ntm 和 maximum_nights_avg_ntm 处理
listings_detailed['minimum_nights_avg_ntm'] = listings_detailed['minimum_nights_avg_ntm'].fillna("missing")
listings_detailed['maximum_nights_avg_ntm'] = listings_detailed['maximum_nights_avg_ntm'].fillna("missing")

# 2.11 review_scores 相关列处理
review_scores_cols = [
    'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness',
    'review_scores_checkin', 'review_scores_communication', 'review_scores_location',
    'review_scores_value'
]
for col in review_scores_cols:
    if col in listings_detailed.columns:
        listings_detailed[col] = listings_detailed[col].fillna("missing")

# 2.12 reviews_per_month 处理
listings_detailed['reviews_per_month'] = listings_detailed['reviews_per_month'].fillna('missing')

# 2.13 license 处理
if 'license' in listings_detailed.columns:
    listings_detailed['license'] = listings_detailed['license'].notna()  # True表示有license，False表示没有license

print("  [OK] 数据清洗完成")

# ============================================================================
# 3. 创建目标变量 / Create Target Variable
# ============================================================================

print("\n3. 创建目标变量 / Creating Target Variable...")

# income = price × (365 - availability_365)
# 处理price列（可能包含$符号和逗号）
if 'price' in listings_detailed.columns:
    if listings_detailed['price'].dtype == 'object':
        listings_detailed['price_clean'] = listings_detailed['price'].astype(str).str.replace('$', '').str.replace(',', '')
        listings_detailed['price_clean'] = pd.to_numeric(listings_detailed['price_clean'], errors='coerce')
    else:
        listings_detailed['price_clean'] = pd.to_numeric(listings_detailed['price'], errors='coerce')
    listings_detailed['price_clean'] = listings_detailed['price_clean'].fillna(0)
else:
    print("  警告: price列不存在")
    listings_detailed['price_clean'] = 0

# 确保availability_365是数值型
if 'availability_365' in listings_detailed.columns:
    listings_detailed['availability_365'] = pd.to_numeric(listings_detailed['availability_365'], errors='coerce').fillna(0)
else:
    print("  警告: availability_365列不存在")
    listings_detailed['availability_365'] = 0

# 创建目标变量
listings_detailed['income'] = listings_detailed['price_clean'] * (365 - listings_detailed['availability_365'])

# 移除income为0或负数的记录（数据质量问题）
initial_count = len(listings_detailed)
listings_detailed = listings_detailed[listings_detailed['income'] > 0].copy()
print(f"  移除了 {initial_count - len(listings_detailed)} 条income <= 0的记录")
print(f"  [OK] 目标变量创建完成，剩余 {len(listings_detailed)} 条记录")

# ============================================================================
# 4. 准备特征 / Prepare Features
# ============================================================================

print("\n4. 准备特征 / Preparing Features...")

# 定义特征列表
feature_columns = [
    'host_response_time', 'host_response_rate', 'host_acceptance_rate', 
    'host_is_superhost', 'host_listings_count', 'host_total_listings_count',
    'host_verifications', 'host_has_profile_pic', 'host_identity_verified',
    'neighbourhood_cleansed', 'room_type', 'accommodates', 'bathrooms_text', 
    'bedrooms', 'beds', 'amenities', 'price', 'minimum_nights', 'maximum_nights',
    'minimum_nights_avg_ntm', 'maximum_nights_avg_ntm', 'has_availability',
    'number_of_reviews', 'number_of_reviews_ltm', 'number_of_reviews_l30d',
    'first_review', 'last_review', 'review_scores_rating', 'review_scores_accuracy',
    'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication',
    'review_scores_location', 'review_scores_value', 'license',
    'calculated_host_listings_count', 'calculated_host_listings_count_entire_homes',
    'calculated_host_listings_count_private_rooms',
    'calculated_host_listings_count_shared_rooms', 'reviews_per_month'
]

# 添加amenity特征列
amenity_cols = [col for col in listings_detailed.columns if col.startswith('amenity_')]
feature_columns.extend(amenity_cols)

# 添加bathrooms列（从bathrooms_text提取的）
if 'bathrooms' in listings_detailed.columns:
    feature_columns.append('bathrooms')

# 检查哪些特征存在
available_features = [col for col in feature_columns if col in listings_detailed.columns]
missing_features = [col for col in feature_columns if col not in listings_detailed.columns]

print(f"  可用特征数: {len(available_features)}")
if missing_features:
    print(f"  缺失特征: {missing_features[:10]}...")  # 只显示前10个

# ============================================================================
# 5. 特征工程 / Feature Engineering
# ============================================================================

print("\n5. 特征工程 / Feature Engineering...")

# 创建特征数据框
X = listings_detailed[available_features].copy()
y = listings_detailed['income'].copy()

# 5.1 处理数值型特征
numeric_features = []
categorical_features = []

for col in X.columns:
    if X[col].dtype in ['int64', 'float64']:
        numeric_features.append(col)
    else:
        categorical_features.append(col)

print(f"  数值型特征: {len(numeric_features)}")
print(f"  分类特征: {len(categorical_features)}")

# 5.2 处理分类特征编码
label_encoders = {}
for col in categorical_features:
    # 如果是 Categorical 类型，先转换为 object 类型
    if pd.api.types.is_categorical_dtype(X[col]):
        X[col] = X[col].astype(str)
    
    # 处理缺失值（避免 downcasting 警告）
    X[col] = X[col].astype(str).fillna('missing')
    
    # 使用LabelEncoder编码
    le = LabelEncoder()
    try:
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
    except Exception as e:
        print(f"  警告: {col} 编码失败: {e}")
        X[col] = 0

# 5.3 处理数值型特征的缺失值
for col in numeric_features:
    X[col] = pd.to_numeric(X[col], errors='coerce')
    median_val = X[col].median()
    if pd.isna(median_val):
        X[col] = X[col].fillna(0)
    else:
        X[col] = X[col].fillna(median_val)

# 5.4 处理日期特征（first_review, last_review）
date_features = ['first_review', 'last_review']
date_new_cols = {}  # 使用字典收集新列，避免碎片化
for col in date_features:
    if col in X.columns:
        # 转换为日期类型
        date_series = pd.to_datetime(X[col], errors='coerce')
        # 提取日期特征
        date_new_cols[f'{col}_year'] = date_series.dt.year.fillna(0).astype(int)
        date_new_cols[f'{col}_month'] = date_series.dt.month.fillna(0).astype(int)
        date_new_cols[f'{col}_day'] = date_series.dt.day.fillna(0).astype(int)
        # 计算距离数据采集日期的天数（假设是2021-09-07）
        reference_date = pd.Timestamp('2021-09-07')
        date_new_cols[f'{col}_days_ago'] = (reference_date - date_series).dt.days.fillna(0).astype(int)
        # 标记删除原始日期列
        X = X.drop(col, axis=1)

# 一次性添加所有日期特征列，避免碎片化
if date_new_cols:
    date_df = pd.DataFrame(date_new_cols, index=X.index)
    X = pd.concat([X, date_df], axis=1)

# 5.5 处理host_verifications（可能是列表或字符串）
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

# 5.6 处理布尔特征
bool_features = ['host_is_superhost', 'host_has_profile_pic', 'host_identity_verified', 
                 'has_availability', 'license']
for col in bool_features:
    if col in X.columns:
        X[col] = X[col].astype(int)

# 5.7 移除包含inf或-inf的列
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(0)

print(f"  [OK] 特征工程完成，最终特征数: {X.shape[1]}")

# ============================================================================
# 6. 数据分割 / Data Splitting
# ============================================================================

print("\n6. 数据分割 / Data Splitting...")

# 移除目标变量中的异常值（使用IQR方法）
Q1 = y.quantile(0.25)
Q3 = y.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# 保留在合理范围内的数据
mask = (y >= lower_bound) & (y <= upper_bound)
X = X[mask].copy()
y = y[mask].copy()

print(f"  移除异常值后剩余: {len(X)} 条记录")

# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"  训练集: {len(X_train)} 条")
print(f"  测试集: {len(X_test)} 条")

# ============================================================================
# 7. 特征标准化 / Feature Standardization
# ============================================================================

print("\n7. 特征标准化 / Feature Standardization...")

# 线性回归和SVM对特征尺度敏感，需要进行标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 转换为DataFrame以保持列名
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

print("  [OK] 特征标准化完成")

# ============================================================================
# 8. 训练线性回归模型 / Train Linear Regression Model
# ============================================================================

print("\n8. 训练线性回归模型 / Training Linear Regression Model...")

# 训练线性回归模型
import time
start_time = time.time()
print(f"  开始训练（训练集大小: {len(X_train_scaled)} 条，特征数: {X_train_scaled.shape[1]}）...")
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
elapsed_time = time.time() - start_time
print(f"  [OK] 线性回归模型训练完成（耗时: {elapsed_time:.2f}秒）")

# 预测
lr_y_train_pred = lr_model.predict(X_train_scaled)
lr_y_test_pred = lr_model.predict(X_test_scaled)

# 计算评估指标
lr_train_rmse = np.sqrt(mean_squared_error(y_train, lr_y_train_pred))
lr_test_rmse = np.sqrt(mean_squared_error(y_test, lr_y_test_pred))
lr_train_mae = mean_absolute_error(y_train, lr_y_train_pred)
lr_test_mae = mean_absolute_error(y_test, lr_y_test_pred)
lr_train_r2 = r2_score(y_train, lr_y_train_pred)
lr_test_r2 = r2_score(y_test, lr_y_test_pred)

print(f"\n线性回归 - 训练集指标 / Linear Regression - Training Metrics:")
print(f"  RMSE: {lr_train_rmse:,.2f}")
print(f"  MAE: {lr_train_mae:,.2f}")
print(f"  R²: {lr_train_r2:.4f}")

print(f"\n线性回归 - 测试集指标 / Linear Regression - Test Metrics:")
print(f"  RMSE: {lr_test_rmse:,.2f}")
print(f"  MAE: {lr_test_mae:,.2f}")
print(f"  R²: {lr_test_r2:.4f}")

# ============================================================================
# 9. 训练支持向量机模型 / Train SVM Model
# ============================================================================

print("\n9. 训练支持向量机模型 / Training SVM Model...")

# 训练SVM模型（使用RBF核，针对回归问题使用SVR）
# 注意：SVM在大数据集上训练较慢，这里使用采样来加速训练
# 如果数据量太大，可以考虑使用LinearSVR或减少训练样本
from sklearn.svm import LinearSVR

# 对于大数据集，使用采样来加速训练
if len(X_train_scaled) > 3000:
    print(f"  训练集较大 ({len(X_train_scaled)} 条)，使用采样训练以加速...")
    # 使用随机采样，保留 3000 条样本进行训练
    sample_size = min(3000, len(X_train_scaled))
    print(f"  从 {len(X_train_scaled)} 条样本中随机采样 {sample_size} 条进行训练...")
    
    from sklearn.utils import resample
    # 随机采样
    indices = np.random.choice(len(X_train_scaled), size=sample_size, replace=False)
    X_train_sampled = X_train_scaled.iloc[indices] if isinstance(X_train_scaled, pd.DataFrame) else X_train_scaled[indices]
    y_train_sampled = y_train.iloc[indices] if isinstance(y_train, pd.Series) else y_train[indices]
    
    print(f"  采样完成，开始训练 LinearSVR...")
    svm_model = LinearSVR(
        C=1.0,
        epsilon=0.1,
        max_iter=5000,
        tol=1e-3,
        random_state=42
    )
    import warnings
    import time
    start_time = time.time()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
        svm_model.fit(X_train_sampled, y_train_sampled)
    elapsed_time = time.time() - start_time
    print(f"  [OK] LinearSVR 模型训练完成（耗时: {elapsed_time:.2f}秒）")
    print(f"  注意: 模型使用采样数据训练，性能可能略低于使用全部数据")
else:
    # 小数据集，使用 RBF 核
    print("  训练集较小，使用 RBF 核 SVM 进行训练...")
    svm_model = SVR(
        kernel='rbf',
        C=1.0,
        epsilon=0.1,
        gamma='scale',
        max_iter=5000,
        tol=1e-3
    )
    import warnings
    import time
    start_time = time.time()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
        svm_model.fit(X_train_scaled, y_train)
    elapsed_time = time.time() - start_time
    print(f"  [OK] RBF 核 SVM 模型训练完成（耗时: {elapsed_time:.2f}秒）")

# 检查是否收敛
if hasattr(svm_model, 'n_iter_') and svm_model.n_iter_ >= 5000:
    print("  警告: SVM模型可能未完全收敛，但已训练完成")
else:
    print(f"  [OK] 支持向量机模型训练完成（迭代次数: {svm_model.n_iter_ if hasattr(svm_model, 'n_iter_') else 'N/A'}）")

# 预测
svm_y_train_pred = svm_model.predict(X_train_scaled)
svm_y_test_pred = svm_model.predict(X_test_scaled)

# 计算评估指标
svm_train_rmse = np.sqrt(mean_squared_error(y_train, svm_y_train_pred))
svm_test_rmse = np.sqrt(mean_squared_error(y_test, svm_y_test_pred))
svm_train_mae = mean_absolute_error(y_train, svm_y_train_pred)
svm_test_mae = mean_absolute_error(y_test, svm_y_test_pred)
svm_train_r2 = r2_score(y_train, svm_y_train_pred)
svm_test_r2 = r2_score(y_test, svm_y_test_pred)

print(f"\n支持向量机 - 训练集指标 / SVM - Training Metrics:")
print(f"  RMSE: {svm_train_rmse:,.2f}")
print(f"  MAE: {svm_train_mae:,.2f}")
print(f"  R²: {svm_train_r2:.4f}")

print(f"\n支持向量机 - 测试集指标 / SVM - Test Metrics:")
print(f"  RMSE: {svm_test_rmse:,.2f}")
print(f"  MAE: {svm_test_mae:,.2f}")
print(f"  R²: {svm_test_r2:.4f}")

# ============================================================================
# 9.5 特征重要性分析 / Feature Importance Analysis
# ============================================================================

print("\n9.5 特征重要性分析 / Feature Importance Analysis...")

# 线性回归特征重要性（使用系数的绝对值）
lr_coefficients = np.abs(lr_model.coef_)
lr_feature_importance = pd.DataFrame({
    'feature': X_train_scaled.columns,
    'importance': lr_coefficients
}).sort_values('importance', ascending=False)

print("\n线性回归 - 前20个最重要特征 / Linear Regression - Top 20 Most Important Features:")
print(lr_feature_importance.head(20).to_string(index=False))

# 保存线性回归特征重要性
lr_feature_importance.to_csv(charts_dir / 'linear_regression_feature_importance.csv', index=False)
print(f"\n  [OK] 线性回归特征重要性已保存到: {charts_dir / 'linear_regression_feature_importance.csv'}")

# SVM特征重要性（使用permutation importance）
print("\n  计算SVM特征重要性（使用permutation importance，可能需要一些时间）...")
svm_perm_importance = permutation_importance(
    svm_model, X_test_scaled, y_test,
    n_repeats=10,
    random_state=42,
    n_jobs=-1,
    scoring='neg_mean_squared_error'
)

svm_feature_importance = pd.DataFrame({
    'feature': X_train_scaled.columns,
    'importance': svm_perm_importance.importances_mean
}).sort_values('importance', ascending=False)

print("\n支持向量机 - 前20个最重要特征 / SVM - Top 20 Most Important Features:")
print(svm_feature_importance.head(20).to_string(index=False))

# 保存SVM特征重要性
svm_feature_importance.to_csv(charts_dir / 'svm_feature_importance.csv', index=False)
print(f"\n  [OK] SVM特征重要性已保存到: {charts_dir / 'svm_feature_importance.csv'}")

# ============================================================================
# 10. 模型对比分析 / Model Comparison
# ============================================================================

print("\n10. 模型对比分析 / Model Comparison...")

# 创建对比表格
comparison_data = {
    'Model': ['Linear Regression', 'SVM'],
    'Train_RMSE': [lr_train_rmse, svm_train_rmse],
    'Test_RMSE': [lr_test_rmse, svm_test_rmse],
    'Train_MAE': [lr_train_mae, svm_train_mae],
    'Test_MAE': [lr_test_mae, svm_test_mae],
    'Train_R²': [lr_train_r2, svm_train_r2],
    'Test_R²': [lr_test_r2, svm_test_r2]
}

comparison_df = pd.DataFrame(comparison_data)
print("\n模型性能对比 / Model Performance Comparison:")
print(comparison_df.to_string(index=False))

# 保存对比结果
comparison_df.to_csv(charts_dir / 'linear_svm_model_comparison.csv', index=False)
print(f"\n  [OK] 模型对比结果已保存到: {charts_dir / 'linear_svm_model_comparison.csv'}")

# 过拟合分析
print("\n过拟合分析 / Overfitting Analysis:")

# 线性回归
lr_rmse_diff = lr_test_rmse - lr_train_rmse
lr_overfitting_ratio = (lr_rmse_diff / lr_train_rmse) * 100
print(f"\n线性回归 / Linear Regression:")
print(f"  训练集RMSE: {lr_train_rmse:,.2f}")
print(f"  测试集RMSE: {lr_test_rmse:,.2f}")
print(f"  RMSE差异: {lr_rmse_diff:,.2f} (测试集 - 训练集)")
print(f"  泛化差距: {lr_overfitting_ratio:.2f}%")

# 支持向量机
svm_rmse_diff = svm_test_rmse - svm_train_rmse
svm_overfitting_ratio = (svm_rmse_diff / svm_train_rmse) * 100
print(f"\n支持向量机 / SVM:")
print(f"  训练集RMSE: {svm_train_rmse:,.2f}")
print(f"  测试集RMSE: {svm_test_rmse:,.2f}")
print(f"  RMSE差异: {svm_rmse_diff:,.2f} (测试集 - 训练集)")
print(f"  泛化差距: {svm_overfitting_ratio:.2f}%")

# ============================================================================
# 11. 可视化结果 / Visualization
# ============================================================================

print("\n11. 生成可视化图表 / Generating Visualizations...")

# 创建2x3的子图布局：线性回归3个图，SVM3个图
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 11.1 线性回归 - 预测值 vs 真实值（训练集）
axes[0, 0].scatter(y_train, lr_y_train_pred, alpha=0.5, s=10, color='blue')
axes[0, 0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
axes[0, 0].set_xlabel('True Income')
axes[0, 0].set_ylabel('Predicted Income')
axes[0, 0].set_title(f'Linear Regression - Training Set (R² = {lr_train_r2:.4f})')
axes[0, 0].grid(True, alpha=0.3)

# 11.2 线性回归 - 预测值 vs 真实值（测试集）
axes[0, 1].scatter(y_test, lr_y_test_pred, alpha=0.5, s=10, color='blue')
axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 1].set_xlabel('True Income')
axes[0, 1].set_ylabel('Predicted Income')
axes[0, 1].set_title(f'Linear Regression - Test Set (R² = {lr_test_r2:.4f})')
axes[0, 1].grid(True, alpha=0.3)

# 11.3 线性回归 - 残差分布
lr_residuals = y_test - lr_y_test_pred
axes[0, 2].hist(lr_residuals, bins=50, edgecolor='black', color='blue', alpha=0.7)
axes[0, 2].set_xlabel('Residuals')
axes[0, 2].set_ylabel('Frequency')
axes[0, 2].set_title('Linear Regression - Residual Distribution')
axes[0, 2].axvline(x=0, color='r', linestyle='--', lw=2)
axes[0, 2].grid(True, alpha=0.3)

# 11.4 支持向量机 - 预测值 vs 真实值（训练集）
axes[1, 0].scatter(y_train, svm_y_train_pred, alpha=0.5, s=10, color='green')
axes[1, 0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
axes[1, 0].set_xlabel('True Income')
axes[1, 0].set_ylabel('Predicted Income')
axes[1, 0].set_title(f'SVM - Training Set (R² = {svm_train_r2:.4f})')
axes[1, 0].grid(True, alpha=0.3)

# 11.5 支持向量机 - 预测值 vs 真实值（测试集）
axes[1, 1].scatter(y_test, svm_y_test_pred, alpha=0.5, s=10, color='green')
axes[1, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[1, 1].set_xlabel('True Income')
axes[1, 1].set_ylabel('Predicted Income')
axes[1, 1].set_title(f'SVM - Test Set (R² = {svm_test_r2:.4f})')
axes[1, 1].grid(True, alpha=0.3)

# 11.6 支持向量机 - 残差分布
svm_residuals = y_test - svm_y_test_pred
axes[1, 2].hist(svm_residuals, bins=50, edgecolor='black', color='green', alpha=0.7)
axes[1, 2].set_xlabel('Residuals')
axes[1, 2].set_ylabel('Frequency')
axes[1, 2].set_title('SVM - Residual Distribution')
axes[1, 2].axvline(x=0, color='r', linestyle='--', lw=2)
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(charts_dir / 'linear_svm_model_results.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"  [OK] 可视化图表已保存到: {charts_dir / 'linear_svm_model_results.png'}")

# 创建模型性能对比图
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# RMSE对比
models = ['Linear\nRegression', 'SVM']
train_rmse_values = [lr_train_rmse, svm_train_rmse]
test_rmse_values = [lr_test_rmse, svm_test_rmse]

x = np.arange(len(models))
width = 0.35
axes[0].bar(x - width/2, train_rmse_values, width, label='Train', alpha=0.8)
axes[0].bar(x + width/2, test_rmse_values, width, label='Test', alpha=0.8)
axes[0].set_xlabel('Model')
axes[0].set_ylabel('RMSE')
axes[0].set_title('RMSE Comparison')
axes[0].set_xticks(x)
axes[0].set_xticklabels(models)
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis='y')

# MAE对比
train_mae_values = [lr_train_mae, svm_train_mae]
test_mae_values = [lr_test_mae, svm_test_mae]

axes[1].bar(x - width/2, train_mae_values, width, label='Train', alpha=0.8)
axes[1].bar(x + width/2, test_mae_values, width, label='Test', alpha=0.8)
axes[1].set_xlabel('Model')
axes[1].set_ylabel('MAE')
axes[1].set_title('MAE Comparison')
axes[1].set_xticks(x)
axes[1].set_xticklabels(models)
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

# R²对比
train_r2_values = [lr_train_r2, svm_train_r2]
test_r2_values = [lr_test_r2, svm_test_r2]

axes[2].bar(x - width/2, train_r2_values, width, label='Train', alpha=0.8)
axes[2].bar(x + width/2, test_r2_values, width, label='Test', alpha=0.8)
axes[2].set_xlabel('Model')
axes[2].set_ylabel('R²')
axes[2].set_title('R² Comparison')
axes[2].set_xticks(x)
axes[2].set_xticklabels(models)
axes[2].legend()
axes[2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(charts_dir / 'linear_svm_model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"  [OK] 模型对比图表已保存到: {charts_dir / 'linear_svm_model_comparison.png'}")

# 创建特征重要性对比图
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# 线性回归特征重要性Top 20
lr_top_features = lr_feature_importance.head(20)
axes[0].barh(range(len(lr_top_features)), lr_top_features['importance'].values, color='blue', alpha=0.7)
axes[0].set_yticks(range(len(lr_top_features)))
axes[0].set_yticklabels(lr_top_features['feature'].values)
axes[0].set_xlabel('Importance (|Coefficient|)')
axes[0].set_title('Linear Regression - Top 20 Feature Importance')
axes[0].invert_yaxis()
axes[0].grid(True, alpha=0.3, axis='x')

# SVM特征重要性Top 20
svm_top_features = svm_feature_importance.head(20)
axes[1].barh(range(len(svm_top_features)), svm_top_features['importance'].values, color='green', alpha=0.7)
axes[1].set_yticks(range(len(svm_top_features)))
axes[1].set_yticklabels(svm_top_features['feature'].values)
axes[1].set_xlabel('Importance (Permutation)')
axes[1].set_title('SVM - Top 20 Feature Importance')
axes[1].invert_yaxis()
axes[1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(charts_dir / 'linear_svm_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"  [OK] 特征重要性图表已保存到: {charts_dir / 'linear_svm_feature_importance.png'}")

# ============================================================================
# 12. 保存模型 / Save Models
# ============================================================================

print("\n12. 保存模型 / Saving Models...")

import pickle

# 保存线性回归模型和标准化器
lr_model_path = charts_dir / 'linear_regression_income_model.pkl'
with open(lr_model_path, 'wb') as f:
    pickle.dump({'model': lr_model, 'scaler': scaler}, f)

print(f"  [OK] 线性回归模型已保存到: {lr_model_path}")

# 保存支持向量机模型和标准化器
svm_model_path = charts_dir / 'svm_income_model.pkl'
with open(svm_model_path, 'wb') as f:
    pickle.dump({'model': svm_model, 'scaler': scaler}, f)

print(f"  [OK] 支持向量机模型已保存到: {svm_model_path}")

print("\n" + "=" * 80)
print("线性回归和支持向量机模型训练完成！/ Linear Regression & SVM Model Training Complete!")
print("=" * 80)

