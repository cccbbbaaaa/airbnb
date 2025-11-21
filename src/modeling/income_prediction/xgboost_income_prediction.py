"""
XGBoost Income Prediction Model
XGBoost 收入预测模型

本脚本使用XGBoost预测房源收入（income = price × (365 - availability_365)）
This script uses XGBoost to predict listing income (income = price × (365 - availability_365))
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
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import sys

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
print("XGBoost Income Prediction Model")
print("XGBoost 收入预测模型")
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

# 创建人均价格变量 / Create Average Price per Person Variable
if 'accommodates' in listings_detailed.columns:
    # 确保accommodates是数值型且大于0
    listings_detailed['accommodates'] = pd.to_numeric(listings_detailed['accommodates'], errors='coerce')
    # 处理accommodates为0或缺失的情况（避免除零错误）
    listings_detailed['accommodates'] = listings_detailed['accommodates'].replace(0, np.nan)
    listings_detailed['accommodates'] = listings_detailed['accommodates'].fillna(1)  # 缺失值默认为1
    
    # 计算人均价格
    listings_detailed['Average_price_per_person'] = listings_detailed['price_clean'] / listings_detailed['accommodates']
    
    # 处理异常值（如果price_clean为0或负数，人均价格设为0）
    listings_detailed['Average_price_per_person'] = listings_detailed['Average_price_per_person'].fillna(0)
    listings_detailed['Average_price_per_person'] = listings_detailed['Average_price_per_person'].replace([np.inf, -np.inf], 0)
    
    print(f"  [OK] 人均价格变量创建完成 (Average_price_per_person = price_clean / accommodates)")
    print(f"  人均价格统计: 均值={listings_detailed['Average_price_per_person'].mean():.2f}, "
          f"中位数={listings_detailed['Average_price_per_person'].median():.2f}")
else:
    print("  警告: accommodates列不存在，无法创建人均价格变量")
    listings_detailed['Average_price_per_person'] = 0

# ============================================================================
# 3.1 Price与Income相关性分析 / Price-Income Correlation Analysis
# ============================================================================

print("\n3.1 Price与Income相关性分析 / Price-Income Correlation Analysis...")

from scipy.stats import pearsonr, spearmanr
from scipy.stats import linregress

# 提取有效数据（price > 0 且 income > 0）
valid_data = listings_detailed[
    (listings_detailed['price_clean'] > 0) & 
    (listings_detailed['income'] > 0)
].copy()

if len(valid_data) > 0:
    price = valid_data['price_clean'].values
    income = valid_data['income'].values
    
    # 1. Pearson相关系数（线性相关）
    pearson_corr, pearson_p = pearsonr(price, income)
    print(f"\n  Pearson相关系数 / Pearson Correlation Coefficient:")
    print(f"    相关系数 / Correlation: {pearson_corr:.4f}")
    print(f"    P值 / P-value: {pearson_p:.2e}")
    if pearson_corr > 0:
        print(f"    [正相关 / Positive Correlation] price越高，income越高")
    else:
        print(f"    [负相关 / Negative Correlation] price越高，income越低")
    
    # 2. Spearman秩相关系数（非线性相关，更稳健）
    spearman_corr, spearman_p = spearmanr(price, income)
    print(f"\n  Spearman秩相关系数 / Spearman Rank Correlation:")
    print(f"    相关系数 / Correlation: {spearman_corr:.4f}")
    print(f"    P值 / P-value: {spearman_p:.2e}")
    if spearman_corr > 0:
        print(f"    [正相关 / Positive Correlation] price越高，income越高")
    else:
        print(f"    [负相关 / Negative Correlation] price越高，income越低")
    
    # 3. 线性回归分析
    slope, intercept, r_value, p_value, std_err = linregress(price, income)
    print(f"\n  线性回归分析 / Linear Regression Analysis:")
    print(f"    斜率 / Slope: {slope:.4f} (price每增加1单位，income增加{slope:.4f})")
    print(f"    截距 / Intercept: {intercept:.2f}")
    print(f"    R²值 / R-squared: {r_value**2:.4f}")
    print(f"    P值 / P-value: {p_value:.2e}")
    
    # 4. 偏相关分析（控制availability_365的影响）
    if 'availability_365' in valid_data.columns:
        availability = valid_data['availability_365'].values
        
        # 计算偏相关系数：price和income的偏相关（控制availability_365）
        # 方法：先分别对availability_365回归，然后计算残差的相关系数
        from sklearn.linear_model import LinearRegression
        
        # price对availability_365的回归
        reg_price = LinearRegression()
        reg_price.fit(availability.reshape(-1, 1), price)
        price_residual = price - reg_price.predict(availability.reshape(-1, 1))
        
        # income对availability_365的回归
        reg_income = LinearRegression()
        reg_income.fit(availability.reshape(-1, 1), income)
        income_residual = income - reg_income.predict(availability.reshape(-1, 1))
        
        # 计算残差的相关系数（偏相关）
        partial_corr, partial_p = pearsonr(price_residual, income_residual)
        print(f"\n  偏相关分析 / Partial Correlation (控制availability_365):")
        print(f"    偏相关系数 / Partial Correlation: {partial_corr:.4f}")
        print(f"    P值 / P-value: {partial_p:.2e}")
        if partial_corr > 0:
            print(f"    [正相关 / Positive Correlation] 在控制availability_365后，price与income仍为正相关")
        else:
            print(f"    [负相关 / Negative Correlation] 在控制availability_365后，price与income为负相关")
    
    # 5. 可视化
    print(f"\n  生成相关性可视化图表...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 5.1 散点图（原始数据）
    axes[0, 0].scatter(price, income, alpha=0.3, s=10)
    axes[0, 0].set_xlabel('Price (price_clean)')
    axes[0, 0].set_ylabel('Income')
    axes[0, 0].set_title(f'Price vs Income (Pearson r={pearson_corr:.3f})')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 添加回归线
    x_line = np.linspace(price.min(), price.max(), 100)
    y_line = slope * x_line + intercept
    axes[0, 0].plot(x_line, y_line, 'r--', lw=2, label=f'y = {slope:.2f}x + {intercept:.2f}')
    axes[0, 0].legend()
    
    # 5.2 对数变换后的散点图（处理长尾分布）
    price_log = np.log1p(price)
    income_log = np.log1p(income)
    log_corr, _ = pearsonr(price_log, income_log)
    axes[0, 1].scatter(price_log, income_log, alpha=0.3, s=10)
    axes[0, 1].set_xlabel('Log(Price + 1)')
    axes[0, 1].set_ylabel('Log(Income + 1)')
    axes[0, 1].set_title(f'Log-Transformed (Pearson r={log_corr:.3f})')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 添加回归线
    slope_log, intercept_log, _, _, _ = linregress(price_log, income_log)
    x_line_log = np.linspace(price_log.min(), price_log.max(), 100)
    y_line_log = slope_log * x_line_log + intercept_log
    axes[0, 1].plot(x_line_log, y_line_log, 'r--', lw=2)
    
    # 5.3 价格分组的收入分布
    price_bins = pd.qcut(price, q=10, duplicates='drop')
    price_group_stats = valid_data.groupby(price_bins).agg({
        'income': ['mean', 'median', 'std'],
        'price_clean': 'mean'
    }).reset_index()
    price_group_stats.columns = ['price_range', 'income_mean', 'income_median', 'income_std', 'price_mean']
    
    axes[1, 0].plot(price_group_stats['price_mean'], price_group_stats['income_mean'], 
                    'o-', label='Mean Income', linewidth=2, markersize=8)
    axes[1, 0].fill_between(price_group_stats['price_mean'], 
                            price_group_stats['income_mean'] - price_group_stats['income_std'],
                            price_group_stats['income_mean'] + price_group_stats['income_std'],
                            alpha=0.3, label='±1 Std Dev')
    axes[1, 0].set_xlabel('Average Price')
    axes[1, 0].set_ylabel('Average Income')
    axes[1, 0].set_title('Income by Price Groups (10 Quantiles)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5.4 相关性热力图（price, income, availability_365）
    if 'availability_365' in valid_data.columns:
        corr_data = valid_data[['price_clean', 'income', 'availability_365']].corr()
        sns.heatmap(corr_data, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                   square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=axes[1, 1])
        axes[1, 1].set_title('Correlation Heatmap')
    
    plt.tight_layout()
    plt.savefig(charts_dir / 'price_income_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  [OK] 相关性分析图表已保存到: {charts_dir / 'price_income_correlation.png'}")
    
    # 6. 总结
    print(f"\n  总结 / Summary:")
    if pearson_corr > 0.5:
        print(f"    - Price与Income存在强正相关关系（相关系数={pearson_corr:.3f}）")
    elif pearson_corr > 0.3:
        print(f"    - Price与Income存在中等正相关关系（相关系数={pearson_corr:.3f}）")
    elif pearson_corr > 0:
        print(f"    - Price与Income存在弱正相关关系（相关系数={pearson_corr:.3f}）")
    elif pearson_corr < -0.5:
        print(f"    - Price与Income存在强负相关关系（相关系数={pearson_corr:.3f}）")
    elif pearson_corr < -0.3:
        print(f"    - Price与Income存在中等负相关关系（相关系数={pearson_corr:.3f}）")
    else:
        print(f"    - Price与Income存在弱负相关关系（相关系数={pearson_corr:.3f}）")
    
    print(f"    - 从数学公式看：income = price × (365 - availability_365)")
    print(f"    - 理论上，如果availability_365保持不变，price越高，income越高（正相关）")
    print(f"    - 但实际中，高price可能导致低availability（低入住率），从而影响income")
else:
    print("  警告: 没有有效数据进行分析")

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
    'bedrooms', 'beds', 'amenities', 'price',     'minimum_nights', 'maximum_nights',
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

# 添加人均价格特征
if 'Average_price_per_person' in listings_detailed.columns:
    feature_columns.append('Average_price_per_person')

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
# 7. 训练XGBoost模型 / Train XGBoost Model
# ============================================================================

print("\n7. 训练XGBoost模型 / Training XGBoost Model...")

# 设置XGBoost参数（优化版本：降低复杂度、增加随机性、添加正则化）
params = {
    'objective': 'reg:squarederror',
    'n_estimators': 300,  # 减少树的数量，配合更低的learning_rate
    'max_depth': 4,  # 降低模型复杂度（从6降到4）
    'learning_rate': 0.05,  # 降低学习率，更稳定的训练
    'subsample': 0.7,  # 增加随机性（从0.8降到0.7），减少过拟合
    'colsample_bytree': 0.7,  # 增加随机性（从0.8降到0.7），减少过拟合
    'colsample_bylevel': 0.7,  # 每层列采样，进一步增加随机性
    'reg_alpha': 0.1,  # L1正则化（Lasso），减少过拟合
    'reg_lambda': 1.0,  # L2正则化（Ridge），减少过拟合
    'min_child_weight': 3,  # 最小叶子节点权重，防止过拟合
    'gamma': 0.1,  # 最小损失减少量，进一步控制复杂度
    'eval_metric': 'rmse',  # 评估指标（放在初始化参数中）
    'early_stopping_rounds': 50,  # 早停轮数（在初始化时设置）
    'random_state': 42,
    'n_jobs': -1
}

# 训练模型（使用早停防止过拟合）
model = xgb.XGBRegressor(**params)
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=10  # 每10轮输出一次评估结果
)

print("  [OK] 模型训练完成")

# ============================================================================
# 8. 模型评估 / Model Evaluation
# ============================================================================

print("\n8. 模型评估 / Model Evaluation...")

# 预测
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# 计算评估指标
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

# 显示最佳迭代次数（如果使用了早停）
if hasattr(model, 'best_iteration') and model.best_iteration is not None:
    print(f"\n最佳迭代次数 / Best Iteration: {model.best_iteration}")

print(f"\n训练集指标 / Training Metrics:")
print(f"  RMSE: {train_rmse:,.2f}")
print(f"  MAE: {train_mae:,.2f}")
print(f"  R²: {train_r2:.4f}")

print(f"\n测试集指标 / Test Metrics:")
print(f"  RMSE: {test_rmse:,.2f}")
print(f"  MAE: {test_mae:,.2f}")
print(f"  R²: {test_r2:.4f}")

# 过拟合分析
rmse_diff = test_rmse - train_rmse
overfitting_ratio = (rmse_diff / train_rmse) * 100
print(f"\n过拟合分析 / Overfitting Analysis:")
print(f"  训练集RMSE: {train_rmse:,.2f}")
print(f"  测试集RMSE: {test_rmse:,.2f}")
print(f"  RMSE差异: {rmse_diff:,.2f} (测试集 - 训练集)")
print(f"  泛化差距: {overfitting_ratio:.2f}%")
if overfitting_ratio < 5:
    print(f"  [OK] 模型泛化能力优秀，过拟合风险很低")
elif overfitting_ratio < 15:
    print(f"  [OK] 模型泛化能力良好，过拟合风险较低")
elif overfitting_ratio < 25:
    print(f"  [WARNING] 存在轻微过拟合，建议进一步调整参数")
else:
    print(f"  [WARNING] 存在明显过拟合，建议降低模型复杂度或增加正则化")

# ============================================================================
# 9. 特征重要性分析 / Feature Importance Analysis
# ============================================================================

print("\n9. 特征重要性分析 / Feature Importance Analysis...")

# 获取特征重要性
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n前20个最重要特征 / Top 20 Most Important Features:")
print(feature_importance.head(20).to_string(index=False))

# 保存特征重要性
feature_importance.to_csv(charts_dir / 'xgboost_feature_importance.csv', index=False)
print(f"\n  [OK] 特征重要性已保存到: {charts_dir / 'xgboost_feature_importance.csv'}")

# ============================================================================
# 10. 可视化结果 / Visualization
# ============================================================================

print("\n10. 生成可视化图表 / Generating Visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 10.1 特征重要性Top 20
top_features = feature_importance.head(20)
axes[0, 0].barh(range(len(top_features)), top_features['importance'].values)
axes[0, 0].set_yticks(range(len(top_features)))
axes[0, 0].set_yticklabels(top_features['feature'].values)
axes[0, 0].set_xlabel('Importance')
axes[0, 0].set_title('Top 20 Feature Importance')
axes[0, 0].invert_yaxis()

# 10.2 预测值 vs 真实值（训练集）
axes[0, 1].scatter(y_train, y_train_pred, alpha=0.5, s=10)
axes[0, 1].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
axes[0, 1].set_xlabel('True Income')
axes[0, 1].set_ylabel('Predicted Income')
axes[0, 1].set_title(f'Training Set (R² = {train_r2:.4f})')

# 10.3 预测值 vs 真实值（测试集）
axes[1, 0].scatter(y_test, y_test_pred, alpha=0.5, s=10)
axes[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[1, 0].set_xlabel('True Income')
axes[1, 0].set_ylabel('Predicted Income')
axes[1, 0].set_title(f'Test Set (R² = {test_r2:.4f})')

# 10.4 残差分布
residuals = y_test - y_test_pred
axes[1, 1].hist(residuals, bins=50, edgecolor='black')
axes[1, 1].set_xlabel('Residuals')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Residual Distribution')
axes[1, 1].axvline(x=0, color='r', linestyle='--', lw=2)

plt.tight_layout()
plt.savefig(charts_dir / 'xgboost_model_results.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"  [OK] 可视化图表已保存到: {charts_dir / 'xgboost_model_results.png'}")

# ============================================================================
# 11. 保存模型 / Save Model
# ============================================================================

print("\n11. 保存模型 / Saving Model...")

import pickle
model_path = charts_dir / 'xgboost_income_model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(model, f)

print(f"  [OK] 模型已保存到: {model_path}")

print("\n" + "=" * 80)
print("XGBoost模型训练完成！/ XGBoost Model Training Complete!")
print("=" * 80)

