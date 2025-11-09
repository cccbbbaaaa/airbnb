"""
Chapter 6: Variable Correlation Analysis
第6章：变量相关性分析

本脚本进行变量相关性分析，包括数值型变量相关性矩阵、分类变量关联分析、关键比率特征分析等。
This script performs variable correlation analysis, including numerical variable correlation matrix, categorical variable association analysis, and key ratio features analysis.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import chi2_contingency
from utils import setup_plotting, get_project_paths

# 设置中文字体支持 / Set Chinese font support
setup_plotting()

# 获取项目根目录路径 / Get project root directory path
project_root, data_dir, charts_dir = get_project_paths()

print("=" * 80)
print("Chapter 6: Variable Correlation Analysis")
print("第6章：变量相关性分析")
print("=" * 80)

# ============================================================================
# 1. 加载数据 / Load Data
# ============================================================================

print("\n1. 加载数据 / Loading Data...")

listings = pd.read_csv(data_dir / 'listings.csv')

# 数据清洗 / Data Cleaning
if 'neighbourhood_group' in listings.columns:
    listings = listings.drop('neighbourhood_group', axis=1)

listings['last_review'] = listings['last_review'].fillna(0)
listings['reviews_per_month'] = listings['reviews_per_month'].fillna(0)
listings['name'] = listings['name'].fillna('blank_name')
listings['host_name'] = listings['host_name'].fillna('blank_host_name')

# 处理异常值 / Handle Outliers
listings.loc[listings['minimum_nights'] > 365, 'minimum_nights'] = 365
listings.loc[listings['price'] == 0, 'price'] = np.nan  # 将价格为0视为缺失值

print(f"  ✅ 数据加载完成: {len(listings)} 行 × {len(listings.columns)} 列")

# ============================================================================
# 2. 数值型变量相关性矩阵 / Numerical Variable Correlation Matrix
# ============================================================================

print("\n2. 数值型变量相关性分析 / Numerical Variable Correlation Analysis...")

# 选择数值型变量 / Select numerical variables
numeric_cols = ['price', 'minimum_nights', 'number_of_reviews', 'reviews_per_month',
                'calculated_host_listings_count', 'availability_365', 'number_of_reviews_ltm']

# 计算相关性矩阵 / Calculate correlation matrix
correlation_matrix = listings[numeric_cols].corr()

print("\n2.1 相关性矩阵 / Correlation Matrix:")
print(correlation_matrix.round(3))

# 识别高度相关的变量对 / Identify highly correlated variable pairs
print("\n2.2 高度相关的变量对（|r| > 0.5）/ Highly Correlated Variable Pairs:")
high_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        corr_value = correlation_matrix.iloc[i, j]
        if abs(corr_value) > 0.5:
            high_corr_pairs.append({
                'var1': correlation_matrix.columns[i],
                'var2': correlation_matrix.columns[j],
                'correlation': corr_value
            })
            print(f"  - {correlation_matrix.columns[i]} ↔ {correlation_matrix.columns[j]}: {corr_value:.3f}")

# ============================================================================
# 3. 价格与其他变量的相关性 / Price Correlation with Other Variables
# ============================================================================

print("\n3. 价格与其他变量的相关性 / Price Correlation with Other Variables:")

price_corr = correlation_matrix['price'].sort_values(ascending=False)
print("\n3.1 价格相关性排序 / Price Correlation Ranking:")
for var, corr in price_corr.items():
    if var != 'price':
        print(f"  - {var}: {corr:.3f}")

# ============================================================================
# 4. 分类变量关联分析 / Categorical Variable Association Analysis
# ============================================================================

print("\n4. 分类变量关联分析 / Categorical Variable Association Analysis...")

# 4.1 房型与价格的关系
print("\n4.1 房型与价格的关系 / Room Type vs Price:")
room_type_price = listings.groupby('room_type')['price'].agg(['mean', 'median', 'count'])
print(room_type_price.round(2))

# 4.2 街区与价格的关系
print("\n4.2 街区与价格的关系（Top 10）/ Neighbourhood vs Price (Top 10):")
neighbourhood_price = listings.groupby('neighbourhood')['price'].agg(['mean', 'median', 'count']).sort_values('mean', ascending=False)
print(neighbourhood_price.head(10).round(2))

# 4.3 房型与评论数的关系
print("\n4.3 房型与评论数的关系 / Room Type vs Reviews:")
room_type_reviews = listings.groupby('room_type')['number_of_reviews'].agg(['mean', 'median', 'count'])
print(room_type_reviews.round(2))

# 4.4 街区与评论数的关系
print("\n4.4 街区与评论数的关系（Top 10）/ Neighbourhood vs Reviews (Top 10):")
neighbourhood_reviews = listings.groupby('neighbourhood')['number_of_reviews'].agg(['mean', 'median', 'count']).sort_values('mean', ascending=False)
print(neighbourhood_reviews.head(10).round(2))

# ============================================================================
# 5. 关键比率特征分析 / Key Ratio Features Analysis
# ============================================================================

print("\n5. 关键比率特征分析 / Key Ratio Features Analysis...")

# 5.1 评论活跃度
listings['review_activity_ratio'] = listings['reviews_per_month'] / (listings['number_of_reviews'] + 1)
print("\n5.1 评论活跃度比率 / Review Activity Ratio:")
print(f"  - 均值: {listings['review_activity_ratio'].mean():.4f}")
print(f"  - 中位数: {listings['review_activity_ratio'].median():.4f}")

# 5.2 价格可用性比率
listings['price_availability_ratio'] = listings['price'] / (listings['availability_365'] + 1)
print("\n5.2 价格可用性比率 / Price-Availability Ratio:")
print(f"  - 均值: {listings['price_availability_ratio'].mean():.2f}")
print(f"  - 中位数: {listings['price_availability_ratio'].median():.2f}")

# 5.3 单房源评论表现
listings['reviews_per_listing_ratio'] = listings['number_of_reviews'] / (listings['calculated_host_listings_count'] + 1)
print("\n5.3 单房源评论表现比率 / Reviews per Listing Ratio:")
print(f"  - 均值: {listings['reviews_per_listing_ratio'].mean():.2f}")
print(f"  - 中位数: {listings['reviews_per_listing_ratio'].median():.2f}")

# 5.4 入住率
listings['occupancy_rate'] = (365 - listings['availability_365']) / 365 * 100
print("\n5.4 入住率 / Occupancy Rate:")
print(f"  - 均值: {listings['occupancy_rate'].mean():.1f}%")
print(f"  - 中位数: {listings['occupancy_rate'].median():.1f}%")

# ============================================================================
# 6. 创建可视化图表 / Create Visualizations
# ============================================================================

print("\n6. 创建可视化图表 / Creating Visualizations...")

# 6.1 相关性热力图
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# 6.1.1 数值型变量相关性热力图
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, mask=mask, ax=axes[0, 0])
axes[0, 0].set_title('Numerical Variables Correlation Heatmap', 
                     fontsize=12, fontweight='bold')

# 6.1.2 价格与其他变量的相关性（条形图）
price_corr_sorted = price_corr.drop('price').sort_values(ascending=True)
axes[0, 1].barh(price_corr_sorted.index, price_corr_sorted.values, 
                color=['red' if x < 0 else 'green' for x in price_corr_sorted.values])
axes[0, 1].axvline(0, color='black', linestyle='--', linewidth=1)
axes[0, 1].set_title('Price Correlation with Other Variables', 
                     fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Correlation Coefficient', fontsize=11)

# 6.1.3 房型与价格的关系
room_type_price_mean = listings.groupby('room_type')['price'].mean().sort_values(ascending=False)
axes[1, 0].bar(room_type_price_mean.index, room_type_price_mean.values, 
               color=['#4CAF50', '#2196F3', '#FF9800', '#F44336'], edgecolor='black')
axes[1, 0].set_title('Average Price by Room Type', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Price (€)', fontsize=11)
axes[1, 0].tick_params(axis='x', rotation=45)
for i, v in enumerate(room_type_price_mean.values):
    axes[1, 0].text(i, v, f'€{v:.0f}', ha='center', va='bottom', fontsize=10)

# 6.1.4 入住率与价格的关系（散点图）
sample_data = listings.sample(min(5000, len(listings)))  # 采样以提高性能
axes[1, 1].scatter(sample_data['occupancy_rate'], sample_data['price'], 
                   alpha=0.3, s=10, color='purple')
axes[1, 1].set_title('Occupancy Rate vs Price', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Occupancy Rate (%)', fontsize=11)
axes[1, 1].set_ylabel('Price (€)', fontsize=11)
axes[1, 1].set_ylim(0, min(500, listings['price'].quantile(0.95)))  # 限制y轴范围

plt.tight_layout()
plt.savefig(charts_dir / 'chapter6_correlation_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✅ 已保存: chapter6_correlation_analysis.png")

# 6.2 分类变量关联分析可视化
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 6.2.1 街区与价格（Top 15）
top_neighbourhoods_price = neighbourhood_price.head(15)
axes[0, 0].barh(range(len(top_neighbourhoods_price)), top_neighbourhoods_price['mean'].values,
                color='coral', edgecolor='black')
axes[0, 0].set_yticks(range(len(top_neighbourhoods_price)))
axes[0, 0].set_yticklabels(top_neighbourhoods_price.index, fontsize=9)
axes[0, 0].set_title('Top 15 Neighbourhoods by Average Price', 
                     fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Average Price (€)', fontsize=11)
axes[0, 0].invert_yaxis()

# 6.2.2 房型与评论数
room_type_reviews_mean = listings.groupby('room_type')['number_of_reviews'].mean().sort_values(ascending=False)
axes[0, 1].bar(room_type_reviews_mean.index, room_type_reviews_mean.values,
               color=['#9C27B0', '#E91E63', '#00BCD4', '#4CAF50'], edgecolor='black')
axes[0, 1].set_title('Average Reviews by Room Type', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Number of Reviews', fontsize=11)
axes[0, 1].tick_params(axis='x', rotation=45)
for i, v in enumerate(room_type_reviews_mean.values):
    axes[0, 1].text(i, v, f'{v:.0f}', ha='center', va='bottom', fontsize=10)

# 6.2.3 街区与评论数（Top 15）
top_neighbourhoods_reviews = neighbourhood_reviews.head(15)
axes[1, 0].barh(range(len(top_neighbourhoods_reviews)), top_neighbourhoods_reviews['mean'].values,
                color='lightblue', edgecolor='black')
axes[1, 0].set_yticks(range(len(top_neighbourhoods_reviews)))
axes[1, 0].set_yticklabels(top_neighbourhoods_reviews.index, fontsize=9)
axes[1, 0].set_title('Top 15 Neighbourhoods by Average Reviews', 
                     fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Average Reviews', fontsize=11)
axes[1, 0].invert_yaxis()

# 6.2.4 评论数与价格的关系（散点图）
sample_data = listings[listings['number_of_reviews'] > 0].sample(min(5000, len(listings)))
axes[1, 1].scatter(np.log1p(sample_data['number_of_reviews']), sample_data['price'],
                   alpha=0.3, s=10, color='orange')
axes[1, 1].set_title('Reviews vs Price (Log Scale)', 
                     fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Log(Number of Reviews + 1)', fontsize=11)
axes[1, 1].set_ylabel('Price (€)', fontsize=11)
axes[1, 1].set_ylim(0, min(500, listings['price'].quantile(0.95)))

plt.tight_layout()
plt.savefig(charts_dir / 'chapter6_categorical_association.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✅ 已保存: chapter6_categorical_association.png")

# ============================================================================
# 7. 输出统计报告 / Output Statistics Report
# ============================================================================

print("\n7. 生成统计报告 / Generating Statistics Report...")

report_lines = []
report_lines.append("=" * 80)
report_lines.append("Chapter 6: Variable Correlation Analysis")
report_lines.append("第6章：变量相关性分析")
report_lines.append("=" * 80)
report_lines.append(f"\n生成时间 / Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")

report_lines.append("\n## 数值型变量相关性矩阵 / Numerical Variable Correlation Matrix")
report_lines.append("\n" + correlation_matrix.round(3).to_string())

report_lines.append("\n## 高度相关的变量对（|r| > 0.5）/ Highly Correlated Variable Pairs")
for pair in high_corr_pairs:
    report_lines.append(f"  - {pair['var1']} ↔ {pair['var2']}: {pair['correlation']:.3f}")

report_lines.append("\n## 价格与其他变量的相关性 / Price Correlation with Other Variables")
for var, corr in price_corr.items():
    if var != 'price':
        report_lines.append(f"  - {var}: {corr:.3f}")

with open(charts_dir / 'chapter6_correlation_statistics.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))

print("  ✅ 已保存: chapter6_correlation_statistics.txt")

print("\n" + "=" * 80)
print("分析完成 / Analysis Complete!")
print("=" * 80)

