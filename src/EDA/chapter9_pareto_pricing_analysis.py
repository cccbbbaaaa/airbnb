"""
Chapter 9.2 & 9.5: Pareto Analysis and Pricing Strategy Analysis
第9.2和9.5章：帕累托分析和价格策略分析

本脚本进行帕累托分析和价格策略分析。
This script performs Pareto analysis and pricing strategy analysis.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from utils import setup_plotting, get_project_paths

# 设置中文字体支持 / Set Chinese font support
setup_plotting()

# 获取项目根目录路径 / Get project root directory path
project_root, data_dir, charts_dir = get_project_paths()

print("=" * 80)
print("Chapter 9.2 & 9.5: Pareto Analysis and Pricing Strategy Analysis")
print("第9.2和9.5章：帕累托分析和价格策略分析")
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
listings.loc[listings['price'] == 0, 'price'] = np.nan

# 计算入住率和收入
listings['occupancy_rate'] = (365 - listings['availability_365']) / 365 * 100
listings['estimated_revenue'] = listings['price'] * (365 - listings['availability_365'])

print(f"  ✅ 数据加载完成: {len(listings)} 行 × {len(listings.columns)} 列")

# ============================================================================
# 2. 帕累托分析 / Pareto Analysis
# ============================================================================

print("\n2. 帕累托分析 / Pareto Analysis...")

# 2.1 TOP 20%房源是否贡献80%评论？
listings_sorted_reviews = listings.sort_values('number_of_reviews', ascending=False)
total_reviews = listings_sorted_reviews['number_of_reviews'].sum()
top_20_pct_count = int(len(listings_sorted_reviews) * 0.2)
top_20_reviews = listings_sorted_reviews.head(top_20_pct_count)['number_of_reviews'].sum()
top_20_reviews_pct = (top_20_reviews / total_reviews * 100) if total_reviews > 0 else 0

print("\n2.1 TOP 20%房源评论贡献分析 / Top 20% Listings Review Contribution:")
print(f"  - TOP 20%房源数: {top_20_pct_count:,} 个")
print(f"  - TOP 20%房源评论数: {top_20_reviews:,} 条")
print(f"  - 总评论数: {total_reviews:,} 条")
print(f"  - TOP 20%房源评论占比: {top_20_reviews_pct:.1f}%")
print(f"  - 是否符合80/20法则: {'✅ 是' if top_20_reviews_pct >= 80 else '❌ 否'}")

# 2.2 TOP 20%房源是否贡献80%收入？
listings_sorted_revenue = listings.dropna(subset=['estimated_revenue']).sort_values('estimated_revenue', ascending=False)
total_revenue = listings_sorted_revenue['estimated_revenue'].sum()
top_20_pct_count_rev = int(len(listings_sorted_revenue) * 0.2)
top_20_revenue = listings_sorted_revenue.head(top_20_pct_count_rev)['estimated_revenue'].sum()
top_20_revenue_pct = (top_20_revenue / total_revenue * 100) if total_revenue > 0 else 0

print("\n2.2 TOP 20%房源收入贡献分析 / Top 20% Listings Revenue Contribution:")
print(f"  - TOP 20%房源数: {top_20_pct_count_rev:,} 个")
print(f"  - TOP 20%房源收入: €{top_20_revenue:,.0f}")
print(f"  - 总收入: €{total_revenue:,.0f}")
print(f"  - TOP 20%房源收入占比: {top_20_revenue_pct:.1f}%")
print(f"  - 是否符合80/20法则: {'✅ 是' if top_20_revenue_pct >= 80 else '❌ 否'}")

# 2.3 帕累托曲线分析
print("\n2.3 帕累托曲线分析 / Pareto Curve Analysis:")
# 评论数帕累托
listings_sorted_reviews['cumulative_reviews'] = listings_sorted_reviews['number_of_reviews'].cumsum()
listings_sorted_reviews['cumulative_reviews_pct'] = listings_sorted_reviews['cumulative_reviews'] / total_reviews * 100
listings_sorted_reviews['cumulative_listings_pct'] = (np.arange(len(listings_sorted_reviews)) + 1) / len(listings_sorted_reviews) * 100

# 收入帕累托
listings_sorted_revenue['cumulative_revenue'] = listings_sorted_revenue['estimated_revenue'].cumsum()
listings_sorted_revenue['cumulative_revenue_pct'] = listings_sorted_revenue['cumulative_revenue'] / total_revenue * 100
listings_sorted_revenue['cumulative_listings_pct'] = (np.arange(len(listings_sorted_revenue)) + 1) / len(listings_sorted_revenue) * 100

# 找到80%评论/收入对应的房源占比
reviews_80_pct_idx = listings_sorted_reviews[listings_sorted_reviews['cumulative_reviews_pct'] >= 80].index[0]
reviews_80_pct_listings_pct = listings_sorted_reviews.loc[reviews_80_pct_idx, 'cumulative_listings_pct']

revenue_80_pct_idx = listings_sorted_revenue[listings_sorted_revenue['cumulative_revenue_pct'] >= 80].index[0]
revenue_80_pct_listings_pct = listings_sorted_revenue.loc[revenue_80_pct_idx, 'cumulative_listings_pct']

print(f"  - 80%评论来自前 {reviews_80_pct_listings_pct:.1f}% 的房源")
print(f"  - 80%收入来自前 {revenue_80_pct_listings_pct:.1f}% 的房源")

# ============================================================================
# 3. 价格策略分析 / Pricing Strategy Analysis
# ============================================================================

print("\n3. 价格策略分析 / Pricing Strategy Analysis...")

# 3.1 价格分布特征
print("\n3.1 价格分布特征 / Price Distribution Characteristics:")
price_stats = listings['price'].describe()
print(f"  - 均值: €{price_stats['mean']:.2f}")
print(f"  - 中位数: €{price_stats['50%']:.2f}")
print(f"  - 25%分位: €{price_stats['25%']:.2f}")
print(f"  - 75%分位: €{price_stats['75%']:.2f}")
print(f"  - 标准差: €{price_stats['std']:.2f}")
print(f"  - 偏度: {listings['price'].skew():.2f}")
print(f"  - 峰度: {listings['price'].kurtosis():.2f}")

# 3.2 价格与房型的关系
print("\n3.2 价格与房型的关系 / Price vs Room Type:")
room_type_price = listings.groupby('room_type')['price'].agg(['mean', 'median', 'std', 'count'])
for room_type, row in room_type_price.iterrows():
    print(f"  - {room_type}: 均值 €{row['mean']:.2f}, 中位数 €{row['median']:.2f}, "
          f"标准差 €{row['std']:.2f}, 房源数 {row['count']:.0f}")

# 3.3 价格与地理位置的关系
print("\n3.3 价格与地理位置的关系（Top 10）/ Price vs Location (Top 10):")
neighbourhood_price = listings.groupby('neighbourhood')['price'].agg(['mean', 'median', 'count']).sort_values('mean', ascending=False)
for i, (neighbourhood, row) in enumerate(neighbourhood_price.head(10).iterrows(), 1):
    print(f"  {i:2d}. {neighbourhood}: €{row['mean']:.2f} (中位数: €{row['median']:.2f}, 房源数: {row['count']:.0f})")

# 3.4 价格与受欢迎度的关系
print("\n3.4 价格与受欢迎度的关系 / Price vs Popularity:")
# 按评论数分组
listings_with_reviews = listings[listings['number_of_reviews'] > 0]
listings_with_reviews['review_category'] = pd.cut(
    listings_with_reviews['number_of_reviews'],
    bins=[0, 5, 10, 20, 50, float('inf')],
    labels=['0-5', '6-10', '11-20', '21-50', '50+']
)
price_by_reviews = listings_with_reviews.groupby('review_category')['price'].agg(['mean', 'median', 'count'])
for category, row in price_by_reviews.iterrows():
    print(f"  - 评论数 {category}: 平均价格 €{row['mean']:.2f}, "
          f"中位数 €{row['median']:.2f}, 房源数 {row['count']:.0f}")

# 3.5 价格与可用性的关系
print("\n3.5 价格与可用性的关系 / Price vs Availability:")
# 按入住率分组
listings['occupancy_category'] = pd.cut(
    listings['occupancy_rate'],
    bins=[0, 20, 50, 80, 100],
    labels=['Low (0-20%)', 'Medium-Low (20-50%)', 'Medium-High (50-80%)', 'High (80-100%)']
)
price_by_occupancy = listings.groupby('occupancy_category')['price'].agg(['mean', 'median', 'count'])
for category, row in price_by_occupancy.iterrows():
    print(f"  - {category}: 平均价格 €{row['mean']:.2f}, "
          f"中位数 €{row['median']:.2f}, 房源数 {row['count']:.0f}")

# 3.6 最优定价区间识别
print("\n3.6 最优定价区间识别 / Optimal Pricing Range Identification:")
# 计算不同价格区间的平均收入
listings['price_range'] = pd.cut(
    listings['price'],
    bins=[0, 50, 100, 150, 200, 300, 500, float('inf')],
    labels=['€0-50', '€50-100', '€100-150', '€150-200', '€200-300', '€300-500', '€500+']
)
revenue_by_price_range = listings.groupby('price_range')['estimated_revenue'].agg(['mean', 'median', 'count'])
revenue_by_price_range = revenue_by_price_range.sort_values('mean', ascending=False)

print("  各价格区间平均收入 / Average Revenue by Price Range:")
for price_range, row in revenue_by_price_range.iterrows():
    print(f"  - {price_range}: 平均收入 €{row['mean']:.2f}, "
          f"中位数 €{row['median']:.2f}, 房源数 {row['count']:.0f}")

# ============================================================================
# 4. 创建可视化图表 / Create Visualizations
# ============================================================================

print("\n4. 创建可视化图表 / Creating Visualizations...")

# 4.1 帕累托分析图表
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# 4.1.1 评论数帕累托曲线
axes[0, 0].plot(listings_sorted_reviews['cumulative_listings_pct'], 
                listings_sorted_reviews['cumulative_reviews_pct'], 
                linewidth=2, color='#2196F3', label='Reviews Pareto Curve')
axes[0, 0].axhline(80, color='red', linestyle='--', alpha=0.5, label='80% Line')
axes[0, 0].axvline(20, color='red', linestyle='--', alpha=0.5, label='20% Line')
axes[0, 0].plot([0, 20, 100], [0, 80, 100], 'g--', alpha=0.5, label='Perfect 80/20')
axes[0, 0].set_title('Reviews Pareto Curve', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Cumulative Listings (%)', fontsize=11)
axes[0, 0].set_ylabel('Cumulative Reviews (%)', fontsize=11)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend()

# 4.1.2 收入帕累托曲线
axes[0, 1].plot(listings_sorted_revenue['cumulative_listings_pct'], 
                listings_sorted_revenue['cumulative_revenue_pct'], 
                linewidth=2, color='#4CAF50', label='Revenue Pareto Curve')
axes[0, 1].axhline(80, color='red', linestyle='--', alpha=0.5, label='80% Line')
axes[0, 1].axvline(20, color='red', linestyle='--', alpha=0.5, label='20% Line')
axes[0, 1].plot([0, 20, 100], [0, 80, 100], 'g--', alpha=0.5, label='Perfect 80/20')
axes[0, 1].set_title('Revenue Pareto Curve', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Cumulative Listings (%)', fontsize=11)
axes[0, 1].set_ylabel('Cumulative Revenue (%)', fontsize=11)
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend()

# 4.1.3 TOP 20%房源评论贡献
top_20_reviews_data = listings_sorted_reviews.head(top_20_pct_count)
bottom_80_reviews_data = listings_sorted_reviews.tail(len(listings_sorted_reviews) - top_20_pct_count)
axes[1, 0].bar(['TOP 20%', 'Bottom 80%'], 
               [top_20_reviews, bottom_80_reviews_data['number_of_reviews'].sum()],
               color=['#FF9800', '#9E9E9E'], edgecolor='black')
axes[1, 0].set_title(f'Review Contribution: Top 20% vs Bottom 80%\n(TOP 20% contributes {top_20_reviews_pct:.1f}%)', 
                     fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Total Reviews', fontsize=11)
for i, v in enumerate([top_20_reviews, bottom_80_reviews_data['number_of_reviews'].sum()]):
    axes[1, 0].text(i, v, f'{v:,}', ha='center', va='bottom', fontsize=10)

# 4.1.4 TOP 20%房源收入贡献
top_20_revenue_data = listings_sorted_revenue.head(top_20_pct_count_rev)
bottom_80_revenue_data = listings_sorted_revenue.tail(len(listings_sorted_revenue) - top_20_pct_count_rev)
axes[1, 1].bar(['TOP 20%', 'Bottom 80%'], 
               [top_20_revenue, bottom_80_revenue_data['estimated_revenue'].sum()],
               color=['#4CAF50', '#9E9E9E'], edgecolor='black')
axes[1, 1].set_title(f'Revenue Contribution: Top 20% vs Bottom 80%\n(TOP 20% contributes {top_20_revenue_pct:.1f}%)', 
                     fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('Total Revenue (€)', fontsize=11)
for i, v in enumerate([top_20_revenue, bottom_80_revenue_data['estimated_revenue'].sum()]):
    axes[1, 1].text(i, v, f'€{v:,.0f}', ha='center', va='bottom', fontsize=9, rotation=90)

plt.tight_layout()
plt.savefig(charts_dir / 'chapter9_pareto_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✅ 已保存: chapter9_pareto_analysis.png")

# 4.2 价格策略分析图表
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 4.2.1 价格与房型的关系
room_type_price_mean = listings.groupby('room_type')['price'].mean().sort_values(ascending=False)
axes[0, 0].bar(room_type_price_mean.index, room_type_price_mean.values,
               color=['#4CAF50', '#2196F3', '#FF9800', '#F44336'], edgecolor='black')
axes[0, 0].set_title('Average Price by Room Type', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Price (€)', fontsize=11)
axes[0, 0].tick_params(axis='x', rotation=45)
for i, v in enumerate(room_type_price_mean.values):
    axes[0, 0].text(i, v, f'€{v:.0f}', ha='center', va='bottom', fontsize=10)

# 4.2.2 价格与评论数的关系
price_by_reviews_mean = price_by_reviews['mean']
axes[0, 1].bar(range(len(price_by_reviews_mean)), price_by_reviews_mean.values,
               color='purple', edgecolor='black')
axes[0, 1].set_title('Average Price by Review Count Category', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Price (€)', fontsize=11)
axes[0, 1].set_xticks(range(len(price_by_reviews_mean)))
axes[0, 1].set_xticklabels(price_by_reviews_mean.index, rotation=45)
for i, v in enumerate(price_by_reviews_mean.values):
    axes[0, 1].text(i, v, f'€{v:.0f}', ha='center', va='bottom', fontsize=9)

# 4.2.3 价格与入住率的关系
price_by_occupancy_mean = price_by_occupancy['mean']
axes[1, 0].bar(range(len(price_by_occupancy_mean)), price_by_occupancy_mean.values,
               color='orange', edgecolor='black')
axes[1, 0].set_title('Average Price by Occupancy Rate Category', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Price (€)', fontsize=11)
axes[1, 0].set_xticks(range(len(price_by_occupancy_mean)))
axes[1, 0].set_xticklabels(price_by_occupancy_mean.index, rotation=45)
for i, v in enumerate(price_by_occupancy_mean.values):
    axes[1, 0].text(i, v, f'€{v:.0f}', ha='center', va='bottom', fontsize=9)

# 4.2.4 各价格区间平均收入
revenue_by_price_range_mean = revenue_by_price_range['mean']
axes[1, 1].bar(range(len(revenue_by_price_range_mean)), revenue_by_price_range_mean.values,
               color='teal', edgecolor='black')
axes[1, 1].set_title('Average Revenue by Price Range', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('Average Revenue (€)', fontsize=11)
axes[1, 1].set_xticks(range(len(revenue_by_price_range_mean)))
axes[1, 1].set_xticklabels(revenue_by_price_range_mean.index, rotation=45)
for i, v in enumerate(revenue_by_price_range_mean.values):
    axes[1, 1].text(i, v, f'€{v:.0f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(charts_dir / 'chapter9_pricing_strategy_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✅ 已保存: chapter9_pricing_strategy_analysis.png")

# ============================================================================
# 5. 输出统计报告 / Output Statistics Report
# ============================================================================

print("\n5. 生成统计报告 / Generating Statistics Report...")

report_lines = []
report_lines.append("=" * 80)
report_lines.append("Chapter 9.2 & 9.5: Pareto Analysis and Pricing Strategy Analysis")
report_lines.append("第9.2和9.5章：帕累托分析和价格策略分析")
report_lines.append("=" * 80)
report_lines.append(f"\n生成时间 / Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")

report_lines.append("\n## 帕累托分析 / Pareto Analysis")
report_lines.append(f"\n### TOP 20%房源评论贡献")
report_lines.append(f"  - TOP 20%房源评论占比: {top_20_reviews_pct:.1f}%")
report_lines.append(f"  - 是否符合80/20法则: {'是' if top_20_reviews_pct >= 80 else '否'}")

report_lines.append(f"\n### TOP 20%房源收入贡献")
report_lines.append(f"  - TOP 20%房源收入占比: {top_20_revenue_pct:.1f}%")
report_lines.append(f"  - 是否符合80/20法则: {'是' if top_20_revenue_pct >= 80 else '否'}")

report_lines.append("\n## 价格策略分析 / Pricing Strategy Analysis")
report_lines.append(f"\n### 价格分布特征")
report_lines.append(f"  - 均值: €{price_stats['mean']:.2f}")
report_lines.append(f"  - 中位数: €{price_stats['50%']:.2f}")
report_lines.append(f"  - 偏度: {listings['price'].skew():.2f}")

with open(charts_dir / 'chapter9_pareto_pricing_statistics.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))

print("  ✅ 已保存: chapter9_pareto_pricing_statistics.txt")

print("\n" + "=" * 80)
print("分析完成 / Analysis Complete!")
print("=" * 80)

