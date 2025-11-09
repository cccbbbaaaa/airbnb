"""
Chapter 5.2: Reviews Dataset Analysis
第5.2章：reviews 数据集详细分析

本脚本对 reviews.csv 进行详细分析，包括时间序列分析、评论模式分析等。
This script performs detailed analysis on reviews.csv, including time series analysis and review pattern analysis.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from utils import setup_plotting, get_project_paths

# 设置中文字体支持 / Set Chinese font support
setup_plotting()

# 获取项目根目录路径 / Get project root directory path
project_root, data_dir, charts_dir = get_project_paths()

print("=" * 80)
print("Chapter 5.2: Reviews Dataset Analysis")
print("第5.2章：reviews 数据集详细分析")
print("=" * 80)

# ============================================================================
# 1. 加载数据 / Load Data
# ============================================================================

print("\n1. 加载数据 / Loading Data...")

reviews = pd.read_csv(data_dir / 'reviews.csv')
print(f"  ✅ 数据加载完成: {len(reviews)} 行 × {len(reviews.columns)} 列")
print(f"  - 字段 / Columns: {list(reviews.columns)}")

# ============================================================================
# 2. 数据集概览 / Dataset Overview
# ============================================================================

print("\n2. 数据集概览 / Dataset Overview...")

dataset_info = {
    'records': len(reviews),
    'columns': len(reviews.columns),
    'unique_listings': reviews['listing_id'].nunique(),
    'duplicate_rows': reviews.duplicated().sum(),
    'missing_values': reviews.isnull().sum().sum()
}

print(f"  - 总记录数 / Total Records: {dataset_info['records']:,}")
print(f"  - 字段数 / Columns: {dataset_info['columns']}")
print(f"  - 唯一房源数 / Unique Listings: {dataset_info['unique_listings']:,}")
print(f"  - 重复行数 / Duplicate Rows: {dataset_info['duplicate_rows']}")
print(f"  - 缺失值总数 / Total Missing Values: {dataset_info['missing_values']}")

# ============================================================================
# 3. 字段详细分析 / Field Analysis
# ============================================================================

print("\n3. 字段详细分析 / Field Analysis...")

# 3.1 listing_id 分析
print("\n3.1 listing_id 字段分析 / listing_id Field Analysis:")
listing_id_stats = reviews['listing_id'].value_counts()
print(f"  - 唯一 listing_id 数: {reviews['listing_id'].nunique():,}")
print(f"  - 平均每个房源的评论数: {len(reviews) / reviews['listing_id'].nunique():.1f}")
print(f"  - 最多评论的房源评论数: {listing_id_stats.max():,}")
print(f"  - 最少评论的房源评论数: {listing_id_stats.min():,}")
print(f"  - 中位数评论数: {listing_id_stats.median():.1f}")

# 3.2 date 字段分析
print("\n3.2 date 字段分析 / date Field Analysis:")
reviews['date'] = pd.to_datetime(reviews['date'], errors='coerce')
reviews_with_date = reviews.dropna(subset=['date'])

print(f"  - 有效日期数: {len(reviews_with_date):,}")
print(f"  - 缺失日期数: {reviews['date'].isna().sum():,}")

if len(reviews_with_date) > 0:
    print(f"  - 最早评论日期: {reviews_with_date['date'].min().strftime('%Y-%m-%d')}")
    print(f"  - 最晚评论日期: {reviews_with_date['date'].max().strftime('%Y-%m-%d')}")
    time_span = (reviews_with_date['date'].max() - reviews_with_date['date'].min()).days
    print(f"  - 时间跨度: {time_span} 天 ({time_span/365:.1f} 年)")

# ============================================================================
# 4. 时间序列分析 / Time Series Analysis
# ============================================================================

print("\n4. 时间序列分析 / Time Series Analysis...")

if len(reviews_with_date) > 0:
    # 4.1 按年统计
    reviews_with_date['year'] = reviews_with_date['date'].dt.year
    reviews_with_date['month'] = reviews_with_date['date'].dt.month
    reviews_with_date['year_month'] = reviews_with_date['date'].dt.to_period('M')
    
    yearly_counts = reviews_with_date['year'].value_counts().sort_index()
    monthly_counts = reviews_with_date['year_month'].value_counts().sort_index()
    
    print("\n4.1 按年评论统计 / Reviews by Year:")
    for year, count in yearly_counts.items():
        print(f"  - {year}: {count:,} 条评论")
    
    print(f"\n4.2 评论增长趋势 / Review Growth Trend:")
    print(f"  - 最早年份评论数: {yearly_counts.min():,}")
    print(f"  - 最多年份评论数: {yearly_counts.max():,}")
    print(f"  - 平均每年评论数: {yearly_counts.mean():.0f}")
    
    # 4.3 季节性分析
    print("\n4.3 季节性分析 / Seasonal Analysis:")
    monthly_avg = reviews_with_date.groupby('month').size()
    print("  各月平均评论数 / Average Reviews by Month:")
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    for month, count in monthly_avg.items():
        print(f"    - {month_names[month-1]}: {count:.0f} 条")

# ============================================================================
# 5. 评论模式分析 / Review Pattern Analysis
# ============================================================================

print("\n5. 评论模式分析 / Review Pattern Analysis...")

# 5.1 房源评论分布
reviews_per_listing = reviews.groupby('listing_id').size()
print("\n5.1 房源评论分布 / Reviews per Listing Distribution:")
print(f"  - 均值 / Mean: {reviews_per_listing.mean():.1f}")
print(f"  - 中位数 / Median: {reviews_per_listing.median():.1f}")
print(f"  - 25%分位 / 25th percentile: {reviews_per_listing.quantile(0.25):.1f}")
print(f"  - 75%分位 / 75th percentile: {reviews_per_listing.quantile(0.75):.1f}")
print(f"  - 90%分位 / 90th percentile: {reviews_per_listing.quantile(0.90):.1f}")
print(f"  - 最大值 / Max: {reviews_per_listing.max():,}")

# 5.2 评论活跃度分析
if len(reviews_with_date) > 0:
    print("\n5.2 评论活跃度分析 / Review Activity Analysis:")
    # 计算每个房源的最后评论日期
    last_review_by_listing = reviews_with_date.groupby('listing_id')['date'].max()
    current_date = reviews_with_date['date'].max()
    days_since_last_review = (current_date - last_review_by_listing).dt.days
    
    print(f"  - 平均距最后评论天数: {days_since_last_review.mean():.0f} 天")
    print(f"  - 中位数距最后评论天数: {days_since_last_review.median():.0f} 天")
    print(f"  - 最近30天有评论的房源数: {(days_since_last_review <= 30).sum():,}")
    print(f"  - 最近90天有评论的房源数: {(days_since_last_review <= 90).sum():,}")
    print(f"  - 超过1年无评论的房源数: {(days_since_last_review > 365).sum():,}")

# ============================================================================
# 6. 创建可视化图表 / Create Visualizations
# ============================================================================

print("\n6. 创建可视化图表 / Creating Visualizations...")

if len(reviews_with_date) > 0:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 6.1 按年评论趋势
    yearly_counts.plot(kind='line', ax=axes[0, 0], marker='o', color='#2196F3', linewidth=2)
    axes[0, 0].set_title('Reviews Trend by Year', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Year', fontsize=11)
    axes[0, 0].set_ylabel('Number of Reviews', fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 6.2 按月评论分布（季节性）
    monthly_avg.plot(kind='bar', ax=axes[0, 1], color='#4CAF50', edgecolor='black')
    axes[0, 1].set_title('Average Reviews by Month', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Month', fontsize=11)
    axes[0, 1].set_ylabel('Average Reviews', fontsize=11)
    axes[0, 1].set_xticklabels(month_names, rotation=45)
    
    # 6.3 房源评论数分布（对数尺度）
    reviews_log = np.log1p(reviews_per_listing)
    axes[1, 0].hist(reviews_log, bins=50, color='#FF9800', edgecolor='black', alpha=0.7)
    axes[1, 0].set_title('Reviews per Listing Distribution (Log Scale)', 
                         fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Log(Number of Reviews + 1)', fontsize=11)
    axes[1, 0].set_ylabel('Frequency', fontsize=11)
    
    # 6.4 评论时间分布（按年份）
    if len(yearly_counts) > 0:
        axes[1, 1].bar(yearly_counts.index, yearly_counts.values, color='#9C27B0', edgecolor='black')
        axes[1, 1].set_title('Reviews Distribution by Year', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Year', fontsize=11)
        axes[1, 1].set_ylabel('Number of Reviews', fontsize=11)
        axes[1, 1].tick_params(axis='x', rotation=45)
        for i, (year, count) in enumerate(yearly_counts.items()):
            axes[1, 1].text(year, count, f'{count:,}', ha='center', va='bottom', fontsize=9, rotation=90)
    
    plt.tight_layout()
    plt.savefig(charts_dir / 'chapter5_reviews_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ 已保存: chapter5_reviews_analysis.png")

# ============================================================================
# 7. 输出统计报告 / Output Statistics Report
# ============================================================================

print("\n7. 生成统计报告 / Generating Statistics Report...")

report_lines = []
report_lines.append("=" * 80)
report_lines.append("Chapter 5.2: Reviews Dataset Analysis")
report_lines.append("第5.2章：reviews 数据集详细分析")
report_lines.append("=" * 80)
report_lines.append(f"\n生成时间 / Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")

report_lines.append("\n## 数据集概览 / Dataset Overview")
for key, value in dataset_info.items():
    if isinstance(value, (int, float)):
        report_lines.append(f"  - {key}: {value:,}")

report_lines.append("\n## 字段详细分析 / Field Analysis")
report_lines.append(f"\n### listing_id")
report_lines.append(f"  - 唯一 listing_id 数: {reviews['listing_id'].nunique():,}")
report_lines.append(f"  - 平均每个房源的评论数: {len(reviews) / reviews['listing_id'].nunique():.1f}")
report_lines.append(f"  - 最多评论数: {listing_id_stats.max():,}")
report_lines.append(f"  - 中位数评论数: {listing_id_stats.median():.1f}")

if len(reviews_with_date) > 0:
    report_lines.append(f"\n### date")
    report_lines.append(f"  - 最早评论日期: {reviews_with_date['date'].min().strftime('%Y-%m-%d')}")
    report_lines.append(f"  - 最晚评论日期: {reviews_with_date['date'].max().strftime('%Y-%m-%d')}")
    report_lines.append(f"  - 时间跨度: {time_span} 天 ({time_span/365:.1f} 年)")

if len(reviews_with_date) > 0:
    report_lines.append("\n## 时间序列分析 / Time Series Analysis")
    report_lines.append("\n### 按年评论统计 / Reviews by Year")
    for year, count in yearly_counts.items():
        report_lines.append(f"  - {year}: {count:,} 条评论")

with open(charts_dir / 'chapter5_reviews_statistics.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))

print("  ✅ 已保存: chapter5_reviews_statistics.txt")

print("\n" + "=" * 80)
print("分析完成 / Analysis Complete!")
print("=" * 80)

