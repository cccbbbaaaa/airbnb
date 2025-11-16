"""
Chapter 5: Detailed Listings Dataset Analysis
第5章：listings 数据集详细分析

本脚本对 listings.csv 进行详细的字段级分析，包括统计汇总、分布特征、业务洞察等。
This script performs detailed field-level analysis on listings.csv, including statistical summaries, distribution characteristics, and business insights.
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
print("Chapter 5: Detailed Listings Dataset Analysis")
print("第5章：listings 数据集详细分析")
print("=" * 80)

# ============================================================================
# 1. 加载和预处理数据 / Load and Preprocess Data
# ============================================================================

print("\n1. 加载数据 / Loading Data...")

listings = pd.read_csv(data_dir / 'listings.csv')

# 数据清洗 / Data Cleaning
# 删除 neighbourhood_group 列（全为空）
if 'neighbourhood_group' in listings.columns:
    listings = listings.drop('neighbourhood_group', axis=1)

# 填充缺失值 / Fill Missing Values
listings['last_review'] = listings['last_review'].fillna(0)
listings['reviews_per_month'] = listings['reviews_per_month'].fillna(0)
listings['name'] = listings['name'].fillna('blank_name')
listings['host_name'] = listings['host_name'].fillna('blank_host_name')

# 处理异常值 / Handle Outliers
listings.loc[listings['minimum_nights'] > 365, 'minimum_nights'] = 365

print(f"  ✅ 数据加载完成: {len(listings)} 行 × {len(listings.columns)} 列")

# ============================================================================
# 2. 数据集概览 / Dataset Overview
# ============================================================================

print("\n2. 数据集概览 / Dataset Overview...")

dataset_info = {
    'records': len(listings),
    'columns': len(listings.columns),
    'memory_usage_mb': listings.memory_usage(deep=True).sum() / (1024 * 1024),
    'duplicate_rows': listings.duplicated().sum(),
    'unique_hosts': listings['host_id'].nunique(),
    'unique_neighbourhoods': listings['neighbourhood'].nunique(),
    'unique_room_types': listings['room_type'].nunique()
}

print(f"  - 记录数 / Records: {dataset_info['records']:,}")
print(f"  - 字段数 / Columns: {dataset_info['columns']}")
print(f"  - 内存使用 / Memory Usage: {dataset_info['memory_usage_mb']:.2f} MB")
print(f"  - 重复行数 / Duplicate Rows: {dataset_info['duplicate_rows']}")
print(f"  - 唯一房东数 / Unique Hosts: {dataset_info['unique_hosts']:,}")
print(f"  - 唯一街区数 / Unique Neighbourhoods: {dataset_info['unique_neighbourhoods']}")
print(f"  - 唯一房型数 / Unique Room Types: {dataset_info['unique_room_types']}")

# ============================================================================
# 3. 字段详细分析 / Field Analysis
# ============================================================================

print("\n3. 字段详细分析 / Field Analysis...")

field_analysis = {}

# 3.1 基础信息字段 / Basic Information Fields
basic_fields = ['id', 'name', 'host_id', 'host_name', 'calculated_host_listings_count']
print("\n3.1 基础信息字段 / Basic Information Fields:")

for field in basic_fields:
    if field in listings.columns:
        field_data = listings[field]
        missing_count = field_data.isnull().sum()
        missing_pct = (missing_count / len(listings) * 100)
        unique_count = field_data.nunique()
        
        field_analysis[field] = {
            'dtype': str(field_data.dtype),
            'missing_count': missing_count,
            'missing_pct': missing_pct,
            'unique_count': unique_count
        }
        
        if field_data.dtype in ['int64', 'float64']:
            field_analysis[field]['statistics'] = {
                'mean': field_data.mean(),
                'median': field_data.median(),
                'min': field_data.min(),
                'max': field_data.max(),
                'std': field_data.std()
            }
        
        print(f"  {field}:")
        print(f"    - 数据类型 / Data Type: {field_data.dtype}")
        print(f"    - 缺失率 / Missing Rate: {missing_pct:.2f}%")
        print(f"    - 唯一值数 / Unique Values: {unique_count:,}")
        if field_data.dtype in ['int64', 'float64']:
            print(f"    - 均值 / Mean: {field_data.mean():.2f}")
            print(f"    - 中位数 / Median: {field_data.median():.2f}")

# 3.2 地理信息字段 / Geographic Information Fields
geo_fields = ['neighbourhood', 'latitude', 'longitude']
print("\n3.2 地理信息字段 / Geographic Information Fields:")

for field in geo_fields:
    if field in listings.columns:
        field_data = listings[field]
        missing_count = field_data.isnull().sum()
        missing_pct = (missing_count / len(listings) * 100)
        unique_count = field_data.nunique()
        
        field_analysis[field] = {
            'dtype': str(field_data.dtype),
            'missing_count': missing_count,
            'missing_pct': missing_pct,
            'unique_count': unique_count
        }
        
        if field_data.dtype in ['float64']:
            field_analysis[field]['statistics'] = {
                'mean': field_data.mean(),
                'median': field_data.median(),
                'min': field_data.min(),
                'max': field_data.max(),
                'std': field_data.std()
            }
        
        print(f"  {field}:")
        print(f"    - 数据类型 / Data Type: {field_data.dtype}")
        print(f"    - 缺失率 / Missing Rate: {missing_pct:.2f}%")
        print(f"    - 唯一值数 / Unique Values: {unique_count:,}")
        if field_data.dtype in ['float64']:
            print(f"    - 均值 / Mean: {field_data.mean():.4f}")
            print(f"    - 范围 / Range: [{field_data.min():.4f}, {field_data.max():.4f}]")

# 3.3 房源特征字段 / Listing Characteristics Fields
print("\n3.3 房源特征字段 / Listing Characteristics Fields:")

# room_type 分析
if 'room_type' in listings.columns:
    room_type_dist = listings['room_type'].value_counts()
    print(f"  room_type 分布 / Distribution:")
    for room_type, count in room_type_dist.items():
        pct = (count / len(listings) * 100)
        print(f"    - {room_type}: {count:,} ({pct:.2f}%)")

# price 分析
if 'price' in listings.columns:
    price_stats = listings['price'].describe()
    print(price_stats)
    print(f"\n  price 统计 / Statistics:")
    print(f"    - 均值 / Mean: €{price_stats['mean']:.2f}")
    print(f"    - 中位数 / Median: €{price_stats['50%']:.2f}")
    print(f"    - 25%分位 / 25th percentile: €{price_stats['25%']:.2f}")
    print(f"    - 75%分位 / 75th percentile: €{price_stats['75%']:.2f}")
    print(f"    - 最小值 / Min: €{price_stats['min']:.2f}")
    print(f"    - 最大值 / Max: €{price_stats['max']:.2f}")
    print(f"    - 标准差 / Std: €{price_stats['std']:.2f}")

# 3.4 预订规则字段 / Booking Rules Fields
print("\n3.4 预订规则字段 / Booking Rules Fields:")

if 'minimum_nights' in listings.columns:
    min_nights_stats = listings['minimum_nights'].describe()
    print(f"  minimum_nights 统计 / Statistics:")
    print(f"    - 均值 / Mean: {min_nights_stats['mean']:.1f} 天")
    print(f"    - 中位数 / Median: {min_nights_stats['50%']:.1f} 天")
    print(f"    - 范围 / Range: [{min_nights_stats['min']:.0f}, {min_nights_stats['max']:.0f}] 天")

# 3.5 评论相关字段 / Review-related Fields
print("\n3.5 评论相关字段 / Review-related Fields:")

review_fields = ['number_of_reviews', 'reviews_per_month', 'number_of_reviews_ltm']
for field in review_fields:
    if field in listings.columns:
        field_data = listings[field]
        stats = field_data.describe()
        print(f"\n  {field} 统计 / Statistics:")
        print(f"    - 均值 / Mean: {stats['mean']:.2f}")
        print(f"    - 中位数 / Median: {stats['50%']:.2f}")
        print(f"    - 最大值 / Max: {stats['max']:.0f}")
        print(f"    - 零值数量 / Zero Values: {(field_data == 0).sum():,} ({(field_data == 0).sum() / len(listings) * 100:.1f}%)")

# 3.6 可用性字段 / Availability Fields
print("\n3.6 可用性字段 / Availability Fields:")

if 'availability_365' in listings.columns:
    availability_stats = listings['availability_365'].describe()
    occupancy_rate = (365 - listings['availability_365']) / 365 * 100
    occupancy_stats = occupancy_rate.describe()
    
    print(f"  availability_365 统计 / Statistics:")
    print(f"    - 均值 / Mean: {availability_stats['mean']:.1f} 天")
    print(f"    - 中位数 / Median: {availability_stats['50%']:.1f} 天")
    print(f"    - 范围 / Range: [{availability_stats['min']:.0f}, {availability_stats['max']:.0f}] 天")
    
    print(f"\n  入住率统计 / Occupancy Rate Statistics:")
    print(f"    - 平均入住率 / Mean Occupancy Rate: {occupancy_stats['mean']:.1f}%")
    print(f"    - 中位数入住率 / Median Occupancy Rate: {occupancy_stats['50%']:.1f}%")
    print(f"    - 全年运营房源数 / Fully Occupied (0 days available): {(listings['availability_365'] == 0).sum():,}")
    print(f"    - 全年可用房源数 / Fully Available (365 days): {(listings['availability_365'] == 365).sum():,}")

# ============================================================================
# 4. 关键洞察分析 / Key Insights Analysis
# ============================================================================

print("\n4. 关键洞察分析 / Key Insights Analysis...")

# 4.1 价格分布特征
print("\n4.1 价格分布特征 / Price Distribution Characteristics:")
if 'price' in listings.columns:
    price_skew = listings['price'].skew()
    price_kurt = listings['price'].kurtosis()
    print(f"  - 偏度 / Skewness: {price_skew:.2f} {'(右偏 / Right-skewed)' if price_skew > 0 else '(左偏 / Left-skewed)'}")
    print(f"  - 峰度 / Kurtosis: {price_kurt:.2f}")
    print(f"  - 价格分布特征: {'长尾分布 / Long-tail distribution' if price_skew > 1 else '接近正态分布 / Near-normal distribution'}")

# 4.2 房型分布特征
print("\n4.2 房型分布特征 / Room Type Distribution Characteristics:")
if 'room_type' in listings.columns:
    room_type_price = listings.groupby('room_type')['price'].agg(['mean', 'median', 'count'])
    print("  各房型平均价格 / Average Price by Room Type:")
    for room_type, row in room_type_price.iterrows():
        print(f"    - {room_type}: €{row['mean']:.2f} (中位数 / Median: €{row['median']:.2f}, 数量 / Count: {row['count']:,})")

# 4.3 街区分布特征
print("\n4.3 街区分布特征 / Neighbourhood Distribution Characteristics:")
if 'neighbourhood' in listings.columns:
    neighbourhood_counts = listings['neighbourhood'].value_counts()
    print(f"  - 房源最多的前5个街区 / Top 5 Neighbourhoods:")
    for i, (neighbourhood, count) in enumerate(neighbourhood_counts.head(5).items(), 1):
        pct = (count / len(listings) * 100)
        print(f"    {i}. {neighbourhood}: {count:,} ({pct:.1f}%)")

# 4.4 评论模式特征
print("\n4.4 评论模式特征 / Review Pattern Characteristics:")
if 'number_of_reviews' in listings.columns:
    has_reviews = listings['number_of_reviews'] > 0
    print(f"  - 有评论的房源占比: {has_reviews.sum() / len(listings) * 100:.1f}%")
    print(f"  - 无评论的房源占比: {(~has_reviews).sum() / len(listings) * 100:.1f}%")
    if 'reviews_per_month' in listings.columns:
        active_listings = listings[listings['reviews_per_month'] > 0]
        print(f"  - 活跃房源数（reviews_per_month > 0）: {len(active_listings):,}")

# 4.5 可用性模式特征
print("\n4.5 可用性模式特征 / Availability Pattern Characteristics:")
if 'availability_365' in listings.columns:
    occupancy_rate = (365 - listings['availability_365']) / 365 * 100
    high_occupancy = occupancy_rate > 80
    low_occupancy = occupancy_rate < 20
    print(f"  - 高入住率房源（>80%）: {high_occupancy.sum():,} ({high_occupancy.sum() / len(listings) * 100:.1f}%)")
    print(f"  - 低入住率房源（<20%）: {low_occupancy.sum():,} ({low_occupancy.sum() / len(listings) * 100:.1f}%)")

# ============================================================================
# 5. 创建可视化图表 / Create Visualizations
# ============================================================================

print("\n5. 创建可视化图表 / Creating Visualizations...")

# 5.1 字段统计汇总图
fig, axes = plt.subplots(3, 2, figsize=(14, 12))

# 5.1.1 房型分布
if 'room_type' in listings.columns:
    room_type_counts = listings['room_type'].value_counts()
    axes[0, 0].bar(room_type_counts.index, room_type_counts.values, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Room Type Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Count', fontsize=11)
    axes[0, 0].tick_params(axis='x', rotation=45)
    for i, v in enumerate(room_type_counts.values):
        axes[0, 0].text(i, v, f'{v:,}', ha='center', va='bottom', fontsize=9)

# 5.1.2 价格分布（对数尺度）
if 'price' in listings.columns:
    price_log = np.log1p(listings['price'])
    axes[0, 1].hist(price_log, bins=50, color='coral', edgecolor='black', alpha=0.7)
    axes[0, 1].set_title('Price Distribution (Log Scale)', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Log(Price + 1)', fontsize=11)
    axes[0, 1].set_ylabel('Frequency', fontsize=11)

# 5.1.3 评论数分布（对数尺度）
if 'number_of_reviews' in listings.columns:
    reviews_log = np.log1p(listings[listings['number_of_reviews'] > 0]['number_of_reviews'])
    axes[1, 0].hist(reviews_log, bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
    axes[1, 0].set_title('Reviews Distribution (Log Scale)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Log(Number of Reviews + 1)', fontsize=11)
    axes[1, 0].set_ylabel('Frequency', fontsize=11)

# 5.1.4 入住率分布
if 'availability_365' in listings.columns:
    occupancy_rate = (365 - listings['availability_365']) / 365 * 100
    axes[1, 1].hist(occupancy_rate, bins=50, color='plum', edgecolor='black', alpha=0.7)
    axes[1, 1].set_title('Occupancy Rate Distribution', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Occupancy Rate (%)', fontsize=11)
    axes[1, 1].set_ylabel('Frequency', fontsize=11)

# 5.1.5 最少入住天数分布
if 'minimum_nights' in listings.columns:
    min_nights_counts = listings['minimum_nights'].value_counts().head(20).sort_index()
    axes[2, 0].bar(min_nights_counts.index, min_nights_counts.values, color='gold', edgecolor='black')
    axes[2, 0].set_title('Minimum Nights Distribution (Top 20)', fontsize=12, fontweight='bold')
    axes[2, 0].set_xlabel('Minimum Nights', fontsize=11)
    axes[2, 0].set_ylabel('Count', fontsize=11)

# 5.1.6 房东房源数分布
if 'calculated_host_listings_count' in listings.columns:
    host_counts = listings['calculated_host_listings_count'].value_counts().head(15).sort_index()
    axes[2, 1].bar(host_counts.index, host_counts.values, color='lightblue', edgecolor='black')
    axes[2, 1].set_title('Host Listings Count Distribution', fontsize=12, fontweight='bold')
    axes[2, 1].set_xlabel('Number of Listings per Host', fontsize=11)
    axes[2, 1].set_ylabel('Number of Hosts', fontsize=11)

plt.tight_layout()
plt.savefig(charts_dir / 'chapter5_listings_field_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✅ 已保存: chapter5_listings_field_analysis.png")

# 5.2 数值型字段统计汇总表可视化
if 'price' in listings.columns and 'number_of_reviews' in listings.columns:
    numeric_fields = ['price', 'minimum_nights', 'number_of_reviews', 'reviews_per_month', 
                     'availability_365', 'calculated_host_listings_count']
    numeric_data = listings[numeric_fields].describe().T
    
    fig, ax = plt.subplots(figsize=(12, 8))
    numeric_data[['mean', '50%', 'min', 'max']].plot(kind='barh', ax=ax, width=0.8)
    ax.set_title('Numeric Fields Summary Statistics', fontsize=14, fontweight='bold')
    ax.set_xlabel('Value', fontsize=12)
    ax.set_ylabel('Field', fontsize=12)
    ax.legend(['Mean', 'Median', 'Min', 'Max'], fontsize=10)
    plt.tight_layout()
    plt.savefig(charts_dir / 'chapter5_listings_numeric_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ 已保存: chapter5_listings_numeric_summary.png")

# ============================================================================
# 6. 输出详细统计报告 / Output Detailed Statistics Report
# ============================================================================

print("\n6. 生成详细统计报告 / Generating Detailed Statistics Report...")

report_lines = []
report_lines.append("=" * 80)
report_lines.append("Chapter 5: Detailed Listings Dataset Analysis")
report_lines.append("第5章：listings 数据集详细分析")
report_lines.append("=" * 80)
report_lines.append(f"\n生成时间 / Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")

report_lines.append("\n## 数据集概览 / Dataset Overview")
for key, value in dataset_info.items():
    if isinstance(value, float):
        report_lines.append(f"  - {key}: {value:.2f}")
    else:
        report_lines.append(f"  - {key}: {value:,}")

report_lines.append("\n## 字段详细分析 / Field Analysis")
for field, analysis in field_analysis.items():
    report_lines.append(f"\n### {field}")
    report_lines.append(f"  - Data Type / 数据类型: {analysis['dtype']}")
    report_lines.append(f"  - Missing Rate / 缺失率: {analysis['missing_pct']:.2f}%")
    report_lines.append(f"  - Unique Values / 唯一值数: {analysis['unique_count']:,}")
    if 'statistics' in analysis:
        stats = analysis['statistics']
        report_lines.append(f"  - Mean / 均值: {stats['mean']:.2f}")
        report_lines.append(f"  - Median / 中位数: {stats['median']:.2f}")

# 保存报告
with open(charts_dir / 'chapter5_listings_statistics.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))

print("  ✅ 已保存: chapter5_listings_statistics.txt")

print("\n" + "=" * 80)
print("分析完成 / Analysis Complete!")
print("=" * 80)

