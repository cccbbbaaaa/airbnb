"""
Chapter 5.3: Calendar Summary Dataset Analysis
第5.3章：calendar_summary 数据集详细分析

本脚本对 calendar_summary.csv 进行详细分析，包括入住率分析、可用性模式分析等。
This script performs detailed analysis on calendar_summary.csv, including occupancy rate analysis and availability pattern analysis.
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
print("Chapter 5.3: Calendar Summary Dataset Analysis")
print("第5.3章：calendar_summary 数据集详细分析")
print("=" * 80)

# ============================================================================
# 1. 加载数据 / Load Data
# ============================================================================

print("\n1. 加载数据 / Loading Data...")

calendar = pd.read_csv(data_dir / 'calendar_summary.csv', sep=';')
print(f"  ✅ 数据加载完成: {len(calendar)} 行 × {len(calendar.columns)} 列")
print(f"  - 字段 / Columns: {list(calendar.columns)}")

# ============================================================================
# 2. 数据集概览 / Dataset Overview
# ============================================================================

print("\n2. 数据集概览 / Dataset Overview...")

dataset_info = {
    'records': len(calendar),
    'columns': len(calendar.columns),
    'unique_listings': calendar['listing_id'].nunique(),
    'duplicate_rows': calendar.duplicated().sum(),
    'missing_values': calendar.isnull().sum().sum()
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
print(f"  - 唯一 listing_id 数: {calendar['listing_id'].nunique():,}")
print(f"  - 平均每个房源的记录数: {len(calendar) / calendar['listing_id'].nunique():.1f}")

# 3.2 available 字段分析
print("\n3.2 available 字段分析 / available Field Analysis:")
available_dist = calendar['available'].value_counts()
print(f"  - 可用状态分布 / Available Status Distribution:")
for status, count in available_dist.items():
    pct = (count / len(calendar) * 100)
    status_name = "可用 / Available" if status == 't' else "不可用 / Unavailable"
    print(f"    - {status_name} ({status}): {count:,} ({pct:.2f}%)")

# 3.3 count 字段分析
print("\n3.3 count 字段分析 / count Field Analysis:")
count_stats = calendar['count'].describe()
print(f"  - 均值 / Mean: {count_stats['mean']:.1f} 天")
print(f"  - 中位数 / Median: {count_stats['50%']:.1f} 天")
print(f"  - 最小值 / Min: {count_stats['min']:.0f} 天")
print(f"  - 最大值 / Max: {count_stats['max']:.0f} 天")
print(f"  - 标准差 / Std: {count_stats['std']:.1f} 天")

# ============================================================================
# 4. 入住率分析 / Occupancy Rate Analysis
# ============================================================================

print("\n4. 入住率分析 / Occupancy Rate Analysis...")

# 计算每个房源的入住率
print(calendar['available'])
calendar_unavailable = calendar[calendar['available'] == 'f'].groupby('listing_id')['count'].sum()
calendar_available = calendar[calendar['available'] == 't'].groupby('listing_id')['count'].sum()

# 合并数据
listing_availability = pd.DataFrame({
    'listing_id': calendar['listing_id'].unique(),
    'unavailable_days': calendar_unavailable,
    'available_days': calendar_available
}).fillna(0)

listing_availability['total_days'] = listing_availability['unavailable_days'] + listing_availability['available_days']
listing_availability['occupancy_rate'] = (listing_availability['unavailable_days'] / listing_availability['total_days'] * 100).fillna(0)

print(f"\n4.1 入住率统计 / Occupancy Rate Statistics:")
print(f"  - 平均入住率: {listing_availability['occupancy_rate'].mean():.1f}%")
print(f"  - 中位数入住率: {listing_availability['occupancy_rate'].median():.1f}%")
print(f"  - 25%分位入住率: {listing_availability['occupancy_rate'].quantile(0.25):.1f}%")
print(f"  - 75%分位入住率: {listing_availability['occupancy_rate'].quantile(0.75):.1f}%")
print(f"  - 90%分位入住率: {listing_availability['occupancy_rate'].quantile(0.90):.1f}%")

print(f"\n4.2 入住率分类 / Occupancy Rate Categories:")
high_occupancy = (listing_availability['occupancy_rate'] > 80).sum()
medium_occupancy = ((listing_availability['occupancy_rate'] >= 20) & (listing_availability['occupancy_rate'] <= 80)).sum()
low_occupancy = (listing_availability['occupancy_rate'] < 20).sum()
full_occupancy = (listing_availability['occupancy_rate'] == 100).sum()
no_occupancy = (listing_availability['occupancy_rate'] == 0).sum()

print(f"  - 高入住率（>80%）: {high_occupancy:,} ({high_occupancy/len(listing_availability)*100:.1f}%)")
print(f"  - 中入住率（20-80%）: {medium_occupancy:,} ({medium_occupancy/len(listing_availability)*100:.1f}%)")
print(f"  - 低入住率（<20%）: {low_occupancy:,} ({low_occupancy/len(listing_availability)*100:.1f}%)")
print(f"  - 全年运营（100%）: {full_occupancy:,} ({full_occupancy/len(listing_availability)*100:.1f}%)")
print(f"  - 全年可用（0%）: {no_occupancy:,} ({no_occupancy/len(listing_availability)*100:.1f}%)")

# ============================================================================
# 5. 可用性模式分析 / Availability Pattern Analysis
# ============================================================================

print("\n5. 可用性模式分析 / Availability Pattern Analysis...")

# 5.1 全年运营 vs 季节性运营
print("\n5.1 运营模式分析 / Operation Mode Analysis:")
year_round = (listing_availability['unavailable_days'] >= 300).sum()
seasonal = ((listing_availability['unavailable_days'] >= 100) & (listing_availability['unavailable_days'] < 300)).sum()
occasional = (listing_availability['unavailable_days'] < 100).sum()

print(f"  - 全年运营（入住天数≥300）: {year_round:,} ({year_round/len(listing_availability)*100:.1f}%)")
print(f"  - 季节性运营（100-300天）: {seasonal:,} ({seasonal/len(listing_availability)*100:.1f}%)")
print(f"  - 偶尔运营（<100天）: {occasional:,} ({occasional/len(listing_availability)*100:.1f}%)")

# ============================================================================
# 6. 创建可视化图表 / Create Visualizations
# ============================================================================

print("\n6. 创建可视化图表 / Creating Visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 6.1 可用性状态分布
available_dist.plot(kind='bar', ax=axes[0, 0], color=['#4CAF50', '#F44336'], edgecolor='black')
axes[0, 0].set_title('Availability Status Distribution', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Available Status', fontsize=11)
axes[0, 0].set_ylabel('Count', fontsize=11)
axes[0, 0].set_xticklabels(['Available', 'Unavailable'], rotation=0)
for i, v in enumerate(available_dist.values):
    axes[0, 0].text(i, v, f'{v:,}', ha='center', va='bottom', fontsize=10)

# 6.2 入住率分布
axes[0, 1].hist(listing_availability['occupancy_rate'], bins=50, color='#2196F3', edgecolor='black', alpha=0.7)
axes[0, 1].set_title('Occupancy Rate Distribution', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Occupancy Rate (%)', fontsize=11)
axes[0, 1].set_ylabel('Frequency', fontsize=11)
axes[0, 1].axvline(listing_availability['occupancy_rate'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {listing_availability["occupancy_rate"].mean():.1f}%')
axes[0, 1].legend()

# 6.3 入住率分类
occupancy_categories = ['High (>80%)', 'Medium (20-80%)', 'Low (<20%)']
occupancy_counts = [high_occupancy, medium_occupancy, low_occupancy]
axes[1, 0].bar(occupancy_categories, occupancy_counts, color=['#4CAF50', '#FF9800', '#F44336'], edgecolor='black')
axes[1, 0].set_title('Occupancy Rate Categories', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Number of Listings', fontsize=11)
axes[1, 0].tick_params(axis='x', rotation=45)
for i, v in enumerate(occupancy_counts):
    axes[1, 0].text(i, v, f'{v:,}', ha='center', va='bottom', fontsize=10)

# 6.4 运营模式分布
operation_modes = ['Year-round', 'Seasonal', 'Occasional']
operation_counts = [year_round, seasonal, occasional]
axes[1, 1].bar(operation_modes, operation_counts, color=['#9C27B0', '#FF9800', '#00BCD4'], edgecolor='black')
axes[1, 1].set_title('Operation Mode Distribution', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('Number of Listings', fontsize=11)
for i, v in enumerate(operation_counts):
    axes[1, 1].text(i, v, f'{v:,}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(charts_dir / 'chapter5_calendar_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✅ 已保存: chapter5_calendar_analysis.png")

# ============================================================================
# 7. 输出统计报告 / Output Statistics Report
# ============================================================================

print("\n7. 生成统计报告 / Generating Statistics Report...")

report_lines = []
report_lines.append("=" * 80)
report_lines.append("Chapter 5.3: Calendar Summary Dataset Analysis")
report_lines.append("第5.3章：calendar_summary 数据集详细分析")
report_lines.append("=" * 80)
report_lines.append(f"\n生成时间 / Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")

report_lines.append("\n## 数据集概览 / Dataset Overview")
for key, value in dataset_info.items():
    if isinstance(value, (int, float)):
        report_lines.append(f"  - {key}: {value:,}")

report_lines.append("\n## 入住率分析 / Occupancy Rate Analysis")
report_lines.append(f"  - 平均入住率: {listing_availability['occupancy_rate'].mean():.1f}%")
report_lines.append(f"  - 中位数入住率: {listing_availability['occupancy_rate'].median():.1f}%")
report_lines.append(f"  - 高入住率房源（>80%）: {high_occupancy:,}")
report_lines.append(f"  - 全年运营房源（100%）: {full_occupancy:,}")

with open(charts_dir / 'chapter5_calendar_statistics.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))

print("  ✅ 已保存: chapter5_calendar_statistics.txt")

print("\n" + "=" * 80)
print("分析完成 / Analysis Complete!")
print("=" * 80)

