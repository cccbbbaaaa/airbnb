"""
Chapter 5.4: Neighbourhoods Dataset Analysis
第5.4章：neighbourhoods 数据集详细分析

本脚本对 neighbourhoods.csv 进行详细分析，这是一个参考数据集。
This script performs detailed analysis on neighbourhoods.csv, which is a reference dataset.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from utils import setup_plotting, get_project_paths

# 设置中文字体支持 / Set Chinese font support
setup_plotting()

# 获取项目根目录路径 / Get project root directory path
project_root, data_dir, charts_eda_dir, charts_model_dir = get_project_paths()
charts_dir = charts_eda_dir  # 使用 EDA 目录 / Use EDA directory

print("=" * 80)
print("Chapter 5.4: Neighbourhoods Dataset Analysis")
print("第5.4章：neighbourhoods 数据集详细分析")
print("=" * 80)

# ============================================================================
# 1. 加载数据 / Load Data
# ============================================================================

print("\n1. 加载数据 / Loading Data...")

neighbourhoods = pd.read_csv(data_dir / 'neighbourhoods.csv')
print(f"  ✅ 数据加载完成: {len(neighbourhoods)} 行 × {len(neighbourhoods.columns)} 列")
print(f"  - 字段 / Columns: {list(neighbourhoods.columns)}")

# ============================================================================
# 2. 数据集概览 / Dataset Overview
# ============================================================================

print("\n2. 数据集概览 / Dataset Overview...")

dataset_info = {
    'records': len(neighbourhoods),
    'columns': len(neighbourhoods.columns),
    'unique_neighbourhoods': neighbourhoods['neighbourhood'].nunique(),
    'missing_values': neighbourhoods.isnull().sum().sum()
}

print(f"  - 总记录数 / Total Records: {dataset_info['records']}")
print(f"  - 字段数 / Columns: {dataset_info['columns']}")
print(f"  - 唯一街区数 / Unique Neighbourhoods: {dataset_info['unique_neighbourhoods']}")
print(f"  - 缺失值总数 / Total Missing Values: {dataset_info['missing_values']}")

# ============================================================================
# 3. 字段详细分析 / Field Analysis
# ============================================================================

print("\n3. 字段详细分析 / Field Analysis...")

# 3.1 neighbourhood_group 字段
print("\n3.1 neighbourhood_group 字段分析 / neighbourhood_group Field Analysis:")
if 'neighbourhood_group' in neighbourhoods.columns:
    missing_count = neighbourhoods['neighbourhood_group'].isna().sum()
    print(f"  - 缺失值数: {missing_count} ({missing_count/len(neighbourhoods)*100:.1f}%)")
    print(f"  - 非缺失值数: {(~neighbourhoods['neighbourhood_group'].isna()).sum()}")

# 3.2 neighbourhood 字段
print("\n3.2 neighbourhood 字段分析 / neighbourhood Field Analysis:")
if 'neighbourhood' in neighbourhoods.columns:
    print(f"  - 唯一值数: {neighbourhoods['neighbourhood'].nunique()}")
    print(f"  - 缺失值数: {neighbourhoods['neighbourhood'].isna().sum()}")
    print(f"\n  所有街区列表 / All Neighbourhoods List:")
    for i, neighbourhood in enumerate(neighbourhoods['neighbourhood'].dropna().unique(), 1):
        print(f"    {i:2d}. {neighbourhood}")

# ============================================================================
# 4. 与 listings 数据关联分析 / Integration with Listings Data
# ============================================================================

print("\n4. 与 listings 数据关联分析 / Integration with Listings Data...")

listings = pd.read_csv(data_dir / 'listings.csv')

# 检查 neighbourhoods 中的街区是否都在 listings 中
neighbourhoods_list = set(neighbourhoods['neighbourhood'].dropna().unique())
listings_neighbourhoods = set(listings['neighbourhood'].unique())

print(f"  - neighbourhoods 中的街区数: {len(neighbourhoods_list)}")
print(f"  - listings 中的街区数: {len(listings_neighbourhoods)}")
print(f"  - 完全匹配: {neighbourhoods_list == listings_neighbourhoods}")

# 统计每个街区的房源数
if 'neighbourhood' in listings.columns:
    listings_by_neighbourhood = listings['neighbourhood'].value_counts().sort_values(ascending=False)
    print(f"\n  各街区房源数统计 / Listings Count by Neighbourhood:")
    for neighbourhood, count in listings_by_neighbourhood.head(10).items():
        pct = (count / len(listings) * 100)
        print(f"    - {neighbourhood}: {count:,} ({pct:.1f}%)")

# ============================================================================
# 5. 创建可视化图表 / Create Visualizations
# ============================================================================

print("\n5. 创建可视化图表 / Creating Visualizations...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 5.1 街区列表（如果有数据）
if 'neighbourhood' in neighbourhoods.columns:
    neighbourhoods_list_sorted = sorted(neighbourhoods['neighbourhood'].dropna().unique())
    axes[0].barh(range(len(neighbourhoods_list_sorted)), [1] * len(neighbourhoods_list_sorted), 
                color='lightblue', edgecolor='black')
    axes[0].set_yticks(range(len(neighbourhoods_list_sorted)))
    axes[0].set_yticklabels(neighbourhoods_list_sorted, fontsize=9)
    axes[0].set_title('All Neighbourhoods', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Count', fontsize=11)
    axes[0].invert_yaxis()

# 5.2 各街区房源数（Top 15）
if 'neighbourhood' in listings.columns:
    top_neighbourhoods = listings_by_neighbourhood.head(15)
    axes[1].barh(range(len(top_neighbourhoods)), top_neighbourhoods.values, 
                 color='coral', edgecolor='black')
    axes[1].set_yticks(range(len(top_neighbourhoods)))
    axes[1].set_yticklabels(top_neighbourhoods.index, fontsize=9)
    axes[1].set_title('Top 15 Neighbourhoods by Listings Count', 
                      fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Number of Listings', fontsize=11)
    axes[1].invert_yaxis()
    for i, v in enumerate(top_neighbourhoods.values):
        axes[1].text(v, i, f' {v:,}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(charts_dir / 'chapter5_neighbourhoods_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✅ 已保存: chapter5_neighbourhoods_analysis.png")

# ============================================================================
# 6. 输出统计报告 / Output Statistics Report
# ============================================================================

print("\n6. 生成统计报告 / Generating Statistics Report...")

report_lines = []
report_lines.append("=" * 80)
report_lines.append("Chapter 5.4: Neighbourhoods Dataset Analysis")
report_lines.append("第5.4章：neighbourhoods 数据集详细分析")
report_lines.append("=" * 80)
report_lines.append(f"\n生成时间 / Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")

report_lines.append("\n## 数据集概览 / Dataset Overview")
for key, value in dataset_info.items():
    report_lines.append(f"  - {key}: {value}")

report_lines.append("\n## 所有街区列表 / All Neighbourhoods List")
for i, neighbourhood in enumerate(neighbourhoods['neighbourhood'].dropna().unique(), 1):
    report_lines.append(f"  {i:2d}. {neighbourhood}")

with open(charts_dir / 'chapter5_neighbourhoods_statistics.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))

print("  ✅ 已保存: chapter5_neighbourhoods_statistics.txt")

print("\n" + "=" * 80)
print("分析完成 / Analysis Complete!")
print("=" * 80)

