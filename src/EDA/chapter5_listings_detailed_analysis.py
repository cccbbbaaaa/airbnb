"""
Chapter 5.5: Listings Detailed Dataset Analysis
第5.5章：listings_detailed 数据集详细分析

本脚本对 listings_detailed.xlsx 进行探索性分析，对比与 listings.csv 的差异。
This script performs exploratory analysis on listings_detailed.xlsx and compares it with listings.csv.
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
project_root, data_dir, charts_eda_dir, charts_model_dir = get_project_paths()
charts_dir = charts_eda_dir  # 使用 EDA 目录 / Use EDA directory

print("=" * 80)
print("Chapter 5.5: Listings Detailed Dataset Analysis")
print("第5.5章：listings_detailed 数据集详细分析")
print("=" * 80)

# ============================================================================
# 1. 加载数据 / Load Data
# ============================================================================

print("\n1. 加载数据 / Loading Data...")

try:
    listings_detailed = pd.read_excel(data_dir / 'listings_detailed.xlsx')
    listings = pd.read_csv(data_dir / 'listings.csv')
    print(f"  ✅ listings_detailed.xlsx 加载完成: {len(listings_detailed)} 行 × {len(listings_detailed.columns)} 列")
    print(f"  ✅ listings.csv 加载完成: {len(listings)} 行 × {len(listings.columns)} 列")
except Exception as e:
    print(f"  ❌ 数据加载失败: {e}")
    exit(1)

# ============================================================================
# 2. 数据集概览 / Dataset Overview
# ============================================================================

print("\n2. 数据集概览 / Dataset Overview...")

dataset_info = {
    'detailed_records': len(listings_detailed),
    'detailed_columns': len(listings_detailed.columns),
    'csv_records': len(listings),
    'csv_columns': len(listings.columns),
    'extra_columns': len(listings_detailed.columns) - len(listings.columns)
}

print(f"  - listings_detailed 记录数: {dataset_info['detailed_records']:,}")
print(f"  - listings_detailed 字段数: {dataset_info['detailed_columns']}")
print(f"  - listings.csv 记录数: {dataset_info['csv_records']:,}")
print(f"  - listings.csv 字段数: {dataset_info['csv_columns']}")
print(f"  - 额外字段数: {dataset_info['extra_columns']}")

# ============================================================================
# 3. 字段对比分析 / Field Comparison Analysis
# ============================================================================

print("\n3. 字段对比分析 / Field Comparison Analysis...")

csv_columns = set(listings.columns)
detailed_columns = set(listings_detailed.columns)

common_columns = csv_columns & detailed_columns
csv_only_columns = csv_columns - detailed_columns
detailed_only_columns = detailed_columns - csv_columns

print(f"\n3.1 共同字段 / Common Columns: {len(common_columns)}")
print(f"3.2 listings.csv 独有字段 / CSV-only Columns: {len(csv_only_columns)}")
if csv_only_columns:
    print(f"  - {', '.join(list(csv_only_columns)[:10])}")

print(f"\n3.3 listings_detailed 独有字段 / Detailed-only Columns: {len(detailed_only_columns)}")
if detailed_only_columns:
    print(f"  前20个额外字段 / Top 20 Extra Fields:")
    for i, col in enumerate(list(detailed_only_columns)[:20], 1):
        print(f"    {i:2d}. {col}")

# ============================================================================
# 4. 数据记录对比 / Record Comparison
# ============================================================================

print("\n4. 数据记录对比 / Record Comparison...")

# 检查记录数是否一致
if 'id' in listings_detailed.columns and 'id' in listings.columns:
    detailed_ids = set(listings_detailed['id'].unique())
    csv_ids = set(listings['id'].unique())
    
    print(f"  - listings_detailed 唯一 id 数: {len(detailed_ids):,}")
    print(f"  - listings.csv 唯一 id 数: {len(csv_ids):,}")
    print(f"  - 共同 id 数: {len(detailed_ids & csv_ids):,}")
    print(f"  - detailed 独有 id 数: {len(detailed_ids - csv_ids):,}")
    print(f"  - csv 独有 id 数: {len(csv_ids - detailed_ids):,}")

# ============================================================================
# 5. 额外字段分析 / Extra Fields Analysis
# ============================================================================

print("\n5. 额外字段分析 / Extra Fields Analysis...")

if detailed_only_columns:
    print(f"\n5.1 额外字段类别 / Extra Field Categories:")
    
    # 分类额外字段
    host_fields = [col for col in detailed_only_columns if 'host' in col.lower()]
    review_fields = [col for col in detailed_only_columns if 'review' in col.lower()]
    amenity_fields = [col for col in detailed_only_columns if 'amenity' in col.lower() or 'amenities' in col.lower()]
    description_fields = [col for col in detailed_only_columns if 'description' in col.lower() or 'name' in col.lower()]
    other_fields = detailed_only_columns - set(host_fields) - set(review_fields) - set(amenity_fields) - set(description_fields)
    
    print(f"  - Host 相关字段: {len(host_fields)}")
    print(f"  - Review 相关字段: {len(review_fields)}")
    print(f"  - Amenity 相关字段: {len(amenity_fields)}")
    print(f"  - Description 相关字段: {len(description_fields)}")
    print(f"  - 其他字段: {len(other_fields)}")
    
    # 分析一些重要字段
    print(f"\n5.2 重要额外字段示例 / Important Extra Fields Examples:")
    important_fields = ['host_since', 'host_response_time', 'host_response_rate', 
                       'host_acceptance_rate', 'host_is_superhost', 'description']
    for field in important_fields:
        if field in listings_detailed.columns:
            missing_pct = (listings_detailed[field].isnull().sum() / len(listings_detailed) * 100)
            print(f"  - {field}: 缺失率 {missing_pct:.1f}%")

# ============================================================================
# 6. 创建可视化图表 / Create Visualizations
# ============================================================================

print("\n6. 创建可视化图表 / Creating Visualizations...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 6.1 字段对比
field_comparison = pd.DataFrame({
    'Category': ['Common', 'CSV-only', 'Detailed-only'],
    'Count': [len(common_columns), len(csv_only_columns), len(detailed_only_columns)]
})
axes[0].bar(field_comparison['Category'], field_comparison['Count'], 
            color=['#4CAF50', '#FF9800', '#2196F3'], edgecolor='black')
axes[0].set_title('Field Comparison', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Number of Fields', fontsize=11)
axes[0].tick_params(axis='x', rotation=45)
for i, v in enumerate(field_comparison['Count']):
    axes[0].text(i, v, f'{v}', ha='center', va='bottom', fontsize=10)

# 6.2 额外字段类别分布
if detailed_only_columns:
    field_categories = ['Host', 'Review', 'Amenity', 'Description', 'Other']
    field_counts = [len(host_fields), len(review_fields), len(amenity_fields), 
                   len(description_fields), len(other_fields)]
    axes[1].bar(field_categories, field_counts, color=['#9C27B0', '#F44336', '#00BCD4', 
                                                      '#FFC107', '#607D8B'], edgecolor='black')
    axes[1].set_title('Extra Fields by Category', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Number of Fields', fontsize=11)
    for i, v in enumerate(field_counts):
        axes[1].text(i, v, f'{v}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(charts_dir / 'chapter5_listings_detailed_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✅ 已保存: chapter5_listings_detailed_analysis.png")

# ============================================================================
# 7. 输出统计报告 / Output Statistics Report
# ============================================================================

print("\n7. 生成统计报告 / Generating Statistics Report...")

report_lines = []
report_lines.append("=" * 80)
report_lines.append("Chapter 5.5: Listings Detailed Dataset Analysis")
report_lines.append("第5.5章：listings_detailed 数据集详细分析")
report_lines.append("=" * 80)
report_lines.append(f"\n生成时间 / Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")

report_lines.append("\n## 数据集概览 / Dataset Overview")
for key, value in dataset_info.items():
    report_lines.append(f"  - {key}: {value:,}")

report_lines.append("\n## 字段对比 / Field Comparison")
report_lines.append(f"  - 共同字段数: {len(common_columns)}")
report_lines.append(f"  - CSV 独有字段数: {len(csv_only_columns)}")
report_lines.append(f"  - Detailed 独有字段数: {len(detailed_only_columns)}")

if detailed_only_columns:
    report_lines.append("\n## 额外字段类别 / Extra Field Categories")
    report_lines.append(f"  - Host 相关: {len(host_fields)}")
    report_lines.append(f"  - Review 相关: {len(review_fields)}")
    report_lines.append(f"  - Amenity 相关: {len(amenity_fields)}")
    report_lines.append(f"  - Description 相关: {len(description_fields)}")
    report_lines.append(f"  - 其他: {len(other_fields)}")

with open(charts_dir / 'chapter5_listings_detailed_statistics.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))

print("  ✅ 已保存: chapter5_listings_detailed_statistics.txt")

print("\n" + "=" * 80)
print("分析完成 / Analysis Complete!")
print("=" * 80)

