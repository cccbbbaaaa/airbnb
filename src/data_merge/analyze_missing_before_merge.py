"""
分析合并前两个数据集的缺失值情况
Analyze Missing Values in Both Datasets Before Merging
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# 添加 EDA 目录到路径，以便导入 utils 模块 / Add EDA directory to path for importing utils module
sys.path.insert(0, str(Path(__file__).parent.parent / "EDA"))
from utils import setup_plotting, get_project_paths

# 设置中文字体支持 / Set Chinese font support
setup_plotting()

# 获取项目路径
project_root, data_dir, charts_eda_dir, charts_model_dir = get_project_paths()
charts_dir = project_root / 'charts' / 'data_merge'
charts_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("合并前数据集缺失值分析 / Missing Values Analysis Before Merging")
print("=" * 80)

# ============================================================================
# 1. 加载数据 / Load Data
# ============================================================================

print("\n1. 加载数据 / Loading Data...")

try:
    listings_2021 = pd.read_excel(data_dir / 'listings_detailed.xlsx')
    print(f"  ✅ 2021年数据: {len(listings_2021):,} 行 × {len(listings_2021.columns)} 列")
    
    listings_2025 = pd.read_csv(project_root / 'data' / '2025' / 'listings_detailed.csv')
    print(f"  ✅ 2025年数据: {len(listings_2025):,} 行 × {len(listings_2025.columns)} 列")
except Exception as e:
    print(f"  ❌ 数据加载失败: {e}")
    raise

# ============================================================================
# 2. 计算缺失值统计 / Calculate Missing Values Statistics
# ============================================================================

print("\n2. 计算缺失值统计 / Calculating Missing Values Statistics...")

def calculate_missing_stats(df, year):
    """计算缺失值统计"""
    stats = []
    total_rows = len(df)
    
    for col in df.columns:
        missing_count = df[col].isna().sum()
        missing_pct = missing_count / total_rows * 100
        non_missing_count = total_rows - missing_count
        
        stats.append({
            'column': col,
            'missing_count': missing_count,
            'missing_pct': missing_pct,
            'non_missing_count': non_missing_count,
            'completeness_pct': 100 - missing_pct
        })
    
    stats_df = pd.DataFrame(stats)
    stats_df = stats_df.sort_values('missing_pct', ascending=False)
    return stats_df

stats_2021 = calculate_missing_stats(listings_2021, 2021)
stats_2025 = calculate_missing_stats(listings_2025, 2025)

# ============================================================================
# 3. 整体缺失值概况 / Overall Missing Values Overview
# ============================================================================

print("\n3. 整体缺失值概况 / Overall Missing Values Overview")

print("\n3.1 2021年数据缺失值概况:")
print(f"  - 总记录数: {len(listings_2021):,}")
print(f"  - 总字段数: {len(listings_2021.columns)}")
print(f"  - 完全缺失的字段（100%）: {(stats_2021['missing_pct'] == 100).sum()}")
print(f"  - 完全完整的字段（0%）: {(stats_2021['missing_pct'] == 0).sum()}")
print(f"  - 高缺失率字段（>50%）: {(stats_2021['missing_pct'] > 50).sum()}")
print(f"  - 中等缺失率字段（10-50%）: {((stats_2021['missing_pct'] > 10) & (stats_2021['missing_pct'] <= 50)).sum()}")
print(f"  - 低缺失率字段（<10%）: {(stats_2021['missing_pct'] < 10).sum()}")

print("\n3.2 2025年数据缺失值概况:")
print(f"  - 总记录数: {len(listings_2025):,}")
print(f"  - 总字段数: {len(listings_2025.columns)}")
print(f"  - 完全缺失的字段（100%）: {(stats_2025['missing_pct'] == 100).sum()}")
print(f"  - 完全完整的字段（0%）: {(stats_2025['missing_pct'] == 0).sum()}")
print(f"  - 高缺失率字段（>50%）: {(stats_2025['missing_pct'] > 50).sum()}")
print(f"  - 中等缺失率字段（10-50%）: {((stats_2025['missing_pct'] > 10) & (stats_2025['missing_pct'] <= 50)).sum()}")
print(f"  - 低缺失率字段（<10%）: {(stats_2025['missing_pct'] < 10).sum()}")

# ============================================================================
# 4. 详细缺失值分析 / Detailed Missing Values Analysis
# ============================================================================

print("\n4. 详细缺失值分析 / Detailed Missing Values Analysis")

print("\n4.1 2021年数据 - 缺失率最高的20个字段:")
top_missing_2021 = stats_2021.head(20)
for idx, row in top_missing_2021.iterrows():
    print(f"  {row['column']:40s} | 缺失: {row['missing_count']:6,} ({row['missing_pct']:6.2f}%) | 完整: {row['completeness_pct']:6.2f}%")

print("\n4.2 2025年数据 - 缺失率最高的20个字段:")
top_missing_2025 = stats_2025.head(20)
for idx, row in top_missing_2025.iterrows():
    print(f"  {row['column']:40s} | 缺失: {row['missing_count']:6,} ({row['missing_pct']:6.2f}%) | 完整: {row['completeness_pct']:6.2f}%")

# ============================================================================
# 5. 字段对比分析 / Column Comparison Analysis
# ============================================================================

print("\n5. 字段对比分析 / Column Comparison Analysis")

common_cols = set(listings_2021.columns) & set(listings_2025.columns)
only_2021_cols = set(listings_2021.columns) - set(listings_2025.columns)
only_2025_cols = set(listings_2025.columns) - set(listings_2021.columns)

print(f"\n5.1 字段集合对比:")
print(f"  - 共同字段数: {len(common_cols)}")
print(f"  - 仅在2021年的字段数: {len(only_2021_cols)}")
print(f"  - 仅在2025年的字段数: {len(only_2025_cols)}")

# 对比共同字段的缺失率
comparison_data = []
for col in sorted(common_cols):
    stat_2021 = stats_2021[stats_2021['column'] == col].iloc[0]
    stat_2025 = stats_2025[stats_2025['column'] == col].iloc[0]
    
    comparison_data.append({
        'column': col,
        'missing_pct_2021': stat_2021['missing_pct'],
        'missing_pct_2025': stat_2025['missing_pct'],
        'missing_count_2021': stat_2021['missing_count'],
        'missing_count_2025': stat_2025['missing_count'],
        'difference': stat_2025['missing_pct'] - stat_2021['missing_pct'],
        'abs_difference': abs(stat_2025['missing_pct'] - stat_2021['missing_pct'])
    })

comparison_df = pd.DataFrame(comparison_data)

print("\n5.2 共同字段缺失率差异统计:")
print(f"  - 平均缺失率差异: {comparison_df['difference'].mean():.2f}%")
print(f"  - 平均绝对差异: {comparison_df['abs_difference'].mean():.2f}%")
print(f"  - 最大差异: {comparison_df['abs_difference'].max():.2f}%")
print(f"  - 差异>5%的字段数: {(comparison_df['abs_difference'] > 5).sum()}")
print(f"  - 差异>10%的字段数: {(comparison_df['abs_difference'] > 10).sum()}")

# ============================================================================
# 6. 可视化 / Visualization
# ============================================================================

print("\n6. 生成可视化图表 / Generating Visualizations...")

# 设置图表样式
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    try:
        plt.style.use('seaborn-whitegrid')
    except OSError:
        sns.set_style("whitegrid")

# 6.1 缺失值概览对比图
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Missing Values Overview: 2021 vs 2025', fontsize=16, fontweight='bold', y=0.995)

# 子图1: 2021年缺失率分布
ax1 = axes[0, 0]
ax1.hist(stats_2021['missing_pct'], bins=30, alpha=0.7, color='#3498db', edgecolor='black')
ax1.set_xlabel('Missing Rate (%)', fontsize=12)
ax1.set_ylabel('Number of Fields', fontsize=12)
ax1.set_title('2021 Data: Missing Rate Distribution', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.axvline(stats_2021['missing_pct'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {stats_2021["missing_pct"].mean():.1f}%')
ax1.legend()

# 子图2: 2025年缺失率分布
ax2 = axes[0, 1]
ax2.hist(stats_2025['missing_pct'], bins=30, alpha=0.7, color='#ff8c42', edgecolor='black')
ax2.set_xlabel('Missing Rate (%)', fontsize=12)
ax2.set_ylabel('Number of Fields', fontsize=12)
ax2.set_title('2025 Data: Missing Rate Distribution', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.axvline(stats_2025['missing_pct'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {stats_2025["missing_pct"].mean():.1f}%')
ax2.legend()

# 子图3: 缺失率对比（Top 30字段）
ax3 = axes[1, 0]
top_30_missing = comparison_df.nlargest(30, 'abs_difference')
x_pos = np.arange(len(top_30_missing))
width = 0.35
ax3.bar(x_pos - width/2, top_30_missing['missing_pct_2021'], width, 
       label='2021', color='#3498db', alpha=0.8, edgecolor='black')
ax3.bar(x_pos + width/2, top_30_missing['missing_pct_2025'], width, 
       label='2025', color='#ff8c42', alpha=0.8, edgecolor='black')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(top_30_missing['column'], rotation=45, ha='right', fontsize=9)
ax3.set_ylabel('Missing Rate (%)', fontsize=12)
ax3.set_title('Top 30 Fields: Missing Rate Comparison', fontsize=13, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# 子图4: 缺失率分类对比
ax4 = axes[1, 1]
categories = ['Complete\n(0%)', 'Low\n(<10%)', 'Medium\n(10-50%)', 'High\n(50-99%)', 'Missing\n(100%)']
categories_2021 = [
    (stats_2021['missing_pct'] == 0).sum(),
    ((stats_2021['missing_pct'] > 0) & (stats_2021['missing_pct'] < 10)).sum(),
    ((stats_2021['missing_pct'] >= 10) & (stats_2021['missing_pct'] < 50)).sum(),
    ((stats_2021['missing_pct'] >= 50) & (stats_2021['missing_pct'] < 100)).sum(),
    (stats_2021['missing_pct'] == 100).sum()
]
categories_2025 = [
    (stats_2025['missing_pct'] == 0).sum(),
    ((stats_2025['missing_pct'] > 0) & (stats_2025['missing_pct'] < 10)).sum(),
    ((stats_2025['missing_pct'] >= 10) & (stats_2025['missing_pct'] < 50)).sum(),
    ((stats_2025['missing_pct'] >= 50) & (stats_2025['missing_pct'] < 100)).sum(),
    (stats_2025['missing_pct'] == 100).sum()
]

x_pos = np.arange(len(categories))
width = 0.35
ax4.bar(x_pos - width/2, categories_2021, width, label='2021', color='#3498db', alpha=0.8, edgecolor='black')
ax4.bar(x_pos + width/2, categories_2025, width, label='2025', color='#ff8c42', alpha=0.8, edgecolor='black')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(categories, fontsize=10)
ax4.set_ylabel('Number of Fields', fontsize=12)
ax4.set_title('Missing Rate Categories Comparison', fontsize=13, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
overview_chart_file = charts_dir / 'missing_values_overview_before_merge.png'
plt.savefig(overview_chart_file, dpi=300, bbox_inches='tight')
print(f"  ✅ 概览图表已保存: {overview_chart_file}")
plt.close()

# 6.2 缺失值热力图
fig, axes = plt.subplots(1, 2, figsize=(20, 12))
fig.suptitle('Missing Values Heatmap: 2021 vs 2025', fontsize=16, fontweight='bold', y=0.98)

# 准备热力图数据（选择缺失率>0的字段）
fields_with_missing_2021 = stats_2021[stats_2021['missing_pct'] > 0].head(40)
fields_with_missing_2025 = stats_2025[stats_2025['missing_pct'] > 0].head(40)

# 2021年热力图
if len(fields_with_missing_2021) > 0:
    heatmap_data_2021 = pd.DataFrame({
        '2021': fields_with_missing_2021['missing_pct'].values
    }, index=fields_with_missing_2021['column'].values)
    
    sns.heatmap(heatmap_data_2021.T, annot=True, fmt='.1f', cmap='YlOrRd', 
                cbar_kws={'label': 'Missing Rate (%)'}, ax=axes[0], 
                linewidths=0.5, linecolor='gray')
    axes[0].set_title('2021 Data: Missing Rate Heatmap (Top 40)', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Year', fontsize=12)
    axes[0].set_xlabel('Field', fontsize=12)

# 2025年热力图
if len(fields_with_missing_2025) > 0:
    heatmap_data_2025 = pd.DataFrame({
        '2025': fields_with_missing_2025['missing_pct'].values
    }, index=fields_with_missing_2025['column'].values)
    
    sns.heatmap(heatmap_data_2025.T, annot=True, fmt='.1f', cmap='YlOrRd', 
                cbar_kws={'label': 'Missing Rate (%)'}, ax=axes[1], 
                linewidths=0.5, linecolor='gray')
    axes[1].set_title('2025 Data: Missing Rate Heatmap (Top 40)', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Year', fontsize=12)
    axes[1].set_xlabel('Field', fontsize=12)

plt.tight_layout()
heatmap_file = charts_dir / 'missing_values_heatmap_before_merge.png'
plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
print(f"  ✅ 热力图已保存: {heatmap_file}")
plt.close()

# ============================================================================
# 7. 保存详细报告 / Save Detailed Reports
# ============================================================================

print("\n7. 保存详细报告 / Saving Detailed Reports...")

# 保存2021年缺失值统计
stats_2021_file = charts_dir / 'missing_values_2021.csv'
stats_2021.to_csv(stats_2021_file, index=False, encoding='utf-8-sig')
print(f"  ✅ 2021年缺失值统计已保存: {stats_2021_file}")

# 保存2025年缺失值统计
stats_2025_file = charts_dir / 'missing_values_2025.csv'
stats_2025.to_csv(stats_2025_file, index=False, encoding='utf-8-sig')
print(f"  ✅ 2025年缺失值统计已保存: {stats_2025_file}")

# 保存对比数据
comparison_file = charts_dir / 'missing_values_comparison_before_merge.csv'
comparison_df_sorted = comparison_df.sort_values('abs_difference', ascending=False)
comparison_df_sorted.to_csv(comparison_file, index=False, encoding='utf-8-sig')
print(f"  ✅ 对比数据已保存: {comparison_file}")

# 生成文本报告
report_lines = []
report_lines.append("=" * 80)
report_lines.append("Missing Values Analysis Report - Before Merging")
report_lines.append("合并前缺失值分析报告")
report_lines.append("=" * 80)
report_lines.append("")

report_lines.append("1. 2021年数据缺失值概况 / 2021 Data Missing Values Overview:")
report_lines.append(f"  - 总记录数: {len(listings_2021):,}")
report_lines.append(f"  - 总字段数: {len(listings_2021.columns)}")
report_lines.append(f"  - 完全缺失的字段（100%）: {(stats_2021['missing_pct'] == 100).sum()}")
report_lines.append(f"  - 完全完整的字段（0%）: {(stats_2021['missing_pct'] == 0).sum()}")
report_lines.append(f"  - 高缺失率字段（>50%）: {(stats_2021['missing_pct'] > 50).sum()}")
report_lines.append(f"  - 中等缺失率字段（10-50%）: {((stats_2021['missing_pct'] > 10) & (stats_2021['missing_pct'] <= 50)).sum()}")
report_lines.append(f"  - 低缺失率字段（<10%）: {(stats_2021['missing_pct'] < 10).sum()}")
report_lines.append(f"  - 平均缺失率: {stats_2021['missing_pct'].mean():.2f}%")
report_lines.append("")

report_lines.append("2. 2025年数据缺失值概况 / 2025 Data Missing Values Overview:")
report_lines.append(f"  - 总记录数: {len(listings_2025):,}")
report_lines.append(f"  - 总字段数: {len(listings_2025.columns)}")
report_lines.append(f"  - 完全缺失的字段（100%）: {(stats_2025['missing_pct'] == 100).sum()}")
report_lines.append(f"  - 完全完整的字段（0%）: {(stats_2025['missing_pct'] == 0).sum()}")
report_lines.append(f"  - 高缺失率字段（>50%）: {(stats_2025['missing_pct'] > 50).sum()}")
report_lines.append(f"  - 中等缺失率字段（10-50%）: {((stats_2025['missing_pct'] > 10) & (stats_2025['missing_pct'] <= 50)).sum()}")
report_lines.append(f"  - 低缺失率字段（<10%）: {(stats_2025['missing_pct'] < 10).sum()}")
report_lines.append(f"  - 平均缺失率: {stats_2025['missing_pct'].mean():.2f}%")
report_lines.append("")

report_lines.append("3. 字段对比 / Column Comparison:")
report_lines.append(f"  - 共同字段数: {len(common_cols)}")
report_lines.append(f"  - 仅在2021年的字段数: {len(only_2021_cols)}")
report_lines.append(f"  - 仅在2025年的字段数: {len(only_2025_cols)}")
report_lines.append("")

if only_2021_cols:
    report_lines.append("仅在2021年的字段 / Columns Only in 2021:")
    for col in sorted(only_2021_cols):
        report_lines.append(f"  - {col}")
    report_lines.append("")

if only_2025_cols:
    report_lines.append("仅在2025年的字段 / Columns Only in 2025:")
    for col in sorted(only_2025_cols):
        report_lines.append(f"  - {col}")
    report_lines.append("")

report_lines.append("4. 共同字段缺失率差异统计 / Common Fields Missing Rate Difference:")
report_lines.append(f"  - 平均缺失率差异: {comparison_df['difference'].mean():.2f}%")
report_lines.append(f"  - 平均绝对差异: {comparison_df['abs_difference'].mean():.2f}%")
report_lines.append(f"  - 最大差异: {comparison_df['abs_difference'].max():.2f}%")
report_lines.append(f"  - 差异>5%的字段数: {(comparison_df['abs_difference'] > 5).sum()}")
report_lines.append(f"  - 差异>10%的字段数: {(comparison_df['abs_difference'] > 10).sum()}")
report_lines.append("")

report_lines.append("5. 2021年数据 - 缺失率最高的20个字段 / Top 20 Fields with Highest Missing Rate (2021):")
for idx, row in stats_2021.head(20).iterrows():
    report_lines.append(f"  {row['column']:40s} | {row['missing_pct']:6.2f}% ({row['missing_count']:,}/{len(listings_2021):,})")
report_lines.append("")

report_lines.append("6. 2025年数据 - 缺失率最高的20个字段 / Top 20 Fields with Highest Missing Rate (2025):")
for idx, row in stats_2025.head(20).iterrows():
    report_lines.append(f"  {row['column']:40s} | {row['missing_pct']:6.2f}% ({row['missing_count']:,}/{len(listings_2025):,})")
report_lines.append("")

report_lines.append("7. 差异最大的20个字段 / Top 20 Fields with Largest Differences:")
for idx, row in comparison_df.nlargest(20, 'abs_difference').iterrows():
    report_lines.append(f"  {row['column']:40s} | 2021: {row['missing_pct_2021']:6.2f}% | 2025: {row['missing_pct_2025']:6.2f}% | Diff: {row['difference']:+7.2f}%")
report_lines.append("")

report_lines.append("=" * 80)

report_file = charts_dir / 'missing_values_analysis_before_merge.txt'
with open(report_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))
print(f"  ✅ 分析报告已保存: {report_file}")

print("\n" + "=" * 80)
print("缺失值分析完成！/ Missing Values Analysis Complete!")
print("=" * 80)
print(f"\n生成的文件 / Generated Files:")
print(f"  1. {overview_chart_file}")
print(f"  2. {heatmap_file}")
print(f"  3. {stats_2021_file}")
print(f"  4. {stats_2025_file}")
print(f"  5. {comparison_file}")
print(f"  6. {report_file}")

