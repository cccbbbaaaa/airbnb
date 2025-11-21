"""
检查2021年和2025年数据的缺失值情况对比
Check Missing Values Comparison between 2021 and 2025 Data
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
print("缺失值情况对比分析 / Missing Values Comparison Analysis")
print("=" * 80)

# ============================================================================
# 1. 加载数据 / Load Data
# ============================================================================

print("\n1. 加载数据 / Loading Data...")

try:
    listings_2021 = pd.read_excel(data_dir / 'listings_detailed.xlsx')
    print(f"  ✅ listings_detailed.xlsx (2021): {len(listings_2021):,} 行 × {len(listings_2021.columns)} 列")
    
    listings_2025 = pd.read_excel(data_dir / 'listings_detailed_2.xlsx')
    print(f"  ✅ listings_detailed_2.xlsx (2025): {len(listings_2025):,} 行 × {len(listings_2025.columns)} 列")
except Exception as e:
    print(f"  ❌ 数据加载失败: {e}")
    exit(1)

# ============================================================================
# 2. 计算缺失值统计 / Calculate Missing Values Statistics
# ============================================================================

print("\n2. 计算缺失值统计 / Calculating Missing Values Statistics...")

def calculate_missing_stats(df, year):
    """计算缺失值统计"""
    stats = []
    for col in df.columns:
        missing_count = df[col].isna().sum()
        missing_pct = missing_count / len(df) * 100
        stats.append({
            'column': col,
            'missing_count': missing_count,
            'missing_pct': missing_pct,
            'total_count': len(df),
            'non_missing_count': len(df) - missing_count
        })
    return pd.DataFrame(stats)

stats_2021 = calculate_missing_stats(listings_2021, 2021)
stats_2025 = calculate_missing_stats(listings_2025, 2025)

# ============================================================================
# 3. 字段对齐和对比 / Column Alignment and Comparison
# ============================================================================

print("\n3. 字段对齐和对比 / Column Alignment and Comparison...")

# 获取共同字段和独有字段
common_cols = set(listings_2021.columns) & set(listings_2025.columns)
only_2021_cols = set(listings_2021.columns) - set(listings_2025.columns)
only_2025_cols = set(listings_2025.columns) - set(listings_2021.columns)

print(f"  - 共同字段数: {len(common_cols)}")
print(f"  - 仅在2021年的字段数: {len(only_2021_cols)}")
print(f"  - 仅在2025年的字段数: {len(only_2025_cols)}")

# 合并统计信息（只对比共同字段）
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

# ============================================================================
# 4. 缺失值差异分析 / Missing Values Difference Analysis
# ============================================================================

print("\n4. 缺失值差异分析 / Missing Values Difference Analysis...")

# 计算差异统计
print("\n4.1 整体差异统计 / Overall Difference Statistics:")
print(f"  - 平均缺失率差异: {comparison_df['difference'].mean():.2f}%")
print(f"  - 平均绝对差异: {comparison_df['abs_difference'].mean():.2f}%")
print(f"  - 最大差异: {comparison_df['abs_difference'].max():.2f}%")
print(f"  - 差异>5%的字段数: {(comparison_df['abs_difference'] > 5).sum()}")
print(f"  - 差异>10%的字段数: {(comparison_df['abs_difference'] > 10).sum()}")

# 找出差异最大的字段
print("\n4.2 差异最大的字段（Top 10）/ Top 10 Fields with Largest Differences:")
top_diff = comparison_df.nlargest(10, 'abs_difference')[['column', 'missing_pct_2021', 'missing_pct_2025', 'difference', 'abs_difference']]
for idx, row in top_diff.iterrows():
    print(f"  {row['column']:40s} | 2021: {row['missing_pct_2021']:6.2f}% | 2025: {row['missing_pct_2025']:6.2f}% | 差异: {row['difference']:+7.2f}%")

# 分类分析
print("\n4.3 缺失值模式分类 / Missing Pattern Categories:")

# 完全缺失的字段（100%缺失）
complete_missing_2021 = comparison_df[comparison_df['missing_pct_2021'] == 100]
complete_missing_2025 = comparison_df[comparison_df['missing_pct_2025'] == 100]
print(f"  - 2021年完全缺失的字段: {len(complete_missing_2021)}")
print(f"  - 2025年完全缺失的字段: {len(complete_missing_2025)}")

# 完全完整的字段（0%缺失）
complete_2021 = comparison_df[comparison_df['missing_pct_2021'] == 0]
complete_2025 = comparison_df[comparison_df['missing_pct_2025'] == 0]
print(f"  - 2021年完全完整的字段: {len(complete_2021)}")
print(f"  - 2025年完全完整的字段: {len(complete_2025)}")

# 高缺失率字段（>50%缺失）
high_missing_2021 = comparison_df[comparison_df['missing_pct_2021'] > 50]
high_missing_2025 = comparison_df[comparison_df['missing_pct_2025'] > 50]
print(f"  - 2021年高缺失率字段（>50%）: {len(high_missing_2021)}")
print(f"  - 2025年高缺失率字段（>50%）: {len(high_missing_2025)}")

# ============================================================================
# 5. 可视化对比 / Visualization Comparison
# ============================================================================

print("\n5. 生成可视化图表 / Generating Visualizations...")

# 设置图表样式
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    try:
        plt.style.use('seaborn-whitegrid')
    except OSError:
        sns.set_style("whitegrid")

# 5.1 缺失率对比散点图
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Missing Values Comparison: 2021 vs 2025', fontsize=16, fontweight='bold', y=0.995)

# 子图1: 缺失率散点图
ax1 = axes[0, 0]
ax1.scatter(comparison_df['missing_pct_2021'], comparison_df['missing_pct_2025'], 
           alpha=0.6, s=50, color='#3498db')
ax1.plot([0, 100], [0, 100], 'r--', linewidth=2, alpha=0.5, label='y=x (完全相同)')
ax1.set_xlabel('2021 Missing Rate (%)', fontsize=12)
ax1.set_ylabel('2025 Missing Rate (%)', fontsize=12)
ax1.set_title('Missing Rate Scatter Plot', fontsize=13, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 添加对角线区域标注
ax1.fill_between([0, 100], [0, 100], [5, 105], alpha=0.1, color='green', label='Similar (±5%)')
ax1.fill_between([0, 100], [0, 100], [-5, 95], alpha=0.1, color='green')

# 子图2: 差异分布直方图
ax2 = axes[0, 1]
ax2.hist(comparison_df['difference'], bins=30, alpha=0.7, color='#e74c3c', edgecolor='black')
ax2.axvline(0, color='black', linestyle='--', linewidth=2, label='No Difference')
ax2.set_xlabel('Difference (2025 - 2021) (%)', fontsize=12)
ax2.set_ylabel('Frequency', fontsize=12)
ax2.set_title('Missing Rate Difference Distribution', fontsize=13, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 子图3: Top 20差异最大的字段
ax3 = axes[1, 0]
top_20_diff = comparison_df.nlargest(20, 'abs_difference')
y_pos = np.arange(len(top_20_diff))
colors = ['#e74c3c' if d > 0 else '#3498db' for d in top_20_diff['difference']]
ax3.barh(y_pos, top_20_diff['difference'], color=colors, alpha=0.7, edgecolor='black')
ax3.set_yticks(y_pos)
ax3.set_yticklabels(top_20_diff['column'], fontsize=9)
ax3.set_xlabel('Difference (2025 - 2021) (%)', fontsize=12)
ax3.set_title('Top 20 Fields with Largest Differences', fontsize=13, fontweight='bold')
ax3.axvline(0, color='black', linestyle='--', linewidth=1)
ax3.grid(True, alpha=0.3, axis='x')

# 子图4: 缺失率对比（按字段排序）
ax4 = axes[1, 1]
# 选择缺失率较高的字段进行展示
high_missing_cols = comparison_df[comparison_df[['missing_pct_2021', 'missing_pct_2025']].max(axis=1) > 10].nlargest(15, 'abs_difference')
if len(high_missing_cols) > 0:
    x_pos = np.arange(len(high_missing_cols))
    width = 0.35
    ax4.bar(x_pos - width/2, high_missing_cols['missing_pct_2021'], width, 
           label='2021', color='#3498db', alpha=0.8, edgecolor='black')
    ax4.bar(x_pos + width/2, high_missing_cols['missing_pct_2025'], width, 
           label='2025', color='#ff8c42', alpha=0.8, edgecolor='black')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(high_missing_cols['column'], rotation=45, ha='right', fontsize=9)
    ax4.set_ylabel('Missing Rate (%)', fontsize=12)
    ax4.set_title('High Missing Rate Fields Comparison', fontsize=13, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
comparison_chart_file = charts_dir / 'missing_values_comparison.png'
plt.savefig(comparison_chart_file, dpi=300, bbox_inches='tight')
print(f"  ✅ 对比图表已保存: {comparison_chart_file}")
plt.close()

# 5.2 缺失值热力图
fig, axes = plt.subplots(1, 2, figsize=(20, 10))
fig.suptitle('Missing Values Heatmap Comparison', fontsize=16, fontweight='bold', y=0.98)

# 准备热力图数据（选择缺失率>0的字段）
fields_with_missing = comparison_df[
    (comparison_df['missing_pct_2021'] > 0) | (comparison_df['missing_pct_2025'] > 0)
].sort_values('abs_difference', ascending=False).head(30)

if len(fields_with_missing) > 0:
    heatmap_data = pd.DataFrame({
        '2021': fields_with_missing['missing_pct_2021'].values,
        '2025': fields_with_missing['missing_pct_2025'].values
    }, index=fields_with_missing['column'].values)
    
    sns.heatmap(heatmap_data.T, annot=True, fmt='.1f', cmap='YlOrRd', 
                cbar_kws={'label': 'Missing Rate (%)'}, ax=axes[0], 
                linewidths=0.5, linecolor='gray')
    axes[0].set_title('Missing Rate Heatmap (Top 30 Fields)', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Year', fontsize=12)
    axes[0].set_xlabel('Field', fontsize=12)
    
    # 差异热力图
    diff_data = pd.DataFrame({
        'Difference': fields_with_missing['difference'].values
    }, index=fields_with_missing['column'].values)
    
    sns.heatmap(diff_data.T, annot=True, fmt='+.1f', cmap='RdBu_r', center=0,
                cbar_kws={'label': 'Difference (%)'}, ax=axes[1],
                linewidths=0.5, linecolor='gray')
    axes[1].set_title('Missing Rate Difference Heatmap', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Year', fontsize=12)
    axes[1].set_xlabel('Field', fontsize=12)

plt.tight_layout()
heatmap_file = charts_dir / 'missing_values_heatmap.png'
plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
print(f"  ✅ 热力图已保存: {heatmap_file}")
plt.close()

# ============================================================================
# 6. 保存详细对比报告 / Save Detailed Comparison Report
# ============================================================================

print("\n6. 保存详细对比报告 / Saving Detailed Comparison Report...")

# 保存对比数据到CSV
comparison_csv_file = charts_dir / 'missing_values_comparison.csv'
comparison_df_sorted = comparison_df.sort_values('abs_difference', ascending=False)
comparison_df_sorted.to_csv(comparison_csv_file, index=False, encoding='utf-8-sig')
print(f"  ✅ 对比数据已保存: {comparison_csv_file}")

# 保存统计报告
report_output = []
report_output.append("=" * 80)
report_output.append("Missing Values Comparison Report")
report_output.append("缺失值情况对比报告")
report_output.append("=" * 80)
report_output.append("")
report_output.append("Overall Statistics / 整体统计:")
report_output.append(f"  Total Common Fields: {len(common_cols)}")
report_output.append(f"  Fields Only in 2021: {len(only_2021_cols)}")
report_output.append(f"  Fields Only in 2025: {len(only_2025_cols)}")
report_output.append("")
report_output.append("Difference Statistics / 差异统计:")
report_output.append(f"  Average Difference: {comparison_df['difference'].mean():.2f}%")
report_output.append(f"  Average Absolute Difference: {comparison_df['abs_difference'].mean():.2f}%")
report_output.append(f"  Max Absolute Difference: {comparison_df['abs_difference'].max():.2f}%")
report_output.append(f"  Fields with Difference > 5%: {(comparison_df['abs_difference'] > 5).sum()}")
report_output.append(f"  Fields with Difference > 10%: {(comparison_df['abs_difference'] > 10).sum()}")
report_output.append("")
report_output.append("Missing Pattern Categories / 缺失模式分类:")
report_output.append(f"  Completely Missing in 2021 (100%): {len(complete_missing_2021)}")
report_output.append(f"  Completely Missing in 2025 (100%): {len(complete_missing_2025)}")
report_output.append(f"  Completely Complete in 2021 (0%): {len(complete_2021)}")
report_output.append(f"  Completely Complete in 2025 (0%): {len(complete_2025)}")
report_output.append(f"  High Missing Rate in 2021 (>50%): {len(high_missing_2021)}")
report_output.append(f"  High Missing Rate in 2025 (>50%): {len(high_missing_2025)}")
report_output.append("")
report_output.append("Top 20 Fields with Largest Differences / 差异最大的20个字段:")
for idx, row in comparison_df.nlargest(20, 'abs_difference').iterrows():
    report_output.append(f"  {row['column']:40s} | 2021: {row['missing_pct_2021']:6.2f}% | 2025: {row['missing_pct_2025']:6.2f}% | Diff: {row['difference']:+7.2f}%")

report_file = charts_dir / 'missing_values_comparison_report.txt'
with open(report_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_output))
print(f"  ✅ 对比报告已保存: {report_file}")

print("\n" + "=" * 80)
print("缺失值对比分析完成！/ Missing Values Comparison Analysis Complete!")
print("=" * 80)
print(f"\n生成的文件 / Generated Files:")
print(f"  1. {comparison_chart_file}")
print(f"  2. {heatmap_file}")
print(f"  3. {comparison_csv_file}")
print(f"  4. {report_file}")

