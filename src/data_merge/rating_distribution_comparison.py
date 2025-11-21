"""
比较 listings_detailed (2021年) 和 listings_detailed_2 (2025年) 的 review_scores_rating 分布
Compare review_scores_rating distribution between listings_detailed (2021) and listings_detailed_2 (2025)
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 添加 EDA 目录到路径，以便导入 utils 模块 / Add EDA directory to path for importing utils module
sys.path.insert(0, str(Path(__file__).parent.parent / "EDA"))
from utils import setup_plotting, get_project_paths

# 设置中文字体支持 / Set Chinese font support
setup_plotting()

# 获取项目根目录路径 / Get project root directory path
project_root, data_dir, charts_eda_dir, charts_model_dir = get_project_paths()
# 创建 data_merge 输出目录 / Create data_merge output directory
charts_dir = project_root / 'charts' / 'data_merge'
os.makedirs(charts_dir, exist_ok=True)

print("=" * 80)
print("Review Scores Rating Distribution Comparison")
print("review_scores_rating 分布比较分析")
print("=" * 80)

# ============================================================================
# 1. 加载数据 / Load Data
# ============================================================================

print("\n1. 加载数据 / Loading Data...")

try:
    # 加载 2021 年数据 / Load 2021 data
    listings_2021 = pd.read_excel(data_dir / 'listings_detailed.xlsx')
    print(f"  ✅ listings_detailed.xlsx (2021): {len(listings_2021)} 行 × {len(listings_2021.columns)} 列")
    
    # 加载 2025 年数据 / Load 2025 data
    listings_2025 = pd.read_excel(data_dir / 'listings_detailed_2.xlsx')
    print(f"  ✅ listings_detailed_2.xlsx (2025): {len(listings_2025)} 行 × {len(listings_2025.columns)} 列")
except Exception as e:
    print(f"  ❌ 数据加载失败: {e}")
    exit(1)

# ============================================================================
# 2. 提取 review_scores_rating 数据 / Extract review_scores_rating Data
# ============================================================================

print("\n2. 提取 review_scores_rating 数据 / Extracting review_scores_rating Data...")

# 检查列是否存在 / Check if column exists
if 'review_scores_rating' not in listings_2021.columns:
    print("  ❌ listings_detailed.xlsx 中未找到 'review_scores_rating' 列")
    print(f"  可用列: {list(listings_2021.columns)[:10]}...")
    exit(1)

if 'review_scores_rating' not in listings_2025.columns:
    print("  ❌ listings_detailed_2.xlsx 中未找到 'review_scores_rating' 列")
    print(f"  可用列: {list(listings_2025.columns)[:10]}...")
    exit(1)

# 提取评分数据 / Extract rating data
rating_2021 = listings_2021['review_scores_rating'].dropna()
rating_2025 = listings_2025['review_scores_rating'].dropna()

print(f"  - 2021年有效评分数量: {len(rating_2021):,} ({len(rating_2021)/len(listings_2021)*100:.1f}%)")
print(f"  - 2025年有效评分数量: {len(rating_2025):,} ({len(rating_2025)/len(listings_2025)*100:.1f}%)")

# ============================================================================
# 3. 统计摘要 / Statistical Summary
# ============================================================================

print("\n3. 统计摘要 / Statistical Summary...")

stats_2021 = rating_2021.describe()
stats_2025 = rating_2025.describe()

print("\n3.1 2021年数据统计 / 2021 Data Statistics:")
print(f"  - 均值 / Mean: {stats_2021['mean']:.2f}")
print(f"  - 中位数 / Median: {stats_2021['50%']:.2f}")
print(f"  - 标准差 / Std: {stats_2021['std']:.2f}")

print("\n3.2 2025年数据统计 / 2025 Data Statistics:")
print(f"  - 均值 / Mean: {stats_2025['mean']:.2f}")
print(f"  - 中位数 / Median: {stats_2025['50%']:.2f}")
print(f"  - 标准差 / Std: {stats_2025['std']:.2f}")

print("\n3.3 变化对比 / Change Comparison:")
print(f"  - 均值变化 / Mean Change: {stats_2025['mean'] - stats_2021['mean']:.2f} ({((stats_2025['mean'] - stats_2021['mean'])/stats_2021['mean']*100):+.1f}%)")
print(f"  - 中位数变化 / Median Change: {stats_2025['50%'] - stats_2021['50%']:.2f} ({((stats_2025['50%'] - stats_2021['50%'])/stats_2021['50%']*100):+.1f}%)")
print(f"  - 标准差变化 / Std Change: {stats_2025['std'] - stats_2021['std']:.2f} ({((stats_2025['std'] - stats_2021['std'])/stats_2021['std']*100):+.1f}%)")

# 计算前30%和前25%的评分（70%和75%分位数）
print("\n3.4 分位数分析 / Quantile Analysis:")
percentile_70_2021 = rating_2021.quantile(0.70)
percentile_75_2021 = rating_2021.quantile(0.75)
percentile_70_2025 = rating_2025.quantile(0.70)
percentile_75_2025 = rating_2025.quantile(0.75)

print(f"\n  前30%评分阈值（70%分位数）/ Top 30% Rating Threshold (70th percentile):")
print(f"    - 2021年: {percentile_70_2021:.2f}")
print(f"    - 2025年: {percentile_70_2025:.2f}")
print(f"    - 变化: {percentile_70_2025 - percentile_70_2021:.2f} ({((percentile_70_2025 - percentile_70_2021)/percentile_70_2021*100):+.1f}%)")

print(f"\n  前25%评分阈值（75%分位数）/ Top 25% Rating Threshold (75th percentile):")
print(f"    - 2021年: {percentile_75_2021:.2f}")
print(f"    - 2025年: {percentile_75_2025:.2f}")
print(f"    - 变化: {percentile_75_2025 - percentile_75_2021:.2f} ({((percentile_75_2025 - percentile_75_2021)/percentile_75_2021*100):+.1f}%)")

# 计算4-5分区间每个0.2区间的累计分布
print("\n3.5 4-5分区间累计分布分析 / 4-5 Rating Range Cumulative Distribution Analysis:")
print("  计算每个0.2区间的累计百分比 / Calculate cumulative percentage for each 0.2 interval")

# 检查评分系统范围
rating_max_check = max(rating_2021.max(), rating_2025.max())

# 定义4-5分的0.2区间
if rating_max_check <= 5:
    intervals = [(4.0, 4.2), (4.2, 4.4), (4.4, 4.6), (4.6, 4.8), (4.8, 5.0)]
    interval_labels = ['4.0-4.2', '4.2-4.4', '4.4-4.6', '4.6-4.8', '4.8-5.0']
else:
    intervals = [(80, 84), (84, 88), (88, 92), (92, 96), (96, 100)]
    interval_labels = ['80-84', '84-88', '88-92', '92-96', '96-100']

cumulative_2021 = []
cumulative_2025 = []
interval_counts_2021 = []
interval_counts_2025 = []

for i, (low, high) in enumerate(intervals):
    # 计算当前区间的数量
    count_2021 = len(rating_2021[(rating_2021 >= low) & (rating_2021 < high)])
    count_2025 = len(rating_2025[(rating_2025 >= low) & (rating_2025 < high)])
    
    # 计算累计到当前区间的数量（包括当前区间）
    cumulative_count_2021 = len(rating_2021[rating_2021 >= low])
    cumulative_count_2025 = len(rating_2025[rating_2025 >= low])
    
    # 转换为百分比
    cumulative_pct_2021 = cumulative_count_2021 / len(rating_2021) * 100
    cumulative_pct_2025 = cumulative_count_2025 / len(rating_2025) * 100
    
    interval_pct_2021 = count_2021 / len(rating_2021) * 100
    interval_pct_2025 = count_2025 / len(rating_2025) * 100
    
    cumulative_2021.append(cumulative_pct_2021)
    cumulative_2025.append(cumulative_pct_2025)
    interval_counts_2021.append(interval_pct_2021)
    interval_counts_2025.append(interval_pct_2025)
    
    print(f"\n  {interval_labels[i]} 区间 / {interval_labels[i]} Interval:")
    print(f"    - 2021年: 区间占比 {interval_pct_2021:.2f}%, 累计占比 {cumulative_pct_2021:.2f}% (累计数量: {cumulative_count_2021:,})")
    print(f"    - 2025年: 区间占比 {interval_pct_2025:.2f}%, 累计占比 {cumulative_pct_2025:.2f}% (累计数量: {cumulative_count_2025:,})")
    print(f"    - 累计占比变化: {cumulative_pct_2025 - cumulative_pct_2021:.2f}%")

# 保存累计分布数据到CSV
cumulative_df = pd.DataFrame({
    'Interval': interval_labels,
    '2021_Interval_Pct': interval_counts_2021,
    '2021_Cumulative_Pct': cumulative_2021,
    '2025_Interval_Pct': interval_counts_2025,
    '2025_Cumulative_Pct': cumulative_2025,
    'Cumulative_Change': [c5 - c1 for c1, c5 in zip(cumulative_2021, cumulative_2025)]
})

cumulative_file = charts_dir / 'rating_4_5_cumulative_distribution.csv'
cumulative_df.to_csv(cumulative_file, index=False, encoding='utf-8-sig')
print(f"\n  ✅ 累计分布数据已保存到: {cumulative_file}")

# ============================================================================
# 4. 可视化：评分分布面积图对比 / Visualization: Rating Distribution Area Chart Comparison
# ============================================================================

print("\n4. 生成可视化图表 / Generating Visualization...")

# 设置图表样式 / Set plot style
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    try:
        plt.style.use('seaborn-whitegrid')
    except OSError:
        sns.set_style("whitegrid")

# 使用原始评分值，不进行转换 / Use original rating values without conversion
rating_max = max(rating_2021.max(), rating_2025.max())
rating_min = min(rating_2021.min(), rating_2025.min())

# 创建更细粒度的bins用于平滑的面积图
# Create finer bins for smooth area chart
if rating_max <= 5:
    # 0-5评分系统，使用0.1的间隔
    bins_full = np.arange(0, 5.1, 0.1)
    # 4-5分区间使用更细的间隔（0.05）用于放大视图
    bins_zoom = np.arange(4.0, 5.05, 0.05)
else:
    # 0-100评分系统，使用1的间隔
    bins_full = np.arange(0, 101, 1)
    # 80-100分区间使用更细的间隔用于放大视图
    bins_zoom = np.arange(80, 101, 0.5)

# 计算全范围的直方图数据 / Calculate histogram data for full range
counts_2021_full, bin_edges_full = np.histogram(rating_2021, bins=bins_full)
counts_2025_full, _ = np.histogram(rating_2025, bins=bins_full)

# 计算4-5分区间的直方图数据 / Calculate histogram data for 4-5 range
if rating_max <= 5:
    rating_2021_zoom = rating_2021[(rating_2021 >= 4.0) & (rating_2021 <= 5.0)]
    rating_2025_zoom = rating_2025[(rating_2025 >= 4.0) & (rating_2025 <= 5.0)]
else:
    rating_2021_zoom = rating_2021[(rating_2021 >= 80) & (rating_2021 <= 100)]
    rating_2025_zoom = rating_2025[(rating_2025 >= 80) & (rating_2025 <= 100)]

counts_2021_zoom, bin_edges_zoom = np.histogram(rating_2021_zoom, bins=bins_zoom)
counts_2025_zoom, _ = np.histogram(rating_2025_zoom, bins=bins_zoom)

# 转换为密度（百分比） / Convert to density (percentage)
density_2021_full = counts_2021_full / len(rating_2021) * 100
density_2025_full = counts_2025_full / len(rating_2025) * 100

if len(rating_2021_zoom) > 0:
    density_2021_zoom = counts_2021_zoom / len(rating_2021) * 100
else:
    density_2021_zoom = np.zeros_like(counts_2021_zoom)

if len(rating_2025_zoom) > 0:
    density_2025_zoom = counts_2025_zoom / len(rating_2025) * 100
else:
    density_2025_zoom = np.zeros_like(counts_2025_zoom)

# 使用bin的中心点作为x轴 / Use bin centers as x-axis
bin_centers_full = (bin_edges_full[:-1] + bin_edges_full[1:]) / 2
bin_centers_zoom = (bin_edges_zoom[:-1] + bin_edges_zoom[1:]) / 2

# 创建包含两个子图的图表 / Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# 使用橙色和蓝色配色方案 / Use orange and blue color scheme
color_2021 = '#3498db'  # 蓝色 / Blue
color_2021_edge = '#2980b9'  # 深蓝色边框 / Dark blue edge
color_2025 = '#ff8c42'  # 橙色 / Orange
color_2025_edge = '#e67e22'  # 深橙色边框 / Dark orange edge

# ===== 第一个子图：全范围分布 / First subplot: Full range distribution =====
ax1.fill_between(bin_centers_full, density_2021_full, alpha=0.7, color=color_2021, label='2021', edgecolor=color_2021_edge, linewidth=1.8)
ax1.fill_between(bin_centers_full, density_2025_full, alpha=0.7, color=color_2025, label='2025', edgecolor=color_2025_edge, linewidth=1.8)

# 添加均值线 / Add mean lines
ax1.axvline(stats_2021['mean'], color=color_2021_edge, linestyle='--', linewidth=2.5, alpha=0.9, label=f'2021 Mean: {stats_2021["mean"]:.2f}')
ax1.axvline(stats_2025['mean'], color=color_2025_edge, linestyle='--', linewidth=2.5, alpha=0.9, label=f'2025 Mean: {stats_2025["mean"]:.2f}')

# 标记4-5分区间 / Mark 4-5 range
if rating_max <= 5:
    ax1.axvspan(4.0, 5.0, alpha=0.15, color='#ffd700', label='Zoomed Range (4-5)')
else:
    ax1.axvspan(80, 100, alpha=0.15, color='#ffd700', label='Zoomed Range (80-100)')

ax1.set_xlabel('Review Scores Rating', fontsize=12)
ax1.set_ylabel('Density (%)', fontsize=12)
ax1.set_title('Full Range Distribution: 2021 vs 2025', fontsize=13, fontweight='bold')
ax1.legend(loc='best', fontsize=9)
ax1.grid(True, alpha=0.3)

# ===== 第二个子图：4-5分区间放大视图 / Second subplot: Zoomed 4-5 range =====
if len(rating_2021_zoom) > 0 or len(rating_2025_zoom) > 0:
    ax2.fill_between(bin_centers_zoom, density_2021_zoom, alpha=0.7, color=color_2021, label='2021', edgecolor=color_2021_edge, linewidth=1.8)
    ax2.fill_between(bin_centers_zoom, density_2025_zoom, alpha=0.7, color=color_2025, label='2025', edgecolor=color_2025_edge, linewidth=1.8)
    
    # 添加均值线 / Add mean lines
    ax2.axvline(stats_2021['mean'], color=color_2021_edge, linestyle='--', linewidth=2.5, alpha=0.9, label=f'2021 Mean: {stats_2021["mean"]:.2f}')
    ax2.axvline(stats_2025['mean'], color=color_2025_edge, linestyle='--', linewidth=2.5, alpha=0.9, label=f'2025 Mean: {stats_2025["mean"]:.2f}')
    
    if rating_max <= 5:
        ax2.set_xlim(4.0, 5.0)
        ax2.set_xlabel('Review Scores Rating (Zoomed: 4.0-5.0)', fontsize=12)
    else:
        ax2.set_xlim(80, 100)
        ax2.set_xlabel('Review Scores Rating (Zoomed: 80-100)', fontsize=12)
    
    ax2.set_ylabel('Density (%)', fontsize=12)
    ax2.set_title('Zoomed View: High Rating Range (4-5)', fontsize=13, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)

plt.tight_layout()
area_chart_file = charts_dir / 'rating_distribution_comparison.png'
plt.savefig(area_chart_file, dpi=300, bbox_inches='tight')
print(f"  ✅ 面积图已保存到: {area_chart_file}")
plt.close()

# ============================================================================
# 5. 统计量对比表格 / Statistics Comparison Table
# ============================================================================

print("\n5. 生成统计量对比表格 / Generating Statistics Comparison Table...")

# 创建统计量对比表格
stats_comparison = pd.DataFrame({
    '2021': [
        f"{stats_2021['mean']:.2f}",
        f"{stats_2021['50%']:.2f}",
        f"{stats_2021['std']:.2f}",
        f"{stats_2021['min']:.2f}",
        f"{stats_2021['max']:.2f}"
    ],
    '2025': [
        f"{stats_2025['mean']:.2f}",
        f"{stats_2025['50%']:.2f}",
        f"{stats_2025['std']:.2f}",
        f"{stats_2025['min']:.2f}",
        f"{stats_2025['max']:.2f}"
    ],
    'Change': [
        f"{stats_2025['mean'] - stats_2021['mean']:.2f} ({((stats_2025['mean'] - stats_2021['mean'])/stats_2021['mean']*100):+.1f}%)",
        f"{stats_2025['50%'] - stats_2021['50%']:.2f} ({((stats_2025['50%'] - stats_2021['50%'])/stats_2021['50%']*100):+.1f}%)",
        f"{stats_2025['std'] - stats_2021['std']:.2f} ({((stats_2025['std'] - stats_2021['std'])/stats_2021['std']*100):+.1f}%)",
        f"{stats_2025['min'] - stats_2021['min']:.2f}",
        f"{stats_2025['max'] - stats_2021['max']:.2f}"
    ]
}, index=['Mean', 'Median', 'Std', 'Min', 'Max'])

# 保存为CSV
stats_table_file = charts_dir / 'rating_statistics_comparison.csv'
stats_comparison.to_csv(stats_table_file, encoding='utf-8-sig')
print(f"  ✅ 统计量对比表格已保存到: {stats_table_file}")

# 创建可视化表格
fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('tight')
ax.axis('off')

table = ax.table(cellText=stats_comparison.values,
                 rowLabels=stats_comparison.index,
                 colLabels=stats_comparison.columns,
                 cellLoc='center',
                 loc='center',
                 bbox=[0, 0, 1, 1])

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2)

# 设置表格样式
# 遍历所有单元格设置样式
for i in range(len(stats_comparison.index) + 1):  # +1 for header row
    for j in range(len(stats_comparison.columns) + 1):  # +1 for row labels column
        try:
            cell = table[(i, j)]
            if i == 0:  # Header row
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(weight='bold', color='white')
            elif j == 0:  # Row labels column
                cell.set_facecolor('#E8F5E9')
                cell.set_text_props(weight='bold')
        except KeyError:
            pass  # Skip if cell doesn't exist

ax.set_title('Review Scores Rating Statistics Comparison', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
stats_table_chart_file = charts_dir / 'rating_statistics_comparison_table.png'
plt.savefig(stats_table_chart_file, dpi=300, bbox_inches='tight')
print(f"  ✅ 统计量对比表格图表已保存到: {stats_table_chart_file}")
plt.close()

# 保存统计信息到文本文件
stats_output = []
stats_output.append("=" * 80)
stats_output.append("Review Scores Rating Distribution Comparison Statistics")
stats_output.append("review_scores_rating 分布比较统计信息")
stats_output.append("=" * 80)
stats_output.append("")
stats_output.append("2021 Data (listings_detailed.xlsx):")
stats_output.append(f"  Total Records: {len(listings_2021):,}")
stats_output.append(f"  Valid Ratings: {len(rating_2021):,} ({len(rating_2021)/len(listings_2021)*100:.1f}%)")
stats_output.append(f"  Mean: {stats_2021['mean']:.2f}")
stats_output.append(f"  Median: {stats_2021['50%']:.2f}")
stats_output.append(f"  Std: {stats_2021['std']:.2f}")
stats_output.append(f"  Min: {stats_2021['min']:.2f}")
stats_output.append(f"  Max: {stats_2021['max']:.2f}")
stats_output.append("")
stats_output.append("2025 Data (listings_detailed_2.xlsx):")
stats_output.append(f"  Total Records: {len(listings_2025):,}")
stats_output.append(f"  Valid Ratings: {len(rating_2025):,} ({len(rating_2025)/len(listings_2025)*100:.1f}%)")
stats_output.append(f"  Mean: {stats_2025['mean']:.2f}")
stats_output.append(f"  Median: {stats_2025['50%']:.2f}")
stats_output.append(f"  Std: {stats_2025['std']:.2f}")
stats_output.append(f"  Min: {stats_2025['min']:.2f}")
stats_output.append(f"  Max: {stats_2025['max']:.2f}")
stats_output.append("")
stats_output.append("Change Comparison:")
stats_output.append(f"  Mean Change: {stats_2025['mean'] - stats_2021['mean']:.2f} ({((stats_2025['mean'] - stats_2021['mean'])/stats_2021['mean']*100):+.1f}%)")
stats_output.append(f"  Median Change: {stats_2025['50%'] - stats_2021['50%']:.2f} ({((stats_2025['50%'] - stats_2021['50%'])/stats_2021['50%']*100):+.1f}%)")
stats_output.append(f"  Std Change: {stats_2025['std'] - stats_2021['std']:.2f} ({((stats_2025['std'] - stats_2021['std'])/stats_2021['std']*100):+.1f}%)")
stats_output.append("")
stats_output.append("Quantile Analysis:")
stats_output.append(f"  Top 30% Threshold (70th percentile):")
stats_output.append(f"    2021: {percentile_70_2021:.2f}")
stats_output.append(f"    2025: {percentile_70_2025:.2f}")
stats_output.append(f"    Change: {percentile_70_2025 - percentile_70_2021:.2f} ({((percentile_70_2025 - percentile_70_2021)/percentile_70_2021*100):+.1f}%)")
stats_output.append(f"  Top 25% Threshold (75th percentile):")
stats_output.append(f"    2021: {percentile_75_2021:.2f}")
stats_output.append(f"    2025: {percentile_75_2025:.2f}")
stats_output.append(f"    Change: {percentile_75_2025 - percentile_75_2021:.2f} ({((percentile_75_2025 - percentile_75_2021)/percentile_75_2021*100):+.1f}%)")
stats_output.append("")
stats_output.append("4-5 Rating Range Cumulative Distribution (0.2 intervals):")
for i, label in enumerate(interval_labels):
    stats_output.append(f"  {label}:")
    stats_output.append(f"    2021 - Interval: {interval_counts_2021[i]:.2f}%, Cumulative: {cumulative_2021[i]:.2f}%")
    stats_output.append(f"    2025 - Interval: {interval_counts_2025[i]:.2f}%, Cumulative: {cumulative_2025[i]:.2f}%")
    stats_output.append(f"    Cumulative Change: {cumulative_2025[i] - cumulative_2021[i]:.2f}%")

stats_file = charts_dir / 'rating_distribution_comparison_statistics.txt'
with open(stats_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(stats_output))
print(f"  ✅ 统计信息已保存到: {stats_file}")

print("\n" + "=" * 80)
print("分析完成！/ Analysis Complete!")
print("=" * 80)
print(f"\n生成的文件 / Generated Files:")
print(f"  1. {area_chart_file}")
print(f"  2. {stats_table_file}")
print(f"  3. {stats_table_chart_file}")
print(f"  4. {stats_file}")
