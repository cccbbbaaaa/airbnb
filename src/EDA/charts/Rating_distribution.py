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
from scipy import stats

# 添加父目录到路径，以便导入 utils 模块 / Add parent directory to path for importing utils module
sys.path.append(str(Path(__file__).parent.parent))
from utils import setup_plotting, get_project_paths

# 设置中文字体支持 / Set Chinese font support
setup_plotting()

# 获取项目根目录路径 / Get project root directory path
project_root, data_dir, charts_eda_dir, charts_model_dir = get_project_paths()
charts_dir = charts_eda_dir  # 使用 EDA 目录 / Use EDA directory

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
print(f"  - 最小值 / Min: {stats_2021['min']:.2f}")
print(f"  - 最大值 / Max: {stats_2021['max']:.2f}")
print(f"  - 25%分位数 / Q1: {stats_2021['25%']:.2f}")
print(f"  - 75%分位数 / Q3: {stats_2021['75%']:.2f}")

print("\n3.2 2025年数据统计 / 2025 Data Statistics:")
print(f"  - 均值 / Mean: {stats_2025['mean']:.2f}")
print(f"  - 中位数 / Median: {stats_2025['50%']:.2f}")
print(f"  - 标准差 / Std: {stats_2025['std']:.2f}")
print(f"  - 最小值 / Min: {stats_2025['min']:.2f}")
print(f"  - 最大值 / Max: {stats_2025['max']:.2f}")
print(f"  - 25%分位数 / Q1: {stats_2025['25%']:.2f}")
print(f"  - 75%分位数 / Q3: {stats_2025['75%']:.2f}")

print("\n3.3 变化对比 / Change Comparison:")
print(f"  - 均值变化 / Mean Change: {stats_2025['mean'] - stats_2021['mean']:.2f} ({((stats_2025['mean'] - stats_2021['mean'])/stats_2021['mean']*100):+.1f}%)")
print(f"  - 中位数变化 / Median Change: {stats_2025['50%'] - stats_2021['50%']:.2f} ({((stats_2025['50%'] - stats_2021['50%'])/stats_2021['50%']*100):+.1f}%)")
print(f"  - 标准差变化 / Std Change: {stats_2025['std'] - stats_2021['std']:.2f} ({((stats_2025['std'] - stats_2021['std'])/stats_2021['std']*100):+.1f}%)")

# 保存统计信息到文件 / Save statistics to file
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
stats_output.append(f"  Q1: {stats_2021['25%']:.2f}")
stats_output.append(f"  Q3: {stats_2021['75%']:.2f}")
stats_output.append("")
stats_output.append("2025 Data (listings_detailed_2.xlsx):")
stats_output.append(f"  Total Records: {len(listings_2025):,}")
stats_output.append(f"  Valid Ratings: {len(rating_2025):,} ({len(rating_2025)/len(listings_2025)*100:.1f}%)")
stats_output.append(f"  Mean: {stats_2025['mean']:.2f}")
stats_output.append(f"  Median: {stats_2025['50%']:.2f}")
stats_output.append(f"  Std: {stats_2025['std']:.2f}")
stats_output.append(f"  Min: {stats_2025['min']:.2f}")
stats_output.append(f"  Max: {stats_2025['max']:.2f}")
stats_output.append(f"  Q1: {stats_2025['25%']:.2f}")
stats_output.append(f"  Q3: {stats_2025['75%']:.2f}")
stats_output.append("")
stats_output.append("Change Comparison:")
stats_output.append(f"  Mean Change: {stats_2025['mean'] - stats_2021['mean']:.2f} ({((stats_2025['mean'] - stats_2021['mean'])/stats_2021['mean']*100):+.1f}%)")
stats_output.append(f"  Median Change: {stats_2025['50%'] - stats_2021['50%']:.2f} ({((stats_2025['50%'] - stats_2021['50%'])/stats_2021['50%']*100):+.1f}%)")
stats_output.append(f"  Std Change: {stats_2025['std'] - stats_2021['std']:.2f} ({((stats_2025['std'] - stats_2021['std'])/stats_2021['std']*100):+.1f}%)")

# ============================================================================
# 3.4 统计显著性检验 / Statistical Significance Tests
# ============================================================================

print("\n3.4 统计显著性检验 / Statistical Significance Tests...")

# 3.4.1 Mann-Whitney U 检验（非参数检验，适用于两个独立样本）
# Mann-Whitney U Test (non-parametric test for two independent samples)
print("\n3.4.1 Mann-Whitney U Test (Non-parametric):")
mw_statistic, mw_pvalue = stats.mannwhitneyu(rating_2021, rating_2025, alternative='two-sided')
print(f"  - Statistic: {mw_statistic:.2f}")
print(f"  - P-value: {mw_pvalue:.6f}")
mw_significant = mw_pvalue < 0.05
print(f"  - Result: {'Significant difference (p < 0.05)' if mw_significant else 'No significant difference (p >= 0.05)'}")

# 3.4.2 Kolmogorov-Smirnov 检验（检验两个分布是否相同）
# Kolmogorov-Smirnov Test (tests if two distributions are identical)
print("\n3.4.2 Kolmogorov-Smirnov Test:")
ks_statistic, ks_pvalue = stats.ks_2samp(rating_2021, rating_2025)
print(f"  - Statistic: {ks_statistic:.6f}")
print(f"  - P-value: {ks_pvalue:.6f}")
ks_significant = ks_pvalue < 0.05
print(f"  - Result: {'Distributions are different (p < 0.05)' if ks_significant else 'Distributions are similar (p >= 0.05)'}")

# 3.4.3 独立样本 t 检验（参数检验，假设数据近似正态）
# Independent Samples t-test (parametric test, assumes approximate normality)
print("\n3.4.3 Independent Samples t-test:")
t_statistic, t_pvalue = stats.ttest_ind(rating_2021, rating_2025)
print(f"  - Statistic: {t_statistic:.4f}")
print(f"  - P-value: {t_pvalue:.6f}")
t_significant = t_pvalue < 0.05
print(f"  - Result: {'Significant difference in means (p < 0.05)' if t_significant else 'No significant difference in means (p >= 0.05)'}")

# 3.4.4 效应量计算（Cohen's d）
# Effect Size Calculation (Cohen's d)
print("\n3.4.4 Effect Size (Cohen's d):")
n1, n2 = len(rating_2021), len(rating_2025)
s1, s2 = rating_2021.std(ddof=1), rating_2025.std(ddof=1)
pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
cohens_d = (stats_2025['mean'] - stats_2021['mean']) / pooled_std
print(f"  - Cohen's d: {cohens_d:.4f}")
if abs(cohens_d) < 0.2:
    effect_size = "negligible"
elif abs(cohens_d) < 0.5:
    effect_size = "small"
elif abs(cohens_d) < 0.8:
    effect_size = "medium"
else:
    effect_size = "large"
print(f"  - Effect Size: {effect_size}")

# 3.4.5 综合结论
# Overall Conclusion
print("\n3.4.5 Overall Conclusion:")
all_tests_significant = mw_significant or ks_significant or t_significant
if all_tests_significant:
    conclusion = "There is a SIGNIFICANT difference between the two distributions."
    conclusion_cn = "两个数据集的评分分布存在显著差异。"
else:
    conclusion = "There is NO significant difference between the two distributions."
    conclusion_cn = "两个数据集的评分分布不存在显著差异。"

print(f"  - {conclusion}")
print(f"  - {conclusion_cn}")

# 更新统计输出文件
stats_output.append("")
stats_output.append("=" * 80)
stats_output.append("Statistical Significance Tests")
stats_output.append("统计显著性检验")
stats_output.append("=" * 80)
stats_output.append("")
stats_output.append("1. Mann-Whitney U Test (Non-parametric):")
stats_output.append(f"   Statistic: {mw_statistic:.2f}")
stats_output.append(f"   P-value: {mw_pvalue:.6f}")
stats_output.append(f"   Result: {'Significant difference (p < 0.05)' if mw_significant else 'No significant difference (p >= 0.05)'}")
stats_output.append("")
stats_output.append("2. Kolmogorov-Smirnov Test:")
stats_output.append(f"   Statistic: {ks_statistic:.6f}")
stats_output.append(f"   P-value: {ks_pvalue:.6f}")
stats_output.append(f"   Result: {'Distributions are different (p < 0.05)' if ks_significant else 'Distributions are similar (p >= 0.05)'}")
stats_output.append("")
stats_output.append("3. Independent Samples t-test:")
stats_output.append(f"   Statistic: {t_statistic:.4f}")
stats_output.append(f"   P-value: {t_pvalue:.6f}")
stats_output.append(f"   Result: {'Significant difference in means (p < 0.05)' if t_significant else 'No significant difference in means (p >= 0.05)'}")
stats_output.append("")
stats_output.append("4. Effect Size (Cohen's d):")
stats_output.append(f"   Cohen's d: {cohens_d:.4f}")
stats_output.append(f"   Effect Size: {effect_size}")
stats_output.append("")
stats_output.append("=" * 80)
stats_output.append("Overall Conclusion")
stats_output.append("综合结论")
stats_output.append("=" * 80)
stats_output.append(conclusion)
stats_output.append(conclusion_cn)

stats_file = charts_dir / 'rating_distribution_comparison_statistics.txt'
with open(stats_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(stats_output))
print(f"\n  ✅ 统计信息已保存到: {stats_file}")

# ============================================================================
# 3.5 数据调整方法 / Data Adjustment Methods
# ============================================================================

print("\n3.5 数据调整方法 / Data Adjustment Methods...")
print("  将2021年数据调整为与2025年分布相似 / Adjusting 2021 data to match 2025 distribution")

# 方法1: 分位数匹配 (Quantile Matching) - 最精确的方法
# Method 1: Quantile Matching - Most accurate method
print("\n3.5.1 方法1: 分位数匹配 / Method 1: Quantile Matching")

def quantile_matching(source_data, target_data):
    """
    分位数匹配：将源数据的分位数映射到目标数据的分位数
    Quantile Matching: Map quantiles of source data to target data quantiles
    """
    # 计算源数据和目标数据的排序索引
    source_sorted_idx = np.argsort(source_data)
    source_sorted = np.sort(source_data)
    target_sorted = np.sort(target_data)
    
    # 创建分位数映射
    n_source = len(source_data)
    n_target = len(target_data)
    
    # 为源数据的每个值找到对应的目标分位数
    adjusted = np.zeros_like(source_data)
    for i, val in enumerate(source_data):
        # 找到源数据中的分位数位置
        quantile = np.searchsorted(source_sorted, val, side='right') / n_source
        # 映射到目标数据的对应分位数值
        target_idx = int(quantile * (n_target - 1))
        target_idx = min(target_idx, n_target - 1)
        adjusted[i] = target_sorted[target_idx]
    
    return adjusted

rating_2021_adjusted_quantile = quantile_matching(rating_2021.values, rating_2025.values)
stats_adjusted_quantile = pd.Series(rating_2021_adjusted_quantile).describe()

print(f"  - 调整后均值: {stats_adjusted_quantile['mean']:.2f} (目标: {stats_2025['mean']:.2f})")
print(f"  - 调整后中位数: {stats_adjusted_quantile['50%']:.2f} (目标: {stats_2025['50%']:.2f})")
print(f"  - 调整后标准差: {stats_adjusted_quantile['std']:.2f} (目标: {stats_2025['std']:.2f})")

# 方法2: 线性变换 (匹配均值和标准差)
# Method 2: Linear Transformation (match mean and std)
print("\n3.5.2 方法2: 线性变换 / Method 2: Linear Transformation")

def linear_transform(source_data, target_mean, target_std):
    """
    线性变换：调整均值和标准差
    Linear Transformation: Adjust mean and standard deviation
    """
    source_mean = source_data.mean()
    source_std = source_data.std()
    
    # 标准化后重新缩放
    standardized = (source_data - source_mean) / source_std
    adjusted = standardized * target_std + target_mean
    
    return adjusted

rating_2021_adjusted_linear = linear_transform(rating_2021.values, stats_2025['mean'], stats_2025['std'])
# 确保值在合理范围内
rating_2021_adjusted_linear = np.clip(rating_2021_adjusted_linear, rating_2025.min(), rating_2025.max())
stats_adjusted_linear = pd.Series(rating_2021_adjusted_linear).describe()

print(f"  - 调整后均值: {stats_adjusted_linear['mean']:.2f} (目标: {stats_2025['mean']:.2f})")
print(f"  - 调整后中位数: {stats_adjusted_linear['50%']:.2f} (目标: {stats_2025['50%']:.2f})")
print(f"  - 调整后标准差: {stats_adjusted_linear['std']:.2f} (目标: {stats_2025['std']:.2f})")

# 方法3: CDF映射 (使用累积分布函数)
# Method 3: CDF Mapping (using cumulative distribution function)
print("\n3.5.3 方法3: CDF映射 / Method 3: CDF Mapping")

def cdf_mapping(source_data, target_data):
    """
    CDF映射：通过累积分布函数进行映射
    CDF Mapping: Map through cumulative distribution function
    """
    # 计算源数据和目标数据的CDF
    source_sorted = np.sort(source_data)
    target_sorted = np.sort(target_data)
    
    # 为源数据的每个值找到对应的目标值
    adjusted = np.zeros_like(source_data)
    for i, val in enumerate(source_data):
        # 找到源数据中的CDF值
        cdf_value = np.searchsorted(source_sorted, val, side='right') / len(source_sorted)
        # 映射到目标数据的对应CDF值
        target_idx = int(cdf_value * (len(target_sorted) - 1))
        target_idx = min(target_idx, len(target_sorted) - 1)
        adjusted[i] = target_sorted[target_idx]
    
    return adjusted

rating_2021_adjusted_cdf = cdf_mapping(rating_2021.values, rating_2025.values)
stats_adjusted_cdf = pd.Series(rating_2021_adjusted_cdf).describe()

print(f"  - 调整后均值: {stats_adjusted_cdf['mean']:.2f} (目标: {stats_2025['mean']:.2f})")
print(f"  - 调整后中位数: {stats_adjusted_cdf['50%']:.2f} (目标: {stats_2025['50%']:.2f})")
print(f"  - 调整后标准差: {stats_adjusted_cdf['std']:.2f} (目标: {stats_2025['std']:.2f})")

# 评估调整效果
print("\n3.5.4 调整效果评估 / Adjustment Effectiveness Evaluation:")

# 计算调整后的KS统计量
ks_quantile = stats.ks_2samp(rating_2021_adjusted_quantile, rating_2025.values)[0]
ks_linear = stats.ks_2samp(rating_2021_adjusted_linear, rating_2025.values)[0]
ks_cdf = stats.ks_2samp(rating_2021_adjusted_cdf, rating_2025.values)[0]

print(f"  - 分位数匹配 KS统计量: {ks_quantile:.6f} (越小越好)")
print(f"  - 线性变换 KS统计量: {ks_linear:.6f} (越小越好)")
print(f"  - CDF映射 KS统计量: {ks_cdf:.6f} (越小越好)")

# 选择最佳方法
best_method = min([('quantile', ks_quantile), ('linear', ks_linear), ('cdf', ks_cdf)], key=lambda x: x[1])
print(f"\n  ✅ 最佳调整方法: {best_method[0]} (KS统计量: {best_method[1]:.6f})")

# 保存调整后的数据（使用最佳方法）
if best_method[0] == 'quantile':
    rating_2021_adjusted = rating_2021_adjusted_quantile
    method_name = "Quantile Matching"
elif best_method[0] == 'linear':
    rating_2021_adjusted = rating_2021_adjusted_linear
    method_name = "Linear Transformation"
else:
    rating_2021_adjusted = rating_2021_adjusted_cdf
    method_name = "CDF Mapping"

# 更新统计输出
stats_output.append("")
stats_output.append("=" * 80)
stats_output.append("Data Adjustment Methods")
stats_output.append("数据调整方法")
stats_output.append("=" * 80)
stats_output.append("")
stats_output.append(f"Best Method: {method_name}")
stats_output.append("")
stats_output.append("Adjusted 2021 Data Statistics (Best Method):")
stats_adjusted_best = pd.Series(rating_2021_adjusted).describe()
stats_output.append(f"  Mean: {stats_adjusted_best['mean']:.2f} (Target: {stats_2025['mean']:.2f})")
stats_output.append(f"  Median: {stats_adjusted_best['50%']:.2f} (Target: {stats_2025['50%']:.2f})")
stats_output.append(f"  Std: {stats_adjusted_best['std']:.2f} (Target: {stats_2025['std']:.2f})")
stats_output.append(f"  KS Statistic: {best_method[1]:.6f}")

# 保存调整后的数据到CSV
# 注意：由于dropna()后长度可能不同，我们需要处理长度不一致的情况
# Note: Lengths may differ after dropna(), so we need to handle length mismatch
len_2021 = len(rating_2021)
len_2025 = len(rating_2025)
len_adjusted = len(rating_2021_adjusted)

print(f"\n  数据长度检查 / Data Length Check:")
print(f"    - 2021年原始数据: {len_2021:,}")
print(f"    - 2021年调整后数据: {len_adjusted:,}")
print(f"    - 2025年目标数据: {len_2025:,}")

# 创建DataFrame，只保存相同长度的数据
# Create DataFrame, only save data with same length
if len_2021 == len_adjusted:
    # 如果长度相同，直接创建DataFrame
    # If lengths are same, create DataFrame directly
    adjusted_data = pd.DataFrame({
        'original_2021': rating_2021.values,
        'adjusted_2021': rating_2021_adjusted
    })
    # 如果2025年数据长度也相同，添加目标列
    # If 2025 data length is also same, add target column
    if len_2025 == len_2021:
        adjusted_data['target_2025'] = rating_2025.values
    else:
        # 如果长度不同，只保存前min(len_2021, len_2025)个值用于对比
        # If lengths differ, only save first min(len_2021, len_2025) values for comparison
        min_len = min(len_2021, len_2025)
        adjusted_data = adjusted_data.iloc[:min_len].copy()
        adjusted_data['target_2025'] = rating_2025.values[:min_len]
else:
    # 如果长度不同，取最小长度创建DataFrame
    # If lengths differ, use minimum length to create DataFrame
    min_len = min(len_2021, len_adjusted, len_2025)
    adjusted_data = pd.DataFrame({
        'original_2021': rating_2021.values[:min_len],
        'adjusted_2021': rating_2021_adjusted[:min_len],
        'target_2025': rating_2025.values[:min_len]
    })

adjusted_data_file = charts_dir / 'rating_2021_adjusted_data.csv'
adjusted_data.to_csv(adjusted_data_file, index=False)
print(f"\n  ✅ 调整后的数据已保存到: {adjusted_data_file} (共 {len(adjusted_data):,} 行)")

# ============================================================================
# 4. 可视化分析 / Visualization Analysis
# ============================================================================

print("\n4. 生成可视化图表 / Generating Visualizations...")

# 设置图表样式 / Set plot style
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    try:
        plt.style.use('seaborn-whitegrid')
    except OSError:
        sns.set_style("whitegrid")
fig_size = (16, 10)

# 4.1 直方图对比 / Histogram Comparison
print("  4.1 生成直方图对比 / Generating Histogram Comparison...")
fig, axes = plt.subplots(2, 2, figsize=fig_size)
fig.suptitle('Review Scores Rating Distribution Comparison', 
             fontsize=16, fontweight='bold', y=0.995)

# 子图1: 2021年直方图 / Subplot 1: 2021 Histogram
axes[0, 0].hist(rating_2021, bins=50, alpha=0.7, color='#3498db', edgecolor='black', linewidth=0.5)
axes[0, 0].axvline(stats_2021['mean'], color='red', linestyle='--', linewidth=2, label=f'Mean: {stats_2021["mean"]:.2f}')
axes[0, 0].axvline(stats_2021['50%'], color='green', linestyle='--', linewidth=2, label=f'Median: {stats_2021["50%"]:.2f}')
axes[0, 0].set_xlabel('Review Scores Rating', fontsize=12)
axes[0, 0].set_ylabel('Frequency', fontsize=12)
axes[0, 0].set_title('2021 Data Distribution', fontsize=14, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 子图2: 2025年直方图 / Subplot 2: 2025 Histogram
axes[0, 1].hist(rating_2025, bins=50, alpha=0.7, color='#e74c3c', edgecolor='black', linewidth=0.5)
axes[0, 1].axvline(stats_2025['mean'], color='red', linestyle='--', linewidth=2, label=f'Mean: {stats_2025["mean"]:.2f}')
axes[0, 1].axvline(stats_2025['50%'], color='green', linestyle='--', linewidth=2, label=f'Median: {stats_2025["50%"]:.2f}')
axes[0, 1].set_xlabel('Review Scores Rating', fontsize=12)
axes[0, 1].set_ylabel('Frequency', fontsize=12)
axes[0, 1].set_title('2025 Data Distribution', fontsize=14, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 子图3: 重叠直方图对比 / Subplot 3: Overlapping Histogram Comparison
axes[1, 0].hist(rating_2021, bins=50, alpha=0.6, color='#3498db', label='2021', edgecolor='black', linewidth=0.5)
axes[1, 0].hist(rating_2025, bins=50, alpha=0.6, color='#e74c3c', label='2025', edgecolor='black', linewidth=0.5)
axes[1, 0].set_xlabel('Review Scores Rating', fontsize=12)
axes[1, 0].set_ylabel('Frequency', fontsize=12)
axes[1, 0].set_title('Overlapping Distribution Comparison', fontsize=14, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 子图4: 密度图对比 / Subplot 4: Density Plot Comparison
axes[1, 1].hist(rating_2021, bins=50, alpha=0.5, color='#3498db', label='2021', density=True, edgecolor='black', linewidth=0.5)
axes[1, 1].hist(rating_2025, bins=50, alpha=0.5, color='#e74c3c', label='2025', density=True, edgecolor='black', linewidth=0.5)
axes[1, 1].set_xlabel('Review Scores Rating', fontsize=12)
axes[1, 1].set_ylabel('Density', fontsize=12)
axes[1, 1].set_title('Normalized Density Comparison', fontsize=14, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
histogram_file = charts_dir / 'rating_distribution_comparison_histogram.png'
plt.savefig(histogram_file, dpi=300, bbox_inches='tight')
print(f"    ✅ 已保存: {histogram_file}")
plt.close()

# 4.2 箱线图对比 / Boxplot Comparison
print("  4.2 生成箱线图对比 / Generating Boxplot Comparison...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Review Scores Rating Boxplot Comparison', 
             fontsize=16, fontweight='bold', y=1.0)

# 准备数据 / Prepare data
box_data = [rating_2021, rating_2025]
box_labels = ['2021', '2025']
box_colors = ['#3498db', '#e74c3c']

# 子图1: 并排箱线图 / Subplot 1: Side-by-side Boxplot
bp = axes[0].boxplot(box_data, labels=box_labels, patch_artist=True, 
                     widths=0.6, showmeans=True, meanline=True)
for patch, color in zip(bp['boxes'], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
    plt.setp(bp[element], color='black', linewidth=1.5)
axes[0].set_ylabel('Review Scores Rating', fontsize=12)
axes[0].set_title('Boxplot Comparison', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='y')

# 添加统计信息文本 / Add statistics text
stats_text_2021 = f"2021:\nMean: {stats_2021['mean']:.2f}\nMedian: {stats_2021['50%']:.2f}"
stats_text_2025 = f"2025:\nMean: {stats_2025['mean']:.2f}\nMedian: {stats_2025['50%']:.2f}"
axes[0].text(0.5, 0.95, stats_text_2021, transform=axes[0].transAxes,
            fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
axes[0].text(0.5, 0.75, stats_text_2025, transform=axes[0].transAxes,
            fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 添加显著性标注 / Add significance annotation
y_max = max(rating_2021.max(), rating_2025.max())
y_min = min(rating_2021.min(), rating_2025.min())
y_range = y_max - y_min
y_pos = y_max + y_range * 0.05

# 绘制显著性标记线 / Draw significance marker line
axes[0].plot([1, 2], [y_pos, y_pos], 'k-', linewidth=1.5)
axes[0].plot([1, 1], [y_pos - y_range*0.01, y_pos], 'k-', linewidth=1.5)
axes[0].plot([2, 2], [y_pos - y_range*0.01, y_pos], 'k-', linewidth=1.5)

# 添加p值标注 / Add p-value annotation
if all_tests_significant:
    sig_text = f"*** p < 0.001" if min(mw_pvalue, ks_pvalue, t_pvalue) < 0.001 else f"** p < 0.01" if min(mw_pvalue, ks_pvalue, t_pvalue) < 0.01 else f"* p < 0.05"
    axes[0].text(1.5, y_pos + y_range*0.01, sig_text, ha='center', va='bottom', 
                fontsize=11, fontweight='bold', color='red')
else:
    axes[0].text(1.5, y_pos + y_range*0.01, f"ns (p = {min(mw_pvalue, ks_pvalue, t_pvalue):.3f})", 
                ha='center', va='bottom', fontsize=10, color='gray')

# 子图2: 小提琴图对比 / Subplot 2: Violin Plot Comparison
df_box = pd.DataFrame({
    'Year': ['2021'] * len(rating_2021) + ['2025'] * len(rating_2025),
    'Rating': list(rating_2021) + list(rating_2025)
})
sns.violinplot(data=df_box, x='Year', y='Rating', palette=box_colors, ax=axes[1])
axes[1].set_ylabel('Review Scores Rating', fontsize=12)
axes[1].set_title('Violin Plot Comparison', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')

# 添加显著性标注到小提琴图 / Add significance annotation to violin plot
y_max_v = df_box['Rating'].max()
y_min_v = df_box['Rating'].min()
y_range_v = y_max_v - y_min_v
y_pos_v = y_max_v + y_range_v * 0.05

axes[1].plot([0, 1], [y_pos_v, y_pos_v], 'k-', linewidth=1.5)
axes[1].plot([0, 0], [y_pos_v - y_range_v*0.01, y_pos_v], 'k-', linewidth=1.5)
axes[1].plot([1, 1], [y_pos_v - y_range_v*0.01, y_pos_v], 'k-', linewidth=1.5)

if all_tests_significant:
    sig_text_v = f"*** p < 0.001" if min(mw_pvalue, ks_pvalue, t_pvalue) < 0.001 else f"** p < 0.01" if min(mw_pvalue, ks_pvalue, t_pvalue) < 0.01 else f"* p < 0.05"
    axes[1].text(0.5, y_pos_v + y_range_v*0.01, sig_text_v, ha='center', va='bottom', 
                fontsize=11, fontweight='bold', color='red')
else:
    axes[1].text(0.5, y_pos_v + y_range_v*0.01, f"ns (p = {min(mw_pvalue, ks_pvalue, t_pvalue):.3f})", 
                ha='center', va='bottom', fontsize=10, color='gray')

plt.tight_layout()
boxplot_file = charts_dir / 'rating_distribution_comparison_boxplot.png'
plt.savefig(boxplot_file, dpi=300, bbox_inches='tight')
print(f"    ✅ 已保存: {boxplot_file}")
plt.close()

# 4.3 累积分布函数对比 / Cumulative Distribution Function Comparison
print("  4.3 生成累积分布函数对比 / Generating CDF Comparison...")
fig, ax = plt.subplots(figsize=(12, 8))

# 计算CDF / Calculate CDF
sorted_2021 = np.sort(rating_2021)
sorted_2025 = np.sort(rating_2025)
y_2021 = np.arange(1, len(sorted_2021) + 1) / len(sorted_2021)
y_2025 = np.arange(1, len(sorted_2025) + 1) / len(sorted_2025)

ax.plot(sorted_2021, y_2021, label='2021', linewidth=2.5, color='#3498db')
ax.plot(sorted_2025, y_2025, label='2025', linewidth=2.5, color='#e74c3c')
ax.set_xlabel('Review Scores Rating', fontsize=12)
ax.set_ylabel('Cumulative Probability', fontsize=12)
ax.set_title('Cumulative Distribution Function Comparison', 
             fontsize=14, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
cdf_file = charts_dir / 'rating_distribution_comparison_cdf.png'
plt.savefig(cdf_file, dpi=300, bbox_inches='tight')
print(f"    ✅ 已保存: {cdf_file}")
plt.close()

# 4.4 评分区间分布对比 / Rating Range Distribution Comparison
print("  4.4 生成评分区间分布对比 / Generating Rating Range Distribution Comparison...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Review Scores Rating Range Distribution', 
             fontsize=16, fontweight='bold', y=1.0)

# 定义评分区间 / Define rating ranges
bins = [0, 40, 50, 60, 70, 80, 90, 95, 100]
bin_labels = ['0-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-95', '95-100']

# 计算各区间的数量 / Calculate counts for each range
counts_2021, _ = np.histogram(rating_2021, bins=bins)
counts_2025, _ = np.histogram(rating_2025, bins=bins)

# 转换为百分比 / Convert to percentages
percent_2021 = counts_2021 / len(rating_2021) * 100
percent_2025 = counts_2025 / len(rating_2025) * 100

# 子图1: 柱状图对比 / Subplot 1: Bar Chart Comparison
x = np.arange(len(bin_labels))
width = 0.35
axes[0].bar(x - width/2, percent_2021, width, label='2021', color='#3498db', alpha=0.8, edgecolor='black')
axes[0].bar(x + width/2, percent_2025, width, label='2025', color='#e74c3c', alpha=0.8, edgecolor='black')
axes[0].set_xlabel('Rating Range', fontsize=12)
axes[0].set_ylabel('Percentage (%)', fontsize=12)
axes[0].set_title('Rating Range Distribution (Bar Chart)', fontsize=14, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(bin_labels, rotation=45, ha='right')
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis='y')

# 添加数值标签 / Add value labels
for i, (p1, p2) in enumerate(zip(percent_2021, percent_2025)):
    axes[0].text(i - width/2, p1 + 0.5, f'{p1:.1f}%', ha='center', va='bottom', fontsize=8)
    axes[0].text(i + width/2, p2 + 0.5, f'{p2:.1f}%', ha='center', va='bottom', fontsize=8)

# 子图2: 堆叠柱状图 / Subplot 2: Stacked Bar Chart
axes[1].bar(bin_labels, percent_2021, label='2021', color='#3498db', alpha=0.8, edgecolor='black')
axes[1].bar(bin_labels, percent_2025, bottom=percent_2021, label='2025', color='#e74c3c', alpha=0.8, edgecolor='black')
axes[1].set_xlabel('Rating Range', fontsize=12)
axes[1].set_ylabel('Percentage (%)', fontsize=12)
axes[1].set_title('Rating Range Distribution (Stacked)', fontsize=14, fontweight='bold')
axes[1].set_xticklabels(bin_labels, rotation=45, ha='right')
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
range_file = charts_dir / 'rating_distribution_comparison_range.png'
plt.savefig(range_file, dpi=300, bbox_inches='tight')
print(f"    ✅ 已保存: {range_file}")
plt.close()

# 4.5 数据调整前后对比可视化 / Before and After Adjustment Comparison
print("  4.5 生成数据调整前后对比图 / Generating Before/After Adjustment Comparison...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Data Adjustment Comparison: 2021 Adjusted to Match 2025 Distribution', 
             fontsize=16, fontweight='bold', y=0.995)

# 子图1: 原始数据对比 / Subplot 1: Original Data Comparison
axes[0, 0].hist(rating_2021, bins=50, alpha=0.6, color='#3498db', label='2021 Original', edgecolor='black', linewidth=0.5)
axes[0, 0].hist(rating_2025, bins=50, alpha=0.6, color='#e74c3c', label='2025 Target', edgecolor='black', linewidth=0.5)
axes[0, 0].set_xlabel('Review Scores Rating', fontsize=12)
axes[0, 0].set_ylabel('Frequency', fontsize=12)
axes[0, 0].set_title('Original Data Comparison', fontsize=14, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 子图2: 调整后数据对比 / Subplot 2: Adjusted Data Comparison
axes[0, 1].hist(rating_2021_adjusted, bins=50, alpha=0.6, color='#2ecc71', label='2021 Adjusted', edgecolor='black', linewidth=0.5)
axes[0, 1].hist(rating_2025, bins=50, alpha=0.6, color='#e74c3c', label='2025 Target', edgecolor='black', linewidth=0.5)
axes[0, 1].set_xlabel('Review Scores Rating', fontsize=12)
axes[0, 1].set_ylabel('Frequency', fontsize=12)
axes[0, 1].set_title(f'Adjusted Data Comparison ({method_name})', fontsize=14, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 子图3: CDF对比 - 原始数据 / Subplot 3: CDF Comparison - Original
sorted_2021_orig = np.sort(rating_2021)
sorted_2025_orig = np.sort(rating_2025)
sorted_2021_adj = np.sort(rating_2021_adjusted)
y_2021_orig = np.arange(1, len(sorted_2021_orig) + 1) / len(sorted_2021_orig)
y_2025_orig = np.arange(1, len(sorted_2025_orig) + 1) / len(sorted_2025_orig)

axes[1, 0].plot(sorted_2021_orig, y_2021_orig, label='2021 Original', linewidth=2.5, color='#3498db')
axes[1, 0].plot(sorted_2025_orig, y_2025_orig, label='2025 Target', linewidth=2.5, color='#e74c3c', linestyle='--')
axes[1, 0].set_xlabel('Review Scores Rating', fontsize=12)
axes[1, 0].set_ylabel('Cumulative Probability', fontsize=12)
axes[1, 0].set_title('CDF Comparison: Original Data', fontsize=14, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 子图4: CDF对比 - 调整后数据 / Subplot 4: CDF Comparison - Adjusted
y_2021_adj = np.arange(1, len(sorted_2021_adj) + 1) / len(sorted_2021_adj)
axes[1, 1].plot(sorted_2021_adj, y_2021_adj, label='2021 Adjusted', linewidth=2.5, color='#2ecc71')
axes[1, 1].plot(sorted_2025_orig, y_2025_orig, label='2025 Target', linewidth=2.5, color='#e74c3c', linestyle='--')
axes[1, 1].set_xlabel('Review Scores Rating', fontsize=12)
axes[1, 1].set_ylabel('Cumulative Probability', fontsize=12)
axes[1, 1].set_title(f'CDF Comparison: Adjusted Data ({method_name})', fontsize=14, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
adjustment_file = charts_dir / 'rating_distribution_adjustment_comparison.png'
plt.savefig(adjustment_file, dpi=300, bbox_inches='tight')
print(f"    ✅ 已保存: {adjustment_file}")
plt.close()

# 4.6 三种调整方法对比 / Three Adjustment Methods Comparison
print("  4.6 生成三种调整方法对比图 / Generating Three Methods Comparison...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Comparison of Three Adjustment Methods', 
             fontsize=16, fontweight='bold', y=0.995)

# 子图1: 三种方法的直方图对比 / Subplot 1: Histogram Comparison
axes[0, 0].hist(rating_2021_adjusted_quantile, bins=50, alpha=0.5, label='Quantile Matching', 
                color='#3498db', edgecolor='black', linewidth=0.5)
axes[0, 0].hist(rating_2021_adjusted_linear, bins=50, alpha=0.5, label='Linear Transform', 
                color='#e74c3c', edgecolor='black', linewidth=0.5)
axes[0, 0].hist(rating_2021_adjusted_cdf, bins=50, alpha=0.5, label='CDF Mapping', 
                color='#2ecc71', edgecolor='black', linewidth=0.5)
axes[0, 0].hist(rating_2025, bins=50, alpha=0.3, label='2025 Target', 
                color='gray', edgecolor='black', linewidth=0.5, linestyle='--')
axes[0, 0].set_xlabel('Review Scores Rating', fontsize=12)
axes[0, 0].set_ylabel('Frequency', fontsize=12)
axes[0, 0].set_title('Histogram Comparison of Three Methods', fontsize=14, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 子图2: 三种方法的箱线图对比 / Subplot 2: Boxplot Comparison
box_data_methods = [rating_2021_adjusted_quantile, rating_2021_adjusted_linear, 
                    rating_2021_adjusted_cdf, rating_2025]
box_labels_methods = ['Quantile', 'Linear', 'CDF', '2025 Target']
box_colors_methods = ['#3498db', '#e74c3c', '#2ecc71', '#95a5a6']

bp_methods = axes[0, 1].boxplot(box_data_methods, labels=box_labels_methods, patch_artist=True, 
                                widths=0.6, showmeans=True, meanline=True)
for patch, color in zip(bp_methods['boxes'], box_colors_methods):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
    plt.setp(bp_methods[element], color='black', linewidth=1.5)
axes[0, 1].set_ylabel('Review Scores Rating', fontsize=12)
axes[0, 1].set_title('Boxplot Comparison of Three Methods', fontsize=14, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3, axis='y')

# 子图3: KS统计量对比 / Subplot 3: KS Statistic Comparison
methods_ks = ['Quantile\nMatching', 'Linear\nTransform', 'CDF\nMapping']
ks_values = [ks_quantile, ks_linear, ks_cdf]
colors_ks = ['#3498db', '#e74c3c', '#2ecc71']
bars = axes[1, 0].bar(methods_ks, ks_values, color=colors_ks, alpha=0.7, edgecolor='black')
axes[1, 0].set_ylabel('KS Statistic (Lower is Better)', fontsize=12)
axes[1, 0].set_title('KS Statistic Comparison', fontsize=14, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3, axis='y')
# 添加数值标签
for bar, val in zip(bars, ks_values):
    height = bar.get_height()
    axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# 子图4: 统计指标对比表 / Subplot 4: Statistics Comparison Table
stats_comparison = pd.DataFrame({
    'Original 2021': [stats_2021['mean'], stats_2021['50%'], stats_2021['std']],
    'Quantile': [stats_adjusted_quantile['mean'], stats_adjusted_quantile['50%'], stats_adjusted_quantile['std']],
    'Linear': [stats_adjusted_linear['mean'], stats_adjusted_linear['50%'], stats_adjusted_linear['std']],
    'CDF': [stats_adjusted_cdf['mean'], stats_adjusted_cdf['50%'], stats_adjusted_cdf['std']],
    'Target 2025': [stats_2025['mean'], stats_2025['50%'], stats_2025['std']]
}, index=['Mean', 'Median', 'Std'])

axes[1, 1].axis('tight')
axes[1, 1].axis('off')
table = axes[1, 1].table(cellText=stats_comparison.round(2).values,
                         rowLabels=stats_comparison.index,
                         colLabels=stats_comparison.columns,
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)
axes[1, 1].set_title('Statistics Comparison Table', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
methods_file = charts_dir / 'rating_distribution_adjustment_methods_comparison.png'
plt.savefig(methods_file, dpi=300, bbox_inches='tight')
print(f"    ✅ 已保存: {methods_file}")
plt.close()

print("\n" + "=" * 80)
print("分析完成！/ Analysis Complete!")
print("=" * 80)
print(f"\n生成的文件 / Generated Files:")
print(f"  1. {histogram_file}")
print(f"  2. {boxplot_file}")
print(f"  3. {cdf_file}")
print(f"  4. {range_file}")
print(f"  5. {stats_file}")
print(f"  6. {adjustment_file}")
print(f"  7. {methods_file}")
print(f"  8. {adjusted_data_file}")

