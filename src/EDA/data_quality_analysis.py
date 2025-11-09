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
print("数据质量与规模分析 / Data Quality & Scale Analysis")
print("=" * 80)

# ============================================================================
# 1. 加载所有数据集 / Load All Datasets
# ============================================================================

print("\n1. 加载数据集 / Loading Datasets...")

datasets_info = {}

# listings.csv
try:
    listings = pd.read_csv(data_dir / 'listings.csv')
    datasets_info['listings'] = {
        'df': listings,
        'records': len(listings),
        'columns': len(listings.columns),
        'size_mb': (data_dir / 'listings.csv').stat().st_size / (1024 * 1024)
    }
    print(f"  ✅ listings.csv: {len(listings)} 行 × {len(listings.columns)} 列")
except Exception as e:
    print(f"  ❌ listings.csv 加载失败: {e}")

# reviews.csv
try:
    reviews = pd.read_csv(data_dir / 'reviews.csv')
    datasets_info['reviews'] = {
        'df': reviews,
        'records': len(reviews),
        'columns': len(reviews.columns),
        'size_mb': (data_dir / 'reviews.csv').stat().st_size / (1024 * 1024)
    }
    print(f"  ✅ reviews.csv: {len(reviews)} 行 × {len(reviews.columns)} 列")
except Exception as e:
    print(f"  ❌ reviews.csv 加载失败: {e}")

# calendar_summary.csv
try:
    calendar = pd.read_csv(data_dir / 'calendar_summary.csv', sep=';')
    datasets_info['calendar'] = {
        'df': calendar,
        'records': len(calendar),
        'columns': len(calendar.columns),
        'size_mb': (data_dir / 'calendar_summary.csv').stat().st_size / (1024 * 1024)
    }
    print(f"  ✅ calendar_summary.csv: {len(calendar)} 行 × {len(calendar.columns)} 列")
except Exception as e:
    print(f"  ❌ calendar_summary.csv 加载失败: {e}")

# neighbourhoods.csv
try:
    neighbourhoods = pd.read_csv(data_dir / 'neighbourhoods.csv')
    datasets_info['neighbourhoods'] = {
        'df': neighbourhoods,
        'records': len(neighbourhoods),
        'columns': len(neighbourhoods.columns),
        'size_mb': (data_dir / 'neighbourhoods.csv').stat().st_size / (1024 * 1024)
    }
    print(f"  ✅ neighbourhoods.csv: {len(neighbourhoods)} 行 × {len(neighbourhoods.columns)} 列")
except Exception as e:
    print(f"  ❌ neighbourhoods.csv 加载失败: {e}")

# ============================================================================
# 2. 数据规模统计 / Data Scale Statistics
# ============================================================================

print("\n2. 数据规模统计 / Data Scale Statistics...")

total_records = sum(info['records'] for info in datasets_info.values())
total_size = sum(info['size_mb'] for info in datasets_info.values())

print(f"\n总体统计 / Overall Statistics:")
print(f"  - 总数据集数 / Total Datasets: {len(datasets_info)}")
print(f"  - 总记录数 / Total Records: {total_records:,}")
print(f"  - 总数据大小 / Total Size: {total_size:.2f} MB")

# ============================================================================
# 3. 数据完整度分析 / Data Completeness Analysis
# ============================================================================

print("\n3. 数据完整度分析 / Data Completeness Analysis...")

completeness_results = {}

for dataset_name, info in datasets_info.items():
    df = info['df']
    missing_stats = df.isnull().sum()
    missing_pct = (missing_stats / len(df) * 100).round(2)
    
    completeness_results[dataset_name] = {
        'total_fields': len(df.columns),
        'missing_fields': (missing_stats > 0).sum(),
        'complete_fields': (missing_stats == 0).sum(),
        'avg_missing_rate': missing_pct.mean(),
        'max_missing_rate': missing_pct.max(),
        'high_missing_fields': missing_pct[missing_pct > 50].to_dict()
    }
    
    print(f"\n{dataset_name.upper()}:")
    print(f"  总字段数 / Total Fields: {len(df.columns)}")
    print(f"  完整字段数 / Complete Fields: {(missing_stats == 0).sum()}")
    print(f"  有缺失字段数 / Fields with Missing: {(missing_stats > 0).sum()}")
    print(f"  平均缺失率 / Avg Missing Rate: {missing_pct.mean():.2f}%")
    if missing_pct.max() > 0:
        print(f"  最高缺失率 / Max Missing Rate: {missing_pct.max():.2f}%")
        high_missing = missing_pct[missing_pct > 50]
        if len(high_missing) > 0:
            print(f"  高缺失率字段 (>50%) / High Missing Fields:")
            for field, pct in high_missing.items():
                print(f"    - {field}: {pct:.2f}%")

# ============================================================================
# 4. 时间跨度分析 / Time Span Analysis
# ============================================================================

print("\n4. 时间跨度分析 / Time Span Analysis...")

# 从 reviews.csv 分析时间跨度
if 'reviews' in datasets_info:
    reviews_df = datasets_info['reviews']['df']
    if 'date' in reviews_df.columns:
        reviews_df['date'] = pd.to_datetime(reviews_df['date'], errors='coerce')
        date_range = reviews_df['date'].dropna()
        if len(date_range) > 0:
            min_date = date_range.min()
            max_date = date_range.max()
            time_span_days = (max_date - min_date).days
            print(f"  Reviews 时间跨度 / Reviews Time Span:")
            print(f"    - 最早日期 / Earliest Date: {min_date.strftime('%Y-%m-%d')}")
            print(f"    - 最晚日期 / Latest Date: {max_date.strftime('%Y-%m-%d')}")
            print(f"    - 时间跨度 / Time Span: {time_span_days} 天 ({time_span_days/365:.1f} 年)")

# 从 listings.csv 的 last_review 分析
if 'listings' in datasets_info:
    listings_df = datasets_info['listings']['df']
    if 'last_review' in listings_df.columns:
        last_review_dates = pd.to_datetime(listings_df['last_review'], errors='coerce').dropna()
        if len(last_review_dates) > 0:
            print(f"  Listings last_review 时间范围 / Last Review Date Range:")
            print(f"    - 最早最后评论 / Earliest Last Review: {last_review_dates.min().strftime('%Y-%m-%d')}")
            print(f"    - 最晚最后评论 / Latest Last Review: {last_review_dates.max().strftime('%Y-%m-%d')}")

# ============================================================================
# 5. 创建可视化图表 / Create Visualizations
# ============================================================================

print("\n5. 创建可视化图表 / Creating Visualizations...")

# 5.1 数据集规模对比图 / Dataset Scale Comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

dataset_names = list(datasets_info.keys())
records = [info['records'] for info in datasets_info.values()]
sizes = [info['size_mb'] for info in datasets_info.values()]

# 记录数对比
ax1.bar(dataset_names, records, color=['#4CAF50', '#2196F3', '#FF9800', '#9C27B0'])
ax1.set_title('Dataset Records Comparison / 数据集记录数对比', fontsize=14, fontweight='bold')
ax1.set_ylabel('Number of Records / 记录数', fontsize=12)
ax1.set_xlabel('Dataset / 数据集', fontsize=12)
ax1.tick_params(axis='x', rotation=45)
for i, v in enumerate(records):
    ax1.text(i, v, f'{v:,}', ha='center', va='bottom', fontsize=10)

# 数据大小对比
ax2.bar(dataset_names, sizes, color=['#4CAF50', '#2196F3', '#FF9800', '#9C27B0'])
ax2.set_title('Dataset Size Comparison / 数据集大小对比', fontsize=14, fontweight='bold')
ax2.set_ylabel('Size (MB) / 大小 (MB)', fontsize=12)
ax2.set_xlabel('Dataset / 数据集', fontsize=12)
ax2.tick_params(axis='x', rotation=45)
for i, v in enumerate(sizes):
    ax2.text(i, v, f'{v:.2f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(charts_dir / 'dataset_scale_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✅ 已保存: dataset_scale_comparison.png")

# 5.2 数据完整度热力图 / Data Completeness Heatmap
if 'listings' in datasets_info:
    listings_df = datasets_info['listings']['df']
    missing_pct = (listings_df.isnull().sum() / len(listings_df) * 100).round(2)
    
    # 选择缺失率 > 0 的字段
    missing_fields = missing_pct[missing_pct > 0].sort_values(ascending=False)
    
    if len(missing_fields) > 0:
        fig, ax = plt.subplots(figsize=(10, max(6, len(missing_fields) * 0.5)))
        colors = ['#4CAF50' if x < 10 else '#FF9800' if x < 50 else '#F44336' for x in missing_fields.values]
        bars = ax.barh(missing_fields.index, missing_fields.values, color=colors)
        ax.set_xlabel('Missing Rate (%) / 缺失率 (%)', fontsize=12)
        ax.set_title('Data Completeness - Missing Rate by Field / 数据完整度 - 字段缺失率', 
                     fontsize=14, fontweight='bold')
        ax.set_xlim(0, max(missing_fields.values) * 1.1)
        
        # 添加数值标签
        for i, (field, pct) in enumerate(missing_fields.items()):
            ax.text(pct, i, f' {pct:.2f}%', va='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(charts_dir / 'data_completeness_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✅ 已保存: data_completeness_heatmap.png")

# ============================================================================
# 6. 输出统计结果到文件 / Output Statistics to File
# ============================================================================

print("\n6. 生成统计报告 / Generating Statistics Report...")

report_lines = []
report_lines.append("=" * 80)
report_lines.append("数据质量与规模统计报告 / Data Quality & Scale Statistics Report")
report_lines.append("=" * 80)
report_lines.append(f"\n生成时间 / Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
report_lines.append("\n" + "=" * 80)

report_lines.append("\n## 1. 数据集规模统计 / Dataset Scale Statistics")
report_lines.append(f"\n总数据集数 / Total Datasets: {len(datasets_info)}")
report_lines.append(f"总记录数 / Total Records: {total_records:,}")
report_lines.append(f"总数据大小 / Total Size: {total_size:.2f} MB")

report_lines.append("\n\n## 2. 各数据集详细信息 / Detailed Dataset Information")
for dataset_name, info in datasets_info.items():
    report_lines.append(f"\n### {dataset_name.upper()}")
    report_lines.append(f"  - 记录数 / Records: {info['records']:,}")
    report_lines.append(f"  - 字段数 / Columns: {info['columns']}")
    report_lines.append(f"  - 大小 / Size: {info['size_mb']:.2f} MB")

report_lines.append("\n\n## 3. 数据完整度分析 / Data Completeness Analysis")
for dataset_name, stats in completeness_results.items():
    report_lines.append(f"\n### {dataset_name.upper()}")
    report_lines.append(f"  - 总字段数 / Total Fields: {stats['total_fields']}")
    report_lines.append(f"  - 完整字段数 / Complete Fields: {stats['complete_fields']}")
    report_lines.append(f"  - 有缺失字段数 / Fields with Missing: {stats['missing_fields']}")
    report_lines.append(f"  - 平均缺失率 / Avg Missing Rate: {stats['avg_missing_rate']:.2f}%")
    if stats['high_missing_fields']:
        report_lines.append(f"  - 高缺失率字段 (>50%) / High Missing Fields:")
        for field, pct in stats['high_missing_fields'].items():
            report_lines.append(f"    - {field}: {pct:.2f}%")

# 保存报告
with open(charts_dir / 'data_quality_statistics.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))

print("  ✅ 已保存: data_quality_statistics.txt")

print("\n" + "=" * 80)
print("分析完成 / Analysis Complete!")
print("=" * 80)

