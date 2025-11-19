"""
Chapter 3: Dataset Relationships & Structure Analysis
第3章：数据集关系与结构分析

本脚本分析各数据集之间的关系，验证数据一致性，并评估数据整合价值。
This script analyzes relationships between datasets, validates data consistency, and evaluates data integration value.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from utils import setup_plotting, get_project_paths, print_section_header

def analyze_dataset_relationships(data_dir=None, charts_dir=None, verbose=True):
    """
    分析数据集关系与结构 / Analyze dataset relationships and structure
    
    Args:
        data_dir: 数据目录路径，如果为None则自动获取 / Data directory path, auto-detect if None
        charts_dir: 图表保存目录，如果为None则自动获取 / Charts directory path, auto-detect if None
        verbose: 是否打印详细信息 / Whether to print detailed information
        
    Returns:
        dict: 分析结果字典 / Analysis results dictionary
    """
    # 设置 / Setup
    setup_plotting()
    if data_dir is None or charts_dir is None:
        _, data_dir, charts_eda_dir, charts_model_dir = get_project_paths()
        charts_dir = charts_eda_dir  # 使用 EDA 目录 / Use EDA directory
    
    if verbose:
        print_section_header(
            "Chapter 3: Dataset Relationships & Structure Analysis",
            "第3章：数据集关系与结构分析"
        )

    # ============================================================================
    # 1. 加载所有数据集 / Load All Datasets
    # ============================================================================

    if verbose:
        print("\n1. 加载数据集 / Loading Datasets...")

    listings = pd.read_csv(data_dir / 'listings.csv')
    reviews = pd.read_csv(data_dir / 'reviews.csv')
    calendar = pd.read_csv(data_dir / 'calendar_summary.csv', sep=';')
    neighbourhoods = pd.read_csv(data_dir / 'neighbourhoods.csv')

    if verbose:
        print(f"  ✅ listings.csv: {len(listings)} 行 × {len(listings.columns)} 列")
        print(f"  ✅ reviews.csv: {len(reviews)} 行 × {len(reviews.columns)} 列")
        print(f"  ✅ calendar_summary.csv: {len(calendar)} 行 × {len(calendar.columns)} 列")
        print(f"  ✅ neighbourhoods.csv: {len(neighbourhoods)} 行 × {len(neighbourhoods.columns)} 列")

    # ============================================================================
    # 2. 数据集关系验证 / Dataset Relationship Validation
    # ============================================================================

    if verbose:
        print("\n2. 数据集关系验证 / Dataset Relationship Validation...")

    # 2.1 reviews.csv → listings.csv 关系验证
    if verbose:
        print("\n2.1 reviews.csv → listings.csv 关系验证 / Relationship Validation:")
        print(f"  - reviews 中的唯一 listing_id 数: {reviews['listing_id'].nunique():,}")
        print(f"  - listings 中的唯一 id 数: {listings['id'].nunique():,}")

    # 检查 reviews 中的 listing_id 是否都在 listings 中
    reviews_listing_ids = set(reviews['listing_id'].unique())
    listings_ids = set(listings['id'].unique())
    missing_in_listings = reviews_listing_ids - listings_ids
    if verbose:
        print(f"  - reviews 中有但 listings 中没有的 listing_id: {len(missing_in_listings)} 个")

    # 验证 number_of_reviews
    if verbose:
        print("\n  验证 number_of_reviews 字段 / Validating number_of_reviews:")
    reviews_count_by_listing = reviews.groupby('listing_id').size()
    merged_reviews = listings.merge(
        reviews_count_by_listing.reset_index(name='actual_reviews'),
        left_on='id', right_on='listing_id', how='left'
    )
    merged_reviews['actual_reviews'] = merged_reviews['actual_reviews'].fillna(0)
    merged_reviews['reviews_match'] = merged_reviews['number_of_reviews'] == merged_reviews['actual_reviews']
    match_rate = merged_reviews['reviews_match'].mean() * 100
    if verbose:
        print(f"  - number_of_reviews 匹配率: {match_rate:.2f}%")
        print(f"  - 不匹配的记录数: {(~merged_reviews['reviews_match']).sum()}")

    # 2.2 calendar_summary.csv → listings.csv 关系验证
    if verbose:
        print("\n2.2 calendar_summary.csv → listings.csv 关系验证 / Relationship Validation:")
        print(f"  - calendar 中的唯一 listing_id 数: {calendar['listing_id'].nunique():,}")
        print(f"  - listings 中的唯一 id 数: {listings['id'].nunique():,}")

    # 检查 calendar 中的 listing_id 是否都在 listings 中
    calendar_listing_ids = set(calendar['listing_id'].unique())
    missing_in_listings_cal = calendar_listing_ids - listings_ids
    if verbose:
        print(f"  - calendar 中有但 listings 中没有的 listing_id: {len(missing_in_listings_cal)} 个")

    # 验证 availability_365
    if verbose:
        print("\n  验证 availability_365 字段 / Validating availability_365:")
    # 计算每个房源的不可用天数（available='f'）
    calendar_unavailable = calendar[calendar['available'] == 'f'].groupby('listing_id')['count'].sum().reset_index(name='unavailable_days')
    # 确保所有 listing_ids 都被包含
    all_listings_df = pd.DataFrame({'listing_id': listings['id'].unique()})
    calendar_unavailable = all_listings_df.merge(calendar_unavailable, on='listing_id', how='left').fillna(0)
    calendar_available_days = 365 - calendar_unavailable['unavailable_days']
    calendar_availability = pd.DataFrame({
        'listing_id': calendar_unavailable['listing_id'],
        'calculated_availability': calendar_available_days
    })

    merged_calendar = listings.merge(
        calendar_availability,
        left_on='id', right_on='listing_id', how='left'
    )
    merged_calendar['availability_match'] = (
        merged_calendar['availability_365'] == merged_calendar['calculated_availability']
    ) | merged_calendar['calculated_availability'].isna()
    match_rate_cal = merged_calendar['availability_match'].mean() * 100
    if verbose:
        print(f"  - availability_365 匹配率: {match_rate_cal:.2f}%")

    # 2.3 neighbourhoods.csv → listings.csv 关系验证
    if verbose:
        print("\n2.3 neighbourhoods.csv → listings.csv 关系验证 / Relationship Validation:")
        print(f"  - neighbourhoods 中的唯一 neighbourhood 数: {neighbourhoods['neighbourhood'].nunique()}")
        print(f"  - listings 中的唯一 neighbourhood 数: {listings['neighbourhood'].nunique()}")

    listings_neighbourhoods = set(listings['neighbourhood'].unique())
    neighbourhoods_list = set(neighbourhoods['neighbourhood'].dropna().unique())
    missing_in_neighbourhoods = listings_neighbourhoods - neighbourhoods_list
    extra_in_neighbourhoods = neighbourhoods_list - listings_neighbourhoods
    if verbose:
        print(f"  - listings 中有但 neighbourhoods 中没有的: {len(missing_in_neighbourhoods)} 个")
        print(f"  - neighbourhoods 中有但 listings 中没有的: {len(extra_in_neighbourhoods)} 个")

    # ============================================================================
    # 3. 数据整合价值分析 / Data Integration Value Analysis
    # ============================================================================

    if verbose:
        print("\n3. 数据整合价值分析 / Data Integration Value Analysis...")

    # 3.1 整合后的数据规模
    if verbose:
        print("\n3.1 整合后的数据规模 / Integrated Data Scale:")
        print(f"  - 可以整合的房源数: {len(listings):,}")
        print(f"  - 可以整合的评论数: {len(reviews):,}")
        print(f"  - 平均每个房源的评论数: {len(reviews) / len(listings):.1f}")
        print(f"  - 有评论的房源数: {listings[listings['number_of_reviews'] > 0].shape[0]:,}")
        print(f"  - 有评论的房源占比: {listings[listings['number_of_reviews'] > 0].shape[0] / len(listings) * 100:.1f}%")

    # 3.2 时间序列分析潜力
    if verbose:
        print("\n3.2 时间序列分析潜力 / Time Series Analysis Potential:")
    reviews['date'] = pd.to_datetime(reviews['date'], errors='coerce')
    reviews_with_date = reviews.dropna(subset=['date'])
    if verbose:
        print(f"  - 有效评论日期数: {len(reviews_with_date):,}")
        print(f"  - 时间跨度: {reviews_with_date['date'].min()} 至 {reviews_with_date['date'].max()}")
        print(f"  - 可以分析的时间序列特征: 季节性、趋势、异常事件影响")

    # 3.3 地理空间分析潜力
    if verbose:
        print("\n3.3 地理空间分析潜力 / Geospatial Analysis Potential:")
        print(f"  - 有地理坐标的房源数: {listings[['latitude', 'longitude']].notna().all(axis=1).sum():,}")
        print(f"  - 可以分析的街区数: {listings['neighbourhood'].nunique()}")
        print(f"  - 可以分析的地理特征: 位置、密度、距离市中心距离")

    # ============================================================================
    # 4. 创建可视化图表 / Create Visualizations
    # ============================================================================

    if verbose:
        print("\n4. 创建可视化图表 / Creating Visualizations...")

    # 4.1 数据集关系图（数据验证结果）
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 4.1.1 Reviews 匹配情况
    reviews_match_data = merged_reviews['reviews_match'].value_counts()
    labels_reviews = []
    colors_reviews = []
    if True in reviews_match_data.index:
        labels_reviews.append('Match')
        colors_reviews.append('#4CAF50')
    if False in reviews_match_data.index:
        labels_reviews.append('Mismatch')
        colors_reviews.append('#005691')
    axes[0, 0].pie(reviews_match_data.values, wedgeprops=dict(width=0.3, edgecolor='w'),labels=labels_reviews, 
                    colors=colors_reviews, textprops={'fontsize': 15})
    axes[0, 0].text(0, 0 , f"{match_rate}%", ha = 'center', va = 'center', fontsize = 28, fontweight = 'bold', color='#005691' )
    axes[0, 0].set_title('Reviews Count Match Rate', fontsize=12, fontweight='bold')

    # 4.1.2 Calendar 匹配情况 - 分离式圆环图
    calendar_match_data = merged_calendar['availability_match'].value_counts()
    labels_cal = []
    colors_cal = []
    explode_cal = []
    values_cal = []
    
    # 确保顺序：True (Match) 在前，False (Mismatch) 在后
    if True in calendar_match_data.index:
        match_count = calendar_match_data[True]
        match_pct = (match_count / len(merged_calendar)) * 100
        labels_cal.append(f'Match {match_pct:.1f}%')  # 标签和百分比同一行
        colors_cal.append('#005691')  # 深蓝色
        values_cal.append(match_count)
        explode_cal.append(0.05)  # 稍微拉出
    if False in calendar_match_data.index:
        mismatch_count = calendar_match_data[False]
        mismatch_pct = (mismatch_count / len(merged_calendar)) * 100
        labels_cal.append(f'Mismatch {mismatch_pct:.1f}%')  # 标签和百分比同一行
        colors_cal.append('#F44336')  # 红色
        values_cal.append(mismatch_count)
        explode_cal.append(0.05)  # 稍微拉出
    
    # 创建分离式圆环图
    wedges, texts, autotexts = axes[0, 1].pie(
        values_cal, 
        labels=labels_cal,
        colors=colors_cal,
        explode=explode_cal,
        wedgeprops=dict(width=0.3, edgecolor='w', linewidth=2),
        autopct='',  # 不显示额外的百分比，因为已经在标签中
        labeldistance=1.1,
        textprops={'fontsize': 12, 'fontweight': 'bold'},
        startangle=90
    )
    
    # 在中心添加匹配率文本
    axes[0, 1].text(0, 0, f"{match_rate_cal:.1f}%", 
                    ha='center', va='center', 
                    fontsize=24, fontweight='bold', color='#005691')
    
    axes[0, 1].set_title('Availability Match rate', fontsize=12, fontweight='bold')
    
    # 在图表下方添加总结文本
    summary_text = f"Over{match_rate_cal:.1f}%of property availability statuses align with actual data."
    axes[0, 1].text(0.5, -0.15, summary_text, 
                    ha='center', va='top', 
                    transform=axes[0, 1].transAxes,
                    fontsize=10, style='italic', color='#666666')

    # 4.1.3 有评论房源分布
    has_reviews = listings['number_of_reviews'] > 0
    has_reviews_count = has_reviews.sum()
    no_reviews_count = (~has_reviews).sum()
    total_count = has_reviews_count + no_reviews_count
    has_reviews_pct = (has_reviews_count / total_count) * 100
    no_reviews_pct = (no_reviews_count / total_count) * 100
    
    axes[1, 0].bar(['Has Reviews', 'No Reviews'], 
                   [has_reviews_count, no_reviews_count],
                   color=['#0066CC', '#CCCCCC'])
    axes[1, 0].set_ylabel('Number of Listings', fontsize=11)
    axes[1, 0].set_title('Listing Review Coverage Rate', fontsize=12, fontweight='bold')
    # 移除边框和网格线
    axes[1, 0].spines['top'].set_visible(False)
    axes[1, 0].spines['right'].set_visible(False)
    axes[1, 0].spines['bottom'].set_visible(False)
    axes[1, 0].spines['left'].set_visible(False)
    axes[1, 0].grid(False)
    # 添加百分比标签
    axes[1, 0].text(0, has_reviews_count, f'{has_reviews_count:,} ({has_reviews_pct:.2f}%)', 
                    ha='center', va='bottom', fontsize=10)
    axes[1, 0].text(1, no_reviews_count, f'{no_reviews_count:,} ({no_reviews_pct:.2f}%)', 
                    ha='center', va='bottom', fontsize=10)

    # 4.1.4 评论数分布（对数尺度）
    reviews_dist = listings[listings['number_of_reviews'] > 0]['number_of_reviews']
    axes[1, 1].hist(np.log1p(reviews_dist), bins=50, color='#9C27B0', edgecolor='black', alpha=0.7)
    axes[1, 1].set_xlabel('Log(Number of Reviews + 1)', fontsize=11)
    axes[1, 1].set_ylabel('Frequency', fontsize=11)
    axes[1, 1].set_title('Distribution of Reviews (Log Scale)', fontsize=12, fontweight='bold')
    axes[1 ,1].grid(False)

    plt.tight_layout()
    plt.savefig(charts_dir / 'chapter3_dataset_relationships.png', dpi=300, bbox_inches='tight')
    plt.close()
    if verbose:
        print("  ✅ 已保存: chapter3_dataset_relationships.png")

    # ============================================================================
    # 5. 输出统计结果 / Output Statistics
    # ============================================================================

    if verbose:
        print("\n5. 生成统计报告 / Generating Statistics Report...")

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("Chapter 3: Dataset Relationships & Structure Analysis")
    report_lines.append("第3章：数据集关系与结构分析")
    report_lines.append("=" * 80)
    report_lines.append(f"\n生成时间 / Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")

    report_lines.append("\n## 数据集关系验证结果 / Dataset Relationship Validation Results")
    report_lines.append(f"\n### reviews.csv → listings.csv")
    report_lines.append(f"  - Reviews 中的唯一 listing_id 数: {reviews['listing_id'].nunique():,}")
    report_lines.append(f"  - Listings 中的唯一 id 数: {listings['id'].nunique():,}")
    report_lines.append(f"  - number_of_reviews 匹配率: {match_rate:.2f}%")

    report_lines.append(f"\n### calendar_summary.csv → listings.csv")
    report_lines.append(f"  - Calendar 中的唯一 listing_id 数: {calendar['listing_id'].nunique():,}")
    report_lines.append(f"  - availability_365 匹配率: {match_rate_cal:.2f}%")

    report_lines.append(f"\n### neighbourhoods.csv → listings.csv")
    report_lines.append(f"  - Neighbourhoods 中的唯一 neighbourhood 数: {neighbourhoods['neighbourhood'].nunique()}")
    report_lines.append(f"  - Listings 中的唯一 neighbourhood 数: {listings['neighbourhood'].nunique()}")

    report_lines.append("\n## 数据整合价值 / Data Integration Value")
    report_lines.append(f"  - 可以整合的房源数: {len(listings):,}")
    report_lines.append(f"  - 可以整合的评论数: {len(reviews):,}")
    report_lines.append(f"  - 平均每个房源的评论数: {len(reviews) / len(listings):.1f}")
    report_lines.append(f"  - 有评论的房源占比: {listings[listings['number_of_reviews'] > 0].shape[0] / len(listings) * 100:.1f}%")

    with open(charts_dir / 'chapter3_statistics.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    if verbose:
        print("  ✅ 已保存: chapter3_statistics.txt")
        print("\n" + "=" * 80)
        print("分析完成 / Analysis Complete!")
        print("=" * 80)
    
    # 返回结果 / Return results
    return {
        'match_rate_reviews': match_rate,
        'match_rate_calendar': match_rate_cal,
        'reviews_listing_ids_count': len(reviews_listing_ids),
        'listings_ids_count': len(listings_ids),
        'calendar_listing_ids_count': len(calendar_listing_ids),
        'neighbourhoods_match': len(missing_in_neighbourhoods) == 0 and len(extra_in_neighbourhoods) == 0
    }

# 主程序入口 / Main entry point
if __name__ == "__main__":
    analyze_dataset_relationships()

