"""
EDA 主函数包装器 / EDA Main Function Wrappers

将所有章节分析脚本封装为函数，便于在 Notebook 中调用
Wraps all chapter analysis scripts as functions for easy calling in Notebooks
"""

import sys
from pathlib import Path

# 添加当前目录到路径，以便导入模块
sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    setup_plotting, get_project_paths,
    load_listings_data, load_reviews_data, 
    load_calendar_data, load_neighbourhoods_data,
    print_section_header
)

def run_chapter2_data_quality():
    """
    运行第2章：数据质量与规模总览分析
    Run Chapter 2: Data Quality & Scale Overview Analysis
    """
    from data_quality_analysis import analyze_data_quality
    return analyze_data_quality()

def run_chapter3_dataset_relationships():
    """
    运行第3章：数据集关系与结构分析
    Run Chapter 3: Dataset Relationships & Structure Analysis
    """
    from chapter3_dataset_relationships import analyze_dataset_relationships
    return analyze_dataset_relationships()

def run_chapter5_listings():
    """
    运行第5.1章：listings 数据集详细分析
    Run Chapter 5.1: Listings Dataset Detailed Analysis
    """
    from chapter5_listings_analysis import analyze_listings
    return analyze_listings()

def run_chapter5_reviews():
    """
    运行第5.2章：reviews 数据集详细分析
    Run Chapter 5.2: Reviews Dataset Detailed Analysis
    """
    from chapter5_reviews_analysis import analyze_reviews
    return analyze_reviews()

def run_chapter5_calendar():
    """
    运行第5.3章：calendar_summary 数据集详细分析
    Run Chapter 5.3: Calendar Summary Dataset Detailed Analysis
    """
    from chapter5_calendar_analysis import analyze_calendar
    return analyze_calendar()

def run_chapter5_neighbourhoods():
    """
    运行第5.4章：neighbourhoods 数据集详细分析
    Run Chapter 5.4: Neighbourhoods Dataset Detailed Analysis
    """
    from chapter5_neighbourhoods_analysis import analyze_neighbourhoods
    return analyze_neighbourhoods()

def run_chapter5_listings_detailed():
    """
    运行第5.5章：listings_detailed 数据集详细分析
    Run Chapter 5.5: Listings Detailed Dataset Detailed Analysis
    """
    from chapter5_listings_detailed_analysis import analyze_listings_detailed
    return analyze_listings_detailed()

def run_chapter6_correlation():
    """
    运行第6章：变量相关性分析
    Run Chapter 6: Variable Correlation Analysis
    """
    from chapter6_correlation_analysis import analyze_correlation
    return analyze_correlation()

def run_chapter7_time_series():
    """
    运行第7章：时间序列分析
    Run Chapter 7: Time Series Analysis
    """
    from chapter7_time_series_analysis import analyze_time_series
    return analyze_time_series()

def run_chapter8_geospatial():
    """
    运行第8章：地理空间分析
    Run Chapter 8: Geospatial Analysis
    """
    from chapter8_geospatial_analysis import analyze_geospatial
    return analyze_geospatial()

def run_chapter9_pareto_pricing():
    """
    运行第9章：帕累托分析和价格策略分析
    Run Chapter 9: Pareto Analysis and Pricing Strategy Analysis
    """
    from chapter9_pareto_pricing_analysis import analyze_pareto_pricing
    return analyze_pareto_pricing()

def run_all_analysis():
    """
    运行所有章节的分析
    Run all chapter analyses
    """
    results = {}
    
    print("=" * 80)
    print("开始运行所有 EDA 分析 / Starting All EDA Analysis")
    print("=" * 80)
    
    # 第2章
    print("\n" + "=" * 80)
    print("Chapter 2: Data Quality & Scale Overview")
    print("=" * 80)
    results['chapter2'] = run_chapter2_data_quality()
    
    # 第3章
    print("\n" + "=" * 80)
    print("Chapter 3: Dataset Relationships & Structure")
    print("=" * 80)
    results['chapter3'] = run_chapter3_dataset_relationships()
    
    # 第5章 - listings
    print("\n" + "=" * 80)
    print("Chapter 5.1: Listings Dataset Analysis")
    print("=" * 80)
    results['chapter5_listings'] = run_chapter5_listings()
    
    # 第5章 - reviews
    print("\n" + "=" * 80)
    print("Chapter 5.2: Reviews Dataset Analysis")
    print("=" * 80)
    results['chapter5_reviews'] = run_chapter5_reviews()
    
    # 第5章 - calendar
    print("\n" + "=" * 80)
    print("Chapter 5.3: Calendar Dataset Analysis")
    print("=" * 80)
    results['chapter5_calendar'] = run_chapter5_calendar()
    
    # 第5章 - neighbourhoods
    print("\n" + "=" * 80)
    print("Chapter 5.4: Neighbourhoods Dataset Analysis")
    print("=" * 80)
    results['chapter5_neighbourhoods'] = run_chapter5_neighbourhoods()
    
    # 第5章 - listings_detailed
    print("\n" + "=" * 80)
    print("Chapter 5.5: Listings Detailed Dataset Analysis")
    print("=" * 80)
    results['chapter5_listings_detailed'] = run_chapter5_listings_detailed()
    
    # 第6章
    print("\n" + "=" * 80)
    print("Chapter 6: Variable Correlation Analysis")
    print("=" * 80)
    results['chapter6'] = run_chapter6_correlation()
    
    # 第7章
    print("\n" + "=" * 80)
    print("Chapter 7: Time Series Analysis")
    print("=" * 80)
    results['chapter7'] = run_chapter7_time_series()
    
    # 第8章
    print("\n" + "=" * 80)
    print("Chapter 8: Geospatial Analysis")
    print("=" * 80)
    results['chapter8'] = run_chapter8_geospatial()
    
    # 第9章
    print("\n" + "=" * 80)
    print("Chapter 9: Pareto Analysis and Pricing Strategy")
    print("=" * 80)
    results['chapter9'] = run_chapter9_pareto_pricing()
    
    print("\n" + "=" * 80)
    print("所有分析完成！/ All Analysis Complete!")
    print("=" * 80)
    
    return results

