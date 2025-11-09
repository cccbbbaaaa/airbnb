"""
EDA 工具函数模块 / EDA Utility Functions Module

提供通用的数据加载、路径设置、可视化配置等功能
Provides common functions for data loading, path settings, visualization configuration, etc.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def setup_plotting():
    """
    设置绘图配置 / Setup plotting configuration
    """
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    sns.set_style("whitegrid")

def get_project_paths():
    """
    获取项目路径 / Get project paths
    
    Returns:
        tuple: (project_root, data_dir, charts_dir)
    """
    # 获取项目根目录路径 / Get project root directory path
    # 处理 __file__ 不存在的情况（如在 Notebook 中使用 exec）
    # Handle case where __file__ doesn't exist (e.g., when using exec in Notebook)
    import sys
    
    # 尝试从当前工作目录推断项目根目录
    # Try to infer project root from current working directory
    cwd = Path.cwd()
    
    # 检查是否在 EDA 目录下（需要向上两级到项目根目录）
    # Check if we're in EDA directory (need to go up two levels to project root)
    if cwd.name == 'EDA' and (cwd.parent / 'src').exists():
        project_root = cwd.parent.parent
    # 检查是否在 src 目录下（需要向上一级到项目根目录）
    # Check if we're in src directory (need to go up one level to project root)
    elif cwd.name == 'src' and (cwd.parent / 'data').exists():
        project_root = cwd.parent
    # 检查是否在项目根目录（有 data 和 src 目录）
    # Check if we're in project root (has data and src directories)
    elif (cwd / 'data').exists() and (cwd / 'src').exists():
        project_root = cwd
    else:
        # 最后的备选方案：尝试向上查找项目根目录
        # Final fallback: try to find project root by going up directories
        current = cwd
        for _ in range(3):  # 最多向上查找3级
            if (current / 'data').exists() and (current / 'src').exists():
                project_root = current
                break
            current = current.parent
        else:
            # 如果找不到，假设当前目录就是项目根目录
            project_root = cwd
    
    data_dir = project_root / 'data'
    charts_dir = project_root / 'charts'
    os.makedirs(charts_dir, exist_ok=True)
    return project_root, data_dir, charts_dir

def load_listings_data(data_dir, clean=True):
    """
    加载并清洗 listings 数据 / Load and clean listings data
    
    Args:
        data_dir: 数据目录路径 / Data directory path
        clean: 是否进行数据清洗 / Whether to clean data
        
    Returns:
        pd.DataFrame: 清洗后的 listings 数据 / Cleaned listings data
    """
    listings = pd.read_csv(data_dir / 'listings.csv')
    
    if clean:
        # 删除 neighbourhood_group 列（全为空）
        if 'neighbourhood_group' in listings.columns:
            listings = listings.drop('neighbourhood_group', axis=1)
        
        # 填充缺失值 / Fill Missing Values
        listings['last_review'] = listings['last_review'].fillna(0)
        listings['reviews_per_month'] = listings['reviews_per_month'].fillna(0)
        listings['name'] = listings['name'].fillna('blank_name')
        listings['host_name'] = listings['host_name'].fillna('blank_host_name')
        
        # 处理异常值 / Handle Outliers
        listings.loc[listings['minimum_nights'] > 365, 'minimum_nights'] = 365
        listings.loc[listings['price'] == 0, 'price'] = np.nan
    
    return listings

def load_reviews_data(data_dir):
    """
    加载 reviews 数据 / Load reviews data
    
    Args:
        data_dir: 数据目录路径 / Data directory path
        
    Returns:
        pd.DataFrame: reviews 数据 / Reviews data
    """
    reviews = pd.read_csv(data_dir / 'reviews.csv')
    reviews['date'] = pd.to_datetime(reviews['date'], errors='coerce')
    return reviews

def load_calendar_data(data_dir):
    """
    加载 calendar_summary 数据 / Load calendar_summary data
    
    Args:
        data_dir: 数据目录路径 / Data directory path
        
    Returns:
        pd.DataFrame: calendar 数据 / Calendar data
    """
    calendar = pd.read_csv(data_dir / 'calendar_summary.csv', sep=';')
    return calendar

def load_neighbourhoods_data(data_dir):
    """
    加载 neighbourhoods 数据 / Load neighbourhoods data
    
    Args:
        data_dir: 数据目录路径 / Data directory path
        
    Returns:
        pd.DataFrame: neighbourhoods 数据 / Neighbourhoods data
    """
    neighbourhoods = pd.read_csv(data_dir / 'neighbourhoods.csv')
    return neighbourhoods

def print_section_header(title_en, title_cn):
    """
    打印章节标题 / Print section header
    
    Args:
        title_en: 英文标题 / English title
        title_cn: 中文标题 / Chinese title
    """
    print("=" * 80)
    print(f"{title_en}")
    print(f"{title_cn}")
    print("=" * 80)

