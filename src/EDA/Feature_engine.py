import os
import re
import json
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
from utils import setup_plotting, get_project_paths
 
# 设置中文字体支持 / Set Chinese font support
setup_plotting()

# 获取项目根目录路径 / Get project root directory path
project_root, data_dir, charts_dir = get_project_paths()
listings = pd.read_csv(data_dir / 'listings.csv')
listings_detailed = pd.read_excel(data_dir / 'listings_detailed.xlsx')

#1 数据清洗
#对有license的数据标记为True，没有为False
listings['license'] = listings['license'].isna()


#对review per month为0的填充为0
listings['reviews_per_month'] = listings['reviews_per_month'].fillna(0)
#对host_response_time空值设置为no_response,并将列设置为有序分类
order = ['no_response','a few day or more','within a day','within a few hours','within an hour']
cat_type = pd.CategoricalDtype(categories=order, ordered=True)
listings_detailed['host_response_time'] = listings_detailed['host_response_time'].fillna('no_response')
listings_detailed['host_response_time'] = listings_detailed['host_response_time'].astype(cat_type)


#对host_response_rate中空值填充为0，并将列转换为浮点类型
listings_detailed['host_response_rate'] = listings_detailed['host_response_rate'].fillna(0)
# 处理字符串百分比和数字，统一转换为浮点
def convert_to_flat(x):
    if pd.isna(x):
        return 0
    if isinstance(x, str):
        # 去除百分号并转换为浮点数
        return float(x.rstrip('%'))/100
    else:
        # 直接转换为浮点数
        return float(x)/100

listings_detailed['host_response_rate'] = listings_detailed['host_response_rate'].apply(convert_to_flat)

#host_is_superhost为空的值设置为False
listings_detailed['host_is_superhost'] = listings_detailed['host_is_superhost'].fillna(False)

#host_total_listings_count和host_listings_count有空白填充为0
listings_detailed['host_listings_count'] = listings_detailed['host_listings_count'].fillna(0)
listings_detailed['host_total_listings_count'] = listings_detailed['host_total_listings_count'].fillna(0)

#bathrooms_text将str转换为数字,空值填充为missing
def extract_bathrooms_number(x):
    """
    从bathrooms_text中提取浴室数量
    处理格式如：'1.5 shared baths', 'Half-bath', 'Private half-bath'等
    """
    #空值填充为missing
    if pd.isna(x):
        return "missing"

    
    # 处理 half-bath 特殊情况（返回0.5）
    if 'half-bath' in x.lower():
        return 0.5
    
    # 使用正则表达式提取数字（包括小数）
    # 匹配模式：可选负号、数字、可选小数点和小数部分
    match = re.search(r'(\d+\.?\d*)', x)
    
    if match:
        return float(match.group(1))
    else:
        # 如果没有找到数字，返回NaN
        return np.nan

# 将bathrooms_text转换为数字
listings_detailed['bathrooms'] = listings_detailed['bathrooms_text'].apply(extract_bathrooms_number)


#bedroom空值填充为missing
listings_detailed['bedrooms'] = listings_detailed['bedrooms'].fillna("missing")


#将bed空值填充为missing
listings_detailed['beds'] = listings_detailed['beds'].fillna("missing")


#将amenities内的值转换成Multi-hot Encoding,删除出现频率 < 5 的罕见 amenity
def parse_amenities(x):
    """
    解析amenities字符串，返回列表
    支持JSON格式和字符串列表格式
    """
    if pd.isna(x) or x == '':
        return []
    
    try:
        # 尝试解析为JSON
        if isinstance(x, str) and x.strip().startswith('['):
            # 使用ast.literal_eval解析Python列表字符串
            return ast.literal_eval(x)
        elif isinstance(x, str) and x.strip().startswith('{'):
            # JSON对象格式
            amenities_dict = json.loads(x)
            if isinstance(amenities_dict, list):
                return amenities_dict
            elif isinstance(amenities_dict, dict):
                return list(amenities_dict.keys())
        else:
            # 尝试直接JSON解析
            parsed = json.loads(x)
            if isinstance(parsed, list):
                return parsed
            elif isinstance(parsed, dict):
                return list(parsed.keys())
    except (json.JSONDecodeError, ValueError, SyntaxError):
        # 如果解析失败，尝试分割字符串
        if isinstance(x, str):
            # 去除方括号和引号，然后分割
            cleaned = x.strip().strip('[]').replace("'", "").replace('"', '')
            if cleaned:
                return [item.strip() for item in cleaned.split(',') if item.strip()]
    
    return []

# 解析amenities列
print("解析 amenities 列...")
listings_detailed['amenities_parsed'] = listings_detailed['amenities'].apply(parse_amenities)

# 统计每个amenity的出现频率
print("统计 amenities 频率...")
all_amenities = []
for amenity_list in listings_detailed['amenities_parsed']:
    all_amenities.extend(amenity_list)

amenity_counts = Counter(all_amenities)

# 过滤掉出现频率 < 5 的罕见 amenity
min_frequency = 5
frequent_amenities = {amenity: count for amenity, count in amenity_counts.items() if count >= min_frequency}

print(f'总amenities数量: {len(amenity_counts)}')
print(f'频率 >= {min_frequency} 的amenities数量: {len(frequent_amenities)}')
print(f'已删除罕见amenities数量: {len(amenity_counts) - len(frequent_amenities)}')

# 创建Multi-hot Encoding
print("创建 Multi-hot Encoding...")
amenity_columns = sorted(frequent_amenities.keys())

# 清理列名：将特殊字符替换为下划线
def clean_column_name(amenity_name):
    """清理amenity名称，用于创建列名"""
    # 替换特殊字符为下划线
    cleaned = re.sub(r'[^\w\s-]', '_', str(amenity_name))
    # 替换空格为下划线
    cleaned = re.sub(r'[\s-]+', '_', cleaned)
    # 移除连续的下划线
    cleaned = re.sub(r'_+', '_', cleaned)
    # 移除首尾下划线
    cleaned = cleaned.strip('_')
    return cleaned.lower()

# 为每个频繁出现的amenity创建一列
for amenity in amenity_columns:
    col_name = f'amenity_{clean_column_name(amenity)}'
    listings_detailed[col_name] = listings_detailed['amenities_parsed'].apply(
        lambda x, a=amenity: 1 if a in x else 0
    )

print(f'创建了 {len(amenity_columns)} 个amenity特征列')
print(f'\n前10个最频繁的amenities:')
for amenity, count in Counter(all_amenities).most_common(10):
    print(f'  {amenity}: {count}')

# 显示创建的列名示例（前5个）
print(f'\n创建的列名示例（前5个）:')
amenity_cols = [col for col in listings_detailed.columns if col.startswith('amenity_')]
for col in sorted(amenity_cols)[:5]:
    print(f'  {col}')



#maximum_nights, minimum_nights大于365的需要改为365
listings_detailed['minimum_nights'] = listings_detailed['minimum_nights'].clip(upper = 365)
listings_detailed['maximum_nights'] = listings_detailed['maximum_nights'].clip(upper = 365)

#minimum_nights_avg_ntm,maximum_nights_avg_ntm空白值填充为missing
listings_detailed['minimum_nights_avg_ntm'] = listings_detailed['minimum_nights_avg_ntm'].fillna("missing")
listings_detailed['maximum_nights_avg_ntm'] = listings_detailed['maximum_nights_avg_ntm'].fillna("missing")

#review_scores相关七列空白值和0填充为missing
review_scores_cols = [
    'review_scores_rating',
    'review_scores_accuracy',
    'review_scores_cleanliness',
    'review_scores_checkin',
    'review_scores_communication',
    'review_scores_location',
    'review_scores_value'
]
for col in review_scores_cols:
    if col in listings_detailed.columns:
        listings_detailed[col] = listings_detailed[col].fillna("missing")
    else:
        print(f'警告: 列 {col} 不存在于数据中')

#reviews_per_month空值填充为missing
listings_detailed['reviews_per_month'] = listings_detailed['reviews_per_month'].fillna('missing')