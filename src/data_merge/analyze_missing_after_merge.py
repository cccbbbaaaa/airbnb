"""
分析合并后数据集的缺失值情况并推荐处理方法
Analyze Missing Values in Merged Dataset and Recommend Handling Methods
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# 添加 EDA 目录到路径，以便导入 utils 模块 / Add EDA directory to path for importing utils module
sys.path.insert(0, str(Path(__file__).parent.parent / "EDA"))
from utils import get_project_paths

# 获取项目路径
project_root, data_dir, charts_eda_dir, charts_model_dir = get_project_paths()
charts_dir = project_root / 'charts' / 'data_merge'
charts_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("合并后数据集缺失值分析 / Missing Values Analysis After Merging")
print("=" * 80)

# ============================================================================
# 1. 加载合并后的数据 / Load Merged Data
# ============================================================================

print("\n1. 加载合并后的数据 / Loading Merged Data...")

try:
    merged_data = pd.read_excel(project_root / 'data' / 'merged' / 'listings_merged_2021_2025.xlsx')
    print(f"  ✅ 合并后数据: {len(merged_data):,} 行 × {len(merged_data.columns)} 列")
except Exception as e:
    print(f"  ❌ 数据加载失败: {e}")
    raise

# ============================================================================
# 2. 计算缺失值统计 / Calculate Missing Values Statistics
# ============================================================================

print("\n2. 计算缺失值统计 / Calculating Missing Values Statistics...")

total_rows = len(merged_data)
missing_stats = []

for col in merged_data.columns:
    missing_count = merged_data[col].isna().sum()
    missing_pct = missing_count / total_rows * 100
    non_missing_count = total_rows - missing_count
    completeness_pct = 100 - missing_pct
    
    # 判断字段类型
    dtype = merged_data[col].dtype
    is_numeric = pd.api.types.is_numeric_dtype(dtype)
    is_datetime = pd.api.types.is_datetime64_any_dtype(dtype)
    is_categorical = dtype == 'object' or dtype.name == 'category'
    
    # 检查唯一值数量（用于判断是否是分类变量）
    if non_missing_count > 0:
        unique_count = merged_data[col].dropna().nunique()
        unique_ratio = unique_count / non_missing_count if non_missing_count > 0 else 0
    else:
        unique_count = 0
        unique_ratio = 0
    
    missing_stats.append({
        'column': col,
        'missing_count': missing_count,
        'missing_pct': missing_pct,
        'non_missing_count': non_missing_count,
        'completeness_pct': completeness_pct,
        'dtype': str(dtype),
        'is_numeric': is_numeric,
        'is_datetime': is_datetime,
        'is_categorical': is_categorical,
        'unique_count': unique_count,
        'unique_ratio': unique_ratio
    })

stats_df = pd.DataFrame(missing_stats)
stats_df = stats_df.sort_values('missing_pct', ascending=False)

# ============================================================================
# 3. 推荐缺失值处理方法 / Recommend Missing Value Handling Methods
# ============================================================================

print("\n3. 推荐缺失值处理方法 / Recommending Missing Value Handling Methods...")

def recommend_handling_method(row):
    """
    根据字段特征推荐缺失值处理方法
    Recommend missing value handling method based on field characteristics
    """
    col = row['column']
    missing_pct = row['missing_pct']
    is_numeric = row['is_numeric']
    is_datetime = row['is_datetime']
    is_categorical = row['is_categorical']
    unique_ratio = row['unique_ratio']
    
    # 特殊字段处理
    if col == 'data_year':
        return '保留 / Keep (标识字段)'
    
    if col == 'id':
        return '必须处理 / Must Handle (关键字段)'
    
    # 完全缺失的字段（100%）
    if missing_pct == 100:
        return '删除字段 / Drop Column (完全缺失)'
    
    # 缺失率极高（>90%）
    if missing_pct > 90:
        return '删除字段 / Drop Column (缺失率>90%)'
    
    # 缺失率很高（70-90%）
    if missing_pct > 70:
        if is_numeric:
            return '删除字段或创建缺失指示变量 / Drop Column or Create Missing Indicator'
        else:
            return '删除字段 / Drop Column (缺失率>70%)'
    
    # 缺失率较高（50-70%）
    if missing_pct > 50:
        if is_numeric:
            return '创建缺失指示变量 + 中位数/均值填充 / Missing Indicator + Median/Mean Fill'
        else:
            return '创建缺失指示变量 + 众数填充 / Missing Indicator + Mode Fill'
    
    # 缺失率中等（20-50%）
    if missing_pct > 20:
        if is_numeric:
            return '中位数/均值填充 + 缺失指示变量 / Median/Mean Fill + Missing Indicator'
        elif is_datetime:
            return '前向填充或删除 / Forward Fill or Drop'
        elif is_categorical:
            if unique_ratio < 0.1:  # 低基数分类变量
                return '众数填充 + 缺失指示变量 / Mode Fill + Missing Indicator'
            else:  # 高基数分类变量
                return '创建"未知"类别 + 缺失指示变量 / Create "Unknown" Category + Missing Indicator'
        else:
            return '众数填充或创建"未知"类别 / Mode Fill or Create "Unknown" Category'
    
    # 缺失率较低（10-20%）
    if missing_pct > 10:
        if is_numeric:
            return '中位数/均值填充 / Median/Mean Fill'
        elif is_datetime:
            return '前向填充或插值 / Forward Fill or Interpolation'
        elif is_categorical:
            if unique_ratio < 0.1:
                return '众数填充 / Mode Fill'
            else:
                return '创建"未知"类别 / Create "Unknown" Category'
        else:
            return '众数填充 / Mode Fill'
    
    # 缺失率很低（<10%）
    if missing_pct > 0:
        if is_numeric:
            return '中位数/均值填充 / Median/Mean Fill'
        elif is_datetime:
            return '前向填充或插值 / Forward Fill or Interpolation'
        elif is_categorical:
            return '众数填充 / Mode Fill'
        else:
            return '众数填充或删除记录 / Mode Fill or Drop Rows'
    
    # 无缺失
    return '无需处理 / No Action Needed'

# 应用推荐方法
stats_df['recommended_method'] = stats_df.apply(recommend_handling_method, axis=1)

# ============================================================================
# 4. 添加详细说明 / Add Detailed Explanations
# ============================================================================

print("\n4. 添加详细说明 / Adding Detailed Explanations...")

def add_method_explanation(row):
    """
    添加方法说明
    Add method explanation
    """
    method = row['recommended_method']
    col = row['column']
    missing_pct = row['missing_pct']
    is_numeric = row['is_numeric']
    
    explanations = {
        '保留 / Keep (标识字段)': '这是标识字段，必须保留',
        '必须处理 / Must Handle (关键字段)': '关键字段，需要检查缺失原因并处理',
        '删除字段 / Drop Column (完全缺失)': '字段完全缺失，建议删除',
        '删除字段 / Drop Column (缺失率>90%)': '缺失率过高，信息价值低，建议删除',
        '删除字段 / Drop Column (缺失率>70%)': '缺失率很高，建议删除字段',
        '删除字段或创建缺失指示变量 / Drop Column or Create Missing Indicator': '缺失率很高，可删除或创建二值指示变量',
        '创建缺失指示变量 + 中位数/均值填充 / Missing Indicator + Median/Mean Fill': '创建二值变量标识缺失，并用中位数/均值填充',
        '创建缺失指示变量 + 众数填充 / Missing Indicator + Mode Fill': '创建二值变量标识缺失，并用众数填充',
        '中位数/均值填充 + 缺失指示变量 / Median/Mean Fill + Missing Indicator': '用中位数/均值填充，并创建缺失指示变量',
        '前向填充或删除 / Forward Fill or Drop': '时间序列字段，可用前向填充或删除缺失记录',
        '众数填充 + 缺失指示变量 / Mode Fill + Missing Indicator': '用众数填充，并创建缺失指示变量',
        '创建"未知"类别 + 缺失指示变量 / Create "Unknown" Category + Missing Indicator': '创建"未知"类别填充，并创建缺失指示变量',
        '众数填充或创建"未知"类别 / Mode Fill or Create "Unknown" Category': '用众数填充，或创建"未知"类别',
        '中位数/均值填充 / Median/Mean Fill': '用中位数（对异常值稳健）或均值填充',
        '前向填充或插值 / Forward Fill or Interpolation': '时间序列字段，可用前向填充或插值',
        '众数填充 / Mode Fill': '用最常见的值填充',
        '创建"未知"类别 / Create "Unknown" Category': '创建"未知"类别来填充缺失值',
        '众数填充或删除记录 / Mode Fill or Drop Rows': '用众数填充，或直接删除缺失记录',
        '无需处理 / No Action Needed': '字段无缺失值，无需处理'
    }
    
    return explanations.get(method, '请根据具体情况选择处理方法')

stats_df['method_explanation'] = stats_df.apply(add_method_explanation, axis=1)

# ============================================================================
# 5. 添加优先级建议 / Add Priority Recommendations
# ============================================================================

print("\n5. 添加优先级建议 / Adding Priority Recommendations...")

def add_priority(row):
    """
    添加处理优先级
    Add processing priority
    """
    missing_pct = row['missing_pct']
    col = row['column']
    
    # 关键字段高优先级
    if col in ['id', 'data_year']:
        return '高 / High'
    
    # 完全缺失或缺失率极高，低优先级（可能直接删除）
    if missing_pct >= 90:
        return '低 / Low (建议删除)'
    
    # 缺失率中等，中等优先级
    if missing_pct >= 20:
        return '中 / Medium'
    
    # 缺失率较低，高优先级（容易处理）
    if missing_pct > 0:
        return '高 / High'
    
    return '无 / None'

stats_df['priority'] = stats_df.apply(add_priority, axis=1)

# ============================================================================
# 6. 重新排列列顺序 / Reorder Columns
# ============================================================================

column_order = [
    'column',
    'missing_count',
    'missing_pct',
    'non_missing_count',
    'completeness_pct',
    'dtype',
    'is_numeric',
    'is_datetime',
    'is_categorical',
    'unique_count',
    'unique_ratio',
    'priority',
    'recommended_method',
    'method_explanation'
]

stats_df = stats_df[column_order]

# ============================================================================
# 7. 保存结果 / Save Results
# ============================================================================

print("\n7. 保存结果 / Saving Results...")

output_file = charts_dir / 'missing_values_analysis_merged.csv'
stats_df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"  ✅ 分析结果已保存: {output_file}")

# 打印摘要统计
print("\n" + "=" * 80)
print("缺失值分析摘要 / Missing Values Analysis Summary")
print("=" * 80)
print(f"\n总字段数: {len(stats_df)}")
print(f"完全缺失的字段（100%）: {(stats_df['missing_pct'] == 100).sum()}")
print(f"完全完整的字段（0%）: {(stats_df['missing_pct'] == 0).sum()}")
print(f"高缺失率字段（>50%）: {(stats_df['missing_pct'] > 50).sum()}")
print(f"中等缺失率字段（20-50%）: {((stats_df['missing_pct'] >= 20) & (stats_df['missing_pct'] <= 50)).sum()}")
print(f"低缺失率字段（<20%）: {(stats_df['missing_pct'] < 20).sum()}")

print("\n推荐方法分布 / Recommended Method Distribution:")
method_counts = stats_df['recommended_method'].value_counts()
for method, count in method_counts.items():
    print(f"  {method}: {count}")

print("\n优先级分布 / Priority Distribution:")
priority_counts = stats_df['priority'].value_counts()
for priority, count in priority_counts.items():
    print(f"  {priority}: {count}")

print("\n" + "=" * 80)
print("分析完成！/ Analysis Complete!")
print("=" * 80)
print(f"\n输出文件: {output_file}")

