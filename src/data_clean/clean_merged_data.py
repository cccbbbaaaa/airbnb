"""
基于缺失值处理策略清洗合并后的数据
Clean Merged Data Based on Missing Value Handling Strategy
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime

# 添加 EDA 目录到路径，以便导入 utils 模块 / Add EDA directory to path for importing utils module
sys.path.insert(0, str(Path(__file__).parent.parent / "EDA"))
from utils import get_project_paths

# 获取项目路径
project_root, data_dir, charts_eda_dir, charts_model_dir = get_project_paths()

# 创建输出目录
output_dir = project_root / 'data' / 'cleaned'
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("数据清洗 / Data Cleaning")
print("=" * 80)

# ============================================================================
# 1. 加载数据和策略 / Load Data and Strategy
# ============================================================================

print("\n1. 加载数据和策略 / Loading Data and Strategy...")

try:
    # 加载合并后的数据
    merged_data = pd.read_excel(project_root / 'data' / 'merged' / 'listings_merged_2021_2025.xlsx')
    print(f"  ✅ 合并后数据: {len(merged_data):,} 行 × {len(merged_data.columns)} 列")
    
    # 加载缺失值处理策略
    strategy_df = pd.read_csv(project_root / 'charts' / 'data_merge' / 'missing_values_analysis_merged.csv')
    print(f"  ✅ 处理策略: {len(strategy_df)} 个字段")
except Exception as e:
    print(f"  ❌ 加载失败: {e}")
    raise

# ============================================================================
# 2. 创建清洗后的数据副本 / Create Cleaned Data Copy
# ============================================================================

print("\n2. 创建清洗后的数据副本 / Creating Cleaned Data Copy...")

cleaned_data = merged_data.copy()
original_shape = cleaned_data.shape
print(f"  ✅ 原始数据形状: {original_shape}")

# 记录处理日志
processing_log = []

# ============================================================================
# 3. 处理缺失值 / Handle Missing Values
# ============================================================================

print("\n3. 处理缺失值 / Handling Missing Values...")

def create_missing_indicator(df, col):
    """创建缺失指示变量"""
    indicator_col = f'{col}_is_missing'
    df[indicator_col] = df[col].isna().astype(int)
    return df, indicator_col

def fill_with_median(df, col):
    """用中位数填充数值型字段"""
    median_val = df[col].median()
    if pd.isna(median_val):
        # 如果中位数也是NaN，尝试用均值
        mean_val = df[col].mean()
        if pd.isna(mean_val):
            df[col] = df[col].fillna(0)
            return df, 0
        else:
            df[col] = df[col].fillna(mean_val)
            return df, mean_val
    else:
        df[col] = df[col].fillna(median_val)
        return df, median_val

def fill_with_mode(df, col):
    """用众数填充分类字段"""
    mode_val = df[col].mode()
    if len(mode_val) > 0:
        fill_value = mode_val[0]
        df[col] = df[col].fillna(fill_value)
        return df, fill_value
    else:
        # 如果没有众数，创建"未知"类别
        df[col] = df[col].fillna('Unknown')
        return df, 'Unknown'

def fill_with_unknown(df, col):
    """用"未知"类别填充"""
    df[col] = df[col].fillna('Unknown')
    return df, 'Unknown'

# 按优先级处理字段
strategy_df_sorted = strategy_df.sort_values('priority', ascending=False)

for idx, row in strategy_df_sorted.iterrows():
    col = row['column']
    method = row['recommended_method']
    missing_pct = row['missing_pct']
    is_numeric = row['is_numeric']
    is_categorical = row['is_categorical']
    
    # 跳过不存在的字段
    if col not in cleaned_data.columns:
        processing_log.append({
            'column': col,
            'action': '跳过 / Skipped',
            'reason': '字段不存在 / Column does not exist'
        })
        continue
    
    # 检查是否有缺失值
    if missing_pct == 0:
        processing_log.append({
            'column': col,
            'action': '无需处理 / No Action',
            'reason': '无缺失值 / No missing values'
        })
        continue
    
    # 根据推荐方法处理
    try:
        if '删除字段' in method or 'Drop Column' in method:
            # 删除字段
            cleaned_data = cleaned_data.drop(columns=[col])
            processing_log.append({
                'column': col,
                'action': '删除字段 / Drop Column',
                'reason': method,
                'missing_pct': missing_pct
            })
            print(f"  ✅ 删除字段: {col} (缺失率: {missing_pct:.2f}%)")
        
        elif '保留' in method or 'Keep' in method:
            # 保留字段，不处理
            processing_log.append({
                'column': col,
                'action': '保留 / Keep',
                'reason': method
            })
        
        elif '必须处理' in method or 'Must Handle' in method:
            # 关键字段，检查是否有缺失
            if cleaned_data[col].isna().any():
                # 对于id字段，不能有缺失值
                if col == 'id':
                    # 删除缺失id的记录
                    before_count = len(cleaned_data)
                    cleaned_data = cleaned_data.dropna(subset=[col])
                    after_count = len(cleaned_data)
                    processing_log.append({
                        'column': col,
                        'action': '删除缺失记录 / Drop Rows with Missing Values',
                        'reason': '关键字段不能缺失 / Key field cannot be missing',
                        'dropped_rows': before_count - after_count
                    })
                    print(f"  ⚠️  删除缺失{col}的记录: {before_count - after_count} 行")
        
        elif '创建缺失指示变量' in method or 'Missing Indicator' in method:
            # 创建缺失指示变量
            cleaned_data, indicator_col = create_missing_indicator(cleaned_data, col)
            
            # 然后填充缺失值
            if '中位数' in method or 'Median' in method or '均值' in method or 'Mean' in method:
                cleaned_data, fill_value = fill_with_median(cleaned_data, col)
                processing_log.append({
                    'column': col,
                    'action': '创建缺失指示变量 + 中位数填充 / Missing Indicator + Median Fill',
                    'indicator_column': indicator_col,
                    'fill_value': fill_value,
                    'missing_pct': missing_pct
                })
                print(f"  ✅ {col}: 创建缺失指示变量 + 中位数填充 ({fill_value:.2f})")
            
            elif '众数' in method or 'Mode' in method:
                cleaned_data, fill_value = fill_with_mode(cleaned_data, col)
                processing_log.append({
                    'column': col,
                    'action': '创建缺失指示变量 + 众数填充 / Missing Indicator + Mode Fill',
                    'indicator_column': indicator_col,
                    'fill_value': fill_value,
                    'missing_pct': missing_pct
                })
                print(f"  ✅ {col}: 创建缺失指示变量 + 众数填充 ({fill_value})")
            
            elif '未知' in method or 'Unknown' in method:
                cleaned_data, fill_value = fill_with_unknown(cleaned_data, col)
                processing_log.append({
                    'column': col,
                    'action': '创建缺失指示变量 + 未知类别填充 / Missing Indicator + Unknown Fill',
                    'indicator_column': indicator_col,
                    'fill_value': fill_value,
                    'missing_pct': missing_pct
                })
                print(f"  ✅ {col}: 创建缺失指示变量 + 未知类别填充")
        
        elif '中位数' in method or 'Median' in method or '均值' in method or 'Mean' in method:
            # 中位数/均值填充
            cleaned_data, fill_value = fill_with_median(cleaned_data, col)
            processing_log.append({
                'column': col,
                'action': '中位数/均值填充 / Median/Mean Fill',
                'fill_value': fill_value,
                'missing_pct': missing_pct
            })
            print(f"  ✅ {col}: 中位数填充 ({fill_value:.2f})")
        
        elif '众数' in method or 'Mode' in method:
            # 众数填充
            cleaned_data, fill_value = fill_with_mode(cleaned_data, col)
            processing_log.append({
                'column': col,
                'action': '众数填充 / Mode Fill',
                'fill_value': fill_value,
                'missing_pct': missing_pct
            })
            print(f"  ✅ {col}: 众数填充 ({fill_value})")
        
        elif '未知' in method or 'Unknown' in method:
            # 创建"未知"类别
            cleaned_data, fill_value = fill_with_unknown(cleaned_data, col)
            processing_log.append({
                'column': col,
                'action': '创建未知类别 / Create Unknown Category',
                'fill_value': fill_value,
                'missing_pct': missing_pct
            })
            print(f"  ✅ {col}: 创建未知类别")
        
        elif '前向填充' in method or 'Forward Fill' in method:
            # 前向填充
            cleaned_data[col] = cleaned_data[col].fillna(method='ffill')
            processing_log.append({
                'column': col,
                'action': '前向填充 / Forward Fill',
                'missing_pct': missing_pct
            })
            print(f"  ✅ {col}: 前向填充")
        
        elif '删除记录' in method or 'Drop Rows' in method:
            # 删除缺失记录
            before_count = len(cleaned_data)
            cleaned_data = cleaned_data.dropna(subset=[col])
            after_count = len(cleaned_data)
            processing_log.append({
                'column': col,
                'action': '删除缺失记录 / Drop Rows',
                'dropped_rows': before_count - after_count,
                'missing_pct': missing_pct
            })
            print(f"  ⚠️  {col}: 删除缺失记录 ({before_count - after_count} 行)")
        
        else:
            # 默认处理：根据字段类型
            if is_numeric:
                cleaned_data, fill_value = fill_with_median(cleaned_data, col)
                processing_log.append({
                    'column': col,
                    'action': '默认处理：中位数填充 / Default: Median Fill',
                    'fill_value': fill_value,
                    'missing_pct': missing_pct
                })
                print(f"  ✅ {col}: 默认处理 - 中位数填充 ({fill_value:.2f})")
            else:
                cleaned_data, fill_value = fill_with_mode(cleaned_data, col)
                processing_log.append({
                    'column': col,
                    'action': '默认处理：众数填充 / Default: Mode Fill',
                    'fill_value': fill_value,
                    'missing_pct': missing_pct
                })
                print(f"  ✅ {col}: 默认处理 - 众数填充 ({fill_value})")
    
    except Exception as e:
        processing_log.append({
            'column': col,
            'action': '处理失败 / Processing Failed',
            'error': str(e),
            'missing_pct': missing_pct
        })
        print(f"  ❌ {col}: 处理失败 - {e}")

# ============================================================================
# 4. 验证清洗结果 / Validate Cleaning Results
# ============================================================================

print("\n4. 验证清洗结果 / Validating Cleaning Results...")

final_shape = cleaned_data.shape
print(f"  ✅ 清洗后数据形状: {final_shape}")
print(f"  - 行数变化: {original_shape[0]} → {final_shape[0]} ({final_shape[0] - original_shape[0]:+d})")
print(f"  - 列数变化: {original_shape[1]} → {final_shape[1]} ({final_shape[1] - original_shape[1]:+d})")

# 检查剩余缺失值
remaining_missing = cleaned_data.isna().sum()
cols_with_missing = remaining_missing[remaining_missing > 0]

if len(cols_with_missing) > 0:
    print(f"\n  ⚠️  仍有缺失值的字段 ({len(cols_with_missing)} 个):")
    for col, count in cols_with_missing.items():
        pct = count / len(cleaned_data) * 100
        print(f"    {col}: {count} ({pct:.2f}%)")
else:
    print(f"\n  ✅ 所有字段已无缺失值")

# ============================================================================
# 5. 保存清洗后的数据 / Save Cleaned Data
# ============================================================================

print("\n5. 保存清洗后的数据 / Saving Cleaned Data...")

output_file = output_dir / 'listings_cleaned.csv'
cleaned_data.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"  ✅ 清洗后的数据已保存: {output_file}")

# ============================================================================
# 6. 保存处理日志 / Save Processing Log
# ============================================================================

print("\n6. 保存处理日志 / Saving Processing Log...")

log_df = pd.DataFrame(processing_log)
log_file = output_dir / 'cleaning_log.csv'
log_df.to_csv(log_file, index=False, encoding='utf-8-sig')
print(f"  ✅ 处理日志已保存: {log_file}")

# ============================================================================
# 7. 生成清洗报告 / Generate Cleaning Report
# ============================================================================

print("\n7. 生成清洗报告 / Generating Cleaning Report...")

report_lines = []
report_lines.append("=" * 80)
report_lines.append("Data Cleaning Report / 数据清洗报告")
report_lines.append("=" * 80)
report_lines.append(f"\n清洗时间 / Cleaning Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
report_lines.append("")

report_lines.append("数据概览 / Data Overview:")
report_lines.append(f"  - 原始数据: {original_shape[0]:,} 行 × {original_shape[1]} 列")
report_lines.append(f"  - 清洗后数据: {final_shape[0]:,} 行 × {final_shape[1]} 列")
report_lines.append(f"  - 删除行数: {original_shape[0] - final_shape[0]:,}")
report_lines.append(f"  - 删除列数: {original_shape[1] - final_shape[1]}")
report_lines.append("")

report_lines.append("处理统计 / Processing Statistics:")
action_counts = log_df['action'].value_counts()
for action, count in action_counts.items():
    report_lines.append(f"  - {action}: {count}")
report_lines.append("")

if len(cols_with_missing) > 0:
    report_lines.append("剩余缺失值 / Remaining Missing Values:")
    for col, count in cols_with_missing.items():
        pct = count / len(cleaned_data) * 100
        report_lines.append(f"  - {col}: {count} ({pct:.2f}%)")
    report_lines.append("")
else:
    report_lines.append("剩余缺失值 / Remaining Missing Values: 无 / None")
    report_lines.append("")

report_lines.append("输出文件 / Output Files:")
report_lines.append(f"  - 清洗后的数据: {output_file}")
report_lines.append(f"  - 处理日志: {log_file}")
report_lines.append("")
report_lines.append("=" * 80)

report_file = output_dir / 'cleaning_report.txt'
with open(report_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))
print(f"  ✅ 清洗报告已保存: {report_file}")

print("\n" + "=" * 80)
print("数据清洗完成！/ Data Cleaning Complete!")
print("=" * 80)
print(f"\n输出文件 / Output Files:")
print(f"  1. 清洗后的数据: {output_file}")
print(f"  2. 处理日志: {log_file}")
print(f"  3. 清洗报告: {report_file}")

