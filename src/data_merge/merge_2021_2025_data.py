"""
合并2021年和2025年的listings数据
Merge 2021 and 2025 listings data
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from datetime import datetime

# 添加 EDA 目录到路径，以便导入 utils 模块 / Add EDA directory to path for importing utils module
sys.path.insert(0, str(Path(__file__).parent.parent / "EDA"))
from utils import get_project_paths

# 获取项目路径
project_root, data_dir, charts_eda_dir, charts_model_dir = get_project_paths()

# 创建输出目录
output_dir = project_root / 'data' / 'merged'
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("合并2021年和2025年数据 / Merge 2021 and 2025 Data")
print("=" * 80)

# ============================================================================
# 1. 加载数据 / Load Data
# ============================================================================

print("\n1. 加载数据 / Loading Data...")

try:
    # 加载2021年数据
    listings_2021 = pd.read_excel(data_dir / 'listings_detailed.xlsx')
    print(f"  ✅ 2021年数据: {len(listings_2021):,} 行 × {len(listings_2021.columns)} 列")
    
    # 加载2025年数据（新下载的CSV文件）
    listings_2025 = pd.read_csv(project_root / 'data' / '2025' / 'listings_detailed.csv')
    print(f"  ✅ 2025年数据: {len(listings_2025):,} 行 × {len(listings_2025.columns)} 列")
except Exception as e:
    print(f"  ❌ 数据加载失败: {e}")
    raise

# ============================================================================
# 2. 字段对比分析 / Column Comparison Analysis
# ============================================================================

print("\n2. 字段对比分析 / Column Comparison Analysis...")

cols_2021 = set(listings_2021.columns)
cols_2025 = set(listings_2025.columns)

common_cols = cols_2021 & cols_2025
only_2021_cols = cols_2021 - cols_2025
only_2025_cols = cols_2025 - cols_2021

print(f"  - 共同字段数: {len(common_cols)}")
print(f"  - 仅在2021年的字段数: {len(only_2021_cols)}")
print(f"  - 仅在2025年的字段数: {len(only_2025_cols)}")

if only_2021_cols:
    print(f"\n  仅在2021年的字段: {sorted(only_2021_cols)}")
if only_2025_cols:
    print(f"\n  仅在2025年的字段: {sorted(only_2025_cols)}")

# ============================================================================
# 3. 添加年份标识 / Add Year Identifier
# ============================================================================

print("\n3. 添加年份标识 / Adding Year Identifier...")

listings_2021['data_year'] = 2021
listings_2025['data_year'] = 2025

print(f"  ✅ 已为2021年数据添加年份标识")
print(f"  ✅ 已为2025年数据添加年份标识")

# ============================================================================
# 4. 字段对齐 / Column Alignment
# ============================================================================

print("\n4. 字段对齐 / Column Alignment...")

# 获取所有字段的并集
all_cols = sorted(list(common_cols | only_2021_cols | only_2025_cols | {'data_year'}))

# 为每个数据集添加缺失的字段（填充NaN）
for col in all_cols:
    if col not in listings_2021.columns:
        listings_2021[col] = np.nan
    if col not in listings_2025.columns:
        listings_2025[col] = np.nan

# 确保列顺序一致
listings_2021 = listings_2021[all_cols]
listings_2025 = listings_2025[all_cols]

print(f"  ✅ 字段已对齐，统一字段数: {len(all_cols)}")

# ============================================================================
# 5. 数据类型统一 / Data Type Unification
# ============================================================================

print("\n5. 数据类型统一 / Data Type Unification...")

# 对于共同字段，尝试统一数据类型
for col in common_cols:
    dtype_2021 = listings_2021[col].dtype
    dtype_2025 = listings_2025[col].dtype
    
    # 如果类型不一致，尝试转换为更通用的类型
    if dtype_2021 != dtype_2025:
        # 如果一个是object，另一个是数值型，保持object
        if dtype_2021 == 'object' or dtype_2025 == 'object':
            continue
        # 如果都是数值型，转换为float64
        elif pd.api.types.is_numeric_dtype(dtype_2021) and pd.api.types.is_numeric_dtype(dtype_2025):
            listings_2021[col] = pd.to_numeric(listings_2021[col], errors='coerce')
            listings_2025[col] = pd.to_numeric(listings_2025[col], errors='coerce')

print(f"  ✅ 数据类型已统一")

# ============================================================================
# 6. 合并数据 / Merge Data
# ============================================================================

print("\n6. 合并数据 / Merging Data...")

merged_data = pd.concat([listings_2021, listings_2025], ignore_index=True)

print(f"  ✅ 合并完成: {len(merged_data):,} 行 × {len(merged_data.columns)} 列")
print(f"  - 2021年记录数: {(merged_data['data_year'] == 2021).sum():,}")
print(f"  - 2025年记录数: {(merged_data['data_year'] == 2025).sum():,}")

# ============================================================================
# 7. 数据验证 / Data Validation
# ============================================================================

print("\n7. 数据验证 / Data Validation...")

# 检查ID重复情况
if 'id' in merged_data.columns:
    total_ids = len(merged_data)
    unique_ids = merged_data['id'].nunique()
    duplicate_ids = total_ids - unique_ids
    
    print(f"  - 总记录数: {total_ids:,}")
    print(f"  - 唯一ID数: {unique_ids:,}")
    print(f"  - 重复ID数: {duplicate_ids:,}")
    
    if duplicate_ids > 0:
        print(f"  ⚠️  发现重复ID，可能是同一房源在不同年份的数据")
        # 显示一些重复ID的示例
        duplicate_id_counts = merged_data['id'].value_counts()
        duplicate_ids_sample = duplicate_id_counts[duplicate_id_counts > 1].head(5)
        print(f"  重复ID示例（前5个）:")
        for id_val, count in duplicate_ids_sample.items():
            print(f"    ID {id_val}: {count} 条记录")

# 检查年份分布
print(f"\n  年份分布:")
year_dist = merged_data['data_year'].value_counts().sort_index()
for year, count in year_dist.items():
    print(f"    {year}年: {count:,} 条记录")

# ============================================================================
# 8. 保存合并后的数据 / Save Merged Data
# ============================================================================

print("\n8. 保存合并后的数据 / Saving Merged Data...")

output_file = output_dir / 'listings_merged_2021_2025.xlsx'

# 保存为Excel格式
merged_data.to_excel(output_file, index=False, engine='openpyxl')
print(f"  ✅ 已保存到: {output_file}")

# 同时保存为CSV格式（便于查看）
output_csv_file = output_dir / 'listings_merged_2021_2025.csv'
merged_data.to_csv(output_csv_file, index=False, encoding='utf-8-sig')
print(f"  ✅ 已保存到: {output_csv_file}")

# ============================================================================
# 9. 生成合并报告 / Generate Merge Report
# ============================================================================

print("\n9. 生成合并报告 / Generating Merge Report...")

report_lines = []
report_lines.append("=" * 80)
report_lines.append("Data Merge Report / 数据合并报告")
report_lines.append("=" * 80)
report_lines.append(f"\n合并时间 / Merge Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
report_lines.append("")

report_lines.append("数据源信息 / Data Source Information:")
report_lines.append(f"  - 2021年数据源: listings_detailed.xlsx")
report_lines.append(f"    - 记录数: {len(listings_2021):,}")
report_lines.append(f"    - 字段数: {len(listings_2021.columns)}")
report_lines.append(f"  - 2025年数据源: data/2025/listings_detailed.csv")
report_lines.append(f"    - 记录数: {len(listings_2025):,}")
report_lines.append(f"    - 字段数: {len(listings_2025.columns)}")
report_lines.append("")

report_lines.append("字段对比 / Column Comparison:")
report_lines.append(f"  - 共同字段数: {len(common_cols)}")
report_lines.append(f"  - 仅在2021年的字段数: {len(only_2021_cols)}")
report_lines.append(f"  - 仅在2025年的字段数: {len(only_2025_cols)}")
report_lines.append(f"  - 合并后总字段数: {len(merged_data.columns)}")
report_lines.append("")

if only_2021_cols:
    report_lines.append("仅在2021年的字段 / Columns Only in 2021:")
    for col in sorted(only_2021_cols):
        report_lines.append(f"  - {col}")
    report_lines.append("")

if only_2025_cols:
    report_lines.append("仅在2025年的字段 / Columns Only in 2025:")
    for col in sorted(only_2025_cols):
        report_lines.append(f"  - {col}")
    report_lines.append("")

report_lines.append("合并结果 / Merge Results:")
report_lines.append(f"  - 总记录数: {len(merged_data):,}")
report_lines.append(f"  - 总字段数: {len(merged_data.columns)}")
report_lines.append(f"  - 2021年记录数: {(merged_data['data_year'] == 2021).sum():,}")
report_lines.append(f"  - 2025年记录数: {(merged_data['data_year'] == 2025).sum():,}")
report_lines.append("")

if 'id' in merged_data.columns:
    report_lines.append("ID统计 / ID Statistics:")
    report_lines.append(f"  - 总记录数: {len(merged_data):,}")
    report_lines.append(f"  - 唯一ID数: {merged_data['id'].nunique():,}")
    report_lines.append(f"  - 重复ID数: {len(merged_data) - merged_data['id'].nunique():,}")
    report_lines.append("")

report_lines.append("输出文件 / Output Files:")
report_lines.append(f"  - Excel: {output_file}")
report_lines.append(f"  - CSV: {output_csv_file}")
report_lines.append("")
report_lines.append("=" * 80)

report_file = output_dir / 'merge_report.txt'
with open(report_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))

print(f"  ✅ 合并报告已保存: {report_file}")

print("\n" + "=" * 80)
print("数据合并完成！/ Data Merge Complete!")
print("=" * 80)
print(f"\n合并后的数据文件:")
print(f"  - Excel: {output_file}")
print(f"  - CSV: {output_csv_file}")
print(f"  - 报告: {report_file}")
