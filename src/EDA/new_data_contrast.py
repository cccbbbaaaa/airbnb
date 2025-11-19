"""
比较 listings_detailed 和 listings_detailed_2 的 id
Compare IDs between listings_detailed and listings_detailed_2
"""

import pandas as pd
from pathlib import Path
from utils import get_project_paths

# 获取项目路径
project_root, data_dir, charts_dir = get_project_paths()

print("=" * 80)
print("比较 listings_detailed 和 listings_detailed_2 的 ID")
print("Compare IDs between listings_detailed and listings_detailed_2")
print("=" * 80)

# 加载两个文件
print("\n正在加载数据文件...")
file1_path = data_dir / 'listings_detailed.xlsx'
file2_path = data_dir / 'listings_detailed_2.xlsx'

df1 = pd.read_excel(file1_path)
df2 = pd.read_excel(file2_path)

print(f"listings_detailed.xlsx: {len(df1)} 条记录")
print(f"listings_detailed_2.xlsx: {len(df2)} 条记录")

# 检查 id 列是否存在
if 'id' not in df1.columns:
    print("\n警告: listings_detailed.xlsx 中没有找到 'id' 列")
    print(f"可用列: {list(df1.columns)}")
    # 尝试查找可能的 id 列名
    possible_id_cols = [col for col in df1.columns if 'id' in col.lower()]
    if possible_id_cols:
        print(f"可能的 id 列: {possible_id_cols}")
        id_col1 = possible_id_cols[0]
    else:
        raise ValueError("无法找到 id 列")
else:
    id_col1 = 'id'

if 'id' not in df2.columns:
    print("\n警告: listings_detailed_2.xlsx 中没有找到 'id' 列")
    print(f"可用列: {list(df2.columns)}")
    # 尝试查找可能的 id 列名
    possible_id_cols = [col for col in df2.columns if 'id' in col.lower()]
    if possible_id_cols:
        print(f"可能的 id 列: {possible_id_cols}")
        id_col2 = possible_id_cols[0]
    else:
        raise ValueError("无法找到 id 列")
else:
    id_col2 = 'id'

# 提取 id 列并转换为集合
ids1 = set(df1[id_col1].dropna().astype(str))
ids2 = set(df2[id_col2].dropna().astype(str))

# 计算统计信息
common_ids = ids1 & ids2  # 交集：两个文件都有的 id
only_in_file1 = ids1 - ids2  # 只在文件1中的 id
only_in_file2 = ids2 - ids1  # 只在文件2中的 id
all_unique_ids = ids1 | ids2  # 并集：所有唯一的 id

# 输出结果
print("\n" + "=" * 80)
print("统计结果 / Statistics")
print("=" * 80)
print(f"\nlistings_detailed.xlsx 中的 ID 数量: {len(ids1):,}")
print(f"listings_detailed_2.xlsx 中的 ID 数量: {len(ids2):,}")
print(f"\n两个文件相同的 ID 数量: {len(common_ids):,}")
print(f"只在 listings_detailed.xlsx 中的 ID 数量: {len(only_in_file1):,}")
print(f"只在 listings_detailed_2.xlsx 中的 ID 数量: {len(only_in_file2):,}")
print(f"所有唯一的 ID 总数: {len(all_unique_ids):,}")

# 计算百分比
if len(ids1) > 0:
    common_pct1 = len(common_ids) / len(ids1) * 100
    only_pct1 = len(only_in_file1) / len(ids1) * 100
    print(f"\n相对于 listings_detailed.xlsx:")
    print(f"  相同 ID 占比: {common_pct1:.2f}%")
    print(f"  仅在文件1中的 ID 占比: {only_pct1:.2f}%")

if len(ids2) > 0:
    common_pct2 = len(common_ids) / len(ids2) * 100
    only_pct2 = len(only_in_file2) / len(ids2) * 100
    print(f"\n相对于 listings_detailed_2.xlsx:")
    print(f"  相同 ID 占比: {common_pct2:.2f}%")
    print(f"  仅在文件2中的 ID 占比: {only_pct2:.2f}%")

# 显示一些示例 ID（如果有不同的）
if only_in_file1:
    print(f"\n只在 listings_detailed.xlsx 中的 ID 示例（前10个）:")
    print(list(only_in_file1)[:10])
    
if only_in_file2:
    print(f"\n只在 listings_detailed_2.xlsx 中的 ID 示例（前10个）:")
    print(list(only_in_file2)[:10])

print("\n" + "=" * 80)
print("比较完成！")
print("=" * 80)

