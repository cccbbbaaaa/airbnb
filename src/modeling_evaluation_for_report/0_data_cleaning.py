"""
基于缺失值处理策略清洗合并后的数据
Clean Merged Data Based on Missing Value Handling Strategy
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime

# 获取项目路径 / Get project paths
project_root = Path(__file__).resolve().parent.parent.parent
data_dir = project_root / 'data'
charts_dir = project_root / 'charts'

# 创建输出目录 / Create output directory
output_dir = data_dir / 'cleaned'
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("数据清洗 / Data Cleaning")
print("=" * 80)

# ============================================================================
# 1. 加载数据 / Load Data
# ============================================================================

print("\n1. 加载合并后数据 / Loading merged data...")

try:
    # 加载合并后的数据（2021 + 2025）
    merged_data = pd.read_excel(
        project_root / 'data' / 'merged' / 'listings_merged_2021_2025.xlsx'
    )
    print(f"  ✅ 合并后数据: {len(merged_data):,} 行 × {len(merged_data.columns)} 列")
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

"""
3. 处理缺失值/ Handle Missing Values
"""

print("\n3. 处理缺失值 / Handling Missing Values...")


def create_missing_indicator(df, col):
    """创建缺失指示变量 / Create missing indicator column"""
    indicator_col = f"{col}_is_missing"
    df[indicator_col] = df[col].isna().astype(int)
    return df, indicator_col


def fill_with_median(df, col):
    """用中位数优先、均值兜底、最后 0 填充数值型字段
    Prefer median, fall back to mean, then 0 for numeric fields.
    """
    median_val = df[col].median()
    if pd.isna(median_val):
        mean_val = df[col].mean()
        if pd.isna(mean_val):
            df[col] = df[col].fillna(0)
            return df, 0
        df[col] = df[col].fillna(mean_val)
        return df, mean_val
    df[col] = df[col].fillna(median_val)
    return df, median_val


def fill_with_mode(df, col):
    """用众数填充分类字段（无众数时使用 'Unknown'）
    Fill categorical field with mode; if no mode, use 'Unknown'.
    """
    mode_val = df[col].mode()
    if len(mode_val) > 0:
        fill_value = mode_val[0]
        df[col] = df[col].fillna(fill_value)
        return df, fill_value
    df[col] = df[col].fillna("Unknown")
    return df, "Unknown"


def fill_with_unknown(df, col):
    """用 'Unknown' 填充缺失文本字段 / Fill missing text with 'Unknown'"""
    df[col] = df[col].fillna("Unknown")
    return df, "Unknown"


# ------------------------- 3.1 删除完全缺失字段 -----------------------------
# Drop columns that are completely missing (100% NaN)
cols_to_drop = [
    "calendar_updated",
    "neighbourhood_group_cleansed",
]

for col in cols_to_drop:
    if col in cleaned_data.columns:
        cleaned_data = cleaned_data.drop(columns=[col])
        processing_log.append(
            {
                "column": col,
                "action": "删除字段 / Drop Column",
                "reason": "字段完全缺失 / Column fully missing",
            }
        )
        print(f"  ✅ 删除字段: {col} (完全缺失 / fully missing)")


# ---------------------- 3.2 关键主键字段 / Key Fields ----------------------
# id: 不能缺失，如有缺失则删除整行
if "id" in cleaned_data.columns and cleaned_data["id"].isna().any():
    before_count = len(cleaned_data)
    cleaned_data = cleaned_data.dropna(subset=["id"])
    after_count = len(cleaned_data)
    dropped = before_count - after_count
    processing_log.append(
        {
            "column": "id",
            "action": "删除缺失记录 / Drop Rows with Missing Values",
            "reason": "关键字段不能缺失 / Key field cannot be missing",
            "dropped_rows": dropped,
        }
    )
    print(f"  ⚠️  删除缺失 id 的记录: {dropped} 行 / rows")


# ---------------------- 3.3 创建缺失指示变量 + 填充 ------------------------
# 3.3.1 高缺失率数值型字段：指示变量 + 中位数/均值填充
indicator_median_cols = [
    "availability_eoy",
    "estimated_occupancy_l365d",
    "number_of_reviews_ly",
]

for col in indicator_median_cols:
    if col in cleaned_data.columns:
        cleaned_data, indicator_col = create_missing_indicator(cleaned_data, col)
        cleaned_data, fill_value = fill_with_median(cleaned_data, col)
        processing_log.append(
            {
                "column": col,
                "action": "创建缺失指示变量 + 中位数/均值填充 / Missing Indicator + Median/Mean Fill",
                "indicator_column": indicator_col,
                "fill_value": fill_value,
            }
        )
        print(
            f"  ✅ {col}: 指示变量 {indicator_col} + 中位数/均值填充 ({fill_value})"
        )


# 3.3.2 高缺失率分类型字段：指示变量 + 众数填充
indicator_mode_cols = [
    "source",
    "host_response_time",
    "host_response_rate",
    "host_neighbourhood",
    "host_acceptance_rate",
    "neighbourhood",
]

for col in indicator_mode_cols:
    if col in cleaned_data.columns:
        cleaned_data, indicator_col = create_missing_indicator(cleaned_data, col)
        cleaned_data, fill_value = fill_with_mode(cleaned_data, col)
        processing_log.append(
            {
                "column": col,
                "action": "创建缺失指示变量 + 众数填充 / Missing Indicator + Mode Fill",
                "indicator_column": indicator_col,
                "fill_value": fill_value,
            }
        )
        print(
            f"  ✅ {col}: 指示变量 {indicator_col} + 众数填充 ({fill_value})"
        )


# 3.3.3 高缺失率文本型字段：指示变量 + 'Unknown' 填充
indicator_unknown_cols = [
    "host_about",
    "license",
    "neighborhood_overview",
]

for col in indicator_unknown_cols:
    if col in cleaned_data.columns:
        cleaned_data, indicator_col = create_missing_indicator(cleaned_data, col)
        cleaned_data, fill_value = fill_with_unknown(cleaned_data, col)
        processing_log.append(
            {
                "column": col,
                "action": "创建缺失指示变量 + 'Unknown' 填充 / Missing Indicator + Unknown Fill",
                "indicator_column": indicator_col,
                "fill_value": fill_value,
            }
        )
        print(f"  ✅ {col}: 指示变量 {indicator_col} + Unknown 填充")


# ---------------------- 3.4 仅填充（不建指示变量）-------------------------
# 3.4.1 数值评分与数值字段：中位数/均值填充
median_fill_numeric_cols = [
    "estimated_revenue_l365d",
    "bathrooms",
    "beds",
    "review_scores_checkin",
    "review_scores_location",
    "review_scores_value",
    "review_scores_communication",
    "review_scores_accuracy",
    "review_scores_cleanliness",
    "review_scores_rating",
    "reviews_per_month",
    "bedrooms",
    "host_total_listings_count",
    "host_listings_count",
    "minimum_minimum_nights",
    "maximum_maximum_nights",
    "maximum_minimum_nights",
    "minimum_maximum_nights",
    "minimum_nights_avg_ntm",
    "maximum_nights_avg_ntm",
]

for col in median_fill_numeric_cols:
    if col in cleaned_data.columns and cleaned_data[col].isna().any():
        cleaned_data, fill_value = fill_with_median(cleaned_data, col)
        processing_log.append(
            {
                "column": col,
                "action": "中位数/均值填充 / Median/Mean Fill",
                "fill_value": fill_value,
            }
        )
        print(f"  ✅ {col}: 中位数/均值填充 ({fill_value})")


# 3.4.2 低缺失率分类型/文本字段：众数填充
mode_fill_categorical_cols = [
    "price",
    "host_location",
    "description",
    "has_availability",
    "host_is_superhost",
    "bathrooms_text",
    "name",
    "host_thumbnail_url",
    "host_verifications",
    "host_since",
    "host_has_profile_pic",
    "host_picture_url",
    "host_identity_verified",
    "host_name",
]

for col in mode_fill_categorical_cols:
    if col in cleaned_data.columns and cleaned_data[col].isna().any():
        cleaned_data, fill_value = fill_with_mode(cleaned_data, col)
        processing_log.append(
            {
                "column": col,
                "action": "众数填充 / Mode Fill",
                "fill_value": fill_value,
            }
        )
        print(f"  ✅ {col}: 众数填充 ({fill_value})")


# ---------------------- 3.5 其余字段：仅记录为“无需处理” -------------------
for col in cleaned_data.columns:
    if cleaned_data[col].isna().sum() == 0:
        processing_log.append(
            {
                "column": col,
                "action": "无需处理 / No Action",
                "reason": "无缺失值 / No missing values",
            }
        )

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

