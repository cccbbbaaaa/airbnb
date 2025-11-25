# EDA for Report / 报告用探索性数据分析

## Overview / 概述

This notebook generates visualizations and insights specifically for the **Data Understanding** section of the project report. It provides a concise, focused analysis suitable for a 2-3 page report section.

本notebook专门为项目报告的**数据理解**部分生成可视化和洞察。它提供了适合2-3页报告章节的简洁、聚焦的分析。

---

## Purpose / 目的

The notebook covers three main sections:
本notebook涵盖三个主要部分：

1. **Dataset Overview** (数据集概述)
   - Data source and collection period
   - Dataset size and structure
   - Temporal distribution (2021 vs 2025)
   - Data quality summary

2. **Business Understanding** (业务理解)
   - Business problem definition
   - Target variable explanation (`review_scores_rating > 4.95`)
   - Rating distribution comparison (2021 vs 2025)
   - Classification rationale

3. **Key Features Analysis** (关键特征分析)
   - Pricing features
   - Host features (Superhost status)
   - Property features (room type)
   - Geographic features
   - Reviews and availability

---

## Generated Visualizations / 生成的可视化图表

The notebook generates **10 high-quality charts** saved to `charts/charts_for_report/`:

本notebook生成**10个高质量图表**，保存到 `charts/charts_for_report/`：

| # | Chart Name | Description |
|---|------------|-------------|
| 1 | `1_temporal_distribution.png` | Year distribution (2021 vs 2025) / 年份分布 |
| 2 | `2_feature_categories.png` | Features by category / 按类别统计的特征 |
| 3 | `3_missing_values_key_features.png` | Missing values in key features / 关键特征缺失值 |
| 4 | `4_rating_distribution_2021_vs_2025.png` | Rating distribution comparison / 评分分布对比 |
| 5 | `5_target_variable_distribution.png` | High vs low rating distribution / 高低评分分布 |
| 6 | `6_price_vs_rating.png` | Price vs rating relationship / 价格与评分关系 |
| 7 | `7_superhost_vs_rating.png` | Superhost impact on ratings / 超级房东对评分的影响 |
| 8 | `8_room_type_analysis.png` | Room type analysis / 房型分析 |
| 9 | `9_geographic_distribution.png` | Geographic distribution / 地理分布 |
| 10 | `10_reviews_analysis.png` | Review count analysis / 评论数分析 |

---

## Prerequisites / 前提条件

### Required Data Files / 需要的数据文件

The notebook requires one of the following data files:
本notebook需要以下数据文件之一：

- **Primary** (首选): `data/cleaned/listings_cleaned.csv` (cleaned and merged data / 清洗和合并后的数据)
- **Fallback** (备选): `data/merged/listings_merged_2021_2025.csv` (merged data / 合并数据)

### Required Columns / 需要的列

Essential columns for full functionality:
完整功能所需的核心列：

- `review_scores_rating` - Target variable / 目标变量
- `data_year` - For temporal comparison / 用于时间对比
- `price` - Pricing analysis / 价格分析
- `host_is_superhost` - Host features / 主机特征
- `room_type` - Property features / 房源特征
- `latitude`, `longitude` - Geographic analysis / 地理分析
- `number_of_reviews` - Review analysis / 评论分析

---

## Usage / 使用方法

### Option 1: Jupyter Notebook (Recommended)

```bash
# Navigate to the directory / 进入目录
cd src/dataset_overview_for_report

# Launch Jupyter Notebook / 启动Jupyter Notebook
jupyter notebook EDA_for_report.ipynb
```

Then **Run All Cells** (Cell → Run All)
然后**运行所有单元格**（单元格 → 运行全部）

### Option 2: Jupyter Lab

```bash
jupyter lab EDA_for_report.ipynb
```

### Option 3: VS Code

1. Open `EDA_for_report.ipynb` in VS Code
2. Ensure Jupyter extension is installed
3. Click "Run All" button at the top

---

## Output / 输出

### Charts / 图表

All charts are saved to:
所有图表保存到：

```
charts/charts_for_report/
├── 1_temporal_distribution.png
├── 2_feature_categories.png
├── 3_missing_values_key_features.png
├── 4_rating_distribution_2021_vs_2025.png
├── 5_target_variable_distribution.png
├── 6_price_vs_rating.png
├── 7_superhost_vs_rating.png
├── 8_room_type_analysis.png
├── 9_geographic_distribution.png
└── 10_reviews_analysis.png
```

**Format**: PNG, 300 DPI (high-resolution for reports)
**格式**: PNG, 300 DPI（适合报告的高分辨率）

### Console Output / 控制台输出

The notebook also prints:
notebook还会打印：

- Dataset statistics / 数据集统计
- Feature category breakdown / 特征类别分解
- Data quality metrics / 数据质量指标
- Key findings summary / 关键发现摘要

---

## Key Findings / 关键发现

The analysis reveals:
分析揭示了：

1. **Dataset Characteristics** / 数据集特征
   - Combined 2021 and 2025 data for temporal comparison
   - Rich feature set across multiple categories

2. **Target Variable** / 目标变量
   - Threshold: 4.95 for high-rating classification
   - Based on industry standards and natural distribution

3. **Rating Distribution** / 评分分布
   - Significant shift between 2021 and 2025
   - 2025 shows higher concentration in high-rating range

4. **Feature Insights** / 特征洞察
   - Superhost status strongly correlates with high ratings
   - Price has non-linear relationship with ratings
   - Room type and location influence ratings
   - Review count patterns differ by rating category

---

## Customization / 自定义

### Modify Classification Threshold / 修改分类阈值

To change the high-rating threshold (default: 4.95):
要更改高评分阈值（默认：4.95）：

```python
# In cell under "Section 2.4"
THRESHOLD = 4.95  # Change this value / 修改此值
```

### Adjust Visualizations / 调整可视化

All visualizations use:
所有可视化使用：

- **Figure size**: 300 DPI for reports
- **Color scheme**: Red (#e74c3c) for low, Green (#2ecc71) for high
- **Style**: Seaborn "husl" palette

Modify these in the setup cell if needed.
如需修改，在设置单元格中调整。

---

## Troubleshooting / 故障排除

### Issue 1: Data file not found / 问题1：找不到数据文件

**Error**: `FileNotFoundError: No data file found!`

**Solution** / 解决方案:
1. Run data merging first: `python src/data_merge/merge_2021_2025_data.py`
2. Or run data cleaning: `python src/data_clean/clean_merged_data.py`

### Issue 2: Missing columns / 问题2：缺少列

**Error**: Column 'xxx' not found

**Solution** / 解决方案:
The notebook will skip visualizations for missing columns. Ensure you have the complete dataset with all required columns.

### Issue 3: Import errors / 问题3：导入错误

**Error**: `ModuleNotFoundError: No module named 'xxx'`

**Solution** / 解决方案:
```bash
pip install -r requirements.txt
```

---

## For Report Writing / 报告撰写建议

### Recommended Charts for Report / 报告推荐使用的图表

For a 2-3 page section, select **6-8 charts**:
对于2-3页的章节，选择**6-8个图表**：

**Essential** (必选):
1. `4_rating_distribution_2021_vs_2025.png` - Shows business context / 展示业务背景
2. `5_target_variable_distribution.png` - Explains target variable / 解释目标变量
3. `6_price_vs_rating.png` - Key relationship / 关键关系

**High Impact** (高影响力):
4. `7_superhost_vs_rating.png` - Clear pattern / 清晰模式
5. `9_geographic_distribution.png` - Spatial insights / 空间洞察

**Supplementary** (补充):
6. `2_feature_categories.png` - Dataset structure / 数据集结构
7. `8_room_type_analysis.png` - Property insights / 房源洞察
8. `10_reviews_analysis.png` - User engagement / 用户参与度

### Text Structure / 文本结构建议

```
Section 1: Dataset Overview (~0.5 page)
├── Data source description
├── Chart 2: Feature categories
└── Brief quality summary

Section 2: Business Understanding (~1 page)
├── Research question
├── Business value proposition
├── Chart 4: Rating distribution 2021 vs 2025
├── Target variable definition
└── Chart 5: Target variable distribution

Section 3: Key Features Analysis (~1.5 pages)
├── Pricing (Chart 6)
├── Host features (Chart 7)
├── Property features (Chart 8)
├── Geographic (Chart 9)
└── Reviews (Chart 10)
```

---

## Notes / 注意事项

1. **Chart Quality** / 图表质量
   - All charts are 300 DPI, suitable for academic reports
   - Use PNG format for best quality in Word/LaTeX

2. **Code Quality** / 代码质量
   - Well-commented and documented
   - Follows PEP 8 style guidelines
   - Safe to submit with report

3. **Reproducibility** / 可重现性
   - All visualizations can be regenerated
   - Consistent random seeds (if applicable)
   - Clear dependencies

---

## Contact / 联系方式

For issues or questions:
如有问题：

- Check the main project README
- Review the data processing scripts in `src/data_merge/` and `src/data_clean/`

---

**Last Updated**: 2025-11-25
**最后更新**: 2025年11月25日
