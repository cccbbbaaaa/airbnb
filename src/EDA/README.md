# 📊 EDA 主分析 Notebook 使用说明 / EDA Main Analysis Notebook Usage Guide

## 📑 概述 / Overview

`EDA_main.ipynb` 是一个汇总所有章节 EDA 分析的 Jupyter Notebook，通过调用各个独立的 Python 脚本文件实现。

`EDA_main.ipynb` is a Jupyter Notebook that summarizes all chapter EDA analyses by calling individual Python script files.

## 🚀 快速开始 / Quick Start

### 方法1：使用 Notebook（推荐）/ Method 1: Using Notebook (Recommended)

1. 打开 `src/EDA/EDA_main.ipynb`
   Open `src/EDA/EDA_main.ipynb`
2. 按顺序执行所有代码单元格
   Execute all code cells in order
3. 查看生成的图表和统计报告（保存在 `charts/` 目录）
   View generated charts and statistics reports (saved in `charts/` directory)

### 方法2：直接运行脚本文件 / Method 2: Running Script Files Directly

每个章节的脚本文件都可以独立运行：
Each chapter's script file can be run independently:

```bash
# 运行第3章分析 / Run Chapter 3 Analysis
python src/EDA/chapter3_dataset_relationships.py

# 运行第5.1章分析 / Run Chapter 5.1 Analysis
python src/EDA/chapter5_listings_analysis.py

# ... 其他章节类似 / ... other chapters similar
```

## 📁 文件结构 / File Structure

```
src/EDA/
├── EDA_main.ipynb                    # 主 Notebook（汇总所有分析）/ Main Notebook (summarizes all analyses)
├── utils.py                          # 工具函数模块 / Utility functions module
├── data_quality_analysis.py          # 第2章：数据质量分析 / Chapter 2: Data Quality Analysis
├── chapter3_dataset_relationships.py # 第3章：数据集关系分析 / Chapter 3: Dataset Relationships Analysis
├── chapter5_listings_analysis.py     # 第5.1章：listings 分析 / Chapter 5.1: Listings Analysis
├── chapter5_reviews_analysis.py      # 第5.2章：reviews 分析 / Chapter 5.2: Reviews Analysis
├── chapter5_calendar_analysis.py      # 第5.3章：calendar 分析 / Chapter 5.3: Calendar Analysis
├── chapter5_neighbourhoods_analysis.py # 第5.4章：neighbourhoods 分析 / Chapter 5.4: Neighbourhoods Analysis
├── chapter5_listings_detailed_analysis.py # 第5.5章：listings_detailed 分析 / Chapter 5.5: Listings Detailed Analysis
├── chapter6_correlation_analysis.py  # 第6章：相关性分析 / Chapter 6: Correlation Analysis
├── chapter7_time_series_analysis.py  # 第7章：时间序列分析 / Chapter 7: Time Series Analysis
├── chapter8_geospatial_analysis.py   # 第8章：地理空间分析 / Chapter 8: Geospatial Analysis
└── chapter9_pareto_pricing_analysis.py # 第9章：帕累托和价格策略分析 / Chapter 9: Pareto and Pricing Strategy Analysis
```

## 🔧 依赖 / Dependencies

确保已安装以下 Python 包：
Make sure the following Python packages are installed:

```bash
pip install pandas numpy matplotlib seaborn scipy openpyxl
```

或者在 Notebook 中运行第一个单元格（安装依赖）：
Or run the first cell in the Notebook (install dependencies):

```python
! pip install -q scipy seaborn numpy matplotlib pandas jupyter ipykernel openpyxl
```

## 📊 输出文件 / Output Files

### 图表文件（保存在 `charts/` 目录）/ Chart Files (saved in `charts/` directory)

- `chapter3_dataset_relationships.png` - 数据集关系分析 / Dataset Relationships Analysis
- `chapter5_listings_field_analysis.png` - listings 字段分析 / Listings Field Analysis
- `chapter5_reviews_analysis.png` - reviews 分析 / Reviews Analysis
- `chapter5_calendar_analysis.png` - calendar 分析 / Calendar Analysis
- `chapter5_neighbourhoods_analysis.png` - neighbourhoods 分析 / Neighbourhoods Analysis
- `chapter5_listings_detailed_analysis.png` - listings_detailed 分析 / Listings Detailed Analysis
- `chapter6_correlation_analysis.png` - 相关性分析 / Correlation Analysis
- `chapter6_categorical_association.png` - 分类变量关联分析 / Categorical Variable Association Analysis
- `chapter7_time_series_analysis.png` - 时间序列分析 / Time Series Analysis
- `chapter8_geospatial_analysis.png` - 地理空间分析 / Geospatial Analysis
- `chapter8_location_price_relationship.png` - 地理位置与价格关系 / Location-Price Relationship
- `chapter9_pareto_analysis.png` - 帕累托分析 / Pareto Analysis
- `chapter9_pricing_strategy_analysis.png` - 价格策略分析 / Pricing Strategy Analysis

### 统计报告（保存在 `charts/` 目录）/ Statistics Reports (saved in `charts/` directory)

- `chapter3_statistics.txt` - 数据集关系统计 / Dataset Relationships Statistics
- `chapter5_*_statistics.txt` - 各数据集统计报告 / Dataset Statistics Reports
- `chapter6_correlation_statistics.txt` - 相关性统计 / Correlation Statistics
- `chapter7_time_series_statistics.txt` - 时间序列统计 / Time Series Statistics
- `chapter8_geospatial_statistics.txt` - 地理空间统计 / Geospatial Statistics
- `chapter9_pareto_pricing_statistics.txt` - 帕累托和价格策略统计 / Pareto and Pricing Strategy Statistics

## 💡 使用技巧 / Usage Tips

1. **按顺序执行**: 建议按 Notebook 中的顺序执行各个章节
   **Execute in order**: It is recommended to execute each chapter in the order shown in the Notebook
2. **单独运行**: 如果需要重新运行某个章节，只需执行对应的代码单元格
   **Run individually**: If you need to re-run a specific chapter, just execute the corresponding code cell
3. **查看结果**: 所有图表和统计报告会自动保存到 `charts/` 目录
   **View results**: All charts and statistics reports will be automatically saved to the `charts/` directory
4. **参考文档**: 详细分析结果请参考 `docs/EDA_Report_Outline.md`
   **Reference documentation**: For detailed analysis results, please refer to `docs/EDA_Report_Outline.md`

## ⚠️ 注意事项 / Notes

- 所有脚本文件使用相对路径，确保在项目根目录或 `src/EDA/` 目录下运行
  All script files use relative paths. Make sure to run from the project root directory or `src/EDA/` directory
- 如果遇到路径问题，请检查当前工作目录
  If you encounter path issues, please check the current working directory
- 首次运行可能需要几分钟时间生成所有图表
  The first run may take several minutes to generate all charts
- 如果遇到 `ModuleNotFoundError`，请先运行 Notebook 中的"安装依赖"单元格
  If you encounter `ModuleNotFoundError`, please run the "Install Dependencies" cell in the Notebook first

## 🔍 故障排除 / Troubleshooting

### 问题1：ModuleNotFoundError / Issue 1: ModuleNotFoundError

**错误信息 / Error Message**: `ModuleNotFoundError: No module named 'scipy'`

**解决方案 / Solution**:
运行 Notebook 中的第一个单元格（安装依赖）：
Run the first cell in the Notebook (install dependencies):

```python
! pip install -q scipy seaborn numpy matplotlib pandas jupyter ipykernel openpyxl
```

### 问题2：路径错误 / Issue 2: Path Error

**错误信息 / Error Message**: `FileNotFoundError` 或路径相关错误
**Error Message**: `FileNotFoundError` or path-related errors

**解决方案 / Solution**:
确保在项目根目录或 `src/EDA/` 目录下运行 Notebook
Make sure to run the Notebook from the project root directory or `src/EDA/` directory
