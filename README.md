# Airbnb Amsterdam 数据分析项目 / Airbnb Amsterdam Data Analysis Project

## 项目简介 / Project Overview

本项目是对 Airbnb 阿姆斯特丹房源数据的探索性数据分析（EDA）项目。项目采用 CRISP-DM 方法论，包含数据质量分析、数据集关系分析、变量相关性分析、时间序列分析、地理空间分析等多个模块，旨在深入理解 Airbnb 房源数据的特征和分布规律，为后续建模和业务决策提供数据支持。

This project is an Exploratory Data Analysis (EDA) project for Airbnb Amsterdam listings data. Following the CRISP-DM methodology, it includes data quality analysis, dataset relationship analysis, variable correlation analysis, time series analysis, geospatial analysis, and other modules, aiming to understand the characteristics and distribution patterns of Airbnb listings data and provide data support for subsequent modeling and business decisions.

## 项目结构 / Project Structure

```
project/
├── data/                              # Raw data files directory
│   ├── listings.csv                   # Listings detailed data (16,116 records)
│   ├── listings_detailed.xlsx         # Listings extended data
│   ├── calendar_summary.csv           # Calendar summary data (21,210 records)
│   ├── reviews.csv                    # Reviews data (397,185 records)
│   ├── neighbourhoods.csv             # Neighbourhoods data (22 neighbourhoods)
│   └── data dictionary.xlsx            # Data dictionary
│
├── src/                               # Source code directory
│   ├── EDA/                           # EDA module
│   │   ├── EDA_main.ipynb             # Main analysis Notebook (summarizes all chapters)
│   │   ├── utils.py                   # Utility functions module
│   │   ├── data_quality_analysis.py   # Chapter 2: Data Quality & Scale Overview
│   │   ├── chapter3_dataset_relationships.py # Chapter 3: Dataset Relationships & Structure
│   │   ├── chapter5_listings_analysis.py      # Chapter 5.1: Listings Dataset Analysis
│   │   ├── chapter5_reviews_analysis.py       # Chapter 5.2: Reviews Dataset Analysis
│   │   ├── chapter5_calendar_analysis.py      # Chapter 5.3: Calendar Dataset Analysis
│   │   ├── chapter5_neighbourhoods_analysis.py  # Chapter 5.4: Neighbourhoods Dataset Analysis
│   │   ├── chapter5_listings_detailed_analysis.py # Chapter 5.5: Listings Detailed Dataset Analysis
│   │   ├── chapter6_correlation_analysis.py    # Chapter 6: Variable Correlation Analysis
│   │   ├── chapter7_time_series_analysis.py    # Chapter 7: Time Series Analysis
│   │   ├── chapter8_geospatial_analysis.py      # Chapter 8: Geospatial Analysis
│   │   ├── chapter9_pareto_pricing_analysis.py # Chapter 9: Pareto & Pricing Strategy Analysis
│   │   ├── eda_main.py                # EDA main function wrapper
│   │   └── README.md                  # EDA module usage guide
│   │
│   ├── modeling/                     # Modeling module (to be developed)
│   │
│   └── old_EDA/                       # Old EDA files (archived)
│       ├── Airbnb_EDA.ipynb          # Original EDA Notebook
│       └── popular_house.py          # Popular listings analysis script
│
├── docs/                              # Documentation directory
│   ├── EDA_Report_Outline.md         # EDA Report Outline (complete analysis results)
│   └── project guidance & requirement.md # Project guidance and requirements
│
├── charts/                            # Charts output directory
│   ├── chapter3_dataset_relationships.png     # Dataset Relationships Chart
│   ├── chapter5_*.png                         # Dataset Analysis Charts
│   ├── chapter6_correlation_analysis.png      # Correlation Analysis Chart
│   ├── chapter6_categorical_association.png   # Categorical Association Chart
│   ├── chapter7_time_series_analysis.png       # Time Series Analysis Chart
│   ├── chapter8_geospatial_analysis.png         # Geospatial Analysis Chart
│   ├── chapter8_location_price_relationship.png # Location-Price Relationship Chart
│   ├── chapter9_pareto_analysis.png            # Pareto Analysis Chart
│   ├── chapter9_pricing_strategy_analysis.png   # Pricing Strategy Analysis Chart
│   └── *.txt                                  # Chapter Statistics Reports
│
├── venv/                              # Python virtual environment
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore configuration
└── README.md                          # Project documentation
```

## 环境要求 / Requirements

- **Python**: 3.8+ / Python 3.8+
- **Jupyter Notebook** 或 **JupyterLab**（用于运行 Notebook）/ Jupyter Notebook or JupyterLab (for running Notebooks)
- **依赖包** / Dependencies: 见 `requirements.txt` / See `requirements.txt`

### 核心依赖包 / Core Dependencies

- pandas >= 1.5.0
- numpy >= 1.23.0
- matplotlib >= 3.5.0
- seaborn >= 0.12.0
- scipy >= 1.9.0
- openpyxl >= 3.0.0
- jupyter >= 1.0.0
- ipykernel >= 6.0.0

## 安装步骤 / Installation

### 1. 克隆仓库 / Clone Repository

```bash
git clone git@github.com:cccbbbaaaa/airbnb.git
cd airbnb
```

### 2. 创建虚拟环境 / Create Virtual Environment

```bash
python3 -m venv venv
```

### 3. 激活虚拟环境 / Activate Virtual Environment

**macOS/Linux:**

```bash
source venv/bin/activate
```

**Windows:**

```bash
venv\Scripts\activate
```

### 4. 安装依赖包 / Install Dependencies

```bash
pip install -r requirements.txt
```

## 使用方法 / Usage

### 方法一：使用主 Notebook（推荐）/ Method 1: Using Main Notebook (Recommended)

本项目提供了一个主 Notebook (`src/EDA/EDA_main.ipynb`) 来汇总所有章节的 EDA 分析。

This project provides a main Notebook (`src/EDA/EDA_main.ipynb`) that summarizes all chapter EDA analyses.

#### 运行步骤 / Running Steps

1. **打开 Notebook** / Open Notebook:

   ```bash
   cd src/EDA
   jupyter notebook EDA_main.ipynb
   ```

   或使用 JupyterLab / Or use JupyterLab:

   ```bash
   jupyter lab EDA_main.ipynb
   ```
2. **按顺序执行单元格** / Execute cells in order:

   - 第一个单元格：安装依赖 / First cell: Install dependencies
   - 第二个单元格：环境设置 / Second cell: Environment setup
   - 后续单元格：各章节分析 / Subsequent cells: Chapter analyses
3. **查看结果** / View results:

   - 图表保存在 `charts/` 目录 / Charts saved in `charts/` directory
   - 统计报告保存在 `charts/*_statistics.txt` / Statistics reports saved in `charts/*_statistics.txt`

### 方法二：直接运行脚本文件 / Method 2: Running Script Files Directly

每个章节的脚本文件都可以独立运行：
Each chapter's script file can be run independently:

```bash
# 运行第2章分析 / Run Chapter 2 Analysis
python src/EDA/data_quality_analysis.py

# 运行第3章分析 / Run Chapter 3 Analysis
python src/EDA/chapter3_dataset_relationships.py

# 运行第5.1章分析 / Run Chapter 5.1 Analysis
python src/EDA/chapter5_listings_analysis.py

# ... 其他章节类似 / ... other chapters similar
```

### 方法三：使用 VS Code / Method 3: Using VS Code

直接在 VS Code 中打开 `src/EDA/EDA_main.ipynb` 文件，确保已安装 Jupyter 扩展。

Open `src/EDA/EDA_main.ipynb` directly in VS Code, make sure the Jupyter extension is installed.

## EDA 分析章节 / EDA Analysis Chapters

主 Notebook 包含以下分析章节，按顺序执行：
The main Notebook includes the following analysis chapters, executed in order:

### 第2章：数据质量与规模总览 / Chapter 2: Data Quality & Scale Overview

- 数据集规模统计 / Dataset scale statistics
- 数据完整度分析 / Data completeness analysis
- 时间跨度分析 / Time span analysis
- 数据质量可视化 / Data quality visualization

### 第3章：数据集关系与结构 / Chapter 3: Dataset Relationships & Structure

- 数据集关系验证 / Dataset relationship validation
- 数据一致性检查 / Data consistency check
- 数据整合价值分析 / Data integration value analysis

### 第5章：逐个数据集详细分析 / Chapter 5: Detailed Dataset Analysis

- **5.1 listings.csv**: 房源主数据表分析 / Main listings dataset analysis
- **5.2 reviews.csv**: 评论数据时间序列分析 / Reviews time series analysis
- **5.3 calendar_summary.csv**: 入住率和可用性分析 / Occupancy rate and availability analysis
- **5.4 neighbourhoods.csv**: 街区参考数据分析 / Neighbourhoods reference data analysis
- **5.5 listings_detailed.xlsx**: 扩展字段分析 / Extended fields analysis

### 第6章：变量相关性分析 / Chapter 6: Variable Correlation Analysis

- 数值型变量相关性矩阵 / Numerical variable correlation matrix
- 分类变量关联分析 / Categorical variable association analysis
- 关键比率特征分析 / Key ratio features analysis

### 第7章：时间序列分析 / Chapter 7: Time Series Analysis

- 评论时间趋势分析 / Review time trend analysis
- 季节性模式识别 / Seasonal pattern identification
- COVID-19 影响分析 / COVID-19 impact analysis
- 房源生命周期模式 / Listing lifecycle patterns

### 第8章：地理空间分析 / Chapter 8: Geospatial Analysis

- 房源地理分布 / Geographic distribution of listings
- 地理位置与价格关系 / Location-price relationship
- 地理位置与受欢迎度关系 / Location-popularity relationship

### 第9章：深度业务洞察 / Chapter 9: Deep Business Insights

- **9.1 帕累托分析** / Pareto Analysis: 评论和收入分布分析 / Review and revenue distribution analysis
- **9.2 价格策略分析** / Pricing Strategy Analysis: 价格影响因素和最优定价区间 / Price influencing factors and optimal pricing ranges

## 输出说明 / Output Description

### 图表文件 / Chart Files

所有图表自动保存到 `charts/` 目录，格式为 PNG（300 DPI 高分辨率）：
All charts are automatically saved to the `charts/` directory in PNG format (300 DPI high resolution):

- `chapter3_dataset_relationships.png` - 数据集关系分析图
- `chapter5_*_analysis.png` - 各数据集分析图表
- `chapter6_correlation_analysis.png` - 相关性分析图
- `chapter6_categorical_association.png` - 分类变量关联图
- `chapter7_time_series_analysis.png` - 时间序列分析图
- `chapter8_geospatial_analysis.png` - 地理空间分析图
- `chapter8_location_price_relationship.png` - 地理位置与价格关系图
- `chapter9_pareto_analysis.png` - 帕累托分析图
- `chapter9_pricing_strategy_analysis.png` - 价格策略分析图

### 统计报告 / Statistics Reports

所有统计报告自动保存到 `charts/` 目录：
All statistics reports are automatically saved to the `charts/` directory:

- `chapter3_statistics.txt` - 数据集关系统计
- `chapter5_*_statistics.txt` - 各数据集统计报告
- `chapter6_correlation_statistics.txt` - 相关性统计
- `chapter7_time_series_statistics.txt` - 时间序列统计
- `chapter8_geospatial_statistics.txt` - 地理空间统计
- `chapter9_pareto_pricing_statistics.txt` - 帕累托和价格策略统计

## 数据说明 / Data Description

项目使用的数据包含以下数据集：
The project uses the following datasets:

### listings.csv（房源主数据表）/ Main Listings Dataset

- **记录数** / Records: 16,116 条
- **字段数** / Fields: 18 个（清洗后17个）
- **主要字段** / Main Fields:
  - `id`: 房源ID / Listing ID
  - `name`: 房源名称 / Listing name
  - `host_id`: 房东ID / Host ID
  - `host_name`: 房东名称 / Host name
  - `neighbourhood`: 所在街区 / Neighbourhood
  - `latitude`, `longitude`: 经纬度坐标 / Geographic coordinates
  - `room_type`: 房型（整租/独立房间/共享房间/酒店房间）/ Room type (Entire home/Private room/Shared room/Hotel room)
  - `price`: 价格（欧元/晚）/ Price (EUR/night)
  - `minimum_nights`: 最少入住天数 / Minimum nights
  - `number_of_reviews`: 评论数量 / Number of reviews
  - `last_review`: 最后评论日期 / Last review date
  - `reviews_per_month`: 每月评论数 / Reviews per month
  - `availability_365`: 一年中的可预订天数 / Available days per year
  - `license`: 许可证信息 / License information

### reviews.csv（评论数据表）/ Reviews Dataset

- **记录数** / Records: 397,185 条
- **时间跨度** / Time Span: 2009-03-30 至 2021-09-07（12.4年）
- **字段** / Fields:
  - `listing_id`: 房源ID / Listing ID
  - `date`: 评论日期 / Review date

### calendar_summary.csv（日历汇总表）/ Calendar Summary Dataset

- **记录数** / Records: 21,210 条
- **字段** / Fields:
  - `listing_id`: 房源ID / Listing ID
  - `available`: 是否可用 / Availability status
  - `count`: 天数统计 / Day count

### neighbourhoods.csv（街区参考表）/ Neighbourhoods Reference Dataset

- **记录数** / Records: 22 条
- **字段** / Fields:
  - `neighbourhood`: 街区名称 / Neighbourhood name
  - `neighbourhood_group`: 街区组（大部分为空）/ Neighbourhood group (mostly empty)

## 注意事项 / Notes

1. **数据文件** / Data Files:
   由于数据文件较大，已通过 `.gitignore` 排除，不会推送到 Git 仓库。请确保在 `data/` 目录下放置相应的数据文件。
   Due to large file sizes, data files are excluded via `.gitignore` and will not be pushed to the Git repository. Please ensure data files are placed in the `data/` directory.
2. **图表输出** / Chart Output:
   所有图表自动保存到 `charts/` 目录，如果目录不存在会自动创建。图表文件已通过 `.gitignore` 排除，可通过运行 Notebook 重新生成。
   All charts are automatically saved to the `charts/` directory. If the directory doesn't exist, it will be created automatically. Chart files are excluded via `.gitignore` and can be regenerated by running the Notebook.
3. **Notebook 路径** / Notebook Paths:
   所有脚本使用相对路径，通过 `utils.py` 中的 `get_project_paths()` 函数自动检测项目根目录。支持在项目根目录或 `src/EDA/` 目录下运行。
   All scripts use relative paths and automatically detect the project root directory via the `get_project_paths()` function in `utils.py`. Supports running from project root or `src/EDA/` directory.
4. **Python 版本** / Python Version:
   建议使用 Python 3.8 或更高版本。
   Python 3.8 or higher is recommended.
5. **依赖安装** / Dependency Installation:
   如果遇到 `ModuleNotFoundError`，请先运行 Notebook 中的"安装依赖"单元格，或运行 `pip install -r requirements.txt`。
   If you encounter `ModuleNotFoundError`, please run the "Install Dependencies" cell in the Notebook first, or run `pip install -r requirements.txt`.
6. **执行顺序** / Execution Order:
   建议按顺序执行 Notebook 中的所有单元格，因为后续分析依赖于前面的数据加载和设置步骤。
   It is recommended to execute all cells in the Notebook in order, as subsequent analyses depend on previous data loading and setup steps.
7. **模块化设计** / Modular Design:
   每个章节的分析都封装在独立的 Python 脚本中，便于维护和复用。主 Notebook 通过 `exec()` 调用这些脚本。
   Each chapter's analysis is encapsulated in an independent Python script for easy maintenance and reuse. The main Notebook calls these scripts via `exec()`.

## 项目进度 / Project Progress

**当前阶段 / Current Phase**: 第3-4周 - 数据理解与探索性数据分析 (Week 3-4 - Data Understanding & EDA)

### ✅ 已完成 / Completed

- [X] 数据集选择和数据加载（Week 1）/ Dataset selection and data loading (Week 1)
- [X] 数据质量与规模总览分析（第2章）/ Data quality and scale overview analysis (Chapter 2)
- [X] 数据集关系与结构分析（第3章）/ Dataset relationships and structure analysis (Chapter 3)
- [X] 核心发现与关键洞察总结（第4章）/ Core findings and key insights summary (Chapter 4)
- [X] 所有数据集详细分析（第5章）/ Detailed analysis of all datasets (Chapter 5)
  - [X] listings.csv 分析
  - [X] reviews.csv 分析
  - [X] calendar_summary.csv 分析
  - [X] neighbourhoods.csv 分析
  - [X] listings_detailed.xlsx 分析
- [X] 变量相关性分析（第6章）/ Variable correlation analysis (Chapter 6)
- [X] 时间序列分析（第7章）/ Time series analysis (Chapter 7)
- [X] 地理空间分析（第8章）/ Geospatial analysis (Chapter 8)
- [X] 深度业务洞察（第9章）/ Deep business insights (Chapter 9)
  - [X] 帕累托分析（9.1）
  - [X] 价格策略分析（9.2）
- [X] 主 Notebook 创建（`EDA_main.ipynb`）/ Main Notebook created (`EDA_main.ipynb`)
- [X] 模块化脚本架构 / Modular script architecture
- [X] 工具函数模块（`utils.py`）/ Utility functions module (`utils.py`)
- [X] 50+ 个可视化图表生成 / 50+ visualization charts generated
- [X] 完整的 EDA 报告大纲（`docs/EDA_Report_Outline.md`）/ Complete EDA report outline (`docs/EDA_Report_Outline.md`)

### ⚠️ 进行中 / In Progress

- [ ] 确定研究主题和业务问题（紧急）/ Determine research topic and business questions (urgent)
- [ ] 完善 EDA 报告（第10-12章）/ Complete EDA report (Chapters 10-12)
  - [ ] 数据质量挑战与处理（第10章）/ Data quality challenges and handling (Chapter 10)
  - [ ] 特征工程建议（第11章）/ Feature engineering suggestions (Chapter 11)
  - [ ] 总结与下一步行动（第12章）/ Summary and next steps (Chapter 12)

### 📋 待开始 / To Do

- [ ] 准备与老师的会面材料 / Prepare materials for meeting with instructor
- [ ] 数据准备阶段（特征工程、数据清洗）/ Data preparation phase (feature engineering, data cleaning)
- [ ] 建模阶段 / Modeling phase

## 相关文档 / Related Documentation

- **EDA 模块使用说明** / EDA Module Usage Guide: `src/EDA/README.md`
- **EDA 报告大纲** / EDA Report Outline: `docs/EDA_Report_Outline.md`
- **项目指导和要求** / Project Guidance and Requirements: `docs/project guidance & requirement.md`
