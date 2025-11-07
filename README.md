# Airbnb Amsterdam 数据分析项目 / Airbnb Amsterdam Data Analysis Project

## 项目简介 / Project Overview

本项目是对 Airbnb 阿姆斯特丹房源数据的探索性数据分析（EDA）项目。项目包含数据清洗、统计分析、可视化等多个模块，旨在深入理解 Airbnb 房源数据的特征和分布规律。

This project is an Exploratory Data Analysis (EDA) project for Airbnb Amsterdam listings data. It includes data cleaning, statistical analysis, visualization modules, aiming to understand the characteristics and distribution patterns of Airbnb listings data.

## 项目结构 / Project Structure

```
project/
├── data/                    # 原始数据文件目录 / Raw data files directory
│   ├── listings.csv         # 房源详细数据 / Listings detailed data
│   ├── calendar_summary.csv # 日历汇总数据 / Calendar summary data
│   ├── reviews.csv          # 评论数据 / Reviews data
│   ├── neighbourhoods.csv  # 街区数据 / Neighbourhoods data
│   └── data dictionary.xlsx # 数据字典 / Data dictionary
│
├── src/                     # 源代码目录 / Source code directory
│   └── EDA/                 # 探索性数据分析 / EDA directory
│       └── Airbnb_EDA.ipynb # 完整的EDA分析Notebook / Complete EDA analysis notebook
│
├── charts/                  # 图表输出目录 / Charts output directory
│   ├── occupancy_distribution.png      # 入住率分布图
│   ├── price_distribution.png          # 价格分布图
│   ├── room_type_distribution.png      # 房型分布图
│   ├── review_distribution.png         # 评论分布图
│   ├── avg_price_by_room_type.png      # 房型平均价格图
│   ├── avg_price_by_neighbourhood.png  # 街区平均价格图
│   └── license_distribution.png        # 许可证分布图
│
├── venv/                    # Python 虚拟环境 / Python virtual environment
├── requirements.txt         # Python 依赖包列表 / Python dependencies
├── .gitignore              # Git 忽略文件配置 / Git ignore configuration
└── README.md               # 项目说明文档 / Project documentation
```

## 环境要求 / Requirements

- Python 3.8+
- Jupyter Notebook 或 JupyterLab（用于运行 Notebook）
- 依赖包见 `requirements.txt`

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

### 运行 Jupyter Notebook / Run Jupyter Notebook

本项目使用 Jupyter Notebook 进行交互式数据分析。所有分析代码已整合在 `src/EDA/Airbnb_EDA.ipynb` 中。

#### 方法一：使用 Jupyter Notebook（推荐）

```bash
cd src/EDA
jupyter notebook Airbnb_EDA.ipynb
```

#### 方法二：使用 JupyterLab

```bash
cd src/EDA
jupyter lab Airbnb_EDA.ipynb
```

#### 方法三：使用 VS Code

直接在 VS Code 中打开 `src/EDA/Airbnb_EDA.ipynb` 文件，确保已安装 Jupyter 扩展。

### Notebook 结构说明 / Notebook Structure

Notebook 包含以下分析模块，按顺序执行：

1. **数据加载** / Data Loading - 读取和查看数据基本信息
2. **缺失值检查** / Missing Value Check - 统计和识别缺失值
3. **数据清洗** / Data Cleaning - 删除列、填充缺失值
4. **数据基本信息描述** / Basic Data Information - 数据概览和统计描述
5. **房源活动分析** / Activity Analysis - 入住率和平均价格分析
6. **描述性统计分析** / Descriptive Statistics - 价格、房型、评论分布分析
7. **许可证分析** / License Analysis - 许可证分类和分布
8. **顶级房东分析** / Top Hosts Analysis - 房源数量排名

### 输出说明 / Output Description

- 所有图表将自动保存到 `charts/` 目录下，文件格式为 PNG（300 DPI 高分辨率）
- 图表会在 Notebook 中直接显示，同时保存到文件
- 分析结果和统计信息会在 Notebook 中显示

All charts will be automatically saved to the `charts/` directory in PNG format (300 DPI high resolution). Charts are displayed in the notebook and saved to files simultaneously.

## Notebook 分析模块说明 / Notebook Analysis Modules

`Airbnb_EDA.ipynb` 包含以下分析模块：

### 1. 数据加载 / Data Loading

- 读取 CSV 数据文件
- 查看数据形状和基本信息
- 预览前几行数据

### 2. 缺失值检查 / Missing Value Check

- 统计每个字段的缺失值数量
- 识别需要处理的缺失值字段
- 显示缺失值统计结果

### 3. 数据清洗 / Data Cleaning

- 删除 `neighbourhood_group` 列（全为空）
- 填充缺失值：
  - review 相关字段用 0 填充
  - name 和 host_name 用 "blank_name" 填充
  - license 字段用 0 填充

### 4. 数据基本信息描述 / Basic Data Information

- 显示数据结构和数据类型
- 计算数值型字段的统计描述（均值、标准差、分位数等）

### 5. 房源活动分析 / Activity Analysis

- 计算平均入住天数（365 - availability_365）
- 计算平均价格
- 生成入住率分布直方图

### 6. 描述性统计分析 / Descriptive Statistics

- **价格分布分析**：价格直方图和统计描述
- **房型分布分析**：房型占比饼图
- **评论数量分布分析**：评论数直方图
- **房型平均价格分析**：不同房型的平均价格柱状图
- **街区平均价格分析**：不同街区的平均价格柱状图

### 7. 许可证分析 / License Analysis

- 许可证分类处理：
  - Unlicensed（0值）
  - Exempt（豁免）
  - Licensed（0363开头）
  - pending（其他）
- 生成许可证分布饼图
- 显示许可证分类统计

### 8. 顶级房东分析 / Top Hosts Analysis

- 使用交叉表统计每个房东的房型分布
- 计算每个房东的总房源数
- 按房源数量排序，显示前10名房东及其房型分布

## 数据说明 / Data Description

项目使用的数据包含以下字段：

- `id`: 房源ID
- `name`: 房源名称
- `host_id`: 房东ID
- `host_name`: 房东名称
- `neighbourhood`: 所在街区
- `latitude`, `longitude`: 经纬度坐标
- `room_type`: 房型（整租/独立房间/共享房间/酒店房间）
- `price`: 价格（欧元/晚）
- `minimum_nights`: 最少入住天数
- `number_of_reviews`: 评论数量
- `last_review`: 最后评论日期
- `reviews_per_month`: 每月评论数
- `availability_365`: 一年中的可预订天数
- `license`: 许可证信息

## 注意事项 / Notes

1. **数据文件**: 由于数据文件较大，已通过 `.gitignore` 排除，不会推送到 Git 仓库。请确保在 `data/` 目录下放置相应的数据文件。

2. **图表输出**: 所有图表自动保存到 `charts/` 目录，如果目录不存在会自动创建。图表文件已通过 `.gitignore` 排除，可通过运行 Notebook 重新生成。

3. **Notebook 路径**: Notebook 使用相对路径读取数据（`../../data/listings.csv`），请确保从 `src/EDA/` 目录打开 Notebook，或调整路径设置。

4. **Python 版本**: 建议使用 Python 3.8 或更高版本。

5. **Jupyter 安装**: 如果未安装 Jupyter，请运行 `pip install jupyter` 或 `pip install -r requirements.txt`。

6. **执行顺序**: 建议按顺序执行 Notebook 中的所有单元格，因为后续分析依赖于前面的数据清洗步骤。

7. **交互式分析**: Notebook 支持交互式分析，可以修改代码、重新运行单元格，方便探索性数据分析。

## 更新日志 / Changelog

### 2024 - v2.0
- ✅ 将所有 Python 脚本整合到单个 Jupyter Notebook (`Airbnb_EDA.ipynb`)
- ✅ 优化代码结构，按分析模块组织
- ✅ 添加交互式数据分析支持
- ✅ 更新项目文档和使用说明

### 2024 - v1.0
- ✅ 完成数据清洗模块
- ✅ 完成探索性数据分析脚本
- ✅ 实现图表自动保存功能
- ✅ 创建项目文档
