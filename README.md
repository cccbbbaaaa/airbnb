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
│   └── EDA/                 # 探索性数据分析脚本 / EDA scripts
│       ├── Activity.py              # 房源活动分析 / Activity analysis
│       ├── data clean.py           # 数据清洗 / Data cleaning
│       ├── Description statistic.py # 描述性统计分析 / Descriptive statistics
│       ├── License.py              # 许可证分析 / License analysis
│       ├── Null value.py           # 缺失值检查 / Null value checking
│       └── Top Hosts.py            # 顶级房东分析 / Top hosts analysis
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

### 运行单个分析脚本 / Run Individual Analysis Scripts

在项目根目录下，进入 `src/EDA` 目录运行脚本：

```bash
cd src/EDA

# 运行房源活动分析 / Run activity analysis
python Activity.py

# 运行描述性统计分析 / Run descriptive statistics
python "Description statistic.py"

# 运行许可证分析 / Run license analysis
python License.py

# 检查缺失值 / Check null values
python "Null value.py"

# 分析顶级房东 / Analyze top hosts
python "Top Hosts.py"

# 执行数据清洗 / Perform data cleaning
python "data clean.py"
```

### 输出说明 / Output Description

所有图表将自动保存到 `charts/` 目录下，文件格式为 PNG（300 DPI 高分辨率）。

All charts will be automatically saved to the `charts/` directory in PNG format (300 DPI high resolution).

## 脚本功能说明 / Script Functions

### 1. Activity.py - 房源活动分析
- 计算平均入住天数
- 计算平均价格
- 生成入住率分布直方图

### 2. Description statistic.py - 描述性统计分析
- 价格分布分析
- 房型分布分析
- 评论数量分布分析
- 不同房型的平均价格分析
- 不同街区的平均价格分析

### 3. License.py - 许可证分析
- 许可证分类（已授权/豁免/未授权/待处理）
- 生成许可证分布饼图

### 4. Null value.py - 缺失值检查
- 统计各字段的缺失值数量
- 识别需要处理的缺失值字段

### 5. Top Hosts.py - 顶级房东分析
- 统计每个房东的房源数量
- 按房源数量排序，显示前10名房东

### 6. data clean.py - 数据清洗
- 删除 `neighbourhood_group` 列
- 填充缺失值（review 相关字段用 0，name 相关字段用 "blank_name"）

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

2. **图表输出**: 所有图表自动保存到 `charts/` 目录，如果目录不存在会自动创建。

3. **相对路径**: 所有脚本使用相对路径读取数据，确保在正确的目录下运行脚本。

4. **Python 版本**: 建议使用 Python 3.8 或更高版本。

## 贡献指南 / Contributing

欢迎提交 Issue 和 Pull Request 来改进项目。

## 许可证 / License

本项目仅供学习和研究使用。

## 更新日志 / Changelog

### 2024 - 初始版本
- 完成数据清洗模块
- 完成探索性数据分析脚本
- 实现图表自动保存功能
- 创建项目文档

