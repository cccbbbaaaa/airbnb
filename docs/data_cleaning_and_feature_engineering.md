## 数据清洗与特征工程说明 / Data Cleaning & Feature Engineering Overview

### 背景 / Background
- 数据源：`data/cleaned/listings_cleaned.csv`（由 `clean_merged_data.py` 生成，整合 2021 与 2025 年 Amsterdam Airbnb Listings）。
- 特征工程脚本：`src/feature_engineering/build_train_features.py`。
- 输出数据：`data/processed/train_data.csv`（共 26,596 行 × 114 列，已排除 `id` 与 `data_year`）。

---

### 数据清洗策略 / Data Cleaning Strategy

1. **全缺失字段删除 / Drop Fully Missing Columns**
   - `calendar_updated`, `neighbourhood_group_cleansed` (100% 缺失) → 直接删除。

2. **高缺失字段处理 / High-Missing Columns**
   - `bathrooms`, `estimated_revenue_l365d` (~78% 缺失) → 删除。
   - 其他缺失率 40-60% 的字段（如 `source`, `host_response_rate`, `license` 等）：
     - 创建缺失指示变量 `_is_missing`；
     - 众数/中位数/Unknown 填充。

3. **中低缺失字段填充 / Medium-Low Missing Columns**
   - 数值型（如 `bedrooms`, `reviews_per_month`, `review_scores_*`）使用中位数；
   - 类别/字符串（如 `host_since`, `name`, `host_location`）使用众数；
   - 日期字段 (`first_review`, `last_review`) 填充为 `Unknown`，后续特征工程根据日期再转换。

4. **保留关键信息 / Preserve Key Signals**
   - 删除 `bathrooms` 但保留 `bathrooms_text`，后续在特征工程中提取 `bathrooms_numeric`；
   - 对 `license`、`host_response_rate` 等高缺失字段保留缺失指示（缺失本身即信息）。

5. **质量校验 / Quality Checks**
   - 清洗后数据无缺失值；
   - 创建缺失日志 `data/cleaned/cleaning_log.csv`（百分比均以 “XX.XX%” 展示）。

---

### 特征工程步骤 / Feature Engineering Steps

1. **基础转换 / Basic Transformations**
   - 价格清洗：`price_clean`, `log_price`, `price_per_person`。
   - 百分比转换：`host_response_rate`, `host_acceptance_rate` (字符串 → 浮点)。
   - 布尔/类别编码：`host_is_superhost_flag`, `instant_bookable_flag`, `room_type_encoded`, `property_type_encoded`, `neighbourhood_encoded`。

2. **缺失信息利用 / Missing Information**
   - 12 个 `_is_missing` 指示变量保留缺失含义（如 `license_is_missing`, `host_response_time_is_missing`）；
   - 对填充值（如 `Unknown`）的列保留缺失程度，同时在特征工程中加入实际值。

3. **评论与时间特征 / Reviews & Time Features**
   - `log_reviews_per_month`, `log_number_of_reviews`, `reviews_growth_ratio`；
   - 近期活跃度：`days_since_last_review`, `recent_review_flag`, `recent_review_score = exp(-days_since_last_review / 365)`；
   - 房源年龄：`listing_age_days`。

4. **房东行为特征 / Host Behaviour**
   - `host_activity_score = 0.4*响应率 + 0.3*接受率 + 0.3*超赞房东`；
   - `host_verifications_count`, `host_has_gov_id`；
   - `host_listings_count`, `host_total_listings_count`。

5. **设施特征 / Amenities**
   - 设施数量：`amenities_count`；
   - 设施得分：`amenity_comfort_score` + 四大类计分 (`luxury/family/business/safety`)；
   - 关键设施 Flag（Wi-Fi、厨房、空调、电视等 12 项）。

6. **文本特征 / Text Features**
   - 文本长度：`description_length`, `neighborhood_desc_length`, `host_about_length`；
   - 文本嵌入：对 `description` 与 `neighborhood_overview` 生成 20 维 TF-IDF+SVD 向量 (`desc_embed_*`, `neighborhood_embed_*`)；
   - 情感分析：`desc_sentiment_compound`, `neighborhood_sentiment_compound`, `host_about_sentiment_compound`, 以及综合 `text_sentiment_score`。

7. **地理位置 / Location**
   - `distance_to_center_km`（与市中心直线距离）；
   - `is_central`：距离 ≤ 5 km 标记为 1。

8. **输出 / Output**
   - 最终训练特征：`data/processed/train_data.csv`（仅保留建模所需特征，剔除标识字段与冗余指示变量）；
   - 特征列表注册：`docs/feature_registry.xlsx`（与实际 114 个特征完全一致）。

---

### 结论 / Conclusion
- **数据清洗**：保留关键信息、缺失处理合理、日志透明；
- **特征工程**：覆盖价格、位置、房东、设施、文本、评论等多维度；包含缺失指示、文本嵌入、情感分析等高级特征；
- **可直接用于建模**：特征全部为数值型，目标变量保留，标识字段已剔除。

