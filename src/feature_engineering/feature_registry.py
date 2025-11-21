"""
特征字典与登记工具 / Feature metadata registry utilities

记录模型当前使用的特征信息，并导出为 Excel
Record metadata for all model features and export to Excel
"""

from pathlib import Path
from typing import List, Dict

import pandas as pd


FEATURE_METADATA: List[Dict[str, str]] = [
    # Target
    {
        "name": "review_scores_rating",
        "description": "房源综合评分（0-5）",
        "feature_type": "原始字段",
        "logic": "来自 listings_detailed 的原始评分，保留 0-5 范围",
        "example": "4.85",
        "source": "listings_cleaned.csv",
    },
    # Pricing & availability
    {
        "name": "price_clean",
        "description": "清洗后的每晚价格（欧元）",
        "feature_type": "衍生字段",
        "logic": "移除价格中的 $ 和逗号并转换为数值，缺失填 0",
        "example": "150",
        "source": "listings_cleaned.csv",
    },
    {
        "name": "log_price",
        "description": "价格的对数变换",
        "feature_type": "衍生字段",
        "logic": "log1p(price_clean)",
        "example": "5.01",
        "source": "派生自 price_clean",
    },
    {
        "name": "price_per_person",
        "description": "人均价格",
        "feature_type": "衍生字段",
        "logic": "price_clean / accommodates（若分母为 0 则用 price_clean）",
        "example": "45",
        "source": "listings_cleaned.csv",
    },
    {
        "name": "availability_ratio",
        "description": "年度可用比例",
        "feature_type": "衍生字段",
        "logic": "availability_365 / 365",
        "example": "0.25",
        "source": "listings_cleaned.csv",
    },
    {
        "name": "occupancy_rate",
        "description": "估算入住率",
        "feature_type": "衍生字段",
        "logic": "1 - availability_ratio",
        "example": "0.75",
        "source": "派生自 availability_ratio",
    },
    # Reviews
    {
        "name": "reviews_per_month",
        "description": "每月平均评论数",
        "feature_type": "原始字段",
        "logic": "缺失值填 0",
        "example": "1.2",
        "source": "listings_cleaned.csv",
    },
    {
        "name": "log_reviews_per_month",
        "description": "每月评论数的对数变换",
        "feature_type": "衍生字段",
        "logic": "log1p(reviews_per_month)",
        "example": "0.79",
        "source": "派生自 reviews_per_month",
    },
    {
        "name": "number_of_reviews",
        "description": "总评论数",
        "feature_type": "原始字段",
        "logic": "缺失值填 0",
        "example": "85",
        "source": "listings_cleaned.csv",
    },
    {
        "name": "log_number_of_reviews",
        "description": "总评论数的对数变换",
        "feature_type": "衍生字段",
        "logic": "log1p(number_of_reviews)",
        "example": "4.45",
        "source": "派生自 number_of_reviews",
    },
    {
        "name": "number_of_reviews_ltm",
        "description": "近 12 个月评论数",
        "feature_type": "原始字段",
        "logic": "缺失值填 0",
        "example": "18",
        "source": "listings_cleaned.csv",
    },
    {
        "name": "reviews_growth_ratio",
        "description": "评论增长比率",
        "feature_type": "衍生字段",
        "logic": "(number_of_reviews_l30d + 1) / (number_of_reviews_ltm + 1)",
        "example": "0.25",
        "source": "listings_cleaned.csv",
    },
    # Host behaviour
    {
        "name": "host_response_rate",
        "description": "房东回复率",
        "feature_type": "衍生字段",
        "logic": "百分比字段转浮点，缺失填 0",
        "example": "0.98",
        "source": "listings_cleaned.csv",
    },
    {
        "name": "host_acceptance_rate",
        "description": "房东接受率",
        "feature_type": "衍生字段",
        "logic": "百分比字段转浮点，缺失填 0",
        "example": "0.90",
        "source": "listings_cleaned.csv",
    },
    {
        "name": "host_activity_score",
        "description": "房东活跃度评分",
        "feature_type": "衍生字段",
        "logic": "0.4*host_response_rate + 0.3*host_acceptance_rate + 0.3*host_is_superhost_flag",
        "example": "0.92",
        "source": "派生自 host 行为字段",
    },
    {
        "name": "host_verifications_count",
        "description": "房东验证方式数量",
        "feature_type": "衍生字段",
        "logic": "解析 host_verifications 列表/字典，统计验证方式数量",
        "example": "3",
        "source": "listings_cleaned.csv",
    },
    {
        "name": "host_has_gov_id",
        "description": "是否提供政府ID验证",
        "feature_type": "衍生字段",
        "logic": "host_verifications 中包含 'government' → 1，否则 0",
        "example": "1",
        "source": "listings_cleaned.csv",
    },
    {
        "name": "host_is_superhost_flag",
        "description": "是否为超赞房东",
        "feature_type": "衍生字段",
        "logic": "host_is_superhost in (t,true,1) → 1，否则 0",
        "example": "1",
        "source": "listings_cleaned.csv",
    },
    {
        "name": "host_listings_count",
        "description": "房东当前房源数",
        "feature_type": "原始字段",
        "logic": "缺失填 0",
        "example": "3",
        "source": "listings_cleaned.csv",
    },
    {
        "name": "host_total_listings_count",
        "description": "房东历史房源数",
        "feature_type": "原始字段",
        "logic": "缺失填 0",
        "example": "6",
        "source": "listings_cleaned.csv",
    },
    # Booking preference
    {
        "name": "instant_bookable_flag",
        "description": "是否支持即时预订",
        "feature_type": "衍生字段",
        "logic": "instant_bookable in (t,true,1) → 1，否则 0",
        "example": "0",
        "source": "listings_cleaned.csv",
    },
    {
        "name": "has_license_info",
        "description": "是否提供许可证信息",
        "feature_type": "衍生字段",
        "logic": "license 字段非空 → 1，否则 0",
        "example": "1",
        "source": "listings_cleaned.csv",
    },
    # Encoded categorical fields
    {
        "name": "room_type_encoded",
        "description": "房型编码",
        "feature_type": "衍生字段",
        "logic": "room_type 转为类别编码（整租/独立房间等）",
        "example": "0=Entire home/apt",
        "source": "listings_cleaned.csv",
    },
    {
        "name": "property_type_encoded",
        "description": "房产类型编码",
        "feature_type": "衍生字段",
        "logic": "property_type 转为类别编码",
        "example": "12=Houseboat",
        "source": "listings_cleaned.csv",
    },
    {
        "name": "neighbourhood_encoded",
        "description": "街区编码",
        "feature_type": "衍生字段",
        "logic": "neighbourhood_cleansed 转为类别编码",
        "example": "5=Centrum-West",
        "source": "listings_cleaned.csv",
    },
    # Capacity fields
    {
        "name": "accommodates",
        "description": "可容纳人数",
        "feature_type": "原始字段",
        "logic": "缺失填 1",
        "example": "4",
        "source": "listings_cleaned.csv",
    },
    {
        "name": "bedrooms",
        "description": "卧室数量",
        "feature_type": "原始字段",
        "logic": "缺失填 'missing'",
        "example": "2",
        "source": "listings_cleaned.csv",
    },
    {
        "name": "beds",
        "description": "床位数量",
        "feature_type": "原始字段",
        "logic": "缺失填 'missing'",
        "example": "3",
        "source": "listings_cleaned.csv",
    },
    {
        "name": "bathrooms_numeric",
        "description": "浴室数量（数值化）",
        "feature_type": "衍生字段",
        "logic": "从 bathrooms_text 中提取数字，缺失填 0",
        "example": "1.5",
        "source": "listings_cleaned.csv",
    },
    {
        "name": "is_shared_bath",
        "description": "是否为共享浴室",
        "feature_type": "衍生字段",
        "logic": "bathrooms_text 中包含 'shared' → 1，否则 0",
        "example": "0",
        "source": "listings_cleaned.csv",
    },
    # Amenities
    {
        "name": "amenities_count",
        "description": "设施数量",
        "feature_type": "衍生字段",
        "logic": "解析 amenities 列表后的元素数量",
        "example": "35",
        "source": "listings_cleaned.csv",
    },
    {
        "name": "amenity_comfort_score",
        "description": "设施舒适度综合得分",
        "feature_type": "衍生字段",
        "logic": "luxury/family/business/safety 计分 + 0.05*amenities_count",
        "example": "4.8",
        "source": "派生自 amenities",
    },
    {
        "name": "amenity_score_luxury",
        "description": "高端设施数量",
        "feature_type": "衍生字段",
        "logic": "统计 hot tub/pool/gym 等关键词出现次数",
        "example": "2",
        "source": "派生自 amenities",
    },
    {
        "name": "amenity_score_family",
        "description": "亲子设施数量",
        "feature_type": "衍生字段",
        "logic": "统计 crib/high chair/baby monitor 等关键词出现次数",
        "example": "1",
        "source": "派生自 amenities",
    },
    {
        "name": "amenity_score_business",
        "description": "商务设施数量",
        "feature_type": "衍生字段",
        "logic": "统计 workspace/desk/printer 等关键词出现次数",
        "example": "3",
        "source": "派生自 amenities",
    },
    {
        "name": "amenity_score_safety",
        "description": "安全设施数量",
        "feature_type": "衍生字段",
        "logic": "统计 smoke detector/first aid kit 等关键词出现次数",
        "example": "2",
        "source": "派生自 amenities",
    },
] + [
    {
        "name": f"amenity_has_{suffix}",
        "description": f"是否提供 {label}",
        "feature_type": "衍生字段",
        "logic": f"amenities 中出现 '{label.lower()}' → 1，否则 0",
        "example": "1",
        "source": "listings_cleaned.csv",
    }
    for suffix, label in [
        ("wifi", "Wi-Fi"),
        ("kitchen", "厨房"),
        ("heating", "暖气"),
        ("air_conditioning", "空调"),
        ("washer", "洗衣机"),
        ("dryer", "烘干机"),
        ("tv", "电视"),
        ("dishwasher", "洗碗机"),
        ("parking", "停车位"),
        ("balcony", "阳台"),
        ("elevator", "电梯"),
        ("coffee_maker", "咖啡机"),
    ]
] + [
    # Text features
    {
        "name": "description_length",
        "description": "房源描述字数",
        "feature_type": "衍生字段",
        "logic": "description 文本长度，缺失按 0 计算",
        "example": "520",
        "source": "listings_cleaned.csv",
    },
    {
        "name": "neighborhood_desc_length",
        "description": "社区介绍字数",
        "feature_type": "衍生字段",
        "logic": "neighborhood_overview 文本长度",
        "example": "180",
        "source": "listings_cleaned.csv",
    },
    {
        "name": "host_about_length",
        "description": "房东自我介绍字数",
        "feature_type": "衍生字段",
        "logic": "host_about 文本长度",
        "example": "75",
        "source": "listings_cleaned.csv",
    },
    # Time features
    {
        "name": "listing_age_days",
        "description": "房源上线天数",
        "feature_type": "衍生字段",
        "logic": "2021-09-07 与 first_review 之间的天数",
        "example": "1800",
        "source": "listings_cleaned.csv",
    },
    {
        "name": "days_since_last_review",
        "description": "距离上一条评论的天数",
        "feature_type": "衍生字段",
        "logic": "2021-09-07 与 last_review 的天数（无评论记 9999）",
        "example": "45",
        "source": "listings_cleaned.csv",
    },
    {
        "name": "recent_review_flag",
        "description": "30 天内有新评论标记",
        "feature_type": "衍生字段",
        "logic": "days_since_last_review ≤ 30 → 1，否则 0",
        "example": "0",
        "source": "派生自 last_review",
    },
    {
        "name": "recent_review_score",
        "description": "近期评论活跃度得分",
        "feature_type": "衍生字段",
        "logic": "exp(-days_since_last_review / 365)",
        "example": "0.8",
        "source": "派生自评论字段",
    },
    # Location
    {
        "name": "distance_to_center_km",
        "description": "距离市中心的直线距离（公里）",
        "feature_type": "衍生字段",
        "logic": "根据 latitude/longitude 与市中心 (52.3676, 4.9041) 计算 haversine 距离",
        "example": "3.2",
        "source": "listings_cleaned.csv",
    },
    {
        "name": "is_central",
        "description": "是否位于中心 5KM 内",
        "feature_type": "衍生字段",
        "logic": "distance_to_center_km ≤ 5 → 1，否则 0",
        "example": "1",
        "source": "派生自 distance_to_center_km",
    },
    # Missing indicator variables
    {
        "name": "availability_eoy_is_missing",
        "description": "availability_eoy 字段是否缺失",
        "feature_type": "缺失指示变量",
        "logic": "原始数据中 availability_eoy 缺失 → 1，否则 0",
        "example": "1",
        "source": "listings_cleaned.csv",
    },
    {
        "name": "estimated_occupancy_l365d_is_missing",
        "description": "estimated_occupancy_l365d 字段是否缺失",
        "feature_type": "缺失指示变量",
        "logic": "原始数据中 estimated_occupancy_l365d 缺失 → 1，否则 0",
        "example": "1",
        "source": "listings_cleaned.csv",
    },
    {
        "name": "host_about_is_missing",
        "description": "host_about 字段是否缺失",
        "feature_type": "缺失指示变量",
        "logic": "原始数据中 host_about 缺失 → 1，否则 0",
        "example": "0",
        "source": "listings_cleaned.csv",
    },
    {
        "name": "host_acceptance_rate_is_missing",
        "description": "host_acceptance_rate 字段是否缺失",
        "feature_type": "缺失指示变量",
        "logic": "原始数据中 host_acceptance_rate 缺失 → 1，否则 0",
        "example": "1",
        "source": "listings_cleaned.csv",
    },
    {
        "name": "host_neighbourhood_is_missing",
        "description": "host_neighbourhood 字段是否缺失",
        "feature_type": "缺失指示变量",
        "logic": "原始数据中 host_neighbourhood 缺失 → 1，否则 0",
        "example": "1",
        "source": "listings_cleaned.csv",
    },
    {
        "name": "host_response_rate_is_missing",
        "description": "host_response_rate 字段是否缺失",
        "feature_type": "缺失指示变量",
        "logic": "原始数据中 host_response_rate 缺失 → 1，否则 0",
        "example": "1",
        "source": "listings_cleaned.csv",
    },
    {
        "name": "host_response_time_is_missing",
        "description": "host_response_time 字段是否缺失",
        "feature_type": "缺失指示变量",
        "logic": "原始数据中 host_response_time 缺失 → 1，否则 0",
        "example": "1",
        "source": "listings_cleaned.csv",
    },
    {
        "name": "neighborhood_overview_is_missing",
        "description": "neighborhood_overview 字段是否缺失",
        "feature_type": "缺失指示变量",
        "logic": "原始数据中 neighborhood_overview 缺失 → 1，否则 0",
        "example": "0",
        "source": "listings_cleaned.csv",
    },
    {
        "name": "neighbourhood_is_missing",
        "description": "neighbourhood 字段是否缺失",
        "feature_type": "缺失指示变量",
        "logic": "原始数据中 neighbourhood 缺失 → 1，否则 0",
        "example": "0",
        "source": "listings_cleaned.csv",
    },
    {
        "name": "number_of_reviews_ly_is_missing",
        "description": "number_of_reviews_ly 字段是否缺失",
        "feature_type": "缺失指示变量",
        "logic": "原始数据中 number_of_reviews_ly 缺失 → 1，否则 0",
        "example": "1",
        "source": "listings_cleaned.csv",
    },
    {
        "name": "source_is_missing",
        "description": "source 字段是否缺失",
        "feature_type": "缺失指示变量",
        "logic": "原始数据中 source 缺失 → 1，否则 0",
        "example": "1",
        "source": "listings_cleaned.csv",
    },
    # Text embeddings
] + [
    {
        "name": f"desc_embed_{i}",
        "description": f"房源描述文本嵌入第{i}维",
        "feature_type": "文本嵌入",
        "logic": "使用 TF-IDF + SVD 对 description 字段进行降维，共20维",
        "example": "0.023",
        "source": "listings_cleaned.csv",
    }
    for i in range(20)
] + [
    {
        "name": f"neighborhood_embed_{i}",
        "description": f"社区介绍文本嵌入第{i}维",
        "feature_type": "文本嵌入",
        "logic": "使用 TF-IDF + SVD 对 neighborhood_overview 字段进行降维，共20维",
        "example": "0.015",
        "source": "listings_cleaned.csv",
    }
    for i in range(20)
] + [
    # Sentiment analysis features
    {
        "name": "desc_sentiment_compound",
        "description": "房源描述情感得分",
        "feature_type": "情感分析",
        "logic": "使用 VADER 情感分析器分析 description 文本的复合情感得分（-1到1）",
        "example": "0.65",
        "source": "listings_cleaned.csv",
    },
    {
        "name": "neighborhood_sentiment_compound",
        "description": "社区介绍情感得分",
        "feature_type": "情感分析",
        "logic": "使用 VADER 情感分析器分析 neighborhood_overview 文本的复合情感得分",
        "example": "0.42",
        "source": "listings_cleaned.csv",
    },
    {
        "name": "host_about_sentiment_compound",
        "description": "房东介绍情感得分",
        "feature_type": "情感分析",
        "logic": "使用 VADER 情感分析器分析 host_about 文本的复合情感得分",
        "example": "0.38",
        "source": "listings_cleaned.csv",
    },
    {
        "name": "text_sentiment_score",
        "description": "综合文本情感得分",
        "feature_type": "情感分析",
        "logic": "0.5*desc_sentiment + 0.3*neighborhood_sentiment + 0.2*host_about_sentiment",
        "example": "0.52",
        "source": "派生自文本情感分析",
    },
    # Additional features
    {
        "name": "amenity_has_long_term_stays_allowed",
        "description": "是否允许长期住宿",
        "feature_type": "衍生字段",
        "logic": "amenities 中出现 'long term stays allowed' → 1，否则 0",
        "example": "1",
        "source": "listings_cleaned.csv",
    },
    {
        "name": "amenity_has_pets_allowed",
        "description": "是否允许宠物",
        "feature_type": "衍生字段",
        "logic": "amenities 中出现 'pets allowed' → 1，否则 0",
        "example": "1",
        "source": "listings_cleaned.csv",
    },
]


_FEATURE_NAME_SET = {entry["name"] for entry in FEATURE_METADATA}


def register_feature(metadata: Dict[str, str]) -> None:
    """
    动态注册新特征 / Register a new feature dynamically.

    Args:
        metadata: 包含 name/description/feature_type/logic/example/source 的字典
    """
    name = metadata.get("name")
    if not name:
        raise ValueError("metadata 必须包含 name 字段 / metadata must include 'name'")
    if name in _FEATURE_NAME_SET:
        return
    FEATURE_METADATA.append(metadata)
    _FEATURE_NAME_SET.add(name)


def write_feature_registry(
    output_path: Path = Path("docs/feature_registry.xlsx"),
) -> Path:
    """
    将特征字典写入 Excel / Write feature metadata to Excel.

    Args:
        output_path: 输出文件路径 / Output Excel path.

    Returns:
        Path: 实际保存路径 / Saved path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(FEATURE_METADATA)
    df = df[
        [
            "name",
            "description",
            "feature_type",
            "logic",
            "example",
            "source",
        ]
    ]
    df.columns = [
        "特征名称",
        "特征含义",
        "字段类型",
        "计算/处理方式",
        "取值示例",
        "数据来源",
    ]
    df.to_excel(output_path, index=False)
    return output_path


if __name__ == "__main__":
    path = write_feature_registry()
    print(f"[OK] 特征登记表已更新: {path}")

