"""
Chapter 8 (Extension): Rating Geospatial Heatmap (Merged 2021 & 2025)
第8章（扩展）：评分地理热力图（合并 2021 & 2025 数据）

本脚本基于合并后的 listings 数据（2021 + 2025），绘制房源评分（review_scores_rating）
在阿姆斯特丹空间上的分布热力图（交互式 HTML + 可选静态 PNG）。
This script uses the merged listings dataset (2021 + 2025) to create a geospatial
heatmap of listing ratings (review_scores_rating) in Amsterdam (interactive HTML
and an optional static PNG version).
"""

import os
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils import setup_plotting, get_project_paths

# ---------------------------------------------------------------------------
# Optional folium import for interactive heatmap
# 可选导入 folium，用于生成交互式热力图
# ---------------------------------------------------------------------------
try:
    import folium
    from folium.plugins import HeatMap
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False
    print("  ⚠️ 警告 / Warning: folium 未安装，将仅生成静态热力图（如有）")
    print("  ⚠️ Warning: folium not installed, only static heatmap will be created (if any)")

warnings.filterwarnings("ignore")

# 设置绘图和路径 / Setup plotting and paths
setup_plotting()
project_root, data_dir, charts_eda_dir, charts_model_dir = get_project_paths()
charts_dir = charts_eda_dir  # 与其他 EDA 图保持一致 / Keep consistent with other EDA charts

print("=" * 80)
print("Chapter 8 Extension: Rating Geospatial Heatmap (Merged 2021 & 2025)")
print("第8章扩展：评分地理热力图（合并 2021 & 2025）")
print("=" * 80)

# ============================================================================
# 1. 加载合并后的数据 / Load merged listings data
# ============================================================================

print("\n1. 加载合并后的 listings 数据 / Loading merged listings data...")

merged_path = project_root / "data" / "merged" / "listings_merged_2021_2025.csv"

if not merged_path.exists():
    raise FileNotFoundError(
        f"找不到合并后的数据文件 / Merged file not found: {merged_path}"
    )

df = pd.read_csv(merged_path)
print(f"  ✅ 数据加载完成 / Loaded data: {len(df):,} rows × {len(df.columns)} columns")

# 只保留有经纬度和评分的数据行 / Keep rows with valid lat, lon and rating
required_cols = ["latitude", "longitude", "review_scores_rating"]
missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    raise KeyError(
        f"数据缺少必要列 / Required columns missing: {missing_cols}"
    )

df_clean = df.dropna(subset=required_cols).copy()

# 评分一般在 0–100，将异常值过滤掉 / Filter out unreasonable rating values
df_clean = df_clean[
    (df_clean["review_scores_rating"] >= 0) & (df_clean["review_scores_rating"] <= 100)
]

print(
    f"  ✅ 清洗后样本数 / Valid samples after cleaning: {len(df_clean):,} "
    f"(占比 / share: {len(df_clean) / len(df):.1%})"
)

# 如果数据量过大，做一点采样以控制文件大小（仅用于静态图）
# If too many rows, optionally subsample for static plot
STATIC_SAMPLE_MAX = 50000
if len(df_clean) > STATIC_SAMPLE_MAX:
    df_static = df_clean.sample(STATIC_SAMPLE_MAX, random_state=42)
else:
    df_static = df_clean

# ============================================================================
# 2. 计算地图中心坐标 / Compute map center coordinates
# ============================================================================

lat_mean = df_clean["latitude"].mean()
lon_mean = df_clean["longitude"].mean()
amsterdam_center = [lat_mean, lon_mean]

print(
    f"\n2. 地图中心坐标 / Map center: "
    f"({amsterdam_center[0]:.4f}, {amsterdam_center[1]:.4f})"
)

# ============================================================================
# 3. 创建交互式评分热力图（HTML）/ Create interactive rating heatmap (HTML)
# ============================================================================

heatmap_created = False
html_path = charts_dir / "chapter8_rating_heatmap_merged.html"

if FOLIUM_AVAILABLE:
    print("\n3. 创建交互式评分热力图 / Creating interactive rating heatmap...")

    # 准备热力图数据：[lat, lon, weight]，这里 weight=评分 / Prepare heatmap data
    # 为了避免个别极端值影响视觉效果，可以进行归一化 / Optionally normalize
    rating = df_clean["review_scores_rating"].astype(float)
    rating_min, rating_max = rating.min(), rating.max()
    if rating_max > rating_min:
        weight = (rating - rating_min) / (rating_max - rating_min)
    else:
        weight = np.ones_like(rating)

    heat_data = [
        [row["latitude"], row["longitude"], w]
        for row, w in zip(df_clean.to_dict("records"), weight)
    ]

    # 创建基础地图 / Base map
    rating_map = folium.Map(
        location=amsterdam_center,
        zoom_start=12,
        tiles="OpenStreetMap",
    )

    # 备选底图样式 / Alternative tile style
    folium.TileLayer(
        tiles="CartoDB positron",
        name="CartoDB Positron",
        overlay=False,
        control=True,
    ).add_to(rating_map)

    # 添加热力图图层 / Add heatmap layer
    HeatMap(
        heat_data,
        min_opacity=0.2,
        max_zoom=18,
        radius=15,
        blur=15,
        gradient={
            0.0: "blue",    # 低评分 / Low rating
            0.3: "cyan",
            0.5: "lime",
            0.7: "yellow",
            1.0: "red",     # 高评分 / High rating
        },
    ).add_to(rating_map)

    # 可选：标出满分/接近满分房源（5 分）/ Optionally highlight near-perfect listings (≈5 stars)
    # 注意：原始字段为 0–100 分，这里将 ≥ 4.95 星约等价为 ≥ 99 分
    # Note: raw field is 0–100; we approximate 4.95–5.0 stars as ≥ 99 points.
    high_rating_threshold = 99.0
    high_rating = df_clean[df_clean["review_scores_rating"] >= high_rating_threshold]
    # 为避免标记过多，只取前 50 个 / To avoid too many markers, keep at most first 50
    high_rating = high_rating.sort_values(
        "review_scores_rating", ascending=False
    ).head(50)

    for _, row in high_rating.iterrows():
        popup_parts = [
            f"Rating: {row['review_scores_rating']:.0f} (≈5 stars)",
        ]
        if "neighbourhood" in row and not pd.isna(row["neighbourhood"]):
            popup_parts.append(f"Neighbourhood: {row['neighbourhood']}")
        popup_html = "<br>".join(popup_parts)

        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=5,
            popup=popup_html,
            color="darkred",
            fill=True,
            fillColor="red",
            fillOpacity=0.8,
        ).add_to(rating_map)

    # 添加图层控制 / Layer control
    folium.LayerControl().add_to(rating_map)

    # 添加标题 / Add title
    title_html = """
    <div style="position: fixed;
                top: 10px; left: 50px; width: 420px; height: 90px;
                background-color: white; z-index:9999; font-size:14px;
                border:2px solid grey; padding: 10px">
      <h4 style="margin-top: 0;">Rating Heatmap (Merged 2021 & 2025)</h4>
      <p style="margin-bottom: 0;">
        Amsterdam Airbnb Listings<br>
        Redder colors indicate higher ratings
      </p>
    </div>
    """
    rating_map.get_root().html.add_child(folium.Element(title_html))

    # 保存 HTML / Save HTML
    rating_map.save(str(html_path))
    heatmap_created = True

    print(f"  ✅ 已保存交互式热力图 / Saved interactive heatmap: {html_path}")
    print(f"  📊 热力图数据点 / Heatmap data points: {len(heat_data):,}")
else:
    print("\n3. 跳过交互式热力图（folium 未安装）/ Skipping interactive heatmap (folium not installed)")

# ============================================================================
# 4. 创建静态评分热力图 PNG（可选）/ Create static rating heatmap PNG (optional)
# ============================================================================

static_png_path = charts_dir / "chapter8_rating_heatmap_merged_static.png"

print("\n4. 创建静态评分热力图 / Creating static rating heatmap (hexbin)...")

fig, ax = plt.subplots(figsize=(14, 10))

hb = ax.hexbin(
    df_static["longitude"],
    df_static["latitude"],
    C=df_static["review_scores_rating"],
    gridsize=50,
    cmap="viridis",
    reduce_C_function=np.mean,
    mincnt=1,
)

ax.set_title(
    "Rating Heatmap of Amsterdam Airbnb Listings (Merged 2021 & 2025)",
    fontsize=14,
    fontweight="bold",
    pad=20,
)
ax.set_xlabel("Longitude", fontsize=12)
ax.set_ylabel("Latitude", fontsize=12)

cb = plt.colorbar(hb, ax=ax, label="Average Rating (0–100)")
ax.grid(True, alpha=0.3, linestyle="--")

plt.tight_layout()
plt.savefig(static_png_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"  ✅ 已保存静态热力图 / Saved static heatmap: {static_png_path}")


# ============================================================================
# 5. 评论数量 vs. 高分关系图 / Reviews vs. High Rating Relationship
# ============================================================================

print("\n5. 绘制评论数量与评分关系图 / Plotting reviews vs. rating relationship...")

reviews_plot_path = charts_dir / "chapter8_rating_vs_reviews.png"

if "number_of_reviews" not in df_clean.columns:
    print("  ⚠️ 缺少 number_of_reviews 列，无法绘制关系图 / Missing number_of_reviews column.")
else:
    # 准备数据：保留评论数 ≥0 的记录，并取样以减轻绘图压力
    # Prepare data: keep non-negative review counts and optionally sample
    df_reviews = df_clean[
        (df_clean["number_of_reviews"].notna())
        & (df_clean["number_of_reviews"] >= 0)
    ].copy()

    # 对评论数做 log10 变换（使用 log10(1 + x) 防止 0 问题）
    # Apply log10 transform on review counts: log10(1 + x) to handle zeros
    df_reviews["log_number_of_reviews"] = np.log10(1.0 + df_reviews["number_of_reviews"])

    REVIEWS_SAMPLE_MAX = 60000
    if len(df_reviews) > REVIEWS_SAMPLE_MAX:
        df_reviews = df_reviews.sample(REVIEWS_SAMPLE_MAX, random_state=42)

    fig, ax = plt.subplots(figsize=(12, 7))

    hb = ax.hexbin(
        df_reviews["log_number_of_reviews"],
        df_reviews["review_scores_rating"],
        gridsize=60,
        cmap="magma",
        mincnt=5,
    )

    ax.axhline(
        high_rating_threshold,
        color="red",
        linestyle="--",
        linewidth=1.2,
        label="≈5-star Threshold (rating ≥ 99)",
    )
    ax.set_xlabel("log10(Number of Reviews + 1)", fontsize=12)
    ax.set_ylabel("Review Scores Rating (0–100)", fontsize=12)
    ax.set_title(
        "Relationship Between Review Counts and Ratings (Merged 2021 & 2025)",
        fontsize=14,
        fontweight="bold",
    )

    cb = plt.colorbar(hb, ax=ax, label="Listings Count")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(reviews_plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"  ✅ 已保存评论-评分关系图 / Saved reviews-rating plot: {reviews_plot_path}")

print("\n" + "=" * 80)
print("评分地理热力图生成完成 / Rating geospatial heatmap generation complete!")
print("=" * 80)


