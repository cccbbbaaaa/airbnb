"""
Chapter 8: Geospatial Analysis
第8章：地理空间分析

本脚本进行地理空间分析，包括房源地理分布、地理位置与价格关系、地理位置与受欢迎度关系等。
This script performs geospatial analysis, including geographic distribution of listings, location-price relationships, and location-popularity relationships.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
from utils import setup_plotting, get_project_paths

# 尝试导入 folium，如果失败则跳过热力图生成
# Try to import folium, skip heatmap generation if failed
try:
    import folium
    from folium.plugins import HeatMap
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False
    print("  ⚠️ 警告 / Warning: folium 未安装，将跳过交互式热力图生成")
    print("  ⚠️ Warning: folium not installed, skipping interactive heatmap generation")

# 设置中文字体支持 / Set Chinese font support
setup_plotting()

# 获取项目根目录路径 / Get project root directory path
project_root, data_dir, charts_eda_dir, charts_model_dir = get_project_paths()
charts_dir = charts_eda_dir  # 使用 EDA 目录 / Use EDA directory

print("=" * 80)
print("Chapter 8: Geospatial Analysis")
print("第8章：地理空间分析")
print("=" * 80)

# ============================================================================
# 1. 加载数据 / Load Data
# ============================================================================

print("\n1. 加载数据 / Loading Data...")

listings = pd.read_csv(data_dir / 'listings.csv')

# 数据清洗 / Data Cleaning
if 'neighbourhood_group' in listings.columns:
    listings = listings.drop('neighbourhood_group', axis=1)

listings['last_review'] = listings['last_review'].fillna(0)
listings['reviews_per_month'] = listings['reviews_per_month'].fillna(0)
listings['name'] = listings['name'].fillna('blank_name')
listings['host_name'] = listings['host_name'].fillna('blank_host_name')

# 处理异常值 / Handle Outliers
listings.loc[listings['minimum_nights'] > 365, 'minimum_nights'] = 365
listings.loc[listings['price'] == 0, 'price'] = np.nan

print(f"  ✅ 数据加载完成: {len(listings)} 行 × {len(listings.columns)} 列")

# ============================================================================
# 2. 房源地理分布分析 / Geographic Distribution Analysis
# ============================================================================

print("\n2. 房源地理分布分析 / Geographic Distribution Analysis...")

# 2.1 街区房源分布
neighbourhood_counts = listings['neighbourhood'].value_counts().sort_values(ascending=False)
print("\n2.1 街区房源分布（Top 10）/ Neighbourhood Distribution (Top 10):")
for i, (neighbourhood, count) in enumerate(neighbourhood_counts.head(10).items(), 1):
    pct = (count / len(listings) * 100)
    print(f"  {i:2d}. {neighbourhood}: {count:,} ({pct:.1f}%)")

# 2.2 地理坐标统计
print("\n2.2 地理坐标统计 / Geographic Coordinates Statistics:")
print(f"  - 纬度范围 / Latitude Range: [{listings['latitude'].min():.4f}, {listings['latitude'].max():.4f}]")
print(f"  - 经度范围 / Longitude Range: [{listings['longitude'].min():.4f}, {listings['longitude'].max():.4f}]")
print(f"  - 纬度均值 / Latitude Mean: {listings['latitude'].mean():.4f}")
print(f"  - 经度均值 / Longitude Mean: {listings['longitude'].mean():.4f}")

# 2.3 房源密度分析（按街区）
neighbourhood_density = listings.groupby('neighbourhood').agg({
    'id': 'count',
    'latitude': 'mean',
    'longitude': 'mean'
}).rename(columns={'id': 'count'})
neighbourhood_density = neighbourhood_density.sort_values('count', ascending=False)

print("\n2.3 房源密度分析（Top 10）/ Listing Density Analysis (Top 10):")
for i, (neighbourhood, row) in enumerate(neighbourhood_density.head(10).iterrows(), 1):
    print(f"  {i:2d}. {neighbourhood}: {row['count']:,} 个房源, "
          f"坐标: ({row['latitude']:.4f}, {row['longitude']:.4f})")

# ============================================================================
# 3. 地理位置与价格关系 / Location-Price Relationship
# ============================================================================

print("\n3. 地理位置与价格关系 / Location-Price Relationship...")

# 3.1 各街区平均价格
neighbourhood_price = listings.groupby('neighbourhood')['price'].agg([
    'mean', 'median', 'count', 'std'
]).sort_values('mean', ascending=False)

print("\n3.1 各街区平均价格（Top 10）/ Average Price by Neighbourhood (Top 10):")
for i, (neighbourhood, row) in enumerate(neighbourhood_price.head(10).iterrows(), 1):
    print(f"  {i:2d}. {neighbourhood}: €{row['mean']:.2f} "
          f"(中位数: €{row['median']:.2f}, 房源数: {row['count']:.0f})")

# 3.2 价格地理梯度分析
print("\n3.2 价格地理梯度分析 / Price Geographic Gradient Analysis:")
price_by_lat = listings.groupby(pd.cut(listings['latitude'], bins=10))['price'].mean()
price_by_lon = listings.groupby(pd.cut(listings['longitude'], bins=10))['price'].mean()

print("  按纬度分组平均价格 / Average Price by Latitude Bins:")
for i, (lat_bin, price) in enumerate(price_by_lat.items(), 1):
    print(f"    {i:2d}. {lat_bin}: €{price:.2f}")

print("\n  按经度分组平均价格 / Average Price by Longitude Bins:")
for i, (lon_bin, price) in enumerate(price_by_lon.items(), 1):
    print(f"    {i:2d}. {lon_bin}: €{price:.2f}")

# ============================================================================
# 4. 地理位置与受欢迎度关系 / Location-Popularity Relationship
# ============================================================================

print("\n4. 地理位置与受欢迎度关系 / Location-Popularity Relationship...")

# 4.1 各街区平均评论数
neighbourhood_reviews = listings.groupby('neighbourhood')['number_of_reviews'].agg([
    'mean', 'median', 'count', 'sum'
]).sort_values('mean', ascending=False)

print("\n4.1 各街区平均评论数（Top 10）/ Average Reviews by Neighbourhood (Top 10):")
for i, (neighbourhood, row) in enumerate(neighbourhood_reviews.head(10).iterrows(), 1):
    print(f"  {i:2d}. {neighbourhood}: {row['mean']:.1f} "
          f"(中位数: {row['median']:.1f}, 总评论数: {row['sum']:.0f})")

# 4.2 热门街区识别
print("\n4.2 热门街区识别 / Popular Neighbourhoods Identification:")
# 综合评分：评论数 + 房源数
neighbourhood_score = pd.DataFrame({
    'neighbourhood': neighbourhood_counts.index,
    'listing_count': neighbourhood_counts.values,
    'avg_reviews': neighbourhood_reviews['mean'],
    'total_reviews': neighbourhood_reviews['sum']
})
neighbourhood_score['popularity_score'] = (
    neighbourhood_score['avg_reviews'] * 0.6 + 
    neighbourhood_score['listing_count'] / 100 * 0.4
)
neighbourhood_score = neighbourhood_score.sort_values('popularity_score', ascending=False)

print("  热门街区排名（Top 10）/ Popular Neighbourhoods Ranking (Top 10):")
for i, (idx, row) in enumerate(neighbourhood_score.head(10).iterrows(), 1):
    print(f"  {i:2d}. {row['neighbourhood']}: "
          f"评分 {row['popularity_score']:.2f} "
          f"(房源数: {row['listing_count']:.0f}, 平均评论: {row['avg_reviews']:.1f})")

# ============================================================================
# 5. 创建可视化图表 / Create Visualizations
# ============================================================================

print("\n5. 创建可视化图表 / Creating Visualizations...")

# 5.1 地理分布散点图
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# 5.1.1 房源地理分布散点图
sample_data = listings.sample(min(5000, len(listings)))
scatter = axes[0, 0].scatter(sample_data['longitude'], sample_data['latitude'], 
                             c=sample_data['price'], cmap='viridis', 
                             s=10, alpha=0.5, edgecolors='none')
axes[0, 0].set_title('Geographic Distribution of Listings (Colored by Price)', 
                     fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Longitude', fontsize=11)
axes[0, 0].set_ylabel('Latitude', fontsize=11)
plt.colorbar(scatter, ax=axes[0, 0], label='Price (€)')

# 5.1.2 各街区房源数分布
top_neighbourhoods = neighbourhood_counts.head(15)
axes[0, 1].barh(range(len(top_neighbourhoods)), top_neighbourhoods.values,
                color='skyblue', edgecolor='black')
axes[0, 1].set_yticks(range(len(top_neighbourhoods)))
axes[0, 1].set_yticklabels(top_neighbourhoods.index, fontsize=9)
axes[0, 1].set_title('Top 15 Neighbourhoods by Listing Count', 
                     fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Number of Listings', fontsize=11)
axes[0, 1].invert_yaxis()
for i, v in enumerate(top_neighbourhoods.values):
    axes[0, 1].text(v, i, f' {v:,}', va='center', fontsize=9)

# 5.1.3 各街区平均价格
top_price_neighbourhoods = neighbourhood_price.head(15)
axes[1, 0].barh(range(len(top_price_neighbourhoods)), top_price_neighbourhoods['mean'].values,
                color='coral', edgecolor='black')
axes[1, 0].set_yticks(range(len(top_price_neighbourhoods)))
axes[1, 0].set_yticklabels(top_price_neighbourhoods.index, fontsize=9)
axes[1, 0].set_title('Top 15 Neighbourhoods by Average Price', 
                     fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Average Price (€)', fontsize=11)
axes[1, 0].invert_yaxis()
for i, v in enumerate(top_price_neighbourhoods['mean'].values):
    axes[1, 0].text(v, i, f' €{v:.0f}', va='center', fontsize=9)

# 5.1.4 各街区平均评论数
top_reviews_neighbourhoods = neighbourhood_reviews.head(15)
axes[1, 1].barh(range(len(top_reviews_neighbourhoods)), top_reviews_neighbourhoods['mean'].values,
                color='lightgreen', edgecolor='black')
axes[1, 1].set_yticks(range(len(top_reviews_neighbourhoods)))
axes[1, 1].set_yticklabels(top_reviews_neighbourhoods.index, fontsize=9)
axes[1, 1].set_title('Top 15 Neighbourhoods by Average Reviews', 
                     fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Average Reviews', fontsize=11)
axes[1, 1].invert_yaxis()
for i, v in enumerate(top_reviews_neighbourhoods['mean'].values):
    axes[1, 1].text(v, i, f' {v:.1f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(charts_dir / 'chapter8_geospatial_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✅ 已保存: chapter8_geospatial_analysis.png")

# 5.2 价格与地理位置关系分析
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 5.2.1 价格 vs 纬度散点图
sample_data = listings.dropna(subset=['price']).sample(min(5000, len(listings)))
axes[0].scatter(sample_data['latitude'], sample_data['price'], 
                alpha=0.3, s=10, color='purple')
axes[0].set_title('Price vs Latitude', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Latitude', fontsize=11)
axes[0].set_ylabel('Price (€)', fontsize=11)
axes[0].set_ylim(0, min(500, listings['price'].quantile(0.95)))
axes[0].grid(True, alpha=0.3)

# 5.2.2 价格 vs 经度散点图
axes[1].scatter(sample_data['longitude'], sample_data['price'], 
                alpha=0.3, s=10, color='orange')
axes[1].set_title('Price vs Longitude', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Longitude', fontsize=11)
axes[1].set_ylabel('Price (€)', fontsize=11)
axes[1].set_ylim(0, min(500, listings['price'].quantile(0.95)))
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(charts_dir / 'chapter8_location_price_relationship.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✅ 已保存: chapter8_location_price_relationship.png")

# 5.3 房价热力图（使用阿姆斯特丹地图底图）/ Price Heatmap (with Amsterdam Map Base)
heatmap_created = False
if FOLIUM_AVAILABLE:
    print("\n5.3 创建房价热力图 / Creating Price Heatmap...")
    
    # 准备数据：过滤掉价格异常值和缺失值
    # Prepare data: filter out price outliers and missing values
    heatmap_data = listings.dropna(subset=['latitude', 'longitude', 'price']).copy()
    # 过滤异常价格（使用95%分位数作为上限）
    # Filter outlier prices (use 95th percentile as upper limit)
    price_95 = heatmap_data['price'].quantile(0.95)
    heatmap_data = heatmap_data[heatmap_data['price'] <= price_95]
    
    # 计算阿姆斯特丹中心坐标（用于地图初始视图）
    # Calculate Amsterdam center coordinates (for initial map view)
    amsterdam_center = [
        heatmap_data['latitude'].mean(),
        heatmap_data['longitude'].mean()
    ]
    
    # 创建基础地图，使用 OpenStreetMap 底图
    # Create base map with OpenStreetMap tile layer
    price_map = folium.Map(
        location=amsterdam_center,
        zoom_start=12,
        tiles='OpenStreetMap'
    )
    
    # 添加备选地图样式（CartoDB Positron，更简洁）
    # Add alternative map style (CartoDB Positron, cleaner)
    folium.TileLayer(
        tiles='CartoDB positron',
        name='CartoDB Positron',
        overlay=False,
        control=True
    ).add_to(price_map)
    
    # 准备热力图数据：格式为 [纬度, 经度, 权重（价格）]
    # Prepare heatmap data: format [latitude, longitude, weight (price)]
    heat_data = [[row['latitude'], row['longitude'], row['price']] 
                 for idx, row in heatmap_data.iterrows()]
    
    # 添加热力图图层
    # Add heatmap layer
    HeatMap(
        heat_data,
        min_opacity=0.2,
        max_zoom=18,
        radius=15,
        blur=15,
        gradient={
            0.0: 'blue',      # 低价格 / Low price
            0.3: 'cyan',      # 中低价格 / Medium-low price
            0.5: 'lime',      # 中等价格 / Medium price
            0.7: 'yellow',    # 中高价格 / Medium-high price
            1.0: 'red'        # 高价格 / High price
        }
    ).add_to(price_map)
    
    # 添加标记点（可选：显示部分高价格房源）
    # Add markers (optional: show some high-price listings)
    high_price_listings = heatmap_data.nlargest(20, 'price')
    for idx, row in high_price_listings.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=5,
            popup=f"Price: €{row['price']:.0f}<br>Neighbourhood: {row['neighbourhood']}",
            color='darkred',
            fill=True,
            fillColor='red',
            fillOpacity=0.6
        ).add_to(price_map)
    
    # 添加图例和控制
    # Add legend and controls
    folium.LayerControl().add_to(price_map)
    
    # 添加标题
    # Add title
    title_html = '''
    <div style="position: fixed; 
                top: 10px; left: 50px; width: 400px; height: 90px; 
                background-color: white; z-index:9999; font-size:14px;
                border:2px solid grey; padding: 10px">
    <h4 style="margin-top: 0;">Price Heatmap</h4>
    <p style="margin-bottom: 0;">Amsterdam Airbnb Listings<br>
    Redder colors indicate higher prices</p>
    </div>
    '''
    price_map.get_root().html.add_child(folium.Element(title_html))
    
    # 保存地图
    # Save map
    map_path = charts_dir / 'chapter8_price_heatmap.html'
    price_map.save(str(map_path))
    print(f"  ✅ 已保存: chapter8_price_heatmap.html")
    print(f"  📍 地图中心坐标 / Map Center: ({amsterdam_center[0]:.4f}, {amsterdam_center[1]:.4f})")
    print(f"  📊 热力图数据点 / Heatmap Data Points: {len(heat_data):,}")
    
    # 创建静态版本的热力图（使用 matplotlib）
    # Create static version of heatmap (using matplotlib)
    print("\n5.4 创建静态房价热力图 / Creating Static Price Heatmap...")
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # 使用 hexbin 创建密度热力图
    # Use hexbin to create density heatmap
    hb = ax.hexbin(
        heatmap_data['longitude'],
        heatmap_data['latitude'],
        C=heatmap_data['price'],
        gridsize=50,
        cmap='YlOrRd',
        reduce_C_function=np.mean,
        mincnt=1
    )
    
    ax.set_title('Price Heatmap of Amsterdam Airbnb Listings', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    
    # 添加颜色条
    # Add colorbar
    cb = plt.colorbar(hb, ax=ax, label='Average Price (€)')
    
    # 添加网格
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    static_map_path = charts_dir / 'chapter8_price_heatmap_static.png'
    plt.savefig(static_map_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ 已保存: chapter8_price_heatmap_static.png")
    
    heatmap_created = True
else:
    print("\n5.3 跳过热力图生成（folium 未安装）/ Skipping heatmap generation (folium not installed)")

# ============================================================================
# 6. 输出统计报告 / Output Statistics Report
# ============================================================================

print("\n6. 生成统计报告 / Generating Statistics Report...")

report_lines = []
report_lines.append("=" * 80)
report_lines.append("Chapter 8: Geospatial Analysis")
report_lines.append("第8章：地理空间分析")
report_lines.append("=" * 80)
report_lines.append(f"\n生成时间 / Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")

report_lines.append("\n## 房源地理分布 / Geographic Distribution")
report_lines.append("\n### Top 10 Neighbourhoods by Listing Count")
for i, (neighbourhood, count) in enumerate(neighbourhood_counts.head(10).items(), 1):
    pct = (count / len(listings) * 100)
    report_lines.append(f"  {i:2d}. {neighbourhood}: {count:,} ({pct:.1f}%)")

report_lines.append("\n## 地理位置与价格关系 / Location-Price Relationship")
report_lines.append("\n### Top 10 Neighbourhoods by Average Price")
for i, (neighbourhood, row) in enumerate(neighbourhood_price.head(10).iterrows(), 1):
    report_lines.append(f"  {i:2d}. {neighbourhood}: €{row['mean']:.2f} (房源数: {row['count']:.0f})")

report_lines.append("\n## 地理位置与受欢迎度关系 / Location-Popularity Relationship")
report_lines.append("\n### Top 10 Neighbourhoods by Average Reviews")
for i, (neighbourhood, row) in enumerate(neighbourhood_reviews.head(10).iterrows(), 1):
    report_lines.append(f"  {i:2d}. {neighbourhood}: {row['mean']:.1f} (总评论数: {row['sum']:.0f})")

if heatmap_created:
    report_lines.append("\n## 房价热力图 / Price Heatmap")
    report_lines.append(f"\n### 热力图统计 / Heatmap Statistics")
    report_lines.append(f"  - 地图中心坐标 / Map Center: ({amsterdam_center[0]:.4f}, {amsterdam_center[1]:.4f})")
    report_lines.append(f"  - 热力图数据点 / Heatmap Data Points: {len(heat_data):,}")
    report_lines.append(f"  - 价格范围 / Price Range: €{heatmap_data['price'].min():.2f} - €{heatmap_data['price'].max():.2f}")
    report_lines.append(f"  - 平均价格 / Average Price: €{heatmap_data['price'].mean():.2f}")
    report_lines.append(f"\n### 生成的文件 / Generated Files")
    report_lines.append(f"  - chapter8_price_heatmap.html (交互式地图 / Interactive Map)")
    report_lines.append(f"  - chapter8_price_heatmap_static.png (静态热力图 / Static Heatmap)")

with open(charts_dir / 'chapter8_geospatial_statistics.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))

print("  ✅ 已保存: chapter8_geospatial_statistics.txt")

print("\n" + "=" * 80)
print("分析完成 / Analysis Complete!")
print("=" * 80)

