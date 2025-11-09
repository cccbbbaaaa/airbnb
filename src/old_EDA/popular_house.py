import os
import pandas as pd
import matplotlib.pyplot as plt

# 设置中文字体支持 / Set Chinese font support
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 获取项目根目录路径 / Get project root directory path
# 从 src/EDA/ 目录到项目根目录 / From src/EDA/ to project root
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_path = os.path.join(project_root, 'data', 'listings.csv')
charts_dir = os.path.join(project_root, 'charts')

# 确保 charts 目录存在 / Ensure charts directory exists
os.makedirs(charts_dir, exist_ok=True)

# 加载数据 / Load data
listing = pd.read_csv(data_path)

# ---删除neighbourhood_group列 / Drop neighbourhood_group column---
listing = listing.drop("neighbourhood_group", axis=1)

# ---用0填充review和reviews_per_month的空值 / Fill missing values with 0---
listing["last_review"] = listing["last_review"].fillna(0)
listing["reviews_per_month"] = listing["reviews_per_month"].fillna(0)

# ---用blank_name填充name和host_name的空值 / Fill missing values with placeholder---
listing["name"] = listing["name"].fillna("blank_name")
listing["host_name"] = listing["host_name"].fillna("blank_host_name")

# ---箱型图显示minimum_nights / Boxplot for minimum_nights---
plt.figure(figsize=(8, 6))
listing["minimum_nights"].plot.box()
plt.title("Minimum Nights boxplot")
plt.ylabel("Minimum Nights")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(charts_dir, 'minimum_nights_boxplot.png'), dpi=300, bbox_inches='tight')
plt.close()

# ---存在异常值,修改异常值为365 / Handle outliers: set values > 365 to 365---
listing.loc[listing["minimum_nights"] > 365, "minimum_nights"] = 365

# ---数据探索：最受欢迎房源 / Data Exploration: Most Popular Listings---
# 根据评价对区域房源进行分析 / Analyze neighborhoods by reviews
# 评论越多越受欢迎 / More reviews indicate higher popularity
neighbourhood_group_reviews = listing['number_of_reviews'].groupby(listing['neighbourhood'])
neighbourhood_group_reviews_data = pd.DataFrame(neighbourhood_group_reviews.sum().sort_values(ascending=False))
neighbourhood_group_reviews_data = neighbourhood_group_reviews_data.astype(int).head(10)  # 将评论数转换为整数,取前10个 / Convert to int, take top 10
print("Top 10 neighborhoods by total reviews:")
print(neighbourhood_group_reviews_data)

# 根据可用天数对区域房源进行分析 / Analyze neighborhoods by availability
# 可用天数越小越受欢迎 / Lower availability indicates higher popularity
neighbourhood_group_availability_365 = listing['availability_365'].groupby(listing['neighbourhood'])
neighbourhood_group_availability_365_data = pd.DataFrame(neighbourhood_group_availability_365.mean().sort_values())
neighbourhood_group_availability_365_data = neighbourhood_group_availability_365_data.astype(int).head(10)  # 将可用天数转换为整数,取前10个 / Convert to int, take top 10
print("\nTop 10 neighborhoods by availability (lowest = most popular):")
print(neighbourhood_group_availability_365_data)

# 根据评论数对区域房源进行可视化 / Visualize neighborhoods by reviews
plt.figure(figsize=(12, 6))
neighbourhood_group_reviews_data.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title("The most popular neighbourhoods by reviews")
plt.xlabel("Neighbourhood")
plt.ylabel("Number of Reviews")
plt.xticks(rotation=45, ha='right')  # 旋转x轴标签，避免重叠 / Rotate x-axis labels to avoid overlap
plt.tight_layout()
plt.savefig(os.path.join(charts_dir, 'popular_neighbourhoods_by_reviews.png'), dpi=300, bbox_inches='tight')
plt.close()

# 根据可用天数对区域房源进行可视化 / Visualize neighborhoods by availability
plt.figure(figsize=(12, 6))
neighbourhood_group_availability_365_data.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title("The most popular neighbourhoods by availability_365")
plt.xlabel("Neighbourhood")
plt.ylabel("Availability_365")
plt.xticks(rotation=45, ha='right')  # 旋转x轴标签，避免重叠 / Rotate x-axis labels to avoid overlap
plt.tight_layout()
plt.savefig(os.path.join(charts_dir, 'popular_neighbourhoods_by_availability.png'), dpi=300, bbox_inches='tight')
plt.close()

# ----对房间类型进行分析 / Analyze room types----
# 建立房型与价格的表 / Create table for room type and price
room_type_price = listing.groupby('room_type')['price'].mean()
# 建立房型与评论数的表 / Create table for room type and reviews
room_type_reviews = listing.groupby('room_type')['number_of_reviews'].mean()
# 建立房型与可用天数的表 / Create table for room type and availability
room_type_availability_365 = listing.groupby('room_type')['availability_365'].mean()
# 将以上三个表合并为一个新的表 / Merge the three tables into one
room_type_data = pd.concat([room_type_price, room_type_reviews, room_type_availability_365], axis=1)
room_type_data.columns = ['price', 'number_of_reviews', 'availability_365']

# 可视化分析,按可用天数排序 / Visualization: sorted by availability
plt.figure(figsize=(12, 6))
room_type_data["availability_365"].sort_values(ascending=True).plot(kind='bar', color='skyblue', edgecolor='black')
plt.title("The most popular room types by availability_365")
plt.xlabel("Room Type")
plt.ylabel("Availability_365")
plt.xticks(rotation=45, ha='right')  # 旋转x轴标签，避免重叠 / Rotate x-axis labels to avoid overlap
plt.tight_layout()
plt.savefig(os.path.join(charts_dir, 'popular_room_types_by_availability.png'), dpi=300, bbox_inches='tight')
plt.close()

# 按评论数排序 / Sorted by number of reviews
plt.figure(figsize=(12, 6))
room_type_data["number_of_reviews"].sort_values(ascending=False).plot(kind='bar', color='skyblue', edgecolor='black')
plt.title("The most popular room types by number_of_reviews")
plt.xlabel("Room Type")
plt.ylabel("Number of Reviews")
plt.xticks(rotation=45, ha='right')  # 旋转x轴标签，避免重叠 / Rotate x-axis labels to avoid overlap
plt.tight_layout()
plt.savefig(os.path.join(charts_dir, 'popular_room_types_by_reviews.png'), dpi=300, bbox_inches='tight')
plt.close()

# 按价格排序 / Sorted by price
plt.figure(figsize=(12, 6))
room_type_data["price"].sort_values(ascending=True).plot(kind='bar', color='skyblue', edgecolor='black')
plt.title("The most popular room types by price")
plt.xlabel("Room Type")
plt.ylabel("Price")
plt.xticks(rotation=45, ha='right')  # 旋转x轴标签，避免重叠 / Rotate x-axis labels to avoid overlap
plt.tight_layout()
plt.savefig(os.path.join(charts_dir, 'popular_room_types_by_price.png'), dpi=300, bbox_inches='tight')
plt.close()

# 地区与房型联系表(评论数量) / Pivot table: neighborhood and room type (reviews)
review_pivot = listing.pivot_table("number_of_reviews", index="neighbourhood", columns="room_type", aggfunc="sum")
review_pivot = review_pivot.fillna(0)
review_pivot = review_pivot.astype(int)
print("\nPivot table: Neighborhood vs Room Type (Total Reviews):")
print(review_pivot)

# 地区与房型联系表可视化（评论数量） / Visualize pivot table (reviews)
plt.figure(figsize=(14, 8))
review_pivot.plot.bar()
plt.title("Neighbourhood vs Room Type: Total Reviews")
plt.xlabel("Neighbourhood")
plt.ylabel("Number of Reviews")
plt.xticks(rotation=45, ha='right')  # 旋转x轴标签，避免重叠 / Rotate x-axis labels to avoid overlap
plt.legend(title="Room Type")
plt.tight_layout()
plt.savefig(os.path.join(charts_dir, 'neighbourhood_roomtype_reviews_pivot.png'), dpi=300, bbox_inches='tight')
plt.close()

# 地区与房型联系表(可用天数) / Pivot table: neighborhood and room type (availability)
availability_pivot = listing.pivot_table("availability_365", index="neighbourhood", columns="room_type", aggfunc="mean")
availability_pivot = availability_pivot.fillna(0)
availability_pivot = availability_pivot.astype(int)
print("\nPivot table: Neighborhood vs Room Type (Average Availability):")
print(availability_pivot)

# 地区与房型联系表可视化（可用天数） / Visualize pivot table (availability)
plt.figure(figsize=(14, 8))
availability_pivot.plot.bar()
plt.title("Neighbourhood vs Room Type: Average Availability")
plt.xlabel("Neighbourhood")
plt.ylabel("Availability_365")
plt.xticks(rotation=45, ha='right')  # 旋转x轴标签，避免重叠 / Rotate x-axis labels to avoid overlap
plt.legend(title="Room Type")
plt.tight_layout()
plt.savefig(os.path.join(charts_dir, 'neighbourhood_roomtype_availability_pivot.png'), dpi=300, bbox_inches='tight')
plt.close()

# 地区与房型联系表(价格) / Pivot table: neighborhood and room type (price)
price_pivot = listing.pivot_table("price", index="neighbourhood", columns="room_type", aggfunc="mean")
price_pivot = price_pivot.fillna(0)
print("\nPivot table: Neighborhood vs Room Type (Average Price):")
print(price_pivot)

# 地区与房型联系表可视化（价格） / Visualize pivot table (price)
plt.figure(figsize=(14, 8))
price_pivot.plot.bar()
plt.title("Neighbourhood vs Room Type: Average Price")
plt.xlabel("Neighbourhood")
plt.ylabel("Price")
plt.xticks(rotation=45, ha='right')  # 旋转x轴标签，避免重叠 / Rotate x-axis labels to avoid overlap
plt.legend(title="Room Type")
plt.tight_layout()
plt.savefig(os.path.join(charts_dir, 'neighbourhood_roomtype_price_pivot.png'), dpi=300, bbox_inches='tight')
plt.close()

# 受欢迎房源特征分析(价格偏低) / Popular listings characteristics (lower price)
top_10 = listing.sort_values(by='number_of_reviews', ascending=False).head(10)
average_price = listing["price"].groupby(listing["room_type"]).mean()
top_10_average_price = top_10["price"].groupby(top_10["room_type"]).mean()
print("\nAverage price by room type (all listings):")
print(average_price)
print("\nAverage price by room type (top 10 most reviewed listings):")
print(top_10_average_price)

print("\n所有图表已保存到 charts/ 文件夹 / All charts saved to charts/ folder")
