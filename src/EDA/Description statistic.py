import pandas as pd
import matplotlib.pyplot as plt
import os

# 读取房源数据 / Read listings data
# 使用相对路径读取数据文件 / Use relative path to read data file
listing = pd.read_csv("../../data/listings.csv")

# 确保输出目录存在 / Ensure output directory exists
os.makedirs("../../charts", exist_ok=True)

# ---数据基本信息描述 / Basic data information ---
print(listing.info())
print(listing.describe())

# ---价格分布分析 / Price distribution analysis ---
print(listing["price"].describe())
plt.figure(figsize=(8, 5))
listing['price'].plot(kind='hist', bins=60, edgecolor='black')
plt.title('Price distribution')
plt.xlabel("price(€)")
plt.ylabel("Number of listing")
plt.tight_layout()
plt.savefig("../../charts/price_distribution.png", dpi=300, bbox_inches='tight')
plt.close()
print("图表已保存至: charts/price_distribution.png")

# ---房型分布分析 / Room type distribution analysis ---
plt.figure(figsize=(6, 6))
listing["room_type"].value_counts().plot(kind='pie', autopct="%1.1f%%")
plt.title('Room Type share')
plt.ylabel("")
plt.tight_layout()
plt.savefig("../../charts/room_type_distribution.png", dpi=300, bbox_inches='tight')
plt.close()
print("图表已保存至: charts/room_type_distribution.png")

# ---评论数量分布分析 / Review distribution analysis ---
plt.figure(figsize=(8, 5))
listing["number_of_reviews"].plot(kind='hist', bins=60, edgecolor="black")
plt.title("Review distribution")
plt.xlabel("Number of Reviews")
plt.ylabel("Number of Listing")
plt.tight_layout()
plt.savefig("../../charts/review_distribution.png", dpi=300, bbox_inches='tight')
plt.close()
print("图表已保存至: charts/review_distribution.png")

# ---不同房型的平均价格分析 / Average price analysis by room type ---
plt.figure(figsize=(10, 10))
listing.groupby("room_type")["price"].mean().sort_values().plot(kind='bar')
plt.title("average price of room type")
plt.xlabel("room type")
plt.ylabel("average price")
plt.tight_layout()
plt.savefig("../../charts/avg_price_by_room_type.png", dpi=300, bbox_inches='tight')
plt.close()
print("图表已保存至: charts/avg_price_by_room_type.png")

# ---不同街区的平均价格分析 / Average price analysis by neighbourhood ---
plt.figure(figsize=(15, 12))
listing.groupby("neighbourhood")["price"].mean().sort_values(ascending=False).plot(kind="bar")
plt.title("Average Price of neighbourhoods")
plt.xlabel("Neighbourhoods")
plt.ylabel("Average Price")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("../../charts/avg_price_by_neighbourhood.png", dpi=300, bbox_inches='tight')
plt.close()
print("图表已保存至: charts/avg_price_by_neighbourhood.png")