import pandas as pd
import matplotlib.pyplot as plt

# 读取房源数据 / Read listings data
# 使用相对路径读取数据文件 / Use relative path to read data file
listing = pd.read_csv("../../data/listings.csv")

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
plt.show()

# ---房型分布分析 / Room type distribution analysis ---
plt.figure(figsize=(6, 6))
listing["room_type"].value_counts().plot(kind='pie', autopct="%1.1f%%")
plt.title('Room Type share')
plt.ylabel("")
plt.show()

# ---评论数量分布分析 / Review distribution analysis ---
plt.figure(figsize=(8, 5))
listing["number_of_reviews"].plot(kind='hist', bins=60, edgecolor="black")
plt.title("Review distribution")
plt.xlabel("Number of Reviews")
plt.ylabel("Number of Listing")
plt.show()

# ---不同房型的平均价格分析 / Average price analysis by room type ---
plt.figure(figsize=(10, 10))
listing.groupby("room_type")["price"].mean().sort_values().plot(kind='bar')
plt.title("average price of room type")
plt.xlabel("room type")
plt.ylabel("average price")
plt.show()

# ---不同街区的平均价格分析 / Average price analysis by neighbourhood ---
plt.figure(figsize=(15, 12))
listing.groupby("neighbourhood")["price"].mean().sort_values(ascending=False).plot(kind="bar")
plt.title("Average Price of neighbourhoods")
plt.xlabel("Neighbourhoods")
plt.ylabel("Average Price")
plt.show()