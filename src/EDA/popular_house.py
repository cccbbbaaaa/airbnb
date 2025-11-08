from numpy import average
import pandas as pd
import matplotlib.pyplot as plt

data_path = "../../data/listings.csv"
listing = pd.read_csv(data_path)



#---删除neighbourhood---
listing = listing.drop("neighbourhood_group", axis= 1 )

#---用0填充review和reviews_per_month---
listing["last_review"] = listing["last_review"].fillna(0)
listing["reviews_per_month"] = listing["reviews_per_month"].fillna(0)

#---用blank_name填充name和host_name的空值---
listing["name"] = listing["name"].fillna("blank_name")
listing["host_name"] = listing["host_name"].fillna("blank_host_name")

#--箱型图显示---
plt.figure(figsize=(8, 6))
listing["minimum_nights"].plot.box()
plt.title("Minimum Nights boxplot")
plt.ylabel("Minimum Nights")
plt.grid(True, alpha=0.3)
plt.show()

#存在异常值,修改异常值为365---
listing.loc[listing["minimum_nights"]>365,"minimum_nights"] = 365   

#---数据探索：最受欢迎房源--
# 根据评价对区域房源进行分析
# 评论越多越受欢迎
neighbourhood_group_reviews = listing['number_of_reviews'].groupby(listing['neighbourhood'])
neighbourhood_group_reviews_data = pd.DataFrame(neighbourhood_group_reviews.sum().sort_values(ascending = False))
neighbourhood_group_reviews_data = neighbourhood_group_reviews_data.astype(int).head(10) #将评论数转换为整数,取前10个
print(neighbourhood_group_reviews_data)

#  根据可用天数对区域房源进行分析
#  可用天数越小越受欢迎
neighbourhood_group_availability_365 = listing['availability_365'].groupby(listing['neighbourhood'])
neighbourhood_group_availability_365_data = pd.DataFrame(neighbourhood_group_availability_365.mean().sort_values())
neighbourhood_group_availability_365_data = neighbourhood_group_availability_365_data.astype(int).head(10) #将可用天数转换为整数,取前10个
print(neighbourhood_group_availability_365_data)

#  根据评论数对区域房源进行可视化
plt.figure(figsize=(12, 6))
neighbourhood_group_reviews_data.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title("The most popular neighbourhoods by reviews")
plt.xlabel("Neighbourhood")
plt.ylabel("Number of Reviews")
plt.xticks(rotation=45, ha='right')  # 旋转x轴标签，避免重叠
plt.tight_layout()  # 自动调整布局，确保所有元素都显示
plt.show()

#  根据可用天数对区域房源进行可视化
plt.figure(figsize=(12, 6))
neighbourhood_group_availability_365_data.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title("The most popular neighbourhoods by availability_365")
plt.xlabel("Neighbourhood")
plt.ylabel("Availability_365")
plt.xticks(rotation=45, ha='right')  # 旋转x轴标签，避免重叠
plt.tight_layout()  # 自动调整布局，确保所有元素都显示
plt.show()


#----对房间类型进行分析----
# 建立房型与价格的表
room_type_price = listing.groupby('room_type')['price'].mean()
#建立房型与评论数的表
room_type_reviews = listing.groupby('room_type')['number_of_reviews'].mean()
#建立房型与可用天数的表
room_type_availability_365 = listing.groupby('room_type')['availability_365'].mean()
#将以上三个表合并为一个新的表
room_type_data = pd.concat([room_type_price, room_type_reviews, room_type_availability_365], axis=1)

#可视化分析,按可用天数排序
plt.figure(figsize=(12, 6))
room_type_data["availability_365"].sort_values(ascending=True).plot(kind='bar', color='skyblue', edgecolor='black')
plt.title("The most popular room types by availability_365")
plt.xlabel("Room Type")
plt.ylabel("Availability_365")
plt.xticks(rotation=45, ha='right')  # 旋转x轴标签，避免重叠
plt.tight_layout()  # 自动调整布局，确保所有元素都显示
plt.show()
#按评论数排序
plt.figure(figsize=(12, 6))
room_type_data["number_of_reviews"].sort_values(ascending=False).plot(kind='bar', color='skyblue', edgecolor='black')
plt.title("The most popular room types by number_of_reviews")
plt.xlabel("Room Type")
plt.ylabel("Number of Reviews")
plt.xticks(rotation=45, ha='right')  # 旋转x轴标签，避免重叠
plt.tight_layout()  # 自动调整布局，确保所有元素都显示
plt.show()
#按价格排序
plt.figure(figsize=(12, 6))
room_type_data["price"].sort_values(ascending=True).plot(kind='bar', color='skyblue', edgecolor='black')
plt.title("The most popular room types by price")
plt.xlabel("Room Type")
plt.ylabel("Price")
plt.xticks(rotation=45, ha='right')  # 旋转x轴标签，避免重叠
plt.tight_layout()  # 自动调整布局，确保所有元素都显示
plt.show()

#地区与房型联系表(评论数量)
review_pivot = listing.pivot_table("number_of_reviews", index="neighbourhood",columns="room_type",aggfunc="sum")
review_pivot = review_pivot.fillna(0)
review_pivot = review_pivot.astype(int)
print(review_pivot)
#地区与房型联系表可视化（评论数量）
review_pivot.plot.bar()
plt.xticks(rotation=45, ha='right')  # 旋转x轴标签，避免重叠
plt.tight_layout()  # 自动调整布局，确保所有元素都显示
plt.show()
#地区与房型联系表(可用天数)
availability_pivot = listing.pivot_table("availability_365", index="neighbourhood",columns="room_type",aggfunc="mean")
availability_pivot = availability_pivot.fillna(0)
availability_pivot = availability_pivot.astype(int)
print(availability_pivot)
#地区与房型联系表可视化（可用天数）
availability_pivot.plot.bar()
plt.xticks(rotation=45, ha='right')  # 旋转x轴标签，避免重叠
plt.tight_layout()  # 自动调整布局，确保所有元素都显示
plt.show()
#地区与房型联系表(价格)
price_pivot = listing.pivot_table("price", index="neighbourhood",columns="room_type",aggfunc="mean")
price_pivot = price_pivot.fillna(0)
print(price_pivot)
#地区与房型联系表可视化（价格）
price_pivot.plot.bar()
plt.xticks(rotation=45, ha='right')  # 旋转x轴标签，避免重叠
plt.tight_layout()  # 自动调整布局，确保所有元素都显示
plt.show()

#受欢迎房源房源特征(价格偏低)
top_10 = listing.sort_values(by='number_of_reviews', ascending=False).head(10)
average_price = listing["price"].groupby(listing["room_type"]).mean()
top_10_average_price = top_10["price"].groupby(top_10["room_type"]).mean()
print(average_price)
print(top_10_average_price)
