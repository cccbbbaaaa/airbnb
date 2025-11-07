import pandas as pd

# 读取房源数据 / Read listings data
# 使用相对路径读取数据文件 / Use relative path to read data file
listing = pd.read_csv("../../data/listings.csv")

# ---删除neighbourhood_group列 / Drop neighbourhood_group column ---
listing = listing.drop("neighbourhood_group", axis=1)

# ---用0填充review相关字段的空值 / Fill null values in review-related fields with 0 ---
listing["last_review"] = listing["last_review"].fillna(0)
listing["reviews_per_month"] = listing["reviews_per_month"].fillna(0)

# ---用blank_name填充name和host_name的空值 / Fill null values in name and host_name with blank_name ---
listing["name"] = listing["name"].fillna("blank_name")
listing["host_name"] = listing["host_name"].fillna("blank_host_name")

# ---顶级房东分析 / Top hosts analysis ---
# 使用交叉表统计每个房东的房型分布 / Use crosstab to count room type distribution for each host
room_count = pd.crosstab(listing["host_name"], listing["room_type"])
# 计算每个房东的总房源数 / Calculate total listings for each host
room_count["listings"] = room_count.sum(axis=1)
# 按房源数量降序排序 / Sort by total listings in descending order
room_count = room_count.sort_values(by="listings", ascending=False)
# 显示前10名房东 / Display top 10 hosts
print(room_count.head(10))