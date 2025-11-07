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

# --license处理 / License processing ---
# TODO: 待实现许可证字段的处理逻辑 / TODO: Implement license field processing logic
