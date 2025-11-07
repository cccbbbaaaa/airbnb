import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

# ---用0填充license字段的空值 / Fill null values in license field with 0 ---
listing["license"] = listing["license"].fillna(0)

# ---许可证分类处理 / License classification processing ---
# 根据license字段的值进行分类：0->Unlicensed, "Exempt"->Exempt, 0363开头->Licensed, 其他->pending
# Classify licenses based on license field: 0->Unlicensed, "Exempt"->Exempt, starts with 0363->Licensed, others->pending
pie_label = np.select(
    [
        listing["license"] == 0,
        listing["license"] == "Exempt",
        listing["license"].astype(str).str.startswith("0363")
    ],
    ["Unlicensed", "Exempt", "Licensed"],
    default="pending"
)
listing["License_label"] = pie_label

# ---许可证分布饼图 / License distribution pie chart ---
plt.figure(figsize=(8, 8))
listing["License_label"].value_counts().plot(kind="pie", autopct="%1.1f%%")
plt.title("Licenses")
plt.show()