import pandas as pd

# 读取房源数据 / Read listings data
# 使用相对路径读取数据文件 / Use relative path to read data file
listing = pd.read_csv("../../data/listings.csv")

# ---查找缺失值 / Find missing values ---
# 统计每个特征的缺失值数量 / Count missing values for each feature
missing_value_count = listing.isnull().sum()
print("---Number of null in each feature")
print(missing_value_count)

# ---仅显示有缺失值的特征 / Display only features with missing values ---
missing_feature = missing_value_count[missing_value_count > 0]
print("---Number of null value in this feature")
if missing_feature.empty:
    print("No null value feature")
else:
    print(missing_feature)