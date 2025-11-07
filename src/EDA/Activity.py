import pandas as pd
import matplotlib.pyplot as plt
import os

# 读取房源数据 / Read listings data
# 使用相对路径读取数据文件 / Use relative path to read data file
listing = pd.read_csv("../../data/listings.csv")

# ---数据基本信息描述 / Basic data information ---
print(listing.info())
print(listing.describe())

# 计算平均入住天数 / Calculate average occupancy days
# occupancy days = 365 - availability_365
listing["occupancy days"] = 365 - listing["availability_365"]
print(f"The average nights booked is {round(listing['occupancy days'].mean())} days")

# 计算平均价格 / Calculate average price per night
print(f"The average price per night is ${round(listing['price'].mean())}")

# 绘制入住天数分布直方图 / Plot occupancy days distribution histogram
# 确保输出目录存在 / Ensure output directory exists
os.makedirs("../../charts", exist_ok=True)

plt.figure(figsize=(8, 8))
listing["occupancy days"].plot(kind="hist", bins=list(range(0, 365, 30)), edgecolor="black")
plt.title("Occupancy distribution")
plt.xlabel("Occupancy (last 12 months)")
plt.ylabel("listing")
plt.tight_layout()
plt.savefig("../../charts/occupancy_distribution.png", dpi=300, bbox_inches='tight')
plt.close()
print("图表已保存至: charts/occupancy_distribution.png")