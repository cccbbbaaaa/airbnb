"""
Chapter 7: Time Series Analysis
第7章：时间序列分析

本脚本进行时间序列分析，包括评论时间趋势、季节性模式识别、COVID-19影响分析等。
This script performs time series analysis, including review time trends, seasonal pattern identification, and COVID-19 impact analysis.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from utils import setup_plotting, get_project_paths

# 设置中文字体支持 / Set Chinese font support
setup_plotting()

# 获取项目根目录路径 / Get project root directory path
project_root, data_dir, charts_dir = get_project_paths()

print("=" * 80)
print("Chapter 7: Time Series Analysis")
print("第7章：时间序列分析")
print("=" * 80)

# ============================================================================
# 1. 加载数据 / Load Data
# ============================================================================

print("\n1. 加载数据 / Loading Data...")

reviews = pd.read_csv(data_dir / 'reviews.csv')
listings = pd.read_csv(data_dir / 'listings.csv')

reviews['date'] = pd.to_datetime(reviews['date'], errors='coerce')
reviews_with_date = reviews.dropna(subset=['date'])

print(f"  ✅ reviews.csv: {len(reviews_with_date):,} 条有效评论")
print(f"  ✅ listings.csv: {len(listings):,} 个房源")

# ============================================================================
# 2. 评论时间趋势分析 / Review Time Trend Analysis
# ============================================================================

print("\n2. 评论时间趋势分析 / Review Time Trend Analysis...")

# 2.1 按年统计
reviews_with_date['year'] = reviews_with_date['date'].dt.year
reviews_with_date['month'] = reviews_with_date['date'].dt.month
reviews_with_date['year_month'] = reviews_with_date['date'].dt.to_period('M')
reviews_with_date['quarter'] = reviews_with_date['date'].dt.quarter

yearly_counts = reviews_with_date['year'].value_counts().sort_index()
monthly_counts = reviews_with_date['year_month'].value_counts().sort_index()
quarterly_counts = reviews_with_date.groupby([reviews_with_date['date'].dt.year, 'quarter']).size()

print("\n2.1 按年评论统计 / Reviews by Year:")
for year, count in yearly_counts.items():
    print(f"  - {year}: {count:,} 条评论")

# 2.2 计算增长率
print("\n2.2 年度增长率 / Year-over-Year Growth Rate:")
yearly_growth = yearly_counts.pct_change() * 100
for year, growth in yearly_growth.items():
    if not pd.isna(growth):
        print(f"  - {year}: {growth:+.1f}%")

# ============================================================================
# 3. 季节性模式识别 / Seasonal Pattern Identification
# ============================================================================

print("\n3. 季节性模式识别 / Seasonal Pattern Identification...")

# 3.1 各月平均评论数
monthly_avg = reviews_with_date.groupby('month').size()
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

print("\n3.1 各月平均评论数 / Average Reviews by Month:")
for month, count in monthly_avg.items():
    print(f"  - {month_names[month-1]}: {count:,.0f} 条")

# 3.2 各季度平均评论数
quarterly_avg = reviews_with_date.groupby('quarter').size()
quarter_names = ['Q1 (Jan-Mar)', 'Q2 (Apr-Jun)', 'Q3 (Jul-Sep)', 'Q4 (Oct-Dec)']

print("\n3.2 各季度平均评论数 / Average Reviews by Quarter:")
for quarter, count in quarterly_avg.items():
    print(f"  - {quarter_names[quarter-1]}: {count:,.0f} 条")

# 3.3 识别旺季和淡季
peak_months = monthly_avg.nlargest(3).index.tolist()
low_months = monthly_avg.nsmallest(3).index.tolist()

print("\n3.3 旺季和淡季识别 / Peak and Low Season Identification:")
print(f"  - 旺季月份（Top 3）: {[month_names[m-1] for m in peak_months]}")
print(f"  - 淡季月份（Bottom 3）: {[month_names[m-1] for m in low_months]}")

# ============================================================================
# 4. COVID-19 影响分析 / COVID-19 Impact Analysis
# ============================================================================

print("\n4. COVID-19 影响分析 / COVID-19 Impact Analysis...")

# 4.1 定义COVID-19期间
pre_covid = reviews_with_date[reviews_with_date['year'] < 2020]
covid_period = reviews_with_date[(reviews_with_date['year'] == 2020) | 
                                  ((reviews_with_date['year'] == 2021) & 
                                   (reviews_with_date['date'] <= pd.Timestamp('2021-09-07')))]

print("\n4.1 COVID-19 前后对比 / Pre-COVID vs COVID Period:")
print(f"  - COVID前（2009-2019）平均每月评论数: {len(pre_covid) / (12 * 11):,.0f} 条")
print(f"  - COVID期间（2020-2021.09）平均每月评论数: {len(covid_period) / 21:,.0f} 条")
covid_impact = (len(covid_period) / 21) / (len(pre_covid) / (12 * 11)) - 1
print(f"  - 影响程度: {covid_impact*100:.1f}%")

# 4.2 2020年各月评论数
reviews_2020 = reviews_with_date[reviews_with_date['year'] == 2020]
monthly_2020 = reviews_2020.groupby('month').size()

print("\n4.2 2020年各月评论数 / Monthly Reviews in 2020:")
for month, count in monthly_2020.items():
    print(f"  - {month_names[month-1]}: {count:,} 条")

# ============================================================================
# 5. 房源生命周期模式 / Listing Lifecycle Patterns
# ============================================================================

print("\n5. 房源生命周期模式 / Listing Lifecycle Patterns...")

# 5.1 计算每个房源的首条和最后评论日期
listing_review_dates = reviews_with_date.groupby('listing_id')['date'].agg(['min', 'max', 'count'])
listing_review_dates.columns = ['first_review', 'last_review', 'review_count']
listing_review_dates['listing_age_days'] = (listing_review_dates['last_review'] - 
                                            listing_review_dates['first_review']).dt.days
listing_review_dates['listing_age_months'] = listing_review_dates['listing_age_days'] / 30

print("\n5.1 房源生命周期统计 / Listing Lifecycle Statistics:")
print(f"  - 平均房源活跃时长: {listing_review_dates['listing_age_months'].mean():.1f} 个月")
print(f"  - 中位数房源活跃时长: {listing_review_dates['listing_age_months'].median():.1f} 个月")
print(f"  - 最长活跃时长: {listing_review_dates['listing_age_months'].max():.1f} 个月")

# 5.2 新房源 vs 成熟房源
new_listings = listing_review_dates[listing_review_dates['listing_age_months'] < 12]
mature_listings = listing_review_dates[listing_review_dates['listing_age_months'] >= 12]

print("\n5.2 新房源 vs 成熟房源 / New vs Mature Listings:")
print(f"  - 新房源（<12个月）: {len(new_listings):,} 个 ({len(new_listings)/len(listing_review_dates)*100:.1f}%)")
print(f"  - 成熟房源（≥12个月）: {len(mature_listings):,} 个 ({len(mature_listings)/len(listing_review_dates)*100:.1f}%)")
print(f"  - 新房源平均评论数: {new_listings['review_count'].mean():.1f} 条")
print(f"  - 成熟房源平均评论数: {mature_listings['review_count'].mean():.1f} 条")

# ============================================================================
# 6. 创建可视化图表 / Create Visualizations
# ============================================================================

print("\n6. 创建可视化图表 / Creating Visualizations...")

fig, axes = plt.subplots(3, 2, figsize=(16, 18))

# 6.1 按年评论趋势
yearly_counts.plot(kind='line', ax=axes[0, 0], marker='o', color='#2196F3', linewidth=2, markersize=8)
axes[0, 0].set_title('Reviews Trend by Year', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Year', fontsize=11)
axes[0, 0].set_ylabel('Number of Reviews', fontsize=11)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].tick_params(axis='x', rotation=45)
# 标注COVID-19影响
axes[0, 0].axvspan(2020, 2021, alpha=0.2, color='red', label='COVID-19 Period')
axes[0, 0].legend()

# 6.2 按月评论分布（季节性）
monthly_avg.plot(kind='bar', ax=axes[0, 1], color='#4CAF50', edgecolor='black')
axes[0, 1].set_title('Average Reviews by Month', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Month', fontsize=11)
axes[0, 1].set_ylabel('Average Reviews', fontsize=11)
axes[0, 1].set_xticklabels(month_names, rotation=45)
for i, v in enumerate(monthly_avg.values):
    axes[0, 1].text(i, v, f'{v:,.0f}', ha='center', va='bottom', fontsize=9, rotation=90)

# 6.3 按季度评论分布
quarterly_avg.plot(kind='bar', ax=axes[1, 0], color='#FF9800', edgecolor='black')
axes[1, 0].set_title('Average Reviews by Quarter', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Quarter', fontsize=11)
axes[1, 0].set_ylabel('Average Reviews', fontsize=11)
axes[1, 0].set_xticklabels(quarter_names, rotation=45)
for i, v in enumerate(quarterly_avg.values):
    axes[1, 0].text(i, v, f'{v:,.0f}', ha='center', va='bottom', fontsize=10)

# 6.4 月度时间序列（2015-2021）
monthly_counts_2015_2021 = monthly_counts[monthly_counts.index >= pd.Period('2015-01')]
monthly_counts_2015_2021.plot(kind='line', ax=axes[1, 1], color='#9C27B0', linewidth=2)
axes[1, 1].set_title('Monthly Reviews Trend (2015-2021)', 
                     fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Year-Month', fontsize=11)
axes[1, 1].set_ylabel('Number of Reviews', fontsize=11)
axes[1, 1].tick_params(axis='x', rotation=45)
axes[1, 1].grid(True, alpha=0.3)
# 标注COVID-19期间
covid_start = pd.Period('2020-01')
covid_end = pd.Period('2021-09')
axes[1, 1].axvspan(covid_start, covid_end, alpha=0.2, color='red')

# 6.5 COVID-19影响对比
pre_covid_yearly = pre_covid.groupby('year').size()
covid_yearly = covid_period.groupby('year').size()
comparison_data = pd.DataFrame({
    'Pre-COVID Avg': [pre_covid_yearly.mean()] * len(yearly_counts),
    'Actual': yearly_counts.values
}, index=yearly_counts.index)
comparison_data.plot(kind='bar', ax=axes[2, 0], color=['gray', '#F44336'], edgecolor='black')
axes[2, 0].set_title('COVID-19 Impact Comparison', fontsize=12, fontweight='bold')
axes[2, 0].set_xlabel('Year', fontsize=11)
axes[2, 0].set_ylabel('Number of Reviews', fontsize=11)
axes[2, 0].tick_params(axis='x', rotation=45)
axes[2, 0].legend()

# 6.6 房源生命周期分布
listing_age_sample = listing_review_dates[listing_review_dates['listing_age_months'] <= 60]
axes[2, 1].hist(listing_age_sample['listing_age_months'], bins=30, color='#00BCD4', 
                edgecolor='black', alpha=0.7)
axes[2, 1].set_title('Listing Lifecycle Distribution', fontsize=12, fontweight='bold')
axes[2, 1].set_xlabel('Listing Age (Months)', fontsize=11)
axes[2, 1].set_ylabel('Frequency', fontsize=11)
axes[2, 1].axvline(listing_review_dates['listing_age_months'].mean(), color='red', 
                   linestyle='--', label=f'Mean: {listing_review_dates["listing_age_months"].mean():.1f} months')
axes[2, 1].legend()

plt.tight_layout()
plt.savefig(charts_dir / 'chapter7_time_series_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✅ 已保存: chapter7_time_series_analysis.png")

# ============================================================================
# 7. 输出统计报告 / Output Statistics Report
# ============================================================================

print("\n7. 生成统计报告 / Generating Statistics Report...")

report_lines = []
report_lines.append("=" * 80)
report_lines.append("Chapter 7: Time Series Analysis")
report_lines.append("第7章：时间序列分析")
report_lines.append("=" * 80)
report_lines.append(f"\n生成时间 / Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")

report_lines.append("\n## 按年评论统计 / Reviews by Year")
for year, count in yearly_counts.items():
    report_lines.append(f"  - {year}: {count:,} 条评论")

report_lines.append("\n## 季节性模式 / Seasonal Pattern")
report_lines.append("\n### 各月平均评论数 / Average Reviews by Month")
for month, count in monthly_avg.items():
    report_lines.append(f"  - {month_names[month-1]}: {count:,.0f} 条")

report_lines.append("\n## COVID-19 影响 / COVID-19 Impact")
report_lines.append(f"  - COVID前平均每月评论数: {len(pre_covid) / (12 * 11):,.0f} 条")
report_lines.append(f"  - COVID期间平均每月评论数: {len(covid_period) / 21:,.0f} 条")
report_lines.append(f"  - 影响程度: {covid_impact*100:.1f}%")

with open(charts_dir / 'chapter7_time_series_statistics.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))

print("  ✅ 已保存: chapter7_time_series_statistics.txt")

print("\n" + "=" * 80)
print("分析完成 / Analysis Complete!")
print("=" * 80)

