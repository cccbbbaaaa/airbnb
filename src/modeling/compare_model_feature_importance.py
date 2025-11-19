"""
Model Feature Importance Comparison
模型特征重要性对比

本脚本读取所有模型的特征重要性结果，生成统一的对比报告
This script reads feature importance results from all models and generates a unified comparison report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# 添加 EDA 目录到路径，以便导入 utils 模块 / Add EDA directory to path for importing utils module
sys.path.insert(0, str(Path(__file__).parent.parent / 'EDA'))
from utils import setup_plotting, get_project_paths

# 设置中文字体支持 / Set Chinese font support
setup_plotting()

# 获取项目根目录路径 / Get project root directory path
project_root, data_dir, charts_eda_dir, charts_model_dir = get_project_paths()
charts_dir = charts_model_dir  # 使用模型目录 / Use model directory

# 设置控制台编码（Windows兼容）
import sys
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

print("=" * 80)
print("Model Feature Importance Comparison")
print("模型特征重要性对比")
print("=" * 80)

# ============================================================================
# 1. 加载各模型的特征重要性 / Load Feature Importance from All Models
# ============================================================================

print("\n1. 加载特征重要性数据 / Loading Feature Importance Data...")

# 文件路径
xgboost_file = charts_dir / 'xgboost_feature_importance.csv'
linear_regression_file = charts_dir / 'linear_regression_feature_importance.csv'
svm_file = charts_dir / 'svm_feature_importance.csv'

# 检查文件是否存在
files_to_load = {
    'XGBoost': xgboost_file,
    'Linear Regression': linear_regression_file,
    'SVM': svm_file
}

feature_importance_dict = {}
for model_name, file_path in files_to_load.items():
    if file_path.exists():
        df = pd.read_csv(file_path)
        # 标准化重要性值（归一化到0-1）
        df['importance_normalized'] = (df['importance'] - df['importance'].min()) / (df['importance'].max() - df['importance'].min() + 1e-10)
        feature_importance_dict[model_name] = df
        print(f"  [OK] {model_name}: {len(df)} 个特征")
    else:
        print(f"  [WARNING] {model_name} 特征重要性文件不存在: {file_path}")

if not feature_importance_dict:
    print("\n错误: 没有找到任何特征重要性文件！")
    print("请先运行 xgboost_income_prediction.py 和 linear_svm_income_prediction.py")
    exit(1)

# ============================================================================
# 2. 提取Top 20特征 / Extract Top 20 Features
# ============================================================================

print("\n2. 提取Top 20特征 / Extracting Top 20 Features...")

top20_dict = {}
for model_name, df in feature_importance_dict.items():
    top20 = df.head(20).copy()
    top20_dict[model_name] = top20
    print(f"\n{model_name} - Top 20 特征:")
    print(top20[['feature', 'importance']].to_string(index=False))

# ============================================================================
# 3. 创建对比表格 / Create Comparison Table
# ============================================================================

print("\n3. 创建对比表格 / Creating Comparison Table...")

# 收集所有Top 20特征
all_top_features = set()
for model_name, df in top20_dict.items():
    all_top_features.update(df['feature'].head(20).tolist())

# 创建对比表格
comparison_data = []
for feature in sorted(all_top_features):
    row = {'feature': feature}
    for model_name in feature_importance_dict.keys():
        df = feature_importance_dict[model_name]
        feature_row = df[df['feature'] == feature]
        if not feature_row.empty:
            row[f'{model_name}_importance'] = feature_row.iloc[0]['importance']
            row[f'{model_name}_rank'] = int(feature_row.index[0]) + 1
        else:
            row[f'{model_name}_importance'] = 0
            row[f'{model_name}_rank'] = None
    
    # 计算平均排名（只考虑存在的排名）
    ranks = [row[f'{model_name}_rank'] for model_name in feature_importance_dict.keys() 
             if row[f'{model_name}_rank'] is not None]
    row['avg_rank'] = np.mean(ranks) if ranks else None
    comparison_data.append(row)

comparison_df = pd.DataFrame(comparison_data)

# 按平均排名排序
comparison_df = comparison_df.sort_values('avg_rank', na_position='last')

# 保存对比表格
comparison_df.to_csv(charts_dir / 'all_models_feature_importance_comparison.csv', index=False)
print(f"\n  [OK] 对比表格已保存到: {charts_dir / 'all_models_feature_importance_comparison.csv'}")

# ============================================================================
# 4. 生成可视化图表 / Generate Visualization
# ============================================================================

print("\n4. 生成可视化图表 / Generating Visualizations...")

# 4.1 各模型Top 20特征重要性对比（并排显示）
n_models = len(feature_importance_dict)
fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 10))

if n_models == 1:
    axes = [axes]

colors = {'XGBoost': 'orange', 'Linear Regression': 'blue', 'SVM': 'green'}

for idx, (model_name, df) in enumerate(top20_dict.items()):
    top20 = df.head(20)
    color = colors.get(model_name, 'gray')
    
    axes[idx].barh(range(len(top20)), top20['importance'].values, color=color, alpha=0.7)
    axes[idx].set_yticks(range(len(top20)))
    axes[idx].set_yticklabels(top20['feature'].values, fontsize=9)
    axes[idx].set_xlabel('Importance', fontsize=10)
    axes[idx].set_title(f'{model_name}\nTop 20 Feature Importance', fontsize=12, fontweight='bold')
    axes[idx].invert_yaxis()
    axes[idx].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(charts_dir / 'all_models_top20_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"  [OK] Top 20特征重要性图表已保存到: {charts_dir / 'all_models_top20_feature_importance.png'}")

# 4.2 创建特征重要性热力图（Top 20特征在各模型中的排名）
top_features_all_models = set()
for model_name, df in top20_dict.items():
    top_features_all_models.update(df['feature'].head(20).tolist())

# 创建排名矩阵
rank_matrix = []
feature_list = sorted(list(top_features_all_models))

for feature in feature_list:
    row = {'feature': feature}
    for model_name in feature_importance_dict.keys():
        df = feature_importance_dict[model_name]
        feature_row = df[df['feature'] == feature]
        if not feature_row.empty:
            rank = int(feature_row.index[0]) + 1
            row[model_name] = rank if rank <= 20 else 21  # 超过20的设为21
        else:
            row[model_name] = 21  # 不在Top 20中的设为21
    rank_matrix.append(row)

rank_df = pd.DataFrame(rank_matrix)
rank_df = rank_df.sort_values(by=[list(feature_importance_dict.keys())[0]])

# 创建热力图
fig, ax = plt.subplots(figsize=(10, max(12, len(rank_df) * 0.3)))
rank_data = rank_df.set_index('feature')[list(feature_importance_dict.keys())].T

# 使用颜色映射：排名越小（越重要）颜色越深
im = ax.imshow(rank_data.values, cmap='RdYlGn_r', aspect='auto', vmin=1, vmax=21)

ax.set_xticks(np.arange(len(rank_data.columns)))
ax.set_yticks(np.arange(len(rank_data.index)))
ax.set_xticklabels(rank_data.columns, rotation=45, ha='right', fontsize=8)
ax.set_yticklabels(rank_data.index, fontsize=10)

# 添加数值标注
for i in range(len(rank_data.index)):
    for j in range(len(rank_data.columns)):
        rank_value = rank_data.iloc[i, j]
        if rank_value <= 20:
            text = ax.text(j, i, int(rank_value),
                          ha="center", va="center", color="black", fontsize=7, fontweight='bold')
        else:
            text = ax.text(j, i, '-',
                          ha="center", va="center", color="gray", fontsize=7)

ax.set_title('Feature Importance Ranking Heatmap\n(1-20: Top features, -: Not in Top 20)', 
             fontsize=12, fontweight='bold', pad=20)
plt.colorbar(im, ax=ax, label='Rank (Lower is Better)')
plt.tight_layout()
plt.savefig(charts_dir / 'feature_importance_ranking_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"  [OK] 特征重要性排名热力图已保存到: {charts_dir / 'feature_importance_ranking_heatmap.png'}")

# 4.3 创建汇总报告
print("\n5. 生成汇总报告 / Generating Summary Report...")

summary_text = []
summary_text.append("=" * 80)
summary_text.append("模型特征重要性对比报告 / Model Feature Importance Comparison Report")
summary_text.append("=" * 80)
summary_text.append("")

for model_name, df in top20_dict.items():
    summary_text.append(f"\n{model_name} - Top 20 最重要特征:")
    summary_text.append("-" * 80)
    for idx, row in df.head(20).iterrows():
        summary_text.append(f"  {idx+1:2d}. {row['feature']:40s}  Importance: {row['importance']:.6f}")

summary_text.append("\n" + "=" * 80)
summary_text.append("报告生成完成 / Report Generation Complete")
summary_text.append("=" * 80)

summary_report = "\n".join(summary_text)
print(summary_report)

# 保存汇总报告
with open(charts_dir / 'feature_importance_summary_report.txt', 'w', encoding='utf-8') as f:
    f.write(summary_report)

print(f"\n  [OK] 汇总报告已保存到: {charts_dir / 'feature_importance_summary_report.txt'}")

print("\n" + "=" * 80)
print("特征重要性对比完成！/ Feature Importance Comparison Complete!")
print("=" * 80)

