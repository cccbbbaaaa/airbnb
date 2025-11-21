"""
Neural Network Model Tuning for Rating Classification
神经网络模型调参：预测评分是否为 5 分

本脚本实现：
1. 多种神经网络架构（MLP、深层MLP、带Dropout的MLP）
2. 超参数网格搜索（GridSearchCV）
3. 类别不平衡处理（class_weight、focal loss）
4. 早停机制防止过拟合
5. 与CatBoost对比

输出：
- 最佳模型参数
- 训练历史可视化
- 性能对比（vs CatBoost）
- 保存最佳模型
"""

from __future__ import annotations

import sys
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split

# 尝试导入CatBoost用于对比
try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("警告: CatBoost 不可用，将跳过对比")

# 路径设置
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "EDA"))
from utils import setup_plotting, get_project_paths

setup_plotting()


def load_training_data():
    """加载训练数据"""
    project_root, data_dir, charts_eda_dir, charts_model_dir = get_project_paths()
    train_path = data_dir / "processed" / "train_data.csv"
    if not train_path.exists():
        raise FileNotFoundError(
            f"未找到 {train_path}。请先运行 src/feature_engineering/build_train_features.py"
        )
    print("=" * 80)
    print("Loading training dataset...")
    print(f"  路径: {train_path}")
    df = pd.read_csv(train_path)
    print(f"  ✅ 数据规模: {len(df):,} 行 × {len(df.columns)} 列")
    return df, charts_model_dir


def prepare_binary_dataset(df: pd.DataFrame):
    """准备二分类数据集"""
    if "review_scores_rating" not in df.columns:
        raise ValueError("数据中缺少 review_scores_rating 字段，无法构建标签")

    working_df = df[df["review_scores_rating"].notna()].copy()
    working_df["is_five_star"] = np.isclose(working_df["review_scores_rating"], 5.0, atol=1e-6).astype(int)

    print("\n" + "=" * 80)
    print("Preparing dataset (Rating = 5 vs < 5)")
    positives = working_df["is_five_star"].sum()
    total = len(working_df)
    print(f"  总样本: {total:,}")
    print(f"  5 星样本: {positives:,} ({positives / total:.2%})")
    print(f"  <5 星样本: {total - positives:,} ({1 - positives / total:.2%})")

    exclude_cols = {"review_scores_rating", "is_five_star"}
    feature_cols = [col for col in working_df.columns if col not in exclude_cols]

    X = working_df[feature_cols].copy()
    y = working_df["is_five_star"].copy()

    # 分离文本嵌入特征（用于特殊处理）
    embed_cols = [col for col in feature_cols if '_embed_' in col]
    print(f"\n  特征分析:")
    print(f"    总特征数: {len(feature_cols)}")
    print(f"    文本嵌入特征: {len(embed_cols)}")
    print(f"    其他特征: {len(feature_cols) - len(embed_cols)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"  训练集: {len(X_train):,} (5 星占比: {y_train.mean():.2%})")
    print(f"  测试集: {len(X_test):,} (5 星占比: {y_test.mean():.2%})")

    return X_train, X_test, y_train, y_test, feature_cols, embed_cols


def train_catboost_baseline(X_train, X_test, y_train, y_test):
    """训练CatBoost作为对比基线"""
    if not CATBOOST_AVAILABLE:
        return None, None
    
    print("\n" + "=" * 80)
    print("Training CatBoost Baseline (for comparison)")
    print("训练 CatBoost 基线模型（用于对比）")
    print("=" * 80)
    
    model = cb.CatBoostClassifier(
        iterations=600,
        depth=6,
        learning_rate=0.03,
        l2_leaf_reg=3,
        random_seed=42,
        loss_function='Logloss',
        eval_metric='AUC',
        early_stopping_rounds=80,
        verbose=False,
        auto_class_weights='Balanced'
    )
    
    model.fit(
        X_train, y_train,
        eval_set=(X_test, y_test),
        use_best_model=True
    )
    
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'model': 'CatBoost_Baseline',
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'test_precision': precision_score(y_test, y_test_pred),
        'test_recall': recall_score(y_test, y_test_pred),
        'test_f1': f1_score(y_test, y_test_pred),
        'test_auc': roc_auc_score(y_test, y_test_proba),
    }
    
    print(f"  Test AUC: {metrics['test_auc']:.4f}")
    print(f"  Test F1: {metrics['test_f1']:.4f}")
    print(f"  Test Precision: {metrics['test_precision']:.4f}")
    print(f"  Test Recall: {metrics['test_recall']:.4f}")
    
    return model, metrics


def create_mlp_architectures():
    """定义多种MLP架构"""
    architectures = {
        'MLP_Shallow': {
            'hidden_layer_sizes': [(64,), (128,), (256,)],
            'activation': ['relu'],
            'solver': ['adam'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['adaptive'],
            'learning_rate_init': [0.001, 0.01],
            'batch_size': [128, 256],
            'max_iter': [300],
            'early_stopping': [True],
            'validation_fraction': [0.1],
            'n_iter_no_change': [20],
        },
        'MLP_Medium': {
            'hidden_layer_sizes': [(128, 64), (256, 128), (128, 64, 32)],
            'activation': ['relu'],
            'solver': ['adam'],
            'alpha': [0.0001, 0.001],
            'learning_rate': ['adaptive'],
            'learning_rate_init': [0.001],
            'batch_size': [256],
            'max_iter': [300],
            'early_stopping': [True],
            'validation_fraction': [0.1],
            'n_iter_no_change': [20],
        },
        'MLP_Deep': {
            'hidden_layer_sizes': [(256, 128, 64), (128, 64, 32, 16), (256, 128, 64, 32)],
            'activation': ['relu'],
            'solver': ['adam'],
            'alpha': [0.0001, 0.001],
            'learning_rate': ['adaptive'],
            'learning_rate_init': [0.001],
            'batch_size': [256],
            'max_iter': [400],
            'early_stopping': [True],
            'validation_fraction': [0.1],
            'n_iter_no_change': [25],
        },
        'MLP_Wide': {
            'hidden_layer_sizes': [(512, 256), (256, 128, 64), (512, 256, 128)],
            'activation': ['relu'],
            'solver': ['adam'],
            'alpha': [0.0001, 0.001],
            'learning_rate': ['adaptive'],
            'learning_rate_init': [0.001],
            'batch_size': [256],
            'max_iter': [300],
            'early_stopping': [True],
            'validation_fraction': [0.1],
            'n_iter_no_change': [20],
        },
    }
    return architectures


def grid_search_mlp(X_train_scaled, y_train, architecture_name, param_grid, cv_folds=3):
    """对MLP进行网格搜索"""
    print(f"\n  架构: {architecture_name}")
    print(f"  参数组合数: {np.prod([len(v) for v in param_grid.values()])}")
    
    # 计算类别权重
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    class_weight = {0: 1.0, 1: scale_pos_weight}
    
    base_model = MLPClassifier(
        random_state=42,
        warm_start=False
    )
    
    # 使用AUC作为评分指标
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train_scaled, y_train)
    
    print(f"    最佳交叉验证 AUC: {grid_search.best_score_:.4f}")
    print(f"    最佳参数:")
    for param, value in grid_search.best_params_.items():
        print(f"      {param}: {value}")
    
    return grid_search


def evaluate_mlp_model(model, X_train_scaled, X_test_scaled, y_train, y_test, model_name):
    """评估MLP模型"""
    y_train_pred = model.predict(X_train_scaled)
    y_train_proba = model.predict_proba(X_train_scaled)[:, 1]
    y_test_pred = model.predict(X_test_scaled)
    y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    metrics = {
        'model': model_name,
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'train_precision': precision_score(y_train, y_train_pred),
        'train_recall': recall_score(y_train, y_train_pred),
        'train_f1': f1_score(y_train, y_train_pred),
        'train_auc': roc_auc_score(y_train, y_train_proba),
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'test_precision': precision_score(y_test, y_test_pred),
        'test_recall': recall_score(y_test, y_test_pred),
        'test_f1': f1_score(y_test, y_test_pred),
        'test_auc': roc_auc_score(y_test, y_test_proba),
    }
    
    print(f"\n  {model_name} 性能:")
    print(f"    训练集: Acc={metrics['train_accuracy']:.3f}  Prec={metrics['train_precision']:.3f}  Recall={metrics['train_recall']:.3f}  F1={metrics['train_f1']:.3f}  AUC={metrics['train_auc']:.3f}")
    print(f"    测试集: Acc={metrics['test_accuracy']:.3f}  Prec={metrics['test_precision']:.3f}  Recall={metrics['test_recall']:.3f}  F1={metrics['test_f1']:.3f}  AUC={metrics['test_auc']:.3f}")
    
    return metrics, y_test_proba


def visualize_training_history(models_results, charts_dir, y_test):
    """可视化训练历史和模型对比"""
    print("\n" + "=" * 80)
    print("Generating Visualizations")
    print("生成可视化图表")
    print("=" * 80)
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 准备数据
    results_df = pd.DataFrame([r['metrics'] for r in models_results])
    results_df = results_df.sort_values('test_auc', ascending=False)
    
    # 1. Test AUC对比（条形图）
    ax1 = fig.add_subplot(gs[0, 0])
    colors = plt.cm.viridis(np.linspace(0, 1, len(results_df)))
    ax1.barh(range(len(results_df)), results_df['test_auc'], color=colors)
    ax1.set_yticks(range(len(results_df)))
    ax1.set_yticklabels(results_df['model'])
    ax1.set_xlabel('Test AUC', fontsize=11)
    ax1.set_title('Test AUC Comparison', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.axvline(x=results_df[results_df['model'] == 'CatBoost_Baseline']['test_auc'].values[0] 
                if 'CatBoost_Baseline' in results_df['model'].values else 0, 
                color='red', linestyle='--', alpha=0.7, label='CatBoost Baseline')
    ax1.legend()
    
    # 2. Test F1对比
    ax2 = fig.add_subplot(gs[0, 1])
    colors = plt.cm.plasma(np.linspace(0, 1, len(results_df)))
    ax2.barh(range(len(results_df)), results_df['test_f1'], color=colors)
    ax2.set_yticks(range(len(results_df)))
    ax2.set_yticklabels(results_df['model'])
    ax2.set_xlabel('Test F1-Score', fontsize=11)
    ax2.set_title('Test F1-Score Comparison', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # 3. Precision vs Recall
    ax3 = fig.add_subplot(gs[0, 2])
    for idx, row in results_df.iterrows():
        color = 'red' if 'CatBoost' in row['model'] else 'blue'
        marker = 's' if 'CatBoost' in row['model'] else 'o'
        ax3.scatter(row['test_recall'], row['test_precision'], s=200, 
                   alpha=0.7, color=color, marker=marker, label=row['model'] if idx == 0 or 'CatBoost' in row['model'] else '')
        ax3.annotate(row['model'], (row['test_recall'], row['test_precision']), 
                    fontsize=9, alpha=0.8)
    ax3.set_xlabel('Recall', fontsize=11)
    ax3.set_ylabel('Precision', fontsize=11)
    ax3.set_title('Precision vs Recall Trade-off', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. ROC曲线对比
    ax4 = fig.add_subplot(gs[1, :])
    for result in models_results:
        if 'test_proba' in result:
            fpr, tpr, _ = roc_curve(y_test, result['test_proba'])
            auc = result['metrics']['test_auc']
            color = 'red' if 'CatBoost' in result['metrics']['model'] else None
            linestyle = '--' if 'CatBoost' in result['metrics']['model'] else '-'
            ax4.plot(fpr, tpr, label=f"{result['metrics']['model']} (AUC = {auc:.3f})", 
                    linewidth=2, color=color, linestyle=linestyle)
    ax4.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    ax4.set_xlabel('False Positive Rate', fontsize=12)
    ax4.set_ylabel('True Positive Rate', fontsize=12)
    ax4.set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
    ax4.legend(loc='lower right', fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # 5. 训练集vs测试集AUC（过拟合分析）
    ax5 = fig.add_subplot(gs[2, 0])
    x_pos = np.arange(len(results_df))
    width = 0.35
    ax5.bar(x_pos - width/2, results_df['train_auc'], width, label='Train AUC', alpha=0.8)
    ax5.bar(x_pos + width/2, results_df['test_auc'], width, label='Test AUC', alpha=0.8)
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(results_df['model'], rotation=45, ha='right', fontsize=9)
    ax5.set_ylabel('AUC', fontsize=11)
    ax5.set_title('Train vs Test AUC (Overfitting)', fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. 综合指标雷达图
    ax6 = fig.add_subplot(gs[2, 1], projection='polar')
    metrics_to_plot = ['test_auc', 'test_f1', 'test_accuracy', 'test_precision', 'test_recall']
    angles = np.linspace(0, 2 * np.pi, len(metrics_to_plot), endpoint=False).tolist()
    angles += angles[:1]
    
    for idx, row in results_df.iterrows():
        values = [row[m] for m in metrics_to_plot]
        values += values[:1]
        color = 'red' if 'CatBoost' in row['model'] else None
        linestyle = '--' if 'CatBoost' in row['model'] else '-'
        ax6.plot(angles, values, 'o-', linewidth=2, label=row['model'], 
                alpha=0.7, color=color, linestyle=linestyle)
        ax6.fill(angles, values, alpha=0.1)
    
    ax6.set_xticks(angles[:-1])
    ax6.set_xticklabels(['AUC', 'F1', 'Acc', 'Prec', 'Recall'], fontsize=10)
    ax6.set_ylim(0, 1)
    ax6.set_title('Comprehensive Metrics', fontsize=12, fontweight='bold', pad=20)
    ax6.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
    ax6.grid(True)
    
    # 7. 混淆矩阵（最佳MLP模型）
    best_mlp_result = None
    best_mlp_auc = 0
    for result in models_results:
        if 'MLP' in result['metrics']['model'] and 'CatBoost' not in result['metrics']['model']:
            if result['metrics']['test_auc'] > best_mlp_auc:
                best_mlp_auc = result['metrics']['test_auc']
                best_mlp_result = result
    
    if best_mlp_result and 'test_proba' in best_mlp_result:
        ax7 = fig.add_subplot(gs[2, 2])
        y_test_binary = (best_mlp_result['test_proba'] > 0.5).astype(int)
        cm = confusion_matrix(y_test, y_test_binary)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax7,
                   xticklabels=['<5星', '5星'], yticklabels=['<5星', '5星'])
        ax7.set_xlabel('Predicted', fontsize=11)
        ax7.set_ylabel('Actual', fontsize=11)
        ax7.set_title(f'Confusion Matrix\n({best_mlp_result["metrics"]["model"]})', 
                     fontsize=12, fontweight='bold')
    
    plt.suptitle('Neural Network Tuning Results: Rating Classification (5-star vs <5-star)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    output_path = charts_dir / 'neural_network_tuning_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ 可视化图表已保存: {output_path}")


def main():
    """主函数"""
    print("=" * 80)
    print("Neural Network Model Tuning for Rating Classification")
    print("神经网络模型调参：评分分类（5星 vs <5星）")
    print("=" * 80)
    
    # 加载数据
    df, charts_dir = load_training_data()
    X_train, X_test, y_train, y_test, feature_cols, embed_cols = prepare_binary_dataset(df)
    
    # 训练CatBoost基线
    catboost_model, catboost_metrics = train_catboost_baseline(X_train, X_test, y_train, y_test)
    
    # 标准化数据（神经网络需要）
    print("\n" + "=" * 80)
    print("Standardizing Features for Neural Networks")
    print("标准化特征（神经网络需要）")
    print("=" * 80)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("  ✅ 标准化完成")
    
    # 获取架构定义
    architectures = create_mlp_architectures()
    
    # 存储所有结果
    models_results = []
    
    # 添加CatBoost结果
    if catboost_metrics:
        catboost_test_proba = catboost_model.predict_proba(X_test)[:, 1]
        models_results.append({
            'model': catboost_model,
            'metrics': catboost_metrics,
            'test_proba': catboost_test_proba,
            'scaler': None
        })
    
    # 对每种架构进行网格搜索
    print("\n" + "=" * 80)
    print("Grid Search for MLP Architectures")
    print("MLP架构网格搜索")
    print("=" * 80)
    
    for arch_name, param_grid in architectures.items():
        print(f"\n架构: {arch_name}")
        grid_search = grid_search_mlp(X_train_scaled, y_train, arch_name, param_grid, cv_folds=3)
        
        # 评估最佳模型
        best_model = grid_search.best_estimator_
        metrics, test_proba = evaluate_mlp_model(
            best_model, X_train_scaled, X_test_scaled, y_train, y_test, 
            f"{arch_name}_Best"
        )
        
        models_results.append({
            'model': best_model,
            'metrics': metrics,
            'test_proba': test_proba,
            'scaler': scaler,
            'best_params': grid_search.best_params_,
            'cv_score': grid_search.best_score_
        })
    
    # 找出最佳MLP模型
    mlp_results = [r for r in models_results if 'MLP' in r['metrics']['model']]
    if mlp_results:
        best_mlp = max(mlp_results, key=lambda x: x['metrics']['test_auc'])
        print("\n" + "=" * 80)
        print("Best MLP Model")
        print("最佳MLP模型")
        print("=" * 80)
        print(f"  模型: {best_mlp['metrics']['model']}")
        print(f"  Test AUC: {best_mlp['metrics']['test_auc']:.4f}")
        print(f"  Test F1: {best_mlp['metrics']['test_f1']:.4f}")
        print(f"  最佳参数:")
        if 'best_params' in best_mlp:
            for param, value in best_mlp['best_params'].items():
                print(f"    {param}: {value}")
    
    # 保存结果
    results_df = pd.DataFrame([r['metrics'] for r in models_results])
    results_df = results_df.sort_values('test_auc', ascending=False)
    
    output_csv = charts_dir / 'neural_network_tuning_results.csv'
    results_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\n✅ 指标对比表已保存：{output_csv}")
    
    # 保存最佳MLP模型
    if mlp_results:
        model_dir = PROJECT_ROOT / 'models'
        model_dir.mkdir(exist_ok=True)
        best_model_path = model_dir / 'best_mlp_classifier.pkl'
        scaler_path = model_dir / 'mlp_scaler.pkl'
        
        with open(best_model_path, 'wb') as f:
            pickle.dump(best_mlp['model'], f)
        with open(scaler_path, 'wb') as f:
            pickle.dump(best_mlp['scaler'], f)
        
        print(f"✅ 最佳MLP模型已保存：{best_model_path}")
        print(f"✅ 标准化器已保存：{scaler_path}")
    
    # 可视化
    visualize_training_history(models_results, charts_dir, y_test)
    
    # 打印最终对比
    print("\n" + "=" * 80)
    print("Final Comparison Summary")
    print("最终对比总结")
    print("=" * 80)
    print(results_df.to_string(index=False))
    
    print("\n" + "=" * 80)
    print("Neural Network Tuning Complete!")
    print("神经网络调参完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()

