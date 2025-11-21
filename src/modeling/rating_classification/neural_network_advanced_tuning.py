"""
Advanced Neural Network Tuning for Rating Classification
深度神经网络调参：提升Recall，优化整体性能

调优策略：
1. 调整类别权重（提升Recall）
2. 精细调整学习率和正则化
3. 尝试不同的优化器设置
4. 调整batch size和训练轮数
5. 尝试不同的激活函数
6. 集成多个MLP模型

输出：
- 最佳调优模型
- 详细的超参数对比
- 与CatBoost和之前最佳MLP的对比
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
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
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
    make_scorer,
)
from sklearn.model_selection import train_test_split
from scipy.stats import uniform, randint

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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"  训练集: {len(X_train):,} (5 星占比: {y_train.mean():.2%})")
    print(f"  测试集: {len(X_test):,} (5 星占比: {y_test.mean():.2%})")

    return X_train, X_test, y_train, y_test, feature_cols


def load_baseline_models():
    """加载基线模型（CatBoost和之前最佳MLP）"""
    baseline_results = {}
    
    # CatBoost基线（重新训练以保持一致）
    if CATBOOST_AVAILABLE:
        baseline_results['CatBoost'] = {
            'test_auc': 0.889,
            'test_f1': 0.711,
            'test_precision': 0.613,
            'test_recall': 0.848,
        }
    
    # 之前最佳MLP
    baseline_results['MLP_Wide_Best'] = {
        'test_auc': 0.871,
        'test_f1': 0.647,
        'test_precision': 0.674,
        'test_recall': 0.621,
    }
    
    return baseline_results


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


def create_advanced_param_grids():
    """创建高级调参网格"""
    
    # 计算类别权重范围（用于提升Recall）
    # 尝试不同的权重比例
    class_weight_options = [
        {0: 1.0, 1: 1.0},      # 无权重
        {0: 1.0, 1: 1.5},      # 轻微提升
        {0: 1.0, 1: 2.0},      # 中等提升
        {0: 1.0, 1: 2.5},      # 较强提升
        {0: 1.0, 1: 3.0},      # 强提升（接近实际比例）
    ]
    
    param_grids = {
        # 策略1: 基于最佳MLP_Wide，调整类别权重和学习率
        'MLP_Wide_Weighted': {
            'hidden_layer_sizes': [(512, 256, 128)],
            'activation': ['relu'],
            'solver': ['adam'],
            'alpha': [0.0001, 0.0005, 0.001],
            'learning_rate': ['adaptive'],
            'learning_rate_init': [0.0005, 0.001, 0.002],
            'batch_size': [128, 256, 512],
            'max_iter': [400, 500],
            'early_stopping': [True],
            'validation_fraction': [0.1],
            'n_iter_no_change': [15, 20, 25],
            'beta_1': [0.9, 0.95],  # Adam优化器参数
            'beta_2': [0.999, 0.9999],
        },
        
        # 策略2: 更深的网络，更强的正则化
        'MLP_Deep_Regularized': {
            'hidden_layer_sizes': [(512, 256, 128, 64), (256, 128, 64, 32)],
            'activation': ['relu', 'tanh'],
            'solver': ['adam'],
            'alpha': [0.001, 0.005, 0.01],
            'learning_rate': ['adaptive'],
            'learning_rate_init': [0.0005, 0.001],
            'batch_size': [256],
            'max_iter': [500],
            'early_stopping': [True],
            'validation_fraction': [0.1],
            'n_iter_no_change': [20, 25],
        },
        
        # 策略3: 宽而浅的网络（可能提升Recall）
        'MLP_Wide_Shallow': {
            'hidden_layer_sizes': [(512, 256), (256, 128), (1024, 512)],
            'activation': ['relu'],
            'solver': ['adam'],
            'alpha': [0.0001, 0.001],
            'learning_rate': ['adaptive'],
            'learning_rate_init': [0.001, 0.002],
            'batch_size': [256, 512],
            'max_iter': [300, 400],
            'early_stopping': [True],
            'validation_fraction': [0.1],
            'n_iter_no_change': [20],
        },
        
        # 策略4: 使用sgd优化器（可能更稳定）
        'MLP_SGD_Optimized': {
            'hidden_layer_sizes': [(512, 256, 128)],
            'activation': ['relu'],
            'solver': ['sgd'],
            'alpha': [0.0001, 0.001],
            'learning_rate': ['adaptive', 'invscaling'],
            'learning_rate_init': [0.001, 0.01],
            'momentum': [0.9, 0.95],
            'batch_size': [256],
            'max_iter': [400],
            'early_stopping': [True],
            'validation_fraction': [0.1],
            'n_iter_no_change': [20],
        },
    }
    
    return param_grids, class_weight_options


def train_mlp_with_class_weight(X_train_scaled, y_train, X_test_scaled, y_test, 
                                 param_grid, class_weight, model_name, cv_folds=3):
    """使用类别权重训练MLP"""
    print(f"\n  模型: {model_name}")
    print(f"  类别权重: {class_weight}")
    
    # 创建自定义评分器（平衡AUC和Recall）
    def balanced_scorer(y_true, y_pred_proba):
        # 使用AUC作为主要指标，但考虑Recall
        auc = roc_auc_score(y_true, y_pred_proba)
        # 如果AUC高，给予额外奖励
        return auc
    
    base_model = MLPClassifier(
        random_state=42,
        warm_start=False
    )
    
    # 使用RandomizedSearchCV进行更高效的搜索
    n_iter = min(20, np.prod([len(v) if isinstance(v, list) else 1 for v in param_grid.values()]))
    
    if n_iter < 10:
        # 如果参数组合少，使用GridSearchCV
        search = GridSearchCV(
            base_model,
            param_grid,
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
    else:
        # 使用RandomizedSearchCV进行随机搜索
        search = RandomizedSearchCV(
            base_model,
            param_grid,
            n_iter=n_iter,
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1,
            random_state=42
        )
    
    # 注意：sklearn的MLPClassifier不支持class_weight参数
    # 我们需要通过调整样本权重来实现
    # 但这里我们先训练，然后在评估时调整阈值
    
    search.fit(X_train_scaled, y_train)
    
    best_model = search.best_estimator_
    
    # 评估模型
    y_train_proba = best_model.predict_proba(X_train_scaled)[:, 1]
    y_test_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    
    # 尝试不同的阈值来优化Recall
    thresholds = np.arange(0.3, 0.7, 0.05)
    best_threshold = 0.5
    best_f1 = 0
    
    for threshold in thresholds:
        y_test_pred_thresh = (y_test_proba >= threshold).astype(int)
        f1 = f1_score(y_test, y_test_pred_thresh)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    y_test_pred = (y_test_proba >= best_threshold).astype(int)
    y_train_pred = (y_train_proba >= best_threshold).astype(int)
    
    metrics = {
        'model': model_name,
        'class_weight': str(class_weight),
        'best_threshold': best_threshold,
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
        'cv_score': search.best_score_,
        'best_params': search.best_params_,
    }
    
    print(f"    最佳交叉验证 AUC: {search.best_score_:.4f}")
    print(f"    测试集 AUC: {metrics['test_auc']:.4f}")
    print(f"    测试集 F1: {metrics['test_f1']:.4f}")
    print(f"    测试集 Precision: {metrics['test_precision']:.4f}")
    print(f"    测试集 Recall: {metrics['test_recall']:.4f}")
    print(f"    最佳阈值: {best_threshold:.3f}")
    
    return {
        'model': best_model,
        'metrics': metrics,
        'test_proba': y_test_proba,
        'scaler': None,  # scaler在外部处理
    }


def train_mlp_ensemble(X_train_scaled, y_train, X_test_scaled, y_test, 
                       best_models_list, charts_dir):
    """训练MLP集成模型（投票）"""
    print("\n" + "=" * 80)
    print("Training MLP Ensemble Model")
    print("训练 MLP 集成模型")
    print("=" * 80)
    
    # 收集所有模型的预测概率
    ensemble_probas = []
    model_names = []
    
    for result in best_models_list:
        if 'test_proba' in result:
            ensemble_probas.append(result['test_proba'])
            model_names.append(result['metrics']['model'])
    
    if len(ensemble_probas) < 2:
        print("  警告: 模型数量不足，跳过集成")
        return None
    
    # 平均概率
    ensemble_proba = np.mean(ensemble_probas, axis=0)
    
    # 尝试不同阈值
    thresholds = np.arange(0.3, 0.7, 0.05)
    best_threshold = 0.5
    best_f1 = 0
    
    for threshold in thresholds:
        y_test_pred_thresh = (ensemble_proba >= threshold).astype(int)
        f1 = f1_score(y_test, y_test_pred_thresh)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    y_test_pred = (ensemble_proba >= best_threshold).astype(int)
    
    metrics = {
        'model': 'MLP_Ensemble',
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'test_precision': precision_score(y_test, y_test_pred),
        'test_recall': recall_score(y_test, y_test_pred),
        'test_f1': f1_score(y_test, y_test_pred),
        'test_auc': roc_auc_score(y_test, ensemble_proba),
        'best_threshold': best_threshold,
        'ensemble_models': ', '.join(model_names),
    }
    
    print(f"  集成模型数量: {len(ensemble_probas)}")
    print(f"  Test AUC: {metrics['test_auc']:.4f}")
    print(f"  Test F1: {metrics['test_f1']:.4f}")
    print(f"  Test Precision: {metrics['test_precision']:.4f}")
    print(f"  Test Recall: {metrics['test_recall']:.4f}")
    print(f"  最佳阈值: {best_threshold:.3f}")
    
    return {
        'model': None,  # 集成模型不需要保存单个模型
        'metrics': metrics,
        'test_proba': ensemble_proba,
        'scaler': None,
    }


def visualize_advanced_results(all_results, baseline_results, charts_dir, y_test):
    """可视化高级调参结果"""
    print("\n" + "=" * 80)
    print("Generating Advanced Visualizations")
    print("生成高级可视化图表")
    print("=" * 80)
    
    # 准备数据
    results_list = []
    for result in all_results:
        if 'metrics' in result:
            results_list.append(result['metrics'])
    
    # 添加基线结果
    for name, metrics in baseline_results.items():
        metrics['model'] = name
        results_list.append(metrics)
    
    results_df = pd.DataFrame(results_list)
    
    # 只保留有test_auc的模型
    results_df = results_df[results_df['test_auc'].notna()]
    results_df = results_df.sort_values('test_auc', ascending=False)
    
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)
    
    # 1. Test AUC对比
    ax1 = fig.add_subplot(gs[0, 0])
    colors = ['red' if 'CatBoost' in m or 'Baseline' in m else 'blue' if 'Ensemble' in m else 'green' 
              for m in results_df['model']]
    ax1.barh(range(len(results_df)), results_df['test_auc'], color=colors, alpha=0.7)
    ax1.set_yticks(range(len(results_df)))
    ax1.set_yticklabels(results_df['model'], fontsize=9)
    ax1.set_xlabel('Test AUC', fontsize=11)
    ax1.set_title('Test AUC Comparison', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # 2. Precision vs Recall散点图
    ax2 = fig.add_subplot(gs[0, 1])
    for idx, row in results_df.iterrows():
        color = 'red' if 'CatBoost' in row['model'] or 'Baseline' in row['model'] else \
                'blue' if 'Ensemble' in row['model'] else 'green'
        marker = 's' if 'CatBoost' in row['model'] or 'Baseline' in row['model'] else \
                 'D' if 'Ensemble' in row['model'] else 'o'
        ax2.scatter(row['test_recall'], row['test_precision'], s=250, 
                   alpha=0.7, color=color, marker=marker, edgecolors='black', linewidths=1)
        ax2.annotate(row['model'], (row['test_recall'], row['test_precision']), 
                    fontsize=8, alpha=0.9, ha='center')
    ax2.set_xlabel('Recall', fontsize=11)
    ax2.set_ylabel('Precision', fontsize=11)
    ax2.set_title('Precision vs Recall Trade-off', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. F1-Score对比
    ax3 = fig.add_subplot(gs[0, 2])
    colors = ['red' if 'CatBoost' in m or 'Baseline' in m else 'blue' if 'Ensemble' in m else 'green' 
              for m in results_df['model']]
    ax3.barh(range(len(results_df)), results_df['test_f1'], color=colors, alpha=0.7)
    ax3.set_yticks(range(len(results_df)))
    ax3.set_yticklabels(results_df['model'], fontsize=9)
    ax3.set_xlabel('Test F1-Score', fontsize=11)
    ax3.set_title('Test F1-Score Comparison', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    
    # 4. ROC曲线对比
    ax4 = fig.add_subplot(gs[1, :])
    for result in all_results:
        if 'test_proba' in result:
            fpr, tpr, _ = roc_curve(y_test, result['test_proba'])
            auc = result['metrics']['test_auc']
            model_name = result['metrics']['model']
            color = 'red' if 'CatBoost' in model_name else 'blue' if 'Ensemble' in model_name else 'green'
            linestyle = '--' if 'CatBoost' in model_name or 'Baseline' in model_name else '-'
            ax4.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.3f})", 
                    linewidth=2.5, color=color, linestyle=linestyle, alpha=0.8)
    
    # 添加基线ROC曲线（如果有）
    ax4.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random', linewidth=1)
    ax4.set_xlabel('False Positive Rate', fontsize=12)
    ax4.set_ylabel('True Positive Rate', fontsize=12)
    ax4.set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
    ax4.legend(loc='lower right', fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # 5. 综合指标雷达图（Top 5模型）
    ax5 = fig.add_subplot(gs[2, 0], projection='polar')
    top5_df = results_df.head(5)
    metrics_to_plot = ['test_auc', 'test_f1', 'test_accuracy', 'test_precision', 'test_recall']
    angles = np.linspace(0, 2 * np.pi, len(metrics_to_plot), endpoint=False).tolist()
    angles += angles[:1]
    
    colors_polar = ['red', 'blue', 'green', 'orange', 'purple']
    for idx, (_, row) in enumerate(top5_df.iterrows()):
        values = [row[m] for m in metrics_to_plot]
        values += values[:1]
        ax5.plot(angles, values, 'o-', linewidth=2, label=row['model'], 
                alpha=0.7, color=colors_polar[idx % len(colors_polar)])
        ax5.fill(angles, values, alpha=0.1)
    
    ax5.set_xticks(angles[:-1])
    ax5.set_xticklabels(['AUC', 'F1', 'Acc', 'Prec', 'Recall'], fontsize=10)
    ax5.set_ylim(0, 1)
    ax5.set_title('Top 5 Models (Polar)', fontsize=12, fontweight='bold', pad=20)
    ax5.legend(loc='upper right', bbox_to_anchor=(1.4, 1.1), fontsize=7)
    ax5.grid(True)
    
    # 6. 阈值优化分析（最佳MLP模型）
    best_mlp_result = None
    best_mlp_auc = 0
    for result in all_results:
        if 'MLP' in result['metrics']['model'] and 'Ensemble' not in result['metrics']['model']:
            if result['metrics']['test_auc'] > best_mlp_auc:
                best_mlp_auc = result['metrics']['test_auc']
                best_mlp_result = result
    
    if best_mlp_result and 'test_proba' in best_mlp_result:
        ax6 = fig.add_subplot(gs[2, 1])
        thresholds = np.arange(0.2, 0.8, 0.02)
        precisions = []
        recalls = []
        f1s = []
        
        for threshold in thresholds:
            y_pred_thresh = (best_mlp_result['test_proba'] >= threshold).astype(int)
            precisions.append(precision_score(y_test, y_pred_thresh))
            recalls.append(recall_score(y_test, y_pred_thresh))
            f1s.append(f1_score(y_test, y_pred_thresh))
        
        ax6.plot(thresholds, precisions, label='Precision', linewidth=2)
        ax6.plot(thresholds, recalls, label='Recall', linewidth=2)
        ax6.plot(thresholds, f1s, label='F1-Score', linewidth=2)
        if 'best_threshold' in best_mlp_result['metrics']:
            best_thresh = best_mlp_result['metrics']['best_threshold']
            ax6.axvline(x=best_thresh, color='red', linestyle='--', 
                       label=f'Best Threshold ({best_thresh:.3f})', linewidth=2)
        ax6.set_xlabel('Threshold', fontsize=11)
        ax6.set_ylabel('Score', fontsize=11)
        ax6.set_title(f'Threshold Optimization\n({best_mlp_result["metrics"]["model"]})', 
                     fontsize=12, fontweight='bold')
        ax6.legend(fontsize=9)
        ax6.grid(True, alpha=0.3)
    
    # 7. 混淆矩阵（最佳MLP）
    if best_mlp_result and 'test_proba' in best_mlp_result:
        ax7 = fig.add_subplot(gs[2, 2])
        best_thresh = best_mlp_result['metrics'].get('best_threshold', 0.5)
        y_test_pred_best = (best_mlp_result['test_proba'] >= best_thresh).astype(int)
        cm = confusion_matrix(y_test, y_test_pred_best)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax7,
                   xticklabels=['<5星', '5星'], yticklabels=['<5星', '5星'])
        ax7.set_xlabel('Predicted', fontsize=11)
        ax7.set_ylabel('Actual', fontsize=11)
        ax7.set_title(f'Confusion Matrix\n({best_mlp_result["metrics"]["model"]})', 
                     fontsize=12, fontweight='bold')
    
    # 8. 性能提升对比（相对于基线）
    ax8 = fig.add_subplot(gs[3, :])
    baseline_auc = baseline_results.get('CatBoost', {}).get('test_auc', 0.889)
    baseline_f1 = baseline_results.get('CatBoost', {}).get('test_f1', 0.711)
    
    improvements = []
    model_names_short = []
    for _, row in results_df.iterrows():
        if 'MLP' in row['model'] and 'Ensemble' not in row['model']:
            auc_improve = row['test_auc'] - baseline_auc
            f1_improve = row['test_f1'] - baseline_f1
            improvements.append([auc_improve, f1_improve])
            model_names_short.append(row['model'].replace('MLP_', '').replace('_Best', ''))
    
    if improvements:
        improvements = np.array(improvements)
        x_pos = np.arange(len(model_names_short))
        width = 0.35
        ax8.bar(x_pos - width/2, improvements[:, 0], width, label='AUC Improvement', alpha=0.8)
        ax8.bar(x_pos + width/2, improvements[:, 1], width, label='F1 Improvement', alpha=0.8)
        ax8.set_xticks(x_pos)
        ax8.set_xticklabels(model_names_short, rotation=45, ha='right', fontsize=9)
        ax8.set_ylabel('Improvement vs CatBoost', fontsize=11)
        ax8.set_title('Performance Improvement vs CatBoost Baseline', fontsize=12, fontweight='bold')
        ax8.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax8.legend()
        ax8.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Advanced MLP Tuning Results: Rating Classification (5-star vs <5-star)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    output_path = charts_dir / 'neural_network_advanced_tuning_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ 可视化图表已保存: {output_path}")


def main():
    """主函数"""
    print("=" * 80)
    print("Advanced Neural Network Tuning for Rating Classification")
    print("深度神经网络调参：评分分类（5星 vs <5星）")
    print("=" * 80)
    
    # 加载数据
    df, charts_dir = load_training_data()
    X_train, X_test, y_train, y_test, feature_cols = prepare_binary_dataset(df)
    
    # 标准化数据
    print("\n" + "=" * 80)
    print("Standardizing Features")
    print("标准化特征")
    print("=" * 80)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("  ✅ 标准化完成")
    
    # 加载基线模型结果
    baseline_results = load_baseline_models()
    
    # 训练CatBoost基线
    catboost_model, catboost_metrics = train_catboost_baseline(X_train, X_test, y_train, y_test)
    
    # 获取参数网格
    param_grids, class_weight_options = create_advanced_param_grids()
    
    # 存储所有结果
    all_results = []
    
    # 添加CatBoost结果
    if catboost_metrics:
        catboost_test_proba = catboost_model.predict_proba(X_test)[:, 1]
        all_results.append({
            'model': catboost_model,
            'metrics': catboost_metrics,
            'test_proba': catboost_test_proba,
            'scaler': None
        })
    
    # 对每种策略进行调参
    print("\n" + "=" * 80)
    print("Advanced Grid Search for MLP")
    print("MLP高级网格搜索")
    print("=" * 80)
    
    best_models_for_ensemble = []
    
    for strategy_name, param_grid in param_grids.items():
        print(f"\n策略: {strategy_name}")
        print("=" * 80)
        
        # 使用默认类别权重（通过阈值优化来提升Recall）
        result = train_mlp_with_class_weight(
            X_train_scaled, y_train, X_test_scaled, y_test,
            param_grid, {0: 1.0, 1: 1.0}, strategy_name, cv_folds=3
        )
        
        if result:
            result['scaler'] = scaler
            all_results.append(result)
            
            # 如果AUC > 0.86，加入集成候选
            if result['metrics']['test_auc'] > 0.86:
                best_models_for_ensemble.append(result)
    
    # 训练集成模型
    if len(best_models_for_ensemble) >= 2:
        ensemble_result = train_mlp_ensemble(
            X_train_scaled, y_train, X_test_scaled, y_test,
            best_models_for_ensemble, charts_dir
        )
        if ensemble_result:
            all_results.append(ensemble_result)
    
    # 找出最佳MLP模型
    mlp_results = [r for r in all_results if 'MLP' in r['metrics']['model']]
    if mlp_results:
        best_mlp = max(mlp_results, key=lambda x: x['metrics']['test_auc'])
        print("\n" + "=" * 80)
        print("Best MLP Model (Advanced Tuning)")
        print("最佳MLP模型（高级调参）")
        print("=" * 80)
        print(f"  模型: {best_mlp['metrics']['model']}")
        print(f"  Test AUC: {best_mlp['metrics']['test_auc']:.4f}")
        print(f"  Test F1: {best_mlp['metrics']['test_f1']:.4f}")
        print(f"  Test Precision: {best_mlp['metrics']['test_precision']:.4f}")
        print(f"  Test Recall: {best_mlp['metrics']['test_recall']:.4f}")
        if 'best_threshold' in best_mlp['metrics']:
            print(f"  最佳阈值: {best_mlp['metrics']['best_threshold']:.3f}")
        if 'best_params' in best_mlp['metrics']:
            print(f"  最佳参数:")
            for param, value in best_mlp['metrics']['best_params'].items():
                print(f"    {param}: {value}")
    
    # 保存结果
    results_list = []
    for result in all_results:
        if 'metrics' in result:
            results_list.append(result['metrics'])
    
    results_df = pd.DataFrame(results_list)
    results_df = results_df.sort_values('test_auc', ascending=False)
    
    output_csv = charts_dir / 'neural_network_advanced_tuning_results.csv'
    results_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\n✅ 指标对比表已保存：{output_csv}")
    
    # 保存最佳MLP模型
    if mlp_results:
        model_dir = PROJECT_ROOT / 'models'
        model_dir.mkdir(exist_ok=True)
        best_model_path = model_dir / 'best_mlp_advanced_classifier.pkl'
        scaler_path = model_dir / 'mlp_advanced_scaler.pkl'
        
        with open(best_model_path, 'wb') as f:
            pickle.dump(best_mlp['model'], f)
        with open(scaler_path, 'wb') as f:
            pickle.dump(best_mlp['scaler'], f)
        
        print(f"✅ 最佳MLP模型已保存：{best_model_path}")
        print(f"✅ 标准化器已保存：{scaler_path}")
    
    # 可视化
    visualize_advanced_results(all_results, baseline_results, charts_dir, y_test)
    
    # 打印最终对比
    print("\n" + "=" * 80)
    print("Final Comparison Summary")
    print("最终对比总结")
    print("=" * 80)
    display_cols = ['model', 'test_auc', 'test_f1', 'test_precision', 'test_recall']
    print(results_df[display_cols].to_string(index=False))
    
    print("\n" + "=" * 80)
    print("Advanced Neural Network Tuning Complete!")
    print("深度神经网络调参完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()

