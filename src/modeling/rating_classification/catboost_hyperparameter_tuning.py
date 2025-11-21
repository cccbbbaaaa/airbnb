"""
CatBoost Hyperparameter Tuning
CatBoost超参数调优：使用网格搜索寻找最佳参数

调优参数范围：
- iterations: [400, 600, 800]
- depth: [4, 6, 8]
- learning_rate: [0.01, 0.03, 0.05]
- l2_leaf_reg: [1, 3, 5]
- random_strength: [0.5, 1.0]

输出：
- 最佳参数组合
- 调优时间
- 性能对比（vs原始CatBoost）
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("错误: CatBoost 不可用，请先安装 catboost")
    sys.exit(1)

# 路径设置
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "EDA"))
from utils import get_project_paths


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

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"  训练集: {len(X_train):,} (5 星占比: {y_train.mean():.2%})")
    print(f"  测试集: {len(X_test):,} (5 星占比: {y_test.mean():.2%})")

    return X_train, X_test, y_train, y_test, feature_cols


def train_baseline_catboost(X_train, X_test, y_train, y_test):
    """训练基线CatBoost模型（原始参数）"""
    print("\n" + "=" * 80)
    print("Training Baseline CatBoost Model")
    print("训练基线CatBoost模型（原始参数）")
    print("=" * 80)
    
    start_time = time.time()
    
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
    
    training_time = time.time() - start_time
    
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'model': 'CatBoost_Baseline',
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'test_precision': precision_score(y_test, y_test_pred),
        'test_recall': recall_score(y_test, y_test_pred),
        'test_f1': f1_score(y_test, y_test_pred),
        'test_auc': roc_auc_score(y_test, y_test_proba),
        'training_time_seconds': training_time,
        'iterations': 600,
        'depth': 6,
        'learning_rate': 0.03,
        'l2_leaf_reg': 3,
    }
    
    print(f"  训练时间: {training_time:.2f}秒")
    print(f"  Test AUC: {metrics['test_auc']:.4f}")
    print(f"  Test F1: {metrics['test_f1']:.4f}")
    print(f"  Test Precision: {metrics['test_precision']:.4f}")
    print(f"  Test Recall: {metrics['test_recall']:.4f}")
    
    return model, metrics


def grid_search_catboost(X_train, X_test, y_train, y_test):
    """对CatBoost进行网格搜索"""
    print("\n" + "=" * 80)
    print("CatBoost Grid Search")
    print("CatBoost网格搜索")
    print("=" * 80)
    
    # 定义参数网格
    param_grid = {
        'iterations': [400, 600, 800],
        'depth': [4, 6, 8],
        'learning_rate': [0.01, 0.03, 0.05],
        'l2_leaf_reg': [1, 3, 5],
    }
    
    total_combinations = np.prod([len(v) for v in param_grid.values()])
    print(f"  参数网格大小: {total_combinations} 种组合")
    print(f"  参数范围:")
    for param, values in param_grid.items():
        print(f"    {param}: {values}")
    
    # 创建基础模型
    base_model = cb.CatBoostClassifier(
        random_seed=42,
        loss_function='Logloss',
        eval_metric='AUC',
        early_stopping_rounds=80,
        verbose=False,
        auto_class_weights='Balanced',
        use_best_model=True
    )
    
    # 使用GridSearchCV
    print(f"\n  开始网格搜索（3折交叉验证）...")
    print(f"  总训练次数: {total_combinations} × 3 = {total_combinations * 3}")
    
    start_time = time.time()
    
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    # 注意：GridSearchCV在交叉验证时无法传递eval_set给每个fold
    # 所以这里只传递训练数据，use_best_model会在交叉验证中失效
    # 但这是GridSearchCV的限制，我们需要接受这个限制
    grid_search.fit(X_train, y_train)
    
    tuning_time = time.time() - start_time
    
    print(f"\n  网格搜索完成！")
    print(f"  调优时间: {tuning_time:.2f}秒 ({tuning_time/60:.2f}分钟)")
    print(f"  最佳交叉验证 AUC: {grid_search.best_score_:.4f}")
    print(f"  最佳参数:")
    for param, value in grid_search.best_params_.items():
        print(f"    {param}: {value}")
    
    # 评估最佳模型
    best_model = grid_search.best_estimator_
    y_test_pred = best_model.predict(X_test)
    y_test_proba = best_model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'model': 'CatBoost_Tuned',
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'test_precision': precision_score(y_test, y_test_pred),
        'test_recall': recall_score(y_test, y_test_pred),
        'test_f1': f1_score(y_test, y_test_pred),
        'test_auc': roc_auc_score(y_test, y_test_proba),
        'training_time_seconds': tuning_time,
        'cv_score': grid_search.best_score_,
        **grid_search.best_params_
    }
    
    print(f"\n  最佳模型测试集性能:")
    print(f"    Test AUC: {metrics['test_auc']:.4f}")
    print(f"    Test F1: {metrics['test_f1']:.4f}")
    print(f"    Test Precision: {metrics['test_precision']:.4f}")
    print(f"    Test Recall: {metrics['test_recall']:.4f}")
    
    return best_model, metrics, grid_search.best_params_


def main():
    """主函数"""
    print("=" * 80)
    print("CatBoost Hyperparameter Tuning")
    print("CatBoost超参数调优")
    print("=" * 80)
    
    # 加载数据
    df, charts_dir = load_training_data()
    X_train, X_test, y_train, y_test, feature_cols = prepare_binary_dataset(df)
    
    # 训练基线模型
    baseline_model, baseline_metrics = train_baseline_catboost(X_train, X_test, y_train, y_test)
    
    # 网格搜索
    tuned_model, tuned_metrics, best_params = grid_search_catboost(X_train, X_test, y_train, y_test)
    
    # 对比结果
    print("\n" + "=" * 80)
    print("Comparison: Baseline vs Tuned")
    print("对比：基线模型 vs 调优模型")
    print("=" * 80)
    
    comparison_df = pd.DataFrame([baseline_metrics, tuned_metrics])
    comparison_df = comparison_df[['model', 'test_auc', 'test_f1', 'test_precision', 'test_recall', 'training_time_seconds']]
    
    print("\n性能对比:")
    print(comparison_df.to_string(index=False))
    
    # 计算改进
    auc_improvement = tuned_metrics['test_auc'] - baseline_metrics['test_auc']
    f1_improvement = tuned_metrics['test_f1'] - baseline_metrics['test_f1']
    recall_improvement = tuned_metrics['test_recall'] - baseline_metrics['test_recall']
    precision_improvement = tuned_metrics['test_precision'] - baseline_metrics['test_precision']
    
    print(f"\n性能提升:")
    print(f"  AUC: {auc_improvement:+.4f} ({auc_improvement/baseline_metrics['test_auc']*100:+.2f}%)")
    print(f"  F1: {f1_improvement:+.4f} ({f1_improvement/baseline_metrics['test_f1']*100:+.2f}%)")
    print(f"  Precision: {precision_improvement:+.4f} ({precision_improvement/baseline_metrics['test_precision']*100:+.2f}%)")
    print(f"  Recall: {recall_improvement:+.4f} ({recall_improvement/baseline_metrics['test_recall']*100:+.2f}%)")
    
    # 保存结果
    results = {
        'baseline': baseline_metrics,
        'tuned': tuned_metrics,
        'best_params': best_params,
        'tuning_time_seconds': tuned_metrics['training_time_seconds'],
        'improvements': {
            'auc': auc_improvement,
            'f1': f1_improvement,
            'precision': precision_improvement,
            'recall': recall_improvement,
        }
    }
    
    output_path = charts_dir / 'catboost_tuning_results.csv'
    results_df = pd.DataFrame([baseline_metrics, tuned_metrics])
    results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ 结果已保存至: {output_path}")
    
    # 保存最佳参数
    params_path = charts_dir / 'catboost_best_params.txt'
    with open(params_path, 'w', encoding='utf-8') as f:
        f.write("CatBoost Best Parameters\n")
        f.write("=" * 50 + "\n\n")
        f.write("Grid Search Results:\n")
        for param, value in best_params.items():
            f.write(f"  {param}: {value}\n")
        f.write(f"\nTuning Time: {tuned_metrics['training_time_seconds']:.2f} seconds ({tuned_metrics['training_time_seconds']/60:.2f} minutes)\n")
        f.write(f"CV Score: {tuned_metrics['cv_score']:.4f}\n")
        f.write(f"Test AUC: {tuned_metrics['test_auc']:.4f}\n")
        f.write(f"\nParameter Grid:\n")
        f.write(f"  iterations: [400, 600, 800]\n")
        f.write(f"  depth: [4, 6, 8]\n")
        f.write(f"  learning_rate: [0.01, 0.03, 0.05]\n")
        f.write(f"  l2_leaf_reg: [1, 3, 5]\n")
    
    print(f"✅ 最佳参数已保存至: {params_path}")
    
    print("\n" + "=" * 80)
    print("CatBoost Hyperparameter Tuning Complete!")
    print("CatBoost超参数调优完成！")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    results = main()

