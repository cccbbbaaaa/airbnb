"""
CatBoost Hyperparameter Tuning (Fixed Version)
CatBoost超参数调优（修复版）：解决GridSearchCV与use_best_model的冲突

问题分析：
- GridSearchCV在交叉验证时无法为每个fold传递eval_set
- CatBoost的use_best_model=True需要eval_set
- 这导致GridSearchCV中的模型行为与基线模型不一致

解决方案：
- 使用CatBoost内置的cv方法进行交叉验证
- 或者手动实现网格搜索，为每个参数组合单独训练和评估
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
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


def manual_grid_search_catboost(X_train, X_test, y_train, y_test):
    """手动实现网格搜索，确保use_best_model正常工作"""
    print("\n" + "=" * 80)
    print("CatBoost Manual Grid Search")
    print("CatBoost手动网格搜索（修复use_best_model问题）")
    print("=" * 80)
    
    # 定义参数网格
    param_grid = {
        'iterations': [400, 600, 800],
        'depth': [4, 6, 8],
        'learning_rate': [0.01, 0.03, 0.05],
        'l2_leaf_reg': [1, 3, 5],
    }
    
    # 生成所有参数组合
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(product(*param_values))
    
    total_combinations = len(param_combinations)
    print(f"  参数网格大小: {total_combinations} 种组合")
    print(f"  参数范围:")
    for param, values in param_grid.items():
        print(f"    {param}: {values}")
    
    print(f"\n  开始手动网格搜索（使用CatBoost的cv方法）...")
    print(f"  总训练次数: {total_combinations} 种参数组合")
    
    start_time = time.time()
    
    # 使用CatBoost的cv方法进行交叉验证
    best_cv_score = -np.inf
    best_params = None
    best_model = None
    results = []
    
    for idx, param_combo in enumerate(param_combinations):
        params = dict(zip(param_names, param_combo))
        
        # 创建模型
        model = cb.CatBoostClassifier(
            **params,
            random_seed=42,
            loss_function='Logloss',
            eval_metric='AUC',
            early_stopping_rounds=80,
            verbose=False,
            auto_class_weights='Balanced'
        )
        
        # 使用CatBoost的cv方法（内部会处理use_best_model）
        # 注意：CatBoost的cv方法会自动创建catboost_info文件夹存储训练日志
        # 可以通过设置logging_level='Silent'来减少输出，但文件夹仍会被创建
        cv_data = cb.Pool(X_train, y_train)
        cv_results = cb.cv(
            cv_data,
            model.get_params(),
            fold_count=3,
            shuffle=True,
            partition_random_seed=42,
            verbose=False,
            early_stopping_rounds=80,
            logging_level='Silent'  # 减少日志输出
        )
        
        # 获取最佳AUC（最后一列的AUC，或使用最佳迭代的AUC）
        # cv方法会自动选择最佳迭代，我们取最后一行的AUC
        cv_auc = cv_results['test-AUC-mean'].iloc[-1]
        
        # 训练最终模型（使用全部训练数据和测试集作为验证集）
        model.fit(
            X_train, y_train,
            eval_set=(X_test, y_test),
            use_best_model=True
        )
        
        # 评估测试集
        y_test_proba = model.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, y_test_proba)
        
        results.append({
            **params,
            'cv_auc': cv_auc,
            'test_auc': test_auc
        })
        
        # 更新最佳参数
        if cv_auc > best_cv_score:
            best_cv_score = cv_auc
            best_params = params.copy()
            best_model = model
        
        if (idx + 1) % 10 == 0:
            print(f"  已完成 {idx + 1}/{total_combinations} 种组合...")
    
    tuning_time = time.time() - start_time
    
    print(f"\n  手动网格搜索完成！")
    print(f"  调优时间: {tuning_time:.2f}秒 ({tuning_time/60:.2f}分钟)")
    print(f"  最佳交叉验证 AUC: {best_cv_score:.4f}")
    print(f"  最佳参数:")
    for param, value in best_params.items():
        print(f"    {param}: {value}")
    
    # 评估最佳模型
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
        'cv_score': best_cv_score,
        **best_params
    }
    
    print(f"\n  最佳模型测试集性能:")
    print(f"    Test AUC: {metrics['test_auc']:.4f}")
    print(f"    Test F1: {metrics['test_f1']:.4f}")
    print(f"    Test Precision: {metrics['test_precision']:.4f}")
    print(f"    Test Recall: {metrics['test_recall']:.4f}")
    
    # 保存所有结果
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('cv_auc', ascending=False)
    
    return best_model, metrics, best_params, results_df


def main():
    """主函数"""
    print("=" * 80)
    print("CatBoost Hyperparameter Tuning (Fixed Version)")
    print("CatBoost超参数调优（修复版）")
    print("=" * 80)
    
    # 加载数据
    df, charts_dir = load_training_data()
    X_train, X_test, y_train, y_test, feature_cols = prepare_binary_dataset(df)
    
    # 训练基线模型
    baseline_model, baseline_metrics = train_baseline_catboost(X_train, X_test, y_train, y_test)
    
    # 手动网格搜索
    tuned_model, tuned_metrics, best_params, all_results_df = manual_grid_search_catboost(
        X_train, X_test, y_train, y_test
    )
    
    # 检查基线参数组合的排名
    baseline_combo = (600, 6, 0.03, 3)
    baseline_row = all_results_df[
        (all_results_df['iterations'] == 600) &
        (all_results_df['depth'] == 6) &
        (all_results_df['learning_rate'] == 0.03) &
        (all_results_df['l2_leaf_reg'] == 3)
    ]
    
    print("\n" + "=" * 80)
    print("Baseline Parameters Ranking")
    print("基线参数组合排名")
    print("=" * 80)
    if not baseline_row.empty:
        baseline_rank = all_results_df.index[all_results_df['cv_auc'] <= baseline_row.iloc[0]['cv_auc']].tolist()[0] + 1
        print(f"基线参数组合 (600, 6, 0.03, 3):")
        print(f"  交叉验证AUC排名: 第 {baseline_rank} 名（共 {len(all_results_df)} 种组合）")
        print(f"  交叉验证AUC: {baseline_row.iloc[0]['cv_auc']:.4f}")
        print(f"  测试集AUC: {baseline_row.iloc[0]['test_auc']:.4f}")
    else:
        print("警告: 未找到基线参数组合的结果")
    
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
    
    output_path = charts_dir / 'catboost_tuning_results_fixed.csv'
    results_df = pd.DataFrame([baseline_metrics, tuned_metrics])
    results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ 结果已保存至: {output_path}")
    
    # 保存所有参数组合的结果
    all_results_path = charts_dir / 'catboost_all_combinations_results.csv'
    all_results_df.to_csv(all_results_path, index=False, encoding='utf-8-sig')
    print(f"✅ 所有参数组合结果已保存至: {all_results_path}")
    
    # 清理CatBoost自动生成的训练日志文件夹
    import shutil
    catboost_info_dir = PROJECT_ROOT / 'catboost_info'
    if catboost_info_dir.exists():
        shutil.rmtree(catboost_info_dir)
        print(f"\n✅ 已清理 catboost_info 文件夹")
    
    print("\n" + "=" * 80)
    print("CatBoost Hyperparameter Tuning Complete!")
    print("CatBoost超参数调优完成！")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    results = main()

