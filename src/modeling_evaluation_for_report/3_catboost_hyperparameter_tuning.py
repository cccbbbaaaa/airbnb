"""
================================================================================
CatBoost 超参数调优
CatBoost Hyperparameter Tuning
================================================================================
本脚本用于:
1. 测试不同迭代次数的CatBoost模型 (550, 600, 650)
2. 进行网格搜索寻找最优超参数组合
3. 保存测试结果和最优模型

This script performs:
1. Testing CatBoost with different iteration counts (550, 600, 650)
2. Grid search for optimal hyperparameter combination
3. Saving test results and optimal model

Author: Data Science Course Project
Date: 2025-11-27
================================================================================
"""

import numpy as np
import pandas as pd
from pathlib import Path
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import time

# ==================== PROJECT PATHS ====================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "charts" / "charts_for_report" / "modeling"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_and_prepare_data():
    """
    Load feature-engineered data and prepare train/test split

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    print("\n" + "=" * 80)
    print("加载特征数据 - LOADING FEATURE DATA")
    print("=" * 80)

    data_path = DATA_DIR / "train_data.csv"
    df = pd.read_csv(data_path)

    # Filter valid ratings and create binary target
    working_df = df[df["review_scores_rating"].notna()].copy()
    working_df["is_five_star"] = np.isclose(
        working_df["review_scores_rating"], 5.0, atol=1e-6
    ).astype(int)

    # Separate features and target
    exclude_cols = {"review_scores_rating", "is_five_star"}
    feature_cols = [col for col in working_df.columns if col not in exclude_cols]
    X = working_df[feature_cols].copy()
    y = working_df["is_five_star"].copy()

    # Train-test split (same as in training script)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n数据集划分:")
    print(f"  训练集: {X_train.shape}")
    print(f"  测试集: {X_test.shape}")
    print(f"  特征数量: {len(feature_cols)}")
    print(f"  5星样本比例: {y.mean():.2%}")

    return X_train, X_test, y_train, y_test


def test_iteration_counts(X_train, X_test, y_train, y_test,
                         iterations_list=[550, 600, 650]):
    """
    Test CatBoost with different iteration counts

    Args:
        X_train, X_test, y_train, y_test: Training and test data
        iterations_list: List of iteration counts to test

    Returns:
        pd.DataFrame: Results comparison
    """
    print("\n" + "=" * 80)
    print("测试不同迭代次数 - TESTING DIFFERENT ITERATION COUNTS")
    print("=" * 80)
    print(f"\n将测试的迭代次数: {iterations_list}")

    results = []

    for iterations in iterations_list:
        print(f"\n{'=' * 80}")
        print(f"测试迭代次数: {iterations}")
        print(f"{'=' * 80}")

        # Train model
        print(f"\n训练 CatBoost (iterations={iterations})...")
        start_time = time.time()

        model = CatBoostClassifier(
            iterations=iterations,
            learning_rate=0.1,
            depth=6,
            l2_leaf_reg=3,
            random_seed=42,
            verbose=False,
            eval_metric='AUC',
            auto_class_weights='Balanced'
        )

        model.fit(X_train, y_train)
        train_time = time.time() - start_time

        # Predictions
        y_train_pred_proba = model.predict_proba(X_train)[:, 1]
        y_test_pred_proba = model.predict_proba(X_test)[:, 1]

        # Metrics
        train_auc = roc_auc_score(y_train, y_train_pred_proba)
        test_auc = roc_auc_score(y_test, y_test_pred_proba)

        # Convert to error percentage
        train_error = (1 - train_auc) * 100
        test_error = (1 - test_auc) * 100
        gap = test_error - train_error

        # Additional metrics
        y_test_pred = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred)

        print(f"\n结果:")
        print(f"  训练时间: {train_time:.2f}s")
        print(f"  Train AUC: {train_auc:.4f} (Error: {train_error:.2f}%)")
        print(f"  Test AUC:  {test_auc:.4f} (Error: {test_error:.2f}%)")
        print(f"  Gap (Test - Train): {gap:.2f}%")
        print(f"  Test Accuracy: {test_accuracy:.4f}")
        print(f"  Test F1: {test_f1:.4f}")

        results.append({
            'iterations': iterations,
            'train_auc': train_auc,
            'test_auc': test_auc,
            'train_error': train_error,
            'test_error': test_error,
            'gap': gap,
            'test_accuracy': test_accuracy,
            'test_f1': test_f1,
            'train_time': train_time
        })

        # Save model
        model_path = MODEL_DIR / f"catboost_model_{iterations}iter.cbm"
        model.save_model(model_path)
        print(f"  模型已保存: {model_path}")

    # Create comparison table
    print("\n" + "=" * 80)
    print("结果对比 - COMPARISON")
    print("=" * 80)

    results_df = pd.DataFrame(results)
    print("\n详细对比:")
    print(results_df.to_string(index=False))

    # Determine optimal
    optimal_idx = results_df['test_auc'].idxmax()
    optimal_config = results_df.loc[optimal_idx]

    print("\n" + "=" * 80)
    print("最优配置 - OPTIMAL CONFIGURATION")
    print("=" * 80)
    print(f"\n🎯 最优迭代次数: {int(optimal_config['iterations'])}")
    print(f"   Test AUC: {optimal_config['test_auc']:.4f}")
    print(f"   Test Error: {optimal_config['test_error']:.2f}%")
    print(f"   Gap: {optimal_config['gap']:.2f}%")
    print(f"   Test F1: {optimal_config['test_f1']:.4f}")
    print(f"   训练时间: {optimal_config['train_time']:.2f}s")

    # Save results
    results_path = MODEL_DIR / "catboost_iteration_comparison.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n✅ 结果已保存: {results_path}")

    # Analysis
    print("\n" + "=" * 80)
    print("分析建议 - ANALYSIS")
    print("=" * 80)

    if optimal_config['iterations'] == min(iterations_list):
        print(f"\n📊 分析: {int(optimal_config['iterations'])}次迭代已经达到最优")
        print(f"   更多迭代存在轻微过拟合")
        print(f"   建议: 使用{int(optimal_config['iterations'])}次迭代作为最终模型")
    elif optimal_config['iterations'] == max(iterations_list):
        print(f"\n📊 分析: {int(optimal_config['iterations'])}次迭代表现最佳")
        print(f"   模型仍在改进中")
        print(f"   建议: 使用{int(optimal_config['iterations'])}次迭代，或考虑进一步增加迭代次数")
    else:
        print(f"\n📊 分析: {int(optimal_config['iterations'])}次迭代表现最佳")
        print(f"   在性能和稳定性之间取得了良好平衡")
        print(f"   建议: 使用{int(optimal_config['iterations'])}次迭代作为最终模型")

    # Check for overfitting trend
    if results_df['gap'].iloc[-1] > results_df['gap'].iloc[0]:
        print(f"\n⚠️  注意: 随着迭代次数增加，过拟合gap从 {results_df['gap'].iloc[0]:.2f}% 增至 {results_df['gap'].iloc[-1]:.2f}%")
    else:
        print(f"\n✅ 良好: 过拟合gap保持稳定或改善")

    return results_df


def perform_grid_search(X_train, X_test, y_train, y_test,
                       param_grid=None, cv_folds=3):
    """
    Perform grid search for optimal hyperparameters

    Args:
        X_train, X_test, y_train, y_test: Training and test data
        param_grid: Dictionary of parameters to search
        cv_folds: Number of cross-validation folds

    Returns:
        dict: Best parameters and results
    """
    print("\n" + "=" * 80)
    print("网格搜索超参数调优 - GRID SEARCH HYPERPARAMETER TUNING")
    print("=" * 80)

    if param_grid is None:
        param_grid = {
            'iterations': [400, 550, 800],
            'depth': [4, 6, 8],
            'learning_rate': [0.01, 0.03, 0.05],
            'l2_leaf_reg': [1, 3, 5],
        }

    print(f"\n参数搜索空间:")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")

    total_combinations = np.prod([len(v) for v in param_grid.values()])
    print(f"\n总组合数: {total_combinations}")
    print(f"交叉验证折数: {cv_folds}")
    print(f"总训练次数: {total_combinations * cv_folds}")

    from sklearn.model_selection import GridSearchCV

    # Create base model
    base_model = CatBoostClassifier(
        random_seed=42,
        verbose=False,
        eval_metric='AUC',
        auto_class_weights='Balanced'
    )

    # Perform grid search
    print(f"\n开始网格搜索...")
    start_time = time.time()

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv_folds,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)
    search_time = time.time() - start_time

    print(f"\n✅ 网格搜索完成！用时: {search_time:.2f}s ({search_time/60:.1f}分钟)")

    # Best parameters
    print("\n" + "=" * 80)
    print("最优参数 - BEST PARAMETERS")
    print("=" * 80)
    print(f"\n{grid_search.best_params_}")

    # Best model performance
    best_model = grid_search.best_estimator_
    y_train_pred = best_model.predict_proba(X_train)[:, 1]
    y_test_pred = best_model.predict_proba(X_test)[:, 1]

    train_auc = roc_auc_score(y_train, y_train_pred)
    test_auc = roc_auc_score(y_test, y_test_pred)

    print(f"\n最优模型性能:")
    print(f"  CV AUC (训练集): {grid_search.best_score_:.4f}")
    print(f"  Train AUC (全训练集): {train_auc:.4f}")
    print(f"  Test AUC (测试集): {test_auc:.4f}")
    print(f"  Gap: {(test_auc - train_auc):.4f}")

    # Save best model
    best_model_path = MODEL_DIR / "catboost_best_model.cbm"
    best_model.save_model(best_model_path)
    print(f"\n✅ 最优模型已保存: {best_model_path}")

    # Save grid search results
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_path = MODEL_DIR / "catboost_grid_search_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"✅ 网格搜索结果已保存: {results_path}")

    return {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'train_auc': train_auc,
        'test_auc': test_auc,
        'search_time': search_time,
        'best_model': best_model
    }


def main():
    """
    Main function for CatBoost hyperparameter tuning
    """
    print("\n" + "=" * 80)
    print("CATBOOST 超参数调优 - CATBOOST HYPERPARAMETER TUNING")
    print("=" * 80)

    # Load data
    X_train, X_test, y_train, y_test = load_and_prepare_data()

    # Test different iteration counts
    print("\n\n")
    print("█" * 80)
    print("█ 第1部分: 测试不同迭代次数")
    print("█" * 80)
    iteration_results = test_iteration_counts(
        X_train, X_test, y_train, y_test,
        iterations_list=[550, 600, 650]
    )

    # Optional: Perform grid search (commented out to save time)
    # Uncomment if you want to perform full grid search
    """
    print("\n\n")
    print("█" * 80)
    print("█ 第2部分: 网格搜索最优参数组合")
    print("█" * 80)
    grid_results = perform_grid_search(X_train, X_test, y_train, y_test)
    """

    print("\n" + "=" * 80)
    print("✅ 超参数调优完成！")
    print("=" * 80)
    print(f"\n所有结果已保存到: {MODEL_DIR}")
    print("\n生成的文件:")
    print(f"  1. catboost_iteration_comparison.csv - 迭代次数对比结果")
    print(f"  2. catboost_model_550iter.cbm - 550次迭代模型")
    print(f"  3. catboost_model_600iter.cbm - 600次迭代模型")
    print(f"  4. catboost_model_650iter.cbm - 650次迭代模型")
    # print(f"  5. catboost_best_model.cbm - 网格搜索最优模型")
    # print(f"  6. catboost_grid_search_results.csv - 网格搜索完整结果")


if __name__ == "__main__":
    main()
