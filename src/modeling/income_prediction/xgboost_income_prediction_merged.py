"""
XGBoost Income Prediction Model (Merged Data)
XGBoost 收入预测模型（整合数据版本）

整合 listings_detailed 和 listings_detailed_2 两个数据文件，然后进行收入预测
Merges listings_detailed and listings_detailed_2, then performs income prediction

收入计算公式：income = price × (365 - availability_365)
Income calculation formula: income = price × (365 - availability_365)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# 添加 EDA 目录到路径 / Add EDA directory to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "EDA"))

from utils import setup_plotting, get_project_paths

# 添加特征工程目录 / Add feature engineering directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "feature_engineering"))
from income_feature_engineering_merged import engineer_income_features_merged

setup_plotting()


def train_merged_income_model():
    """训练整合数据版本的收入预测模型 / Train merged data income prediction model."""
    print("=" * 80)
    print("XGBoost Income Prediction Model (Merged Data)")
    print("XGBoost 收入预测模型（整合数据版本）")
    print("=" * 80)

    # =========================================================================
    # 1. 构建特征 / Build Features
    # =========================================================================
    print("\n1. 构建整合特征 / Building Merged Features...")
    try:
        processed_df = engineer_income_features_merged(save_processed=True)
        print(f"  [OK] 特征构建完成: {len(processed_df)} 行 × {processed_df.shape[1]} 列")
    except Exception as e:
        print(f"  [ERROR] 特征构建失败 / Feature engineering failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # =========================================================================
    # 2. 准备数据 / Prepare Data
    # =========================================================================
    print("\n2. 准备数据 / Preparing Data...")
    target_col = "income"
    if target_col not in processed_df.columns:
        print(f"  [ERROR] 目标列 '{target_col}' 不在数据中！")
        return
    
    # 排除目标列和数据源列
    feature_cols = [col for col in processed_df.columns 
                   if col not in [target_col, 'data_source']]

    X = processed_df[feature_cols].copy()
    y = processed_df[target_col].copy()
    
    # 只保留有效的收入数据
    mask = y.notna() & (y > 0)
    X = X[mask].copy()
    y = y[mask].copy()
    
    print(f"  有效数据: {len(X)} 条记录")
    print(f"  收入统计: 均值={y.mean():,.2f}, 中位数={y.median():,.2f}, "
          f"范围=[{y.min():,.2f}, {y.max():,.2f}]")

    # 异常值处理 / Outlier handling (使用IQR方法)
    Q1 = y.quantile(0.25)
    Q3 = y.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    mask = (y >= lower_bound) & (y <= upper_bound)
    X = X[mask].copy()
    y = y[mask].copy()
    
    print(f"  移除异常值后剩余: {len(X)} 条记录")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"  训练集: {len(X_train)} 条")
    print(f"  测试集: {len(X_test)} 条")

    # =========================================================================
    # 3. 训练模型 / Train Model
    # =========================================================================
    print("\n3. 训练 XGBoost 模型 / Training XGBoost Model...")
    # 使用优化的参数
    params = {
        "objective": "reg:squarederror",
        "n_estimators": 600,
        "max_depth": 6,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "colsample_bylevel": 0.8,
        "min_child_weight": 2,
        "gamma": 0.1,
        "reg_alpha": 0.2,
        "reg_lambda": 1.2,
        "eval_metric": "rmse",
        "early_stopping_rounds": 80,
        "random_state": 42,
        "n_jobs": -1,
    }

    model = xgb.XGBRegressor(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=50,
    )
    print("  [OK] 模型训练完成")

    # =========================================================================
    # 4. 模型评估 / Evaluation
    # =========================================================================
    print("\n4. 模型评估 / Model Evaluation...")
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    if hasattr(model, "best_iteration") and model.best_iteration is not None:
        print(f"  最佳迭代次数 / Best Iteration: {model.best_iteration}")

    print("\n训练集指标 / Training Metrics:")
    print(f"  RMSE: {train_rmse:,.2f}")
    print(f"  MAE: {train_mae:,.2f}")
    print(f"  R2: {train_r2:.4f}")

    print("\n测试集指标 / Test Metrics:")
    print(f"  RMSE: {test_rmse:,.2f}")
    print(f"  MAE: {test_mae:,.2f}")
    print(f"  R2: {test_r2:.4f}")

    rmse_diff = test_rmse - train_rmse
    overfit_ratio = (rmse_diff / train_rmse) * 100 if train_rmse != 0 else 0
    print("\n过拟合分析 / Overfitting Analysis:")
    print(f"  训练集RMSE: {train_rmse:,.2f}")
    print(f"  测试集RMSE: {test_rmse:,.2f}")
    print(f"  RMSE差异: {rmse_diff:,.2f}")
    print(f"  泛化差距: {overfit_ratio:.2f}%")
    if overfit_ratio < 5:
        print(f"  [OK] 模型泛化能力优秀，过拟合风险很低")
    elif overfit_ratio < 15:
        print(f"  [OK] 模型泛化能力良好，过拟合风险较低")
    elif overfit_ratio < 25:
        print(f"  [WARNING] 存在轻微过拟合，建议进一步调整参数")
    else:
        print(f"  [WARNING] 存在明显过拟合，建议降低模型复杂度或增加正则化")

    # =========================================================================
    # 5. 特征重要性 / Feature Importance
    # =========================================================================
    print("\n5. 特征重要性分析 / Feature Importance Analysis...")
    project_root, _, charts_eda_dir, charts_model_dir = get_project_paths()
    charts_dir = charts_model_dir
    
    importance_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    
    print("\n前20个最重要特征 / Top 20 Most Important Features:")
    print(importance_df.head(20).to_string(index=False))
    
    output_importance = charts_dir / "xgboost_income_feature_importance_merged.csv"
    importance_df.to_csv(output_importance, index=False)
    print(f"\n  [OK] 特征重要性已保存到: {output_importance}")

    # =========================================================================
    # 6. 可视化结果 / Visualization
    # =========================================================================
    print("\n6. 生成可视化图表 / Generating Visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 6.1 特征重要性Top 20
    top_features = importance_df.head(20)
    axes[0, 0].barh(range(len(top_features)), top_features['importance'].values)
    axes[0, 0].set_yticks(range(len(top_features)))
    axes[0, 0].set_yticklabels(top_features['feature'].values)
    axes[0, 0].set_xlabel('Importance')
    axes[0, 0].set_title('Top 20 Feature Importance (Merged Data - Income)')
    axes[0, 0].invert_yaxis()

    # 6.2 预测值 vs 真实值（训练集）
    axes[0, 1].scatter(y_train, y_train_pred, alpha=0.5, s=10)
    axes[0, 1].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
    axes[0, 1].set_xlabel('True Income')
    axes[0, 1].set_ylabel('Predicted Income')
    axes[0, 1].set_title(f'Training Set (R2 = {train_r2:.4f})')

    # 6.3 预测值 vs 真实值（测试集）
    axes[1, 0].scatter(y_test, y_test_pred, alpha=0.5, s=10)
    axes[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[1, 0].set_xlabel('True Income')
    axes[1, 0].set_ylabel('Predicted Income')
    axes[1, 0].set_title(f'Test Set (R2 = {test_r2:.4f})')

    # 6.4 残差分布
    residuals = y_test - y_test_pred
    axes[1, 1].hist(residuals, bins=50, edgecolor='black')
    axes[1, 1].set_xlabel('Residuals')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Residual Distribution')
    axes[1, 1].axvline(x=0, color='r', linestyle='--', lw=2)

    plt.tight_layout()
    output_chart = charts_dir / 'xgboost_income_model_results_merged.png'
    plt.savefig(output_chart, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  [OK] 可视化图表已保存到: {output_chart}")

    # =========================================================================
    # 7. 保存模型 / Save Model
    # =========================================================================
    print("\n7. 保存模型 / Saving Model...")
    model_path = charts_dir / "xgboost_income_model_merged.json"
    try:
        model.save_model(str(model_path))
        print(f"  [OK] 模型已保存到: {model_path}")
    except Exception:
        import pickle
        pkl_path = charts_dir / "xgboost_income_model_merged.pkl"
        with open(pkl_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"  [OK] 模型已保存到 (fallback): {pkl_path}")

    return {
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "train_mae": train_mae,
        "test_mae": test_mae,
        "train_r2": train_r2,
        "test_r2": test_r2,
        "overfitting_ratio": overfit_ratio,
    }


def main():
    metrics = train_merged_income_model()
    print("\n" + "=" * 80)
    print("最终指标 / Final Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            if 'rmse' in key.lower() or 'mae' in key.lower():
                print(f"  {key}: {value:,.2f}")
            else:
                print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print("=" * 80)
    print("整合数据收入预测模型完成 / Merged data income model training complete")
    print("=" * 80)


if __name__ == "__main__":
    main()

