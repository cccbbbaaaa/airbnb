"""
XGBoost Rating Prediction Model (Enhanced Features)
XGBoost 评分预测模型（强化特征工程版本）

通过专门的特征工程提升评分预测效果
Improve rating prediction performance with dedicated feature engineering
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# 添加 EDA 目录到路径 / Add EDA directory to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "EDA"))

from utils import setup_plotting, get_project_paths

# 添加特征工程目录 / Add feature engineering directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "feature_engineering"))
from rating_feature_engineering import engineer_rating_features

setup_plotting()


def train_enhanced_rating_model():
    """训练强化版评分预测模型 / Train enhanced rating prediction model."""
    print("=" * 80)
    print("XGBoost Rating Prediction Model (Enhanced Features)")
    print("XGBoost 评分预测模型（强化特征工程版本）")
    print("=" * 80)

    # =========================================================================
    # 1. 构建特征 / Build Features
    # =========================================================================
    print("\n1. 构建特征 / Building Features...")
    processed_df = engineer_rating_features(save_processed=True)
    print(f"  [OK] 特征构建完成: {len(processed_df)} 行 × {processed_df.shape[1]} 列")

    # =========================================================================
    # 2. 准备数据 / Prepare Data
    # =========================================================================
    print("\n2. 准备数据 / Preparing Data...")
    target_col = "review_scores_rating"
    feature_cols = [col for col in processed_df.columns if col != target_col]

    X = processed_df[feature_cols].copy()
    y = processed_df[target_col].copy()

    # 异常值处理 / Outlier handling
    Q1 = y.quantile(0.25)
    Q3 = y.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    mask = (y >= lower_bound) & (y <= upper_bound)
    X = X[mask].copy()
    y = y[mask].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"  训练集: {len(X_train)} 条")
    print(f"  测试集: {len(X_test)} 条")

    # =========================================================================
    # 3. 训练模型 / Train Model
    # =========================================================================
    print("\n3. 训练 XGBoost 模型 / Training XGBoost Model...")
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
    print(f"  RMSE: {train_rmse:.4f}")
    print(f"  MAE: {train_mae:.4f}")
    print(f"  R²: {train_r2:.4f}")

    print("\n测试集指标 / Test Metrics:")
    print(f"  RMSE: {test_rmse:.4f}")
    print(f"  MAE: {test_mae:.4f}")
    print(f"  R²: {test_r2:.4f}")

    rmse_diff = test_rmse - train_rmse
    overfit_ratio = (rmse_diff / train_rmse) * 100
    print("\n过拟合分析 / Overfitting Analysis:")
    print(f"  RMSE 差异: {rmse_diff:.4f}")
    print(f"  泛化差距: {overfit_ratio:.2f}%")

    # =========================================================================
    # 5. 特征重要性 / Feature Importance
    # =========================================================================
    project_root, _, charts_eda_dir, charts_model_dir = get_project_paths()
    charts_dir = charts_model_dir  # 使用模型目录 / Use model directory
    importance_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    output_importance = charts_dir / "xgboost_rating_feature_importance_v2.csv"
    importance_df.to_csv(output_importance, index=False)
    print(f"\n  [OK] 特征重要性已保存到: {output_importance}")

    # =========================================================================
    # 6. 保存模型 / Save Model
    # =========================================================================
    model_path = charts_dir / "xgboost_rating_model_v2.pkl"
    importance_df.to_csv(output_importance, index=False)
    model.save_model(model_path)
    print(f"  [OK] 模型已保存到: {model_path}")

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
    metrics = train_enhanced_rating_model()
    print("\n最终指标 / Final Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    print("=" * 80)
    print("强化版评分预测模型完成 / Enhanced rating model training complete")
    print("=" * 80)


if __name__ == "__main__":
    main()

