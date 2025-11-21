"""
Model Metrics Collector
收集模型训练时间、复杂度等指标

用于生成模型对比文档
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# 路径设置
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "EDA"))
from utils import get_project_paths


def load_training_data():
    """加载训练数据"""
    project_root, data_dir, charts_eda_dir, charts_model_dir = get_project_paths()
    train_path = data_dir / "processed" / "train_data.csv"
    df = pd.read_csv(train_path)
    
    working_df = df[df["review_scores_rating"].notna()].copy()
    working_df["is_five_star"] = np.isclose(working_df["review_scores_rating"], 5.0, atol=1e-6).astype(int)
    
    exclude_cols = {"review_scores_rating", "is_five_star"}
    feature_cols = [col for col in working_df.columns if col not in exclude_cols]
    
    X = working_df[feature_cols].copy()
    y = working_df["is_five_star"].copy()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, feature_cols


def count_model_parameters(model, model_name):
    """估算模型参数数量"""
    if 'LogisticRegression' in model_name:
        # 逻辑回归: n_features + 1 (bias)
        return model.n_features_in_ + 1
    
    elif 'LinearSVM' in model_name or 'SVM' in model_name:
        # Linear SVM: n_features (支持向量权重)
        if hasattr(model, 'coef_'):
            return model.coef_.size + 1  # +1 for bias
        return 0
    
    elif 'RandomForest' in model_name:
        # 随机森林: n_estimators * avg_tree_size
        n_estimators = model.n_estimators
        # 估算每棵树的平均节点数（简化估算）
        avg_nodes = 100  # 保守估计
        return n_estimators * avg_nodes
    
    elif 'XGBoost' in model_name or 'CatBoost' in model_name:
        # 梯度提升树: 类似随机森林
        if hasattr(model, 'n_estimators'):
            n_estimators = model.n_estimators
        elif hasattr(model, 'tree_count_'):
            # CatBoost的tree_count_是属性，不是方法
            n_estimators = model.tree_count_ if isinstance(model.tree_count_, int) else 100
        elif hasattr(model, 'get_tree_count'):
            n_estimators = model.get_tree_count()
        else:
            n_estimators = 100  # 默认值
        avg_nodes = 100
        return n_estimators * avg_nodes
    
    elif 'MLP' in model_name:
        # MLP: 计算所有层的参数
        total_params = 0
        if hasattr(model, 'coefs_'):
            for i, coef in enumerate(model.coefs_):
                total_params += coef.size
            if hasattr(model, 'intercepts_'):
                for intercept in model.intercepts_:
                    total_params += intercept.size
        return total_params
    
    return 0


def measure_model_complexity(model, model_name, X_train):
    """测量模型复杂度"""
    complexity = {
        'n_parameters': count_model_parameters(model, model_name),
        'model_size_mb': 0,  # 可以后续添加
    }
    
    # 估算模型大小（MB）
    if hasattr(model, 'coefs_'):
        # MLP
        total_size = 0
        for coef in model.coefs_:
            total_size += coef.nbytes
        if hasattr(model, 'intercepts_'):
            for intercept in model.intercepts_:
                total_size += intercept.nbytes
        complexity['model_size_mb'] = total_size / (1024 * 1024)
    elif hasattr(model, 'coef_'):
        # 线性模型
        complexity['model_size_mb'] = model.coef_.nbytes / (1024 * 1024)
    else:
        # 树模型（估算）
        complexity['model_size_mb'] = complexity['n_parameters'] * 8 / (1024 * 1024)  # 假设每个参数8字节
    
    return complexity


def train_and_measure_model(model_class, model_params, model_name, 
                           X_train, X_test, y_train, y_test, scaler=None):
    """训练模型并测量时间和性能"""
    print(f"\n训练模型: {model_name}")
    
    # 准备数据
    if scaler:
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test
    
    # 创建模型
    model = model_class(**model_params)
    
    # 测量训练时间
    start_time = time.time()
    model.fit(X_train_scaled, y_train)
    training_time = time.time() - start_time
    
    # 预测
    if hasattr(model, 'predict_proba'):
        y_train_proba = model.predict_proba(X_train_scaled)[:, 1]
        y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
    elif hasattr(model, 'decision_function'):
        from scipy.special import expit
        train_scores = model.decision_function(X_train_scaled)
        test_scores = model.decision_function(X_test_scaled)
        y_train_proba = expit(train_scores)
        y_test_proba = expit(test_scores)
    else:
        y_train_proba = np.zeros(len(y_train))
        y_test_proba = np.zeros(len(y_test))
    
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # 计算指标
    metrics = {
        'model_name': model_name,
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
        'training_time_seconds': training_time,
    }
    
    # 测量复杂度
    complexity = measure_model_complexity(model, model_name, X_train_scaled)
    metrics.update(complexity)
    
    print(f"  训练时间: {training_time:.2f}秒")
    print(f"  参数数量: {complexity['n_parameters']:,}")
    print(f"  模型大小: {complexity['model_size_mb']:.2f} MB")
    
    return metrics, model


def collect_all_metrics():
    """收集所有模型的指标"""
    print("=" * 80)
    print("Collecting Model Metrics")
    print("收集模型指标")
    print("=" * 80)
    
    # 加载数据
    X_train, X_test, y_train, y_test, feature_cols = load_training_data()
    print(f"\n数据规模: {len(X_train):,} 训练样本, {len(X_test):,} 测试样本")
    print(f"特征数量: {len(feature_cols)}")
    
    all_metrics = []
    
    # 1. Logistic Regression
    scaler_lr = StandardScaler()
    metrics_lr, _ = train_and_measure_model(
        LogisticRegression,
        {'max_iter': 2000, 'class_weight': 'balanced', 'n_jobs': -1, 'random_state': 42},
        'LogisticRegression',
        X_train, X_test, y_train, y_test, scaler_lr
    )
    all_metrics.append(metrics_lr)
    
    # 2. Linear SVM
    scaler_svm = StandardScaler()
    metrics_svm, _ = train_and_measure_model(
        LinearSVC,
        {'class_weight': 'balanced', 'max_iter': 5000, 'dual': False, 'random_state': 42},
        'LinearSVM',
        X_train, X_test, y_train, y_test, scaler_svm
    )
    all_metrics.append(metrics_svm)
    
    # 3. Random Forest
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    class_weight = {0: 1.0, 1: scale_pos_weight}
    metrics_rf, _ = train_and_measure_model(
        RandomForestClassifier,
        {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'class_weight': class_weight,
            'random_state': 42,
            'n_jobs': -1
        },
        'RandomForest',
        X_train, X_test, y_train, y_test, None
    )
    all_metrics.append(metrics_rf)
    
    # 4. CatBoost
    if CATBOOST_AVAILABLE:
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        metrics_cat, _ = train_and_measure_model(
            cb.CatBoostClassifier,
            {
                'iterations': 600,
                'depth': 6,
                'learning_rate': 0.03,
                'l2_leaf_reg': 3,
                'random_seed': 42,
                'loss_function': 'Logloss',
                'eval_metric': 'AUC',
                'early_stopping_rounds': 80,
                'verbose': False,
                'auto_class_weights': 'Balanced'
            },
            'CatBoost',
            X_train, X_test, y_train, y_test, None
        )
        all_metrics.append(metrics_cat)
    
    # 5. MLP (最佳模型 - MLP_Ensemble的单个最佳模型)
    scaler_mlp = StandardScaler()
    X_train_scaled_mlp = scaler_mlp.fit_transform(X_train)
    X_test_scaled_mlp = scaler_mlp.transform(X_test)
    
    # 使用MLP_Wide_Weighted的最佳参数（集成模型的基础）
    metrics_mlp, _ = train_and_measure_model(
        MLPClassifier,
        {
            'hidden_layer_sizes': (512, 256, 128),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.0001,
            'learning_rate': 'adaptive',
            'learning_rate_init': 0.002,
            'batch_size': 128,
            'max_iter': 500,
            'early_stopping': True,
            'validation_fraction': 0.1,
            'n_iter_no_change': 15,
            'beta_1': 0.95,
            'beta_2': 0.9999,
            'random_state': 42
        },
        'MLP_Best',
        X_train_scaled_mlp, X_test_scaled_mlp, y_train, y_test, None
    )
    all_metrics.append(metrics_mlp)
    
    # 保存结果
    df_metrics = pd.DataFrame(all_metrics)
    output_path = PROJECT_ROOT / 'charts' / 'model' / 'model_metrics_detailed.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_metrics.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print("\n" + "=" * 80)
    print("Metrics Collection Complete")
    print("指标收集完成")
    print("=" * 80)
    print(f"\n结果已保存至: {output_path}")
    
    return df_metrics


if __name__ == "__main__":
    metrics_df = collect_all_metrics()
    print("\n模型指标汇总:")
    print(metrics_df[['model_name', 'test_auc', 'test_f1', 'training_time_seconds', 'n_parameters']].to_string(index=False))

