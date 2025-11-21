"""
Extended Rating Classification Model Comparison (Rating = 5 vs < 5)
扩展评分分类模型对比：预测评分是否为 5 分

模型列表：
1. Logistic Regression（线性基线）
2. Linear SVM（线性基线）
3. Random Forest（树模型基线）
4. XGBoost（当前最佳）
5. LightGBM（推荐：速度快、性能好）
6. CatBoost（推荐：自动化程度高）
7. MLP Neural Network（推荐：适合文本嵌入特征）

输出内容：
- 控制台打印每个模型的训练/测试指标
- `charts/model/rating_classifier_extended_comparison.csv` 保存指标对比表
- `charts/model/rating_classifier_extended_comparison.png` 可视化对比图表
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier

# 尝试导入梯度提升库
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("警告: XGBoost 不可用，将跳过 XGBoost 模型")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("警告: LightGBM 不可用，将跳过 LightGBM 模型")

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("警告: CatBoost 不可用，将跳过 CatBoost 模型")

# 路径设置
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "EDA"))
from utils import setup_plotting, get_project_paths

setup_plotting()


def load_training_data() -> pd.DataFrame:
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


def train_logistic_regression(X_train, X_test, y_train, y_test):
    """训练逻辑回归模型"""
    print("\n训练模型: LogisticRegression")
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'model': 'LogisticRegression',
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
    
    print(f"  训练集: Acc={metrics['train_accuracy']:.3f}  Prec={metrics['train_precision']:.3f}  Recall={metrics['train_recall']:.3f}  F1={metrics['train_f1']:.3f}  AUC={metrics['train_auc']:.3f}")
    print(f"  测试集: Acc={metrics['test_accuracy']:.3f}  Prec={metrics['test_precision']:.3f}  Recall={metrics['test_recall']:.3f}  F1={metrics['test_f1']:.3f}  AUC={metrics['test_auc']:.3f}")
    
    return model, metrics, y_test_proba


def train_linear_svm(X_train, X_test, y_train, y_test):
    """训练线性SVM模型"""
    print("\n训练模型: LinearSVM")
    
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LinearSVC(
            class_weight='balanced',
            max_iter=2000,
            random_state=42,
            dual=False
        ))
    ])
    
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # LinearSVC没有predict_proba，使用decision_function
    y_train_scores = model.decision_function(X_train)
    y_test_scores = model.decision_function(X_test)
    
    # 将decision_function转换为概率（使用sigmoid近似）
    from scipy.special import expit
    y_train_proba = expit(y_train_scores)
    y_test_proba = expit(y_test_scores)
    
    metrics = {
        'model': 'LinearSVM',
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
    
    print(f"  训练集: Acc={metrics['train_accuracy']:.3f}  Prec={metrics['train_precision']:.3f}  Recall={metrics['train_recall']:.3f}  F1={metrics['train_f1']:.3f}  AUC={metrics['train_auc']:.3f}")
    print(f"  测试集: Acc={metrics['test_accuracy']:.3f}  Prec={metrics['test_precision']:.3f}  Recall={metrics['test_recall']:.3f}  F1={metrics['test_f1']:.3f}  AUC={metrics['test_auc']:.3f}")
    
    return model, metrics, y_test_proba


def train_random_forest(X_train, X_test, y_train, y_test):
    """训练随机森林模型"""
    print("\n训练模型: RandomForest")
    
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    class_weight = {0: 1.0, 1: scale_pos_weight}
    
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight=class_weight,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'model': 'RandomForest',
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
    
    print(f"  训练集: Acc={metrics['train_accuracy']:.3f}  Prec={metrics['train_precision']:.3f}  Recall={metrics['train_recall']:.3f}  F1={metrics['train_f1']:.3f}  AUC={metrics['train_auc']:.3f}")
    print(f"  测试集: Acc={metrics['test_accuracy']:.3f}  Prec={metrics['test_precision']:.3f}  Recall={metrics['test_recall']:.3f}  F1={metrics['test_f1']:.3f}  AUC={metrics['test_auc']:.3f}")
    
    return model, metrics, y_test_proba


def train_xgboost(X_train, X_test, y_train, y_test):
    """训练XGBoost模型"""
    if not XGBOOST_AVAILABLE:
        return None, None, None
    
    print("\n训练模型: XGBoost")
    
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=600,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=2,
        gamma=0.1,
        reg_alpha=0.2,
        reg_lambda=1.2,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        early_stopping_rounds=80,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False
    )
    
    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'model': 'XGBoost',
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
    
    print(f"  训练集: Acc={metrics['train_accuracy']:.3f}  Prec={metrics['train_precision']:.3f}  Recall={metrics['train_recall']:.3f}  F1={metrics['train_f1']:.3f}  AUC={metrics['train_auc']:.3f}")
    print(f"  测试集: Acc={metrics['test_accuracy']:.3f}  Prec={metrics['test_precision']:.3f}  Recall={metrics['test_recall']:.3f}  F1={metrics['test_f1']:.3f}  AUC={metrics['test_auc']:.3f}")
    
    return model, metrics, y_test_proba


def train_lightgbm(X_train, X_test, y_train, y_test):
    """训练LightGBM模型"""
    if not LIGHTGBM_AVAILABLE:
        return None, None, None
    
    print("\n训练模型: LightGBM")
    
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    model = lgb.LGBMClassifier(
        objective='binary',
        n_estimators=600,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_alpha=0.2,
        reg_lambda=1.2,
        scale_pos_weight=scale_pos_weight,
        metric='binary_logloss',
        early_stopping_rounds=80,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        eval_names=['train', 'valid'],
        callbacks=[lgb.early_stopping(stopping_rounds=80), lgb.log_evaluation(period=0)]
    )
    
    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'model': 'LightGBM',
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
    
    print(f"  训练集: Acc={metrics['train_accuracy']:.3f}  Prec={metrics['train_precision']:.3f}  Recall={metrics['train_recall']:.3f}  F1={metrics['train_f1']:.3f}  AUC={metrics['train_auc']:.3f}")
    print(f"  测试集: Acc={metrics['test_accuracy']:.3f}  Prec={metrics['test_precision']:.3f}  Recall={metrics['test_recall']:.3f}  F1={metrics['test_f1']:.3f}  AUC={metrics['test_auc']:.3f}")
    
    return model, metrics, y_test_proba


def train_catboost(X_train, X_test, y_train, y_test):
    """训练CatBoost模型"""
    if not CATBOOST_AVAILABLE:
        return None, None, None
    
    print("\n训练模型: CatBoost")
    
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
        auto_class_weights='Balanced'  # 自动处理类别不平衡
    )
    
    model.fit(
        X_train, y_train,
        eval_set=(X_test, y_test),
        use_best_model=True
    )
    
    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'model': 'CatBoost',
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
    
    print(f"  训练集: Acc={metrics['train_accuracy']:.3f}  Prec={metrics['train_precision']:.3f}  Recall={metrics['train_recall']:.3f}  F1={metrics['train_f1']:.3f}  AUC={metrics['train_auc']:.3f}")
    print(f"  测试集: Acc={metrics['test_accuracy']:.3f}  Prec={metrics['test_precision']:.3f}  Recall={metrics['test_recall']:.3f}  F1={metrics['test_f1']:.3f}  AUC={metrics['test_auc']:.3f}")
    
    return model, metrics, y_test_proba


def train_mlp(X_train, X_test, y_train, y_test):
    """训练MLP神经网络模型"""
    print("\n训练模型: MLP Neural Network")
    
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    class_weight = {0: 1.0, 1: scale_pos_weight}
    
    # 标准化数据（神经网络需要）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        alpha=0.001,
        batch_size=256,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42
    )
    
    model.fit(X_train_scaled, y_train)
    
    y_train_pred = model.predict(X_train_scaled)
    y_train_proba = model.predict_proba(X_train_scaled)[:, 1]
    y_test_pred = model.predict(X_test_scaled)
    y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    metrics = {
        'model': 'MLP',
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
    
    print(f"  训练集: Acc={metrics['train_accuracy']:.3f}  Prec={metrics['train_precision']:.3f}  Recall={metrics['train_recall']:.3f}  F1={metrics['train_f1']:.3f}  AUC={metrics['train_auc']:.3f}")
    print(f"  测试集: Acc={metrics['test_accuracy']:.3f}  Prec={metrics['test_precision']:.3f}  Recall={metrics['test_recall']:.3f}  F1={metrics['test_f1']:.3f}  AUC={metrics['test_auc']:.3f}")
    
    return model, metrics, y_test_proba, scaler


def visualize_comparison(results_df, roc_data, charts_dir):
    """可视化模型对比结果"""
    print("\n" + "=" * 80)
    print("Generating Visualizations")
    print("生成可视化图表")
    print("=" * 80)
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. ROC曲线对比
    ax1 = fig.add_subplot(gs[0, :])
    for model_name, (fpr, tpr, auc) in roc_data.items():
        ax1.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})', linewidth=2)
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. Test AUC对比（条形图）
    ax2 = fig.add_subplot(gs[1, 0])
    test_auc_sorted = results_df.sort_values('test_auc', ascending=True)
    colors = plt.cm.viridis(np.linspace(0, 1, len(test_auc_sorted)))
    ax2.barh(range(len(test_auc_sorted)), test_auc_sorted['test_auc'], color=colors)
    ax2.set_yticks(range(len(test_auc_sorted)))
    ax2.set_yticklabels(test_auc_sorted['model'])
    ax2.set_xlabel('Test AUC', fontsize=11)
    ax2.set_title('Test AUC Comparison', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # 3. Test F1对比
    ax3 = fig.add_subplot(gs[1, 1])
    test_f1_sorted = results_df.sort_values('test_f1', ascending=True)
    colors = plt.cm.plasma(np.linspace(0, 1, len(test_f1_sorted)))
    ax3.barh(range(len(test_f1_sorted)), test_f1_sorted['test_f1'], color=colors)
    ax3.set_yticks(range(len(test_f1_sorted)))
    ax3.set_yticklabels(test_f1_sorted['model'])
    ax3.set_xlabel('Test F1-Score', fontsize=11)
    ax3.set_title('Test F1-Score Comparison', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    
    # 4. Test Accuracy对比
    ax4 = fig.add_subplot(gs[1, 2])
    test_acc_sorted = results_df.sort_values('test_accuracy', ascending=True)
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(test_acc_sorted)))
    ax4.barh(range(len(test_acc_sorted)), test_acc_sorted['test_accuracy'], color=colors)
    ax4.set_yticks(range(len(test_acc_sorted)))
    ax4.set_yticklabels(test_acc_sorted['model'])
    ax4.set_xlabel('Test Accuracy', fontsize=11)
    ax4.set_title('Test Accuracy Comparison', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')
    
    # 5. 训练集vs测试集AUC对比（过拟合分析）
    ax5 = fig.add_subplot(gs[2, 0])
    x_pos = np.arange(len(results_df))
    width = 0.35
    ax5.bar(x_pos - width/2, results_df['train_auc'], width, label='Train AUC', alpha=0.8)
    ax5.bar(x_pos + width/2, results_df['test_auc'], width, label='Test AUC', alpha=0.8)
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(results_df['model'], rotation=45, ha='right', fontsize=9)
    ax5.set_ylabel('AUC', fontsize=11)
    ax5.set_title('Train vs Test AUC (Overfitting Analysis)', fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Precision vs Recall散点图
    ax6 = fig.add_subplot(gs[2, 1])
    for idx, row in results_df.iterrows():
        ax6.scatter(row['test_recall'], row['test_precision'], s=200, alpha=0.7, label=row['model'])
        ax6.annotate(row['model'], (row['test_recall'], row['test_precision']), 
                    fontsize=9, alpha=0.8)
    ax6.set_xlabel('Recall', fontsize=11)
    ax6.set_ylabel('Precision', fontsize=11)
    ax6.set_title('Precision vs Recall Trade-off', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    # 7. 综合指标雷达图（简化版：AUC, F1, Accuracy）
    ax7 = fig.add_subplot(gs[2, 2], projection='polar')
    metrics_to_plot = ['test_auc', 'test_f1', 'test_accuracy']
    angles = np.linspace(0, 2 * np.pi, len(metrics_to_plot), endpoint=False).tolist()
    angles += angles[:1]  # 闭合
    
    for idx, row in results_df.iterrows():
        values = [row[m] for m in metrics_to_plot]
        values += values[:1]  # 闭合
        ax7.plot(angles, values, 'o-', linewidth=2, label=row['model'], alpha=0.7)
        ax7.fill(angles, values, alpha=0.1)
    
    ax7.set_xticks(angles[:-1])
    ax7.set_xticklabels(['AUC', 'F1', 'Accuracy'], fontsize=10)
    ax7.set_ylim(0, 1)
    ax7.set_title('Comprehensive Metrics (Polar)', fontsize=12, fontweight='bold', pad=20)
    ax7.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
    ax7.grid(True)
    
    plt.suptitle('Extended Model Comparison: Rating Classification (5-star vs <5-star)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    output_path = charts_dir / 'rating_classifier_extended_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ 可视化图表已保存: {output_path}")


def main():
    """主函数"""
    print("=" * 80)
    print("Extended Rating Classification Model Comparison")
    print("扩展评分分类模型对比（Rating = 5 vs < 5）")
    print("=" * 80)
    
    # 加载数据
    df, charts_dir = load_training_data()
    X_train, X_test, y_train, y_test, feature_cols = prepare_binary_dataset(df)
    
    # 训练所有模型
    all_results = []
    roc_data = {}
    
    # 基线模型
    _, metrics_lr, proba_lr = train_logistic_regression(X_train, X_test, y_train, y_test)
    if metrics_lr:
        all_results.append(metrics_lr)
        fpr, tpr, _ = roc_curve(y_test, proba_lr)
        roc_data['LogisticRegression'] = (fpr, tpr, metrics_lr['test_auc'])
    
    _, metrics_svm, proba_svm = train_linear_svm(X_train, X_test, y_train, y_test)
    if metrics_svm:
        all_results.append(metrics_svm)
        fpr, tpr, _ = roc_curve(y_test, proba_svm)
        roc_data['LinearSVM'] = (fpr, tpr, metrics_svm['test_auc'])
    
    _, metrics_rf, proba_rf = train_random_forest(X_train, X_test, y_train, y_test)
    if metrics_rf:
        all_results.append(metrics_rf)
        fpr, tpr, _ = roc_curve(y_test, proba_rf)
        roc_data['RandomForest'] = (fpr, tpr, metrics_rf['test_auc'])
    
    # 梯度提升模型
    _, metrics_xgb, proba_xgb = train_xgboost(X_train, X_test, y_train, y_test)
    if metrics_xgb:
        all_results.append(metrics_xgb)
        fpr, tpr, _ = roc_curve(y_test, proba_xgb)
        roc_data['XGBoost'] = (fpr, tpr, metrics_xgb['test_auc'])
    
    _, metrics_lgb, proba_lgb = train_lightgbm(X_train, X_test, y_train, y_test)
    if metrics_lgb:
        all_results.append(metrics_lgb)
        fpr, tpr, _ = roc_curve(y_test, proba_lgb)
        roc_data['LightGBM'] = (fpr, tpr, metrics_lgb['test_auc'])
    
    _, metrics_cat, proba_cat = train_catboost(X_train, X_test, y_train, y_test)
    if metrics_cat:
        all_results.append(metrics_cat)
        fpr, tpr, _ = roc_curve(y_test, proba_cat)
        roc_data['CatBoost'] = (fpr, tpr, metrics_cat['test_auc'])
    
    # 神经网络
    result_mlp = train_mlp(X_train, X_test, y_train, y_test)
    if result_mlp[1]:
        _, metrics_mlp, proba_mlp, _ = result_mlp
        all_results.append(metrics_mlp)
        fpr, tpr, _ = roc_curve(y_test, proba_mlp)
        roc_data['MLP'] = (fpr, tpr, metrics_mlp['test_auc'])
    
    # 汇总结果
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('test_auc', ascending=False)
    
    print("\n" + "=" * 80)
    print("Model Comparison Summary")
    print("模型对比总结")
    print("=" * 80)
    print(results_df.to_string(index=False))
    
    # 保存结果
    output_csv = charts_dir / 'rating_classifier_extended_comparison.csv'
    results_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\n✅ 指标对比表已保存：{output_csv}")
    
    # 可视化
    visualize_comparison(results_df, roc_data, charts_dir)
    
    print("\n" + "=" * 80)
    print("Extended Model Comparison Complete!")
    print("扩展模型对比完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()

