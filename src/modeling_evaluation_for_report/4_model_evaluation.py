"""
=====================================================================================
Final Model Evaluation Script for Rating Classification
评分分类模型评估最终版本

This script provides comprehensive evaluation and visualization for the trained
models, including:
- ROC curves comparison
- Confusion matrices
- Feature importance analysis
- Model performance metrics
- Business insights

Author: Data Science Course Project
Date: 2025
=====================================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

# Path setup
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CHARTS_DIR = PROJECT_ROOT / "charts"

# Plotting setup
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


# ==================== VISUALIZATION FUNCTIONS ====================
def plot_roc_curves(roc_data: dict, y_test, charts_dir: Path):
    """
    Plot ROC curves for all models

    Args:
        roc_data: Dictionary with model_name -> (fpr, tpr, auc)
        y_test: True test labels
        charts_dir: Directory to save charts
    """
    plt.figure(figsize=(10, 8))

    # Plot each model's ROC curve
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for idx, (model_name, (fpr, tpr, auc)) in enumerate(roc_data.items()):
        color = colors[idx % len(colors)]
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})',
                linewidth=2.5, color=color)

    # Plot random baseline
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=2, label='Random (AUC = 0.500)')

    plt.xlabel('False Positive Rate', fontsize=13, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=13, fontweight='bold')
    plt.title('ROC Curves Comparison - Rating Classification (5-star vs <5-star)',
             fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc='lower right', fontsize=11, framealpha=0.95)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = charts_dir / 'roc_curves_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ ROC curves saved to: {output_path}")


def plot_confusion_matrix(y_true, y_pred, model_name: str, charts_dir: Path):
    """
    Plot confusion matrix for a specific model

    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model
        charts_dir: Directory to save charts
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                square=True, linewidths=1, linecolor='gray',
                annot_kws={'size': 14, 'weight': 'bold'})

    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold', pad=15)

    # Add labels
    plt.xticks([0.5, 1.5], ['<5-star (0)', '5-star (1)'])
    plt.yticks([0.5, 1.5], ['<5-star (0)', '5-star (1)'], rotation=0)

    plt.tight_layout()
    output_path = charts_dir / f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Confusion matrix saved to: {output_path}")


def plot_model_comparison_bars(results_df: pd.DataFrame, charts_dir: Path):
    """
    Plot bar charts comparing models across different metrics

    Args:
        results_df: DataFrame with model comparison results
        charts_dir: Directory to save charts
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Performance Comparison Across Metrics',
                fontsize=16, fontweight='bold', y=0.995)

    metrics = [
        ('test_auc', 'Test AUC-ROC', axes[0, 0]),
        ('test_f1', 'Test F1-Score', axes[0, 1]),
        ('test_precision', 'Test Precision', axes[1, 0]),
        ('test_recall', 'Test Recall', axes[1, 1])
    ]

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for metric_col, metric_name, ax in metrics:
        sorted_df = results_df.sort_values(metric_col, ascending=True)
        model_colors = [colors[i % len(colors)] for i in range(len(sorted_df))]

        bars = ax.barh(range(len(sorted_df)), sorted_df[metric_col], color=model_colors, alpha=0.8)
        ax.set_yticks(range(len(sorted_df)))
        ax.set_yticklabels(sorted_df['model'], fontsize=11)
        ax.set_xlabel(metric_name, fontsize=12, fontweight='bold')
        ax.set_title(metric_name, fontsize=13, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, axis='x')

        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, sorted_df[metric_col])):
            ax.text(value + 0.01, i, f'{value:.3f}', va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    output_path = charts_dir / 'model_comparison_bars.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Model comparison bars saved to: {output_path}")


def plot_train_vs_test_performance(results_df: pd.DataFrame, charts_dir: Path):
    """
    Plot train vs test performance (overfitting analysis)

    Args:
        results_df: DataFrame with model comparison results
        charts_dir: Directory to save charts
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Train vs Test Performance - Overfitting Analysis',
                fontsize=15, fontweight='bold')

    # AUC comparison
    x_pos = np.arange(len(results_df))
    width = 0.35

    axes[0].bar(x_pos - width/2, results_df['train_auc'], width,
               label='Train AUC', alpha=0.8, color='#2ca02c')
    axes[0].bar(x_pos + width/2, results_df['test_auc'], width,
               label='Test AUC', alpha=0.8, color='#d62728')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(results_df['model'], rotation=45, ha='right', fontsize=10)
    axes[0].set_ylabel('AUC Score', fontsize=12, fontweight='bold')
    axes[0].set_title('AUC: Train vs Test', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3, axis='y')

    # F1 comparison
    axes[1].bar(x_pos - width/2, results_df['train_f1'], width,
               label='Train F1', alpha=0.8, color='#2ca02c')
    axes[1].bar(x_pos + width/2, results_df['test_f1'], width,
               label='Test F1', alpha=0.8, color='#d62728')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(results_df['model'], rotation=45, ha='right', fontsize=10)
    axes[1].set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    axes[1].set_title('F1-Score: Train vs Test', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = charts_dir / 'train_vs_test_performance.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Train vs test performance plot saved to: {output_path}")


def plot_precision_recall_tradeoff(results_df: pd.DataFrame, charts_dir: Path):
    """
    Plot precision vs recall scatter plot

    Args:
        results_df: DataFrame with model comparison results
        charts_dir: Directory to save charts
    """
    plt.figure(figsize=(10, 8))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for idx, row in results_df.iterrows():
        color = colors[idx % len(colors)]
        plt.scatter(row['test_recall'], row['test_precision'],
                   s=300, alpha=0.7, color=color, edgecolors='black', linewidth=1.5)
        plt.annotate(row['model'],
                    (row['test_recall'], row['test_precision']),
                    fontsize=11, fontweight='bold',
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.3))

    plt.xlabel('Recall (Sensitivity)', fontsize=13, fontweight='bold')
    plt.ylabel('Precision', fontsize=13, fontweight='bold')
    plt.title('Precision vs Recall Trade-off', fontsize=14, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = charts_dir / 'precision_recall_tradeoff.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Precision-recall trade-off plot saved to: {output_path}")


def plot_feature_importance(model, feature_names: list, model_name: str, charts_dir: Path, top_n: int = 20):
    """
    Plot feature importance for tree-based models

    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        model_name: Name of the model
        charts_dir: Directory to save charts
        top_n: Number of top features to display
    """
    if not hasattr(model, 'feature_importances_'):
        print(f"⚠️  Model {model_name} does not support feature importance")
        return

    # Get feature importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    plt.figure(figsize=(12, 8))
    plt.barh(range(top_n), importances[indices], color='steelblue', alpha=0.8)
    plt.yticks(range(top_n), [feature_names[i] for i in indices], fontsize=10)
    plt.xlabel('Feature Importance', fontsize=12, fontweight='bold')
    plt.title(f'Top {top_n} Most Important Features - {model_name}',
             fontsize=14, fontweight='bold', pad=15)
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()

    output_path = charts_dir / f'feature_importance_{model_name.lower().replace(" ", "_")}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Feature importance plot saved to: {output_path}")


def generate_classification_report_text(y_true, y_pred, model_name: str, charts_dir: Path):
    """
    Generate and save classification report as text file

    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model
        charts_dir: Directory to save reports
    """
    report = classification_report(y_true, y_pred, target_names=['<5-star', '5-star'])

    output_path = charts_dir / f'classification_report_{model_name.lower().replace(" ", "_")}.txt'
    with open(output_path, 'w') as f:
        f.write(f"Classification Report - {model_name}\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)

    print(f"✅ Classification report saved to: {output_path}")


# ==================== MAIN EVALUATION PIPELINE ====================
def main():
    """
    Main evaluation pipeline: Generate all visualizations and reports
    """
    print("\n" + "=" * 80)
    print("MODEL EVALUATION - GENERATING VISUALIZATIONS")
    print("=" * 80)

    # Import and run model training
    sys.path.insert(0, str(CURRENT_DIR))
    # Import with underscore prefix (2_model_training_final)
    import importlib.util
    spec = importlib.util.spec_from_file_location("model_training_final",
                                                    CURRENT_DIR / "2_model_training_final.py")
    model_training_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_training_module)

    train_models = model_training_module.main
    load_training_data = model_training_module.load_training_data
    prepare_binary_dataset = model_training_module.prepare_binary_dataset

    # Train models and get results
    results_df, roc_data, best_model = train_models()

    # Load data for additional evaluations
    df, charts_dir = load_training_data()
    X_train, X_test, y_train, y_test, feature_names = prepare_binary_dataset(df)

    print("\n" + "=" * 80)
    print("GENERATING EVALUATION VISUALIZATIONS")
    print("=" * 80)

    # 1. ROC Curves
    print("\n[1/6] Generating ROC curves...")
    plot_roc_curves(roc_data, y_test, charts_dir)

    # 2. Model Comparison Bars
    print("[2/6] Generating model comparison bar charts...")
    plot_model_comparison_bars(results_df, charts_dir)

    # 3. Train vs Test Performance
    print("[3/6] Generating train vs test performance analysis...")
    plot_train_vs_test_performance(results_df, charts_dir)

    # 4. Precision-Recall Trade-off
    print("[4/6] Generating precision-recall trade-off plot...")
    plot_precision_recall_tradeoff(results_df, charts_dir)

    # 5. Confusion Matrix (for best model)
    print("[5/6] Generating confusion matrix for best model...")
    if best_model is not None:
        best_model_name = results_df.iloc[0]['model']
        y_pred_best = best_model.predict(X_test)
        plot_confusion_matrix(y_test, y_pred_best, best_model_name, charts_dir)
        generate_classification_report_text(y_test, y_pred_best, best_model_name, charts_dir)

    # 6. Feature Importance (for best model if applicable)
    print("[6/6] Generating feature importance plot...")
    if best_model is not None and hasattr(best_model, 'feature_importances_'):
        best_model_name = results_df.iloc[0]['model']
        plot_feature_importance(best_model, feature_names, best_model_name, charts_dir, top_n=20)

    # Summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Best Model: {results_df.iloc[0]['model']}")
    print(f"  Test AUC: {results_df.iloc[0]['test_auc']:.4f}")
    print(f"  Test F1: {results_df.iloc[0]['test_f1']:.4f}")
    print(f"  Test Precision: {results_df.iloc[0]['test_precision']:.4f}")
    print(f"  Test Recall: {results_df.iloc[0]['test_recall']:.4f}")
    print(f"\nAll visualizations saved to: {charts_dir}")

    print("\n" + "=" * 80)
    print("MODEL EVALUATION COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()
