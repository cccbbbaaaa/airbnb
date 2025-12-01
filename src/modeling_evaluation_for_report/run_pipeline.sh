#!/bin/bash

################################################################################
# Complete Modeling Pipeline Runner
# Executes all three steps: Feature Engineering → Model Training → Evaluation
################################################################################

set -e  # Exit on error

echo "================================================================================"
echo "AIRBNB RATING CLASSIFICATION - COMPLETE PIPELINE"
echo "================================================================================"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

echo "Project Root: $PROJECT_ROOT"
echo "Script Directory: $SCRIPT_DIR"
echo ""

# Change to project root
cd "$PROJECT_ROOT"

# Check Python version
echo "Checking Python version..."
python --version
echo ""

# Check if data exists
if [ ! -f "data/cleaned/listings_cleaned.csv" ]; then
    echo "❌ ERROR: data/cleaned/listings_cleaned.csv not found!"
    echo "Please run data cleaning first:"
    echo "  python src/data_clean/clean_merged_data.py"
    exit 1
fi

echo "✅ Cleaned data found"
echo ""

# Step 1: Feature Engineering
echo "================================================================================"
echo "STEP 1/3: FEATURE ENGINEERING"
echo "================================================================================"
echo ""
python src/modeling_evaluation_for_report/1_feature_engineering.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Feature engineering failed!"
    exit 1
fi

echo ""
echo "✅ Feature engineering completed"
echo ""

# Check if train_data.csv was created
if [ ! -f "data/processed/train_data.csv" ]; then
    echo "❌ ERROR: train_data.csv was not created!"
    exit 1
fi

echo "✅ train_data.csv created successfully"
echo ""

# Step 2: Model Training
echo "================================================================================"
echo "STEP 2/3: MODEL TRAINING"
echo "================================================================================"
echo ""
python src/modeling_evaluation_for_report/2_model_training.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Model training failed!"
    exit 1
fi

echo ""
echo "✅ Model training completed"
echo ""

# Step 3: Model Evaluation & Report Charts
echo "================================================================================"
echo "STEP 3/3: MODEL EVALUATION & VISUALIZATION"
echo "================================================================================"
echo ""
python src/modeling_evaluation_for_report/4_model_evaluation.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Model evaluation failed!"
    exit 1
fi

echo ""
echo "✅ Model evaluation completed"
echo ""

echo "生成报告用美化图表 / Generating beautified report charts..."
python src/modeling_evaluation_for_report/5_generate_report_charts.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Report chart generation failed!"
    exit 1
fi

echo ""
echo "✅ Model evaluation completed"
echo ""

# Summary
echo "================================================================================"
echo "PIPELINE COMPLETED SUCCESSFULLY!"
echo "================================================================================"
echo ""
echo "📊 Generated Files:"
echo ""
echo "Data:"
echo "  ✅ data/processed/train_data.csv"
echo ""
echo "Results (CSV):"
echo "  ✅ charts/charts_for_report/modeling/model_comparison_results.csv"
echo "  ✅ charts/charts_for_report/modeling/catboost_iteration_comparison.csv (if tuned)"
echo ""
echo "Visualizations (evaluation + report-ready):"
echo "  ✅ charts/charts_for_report/modeling/roc_curves_comparison.png"
echo "  ✅ charts/charts_for_report/modeling/model_comparison_bars.png"
echo "  ✅ charts/charts_for_report/modeling/train_vs_test_performance.png"
echo "  ✅ charts/charts_for_report/modeling/precision_recall_tradeoff.png"
echo "  ✅ charts/charts_for_report/modeling/confusion_matrix_*.png"
echo "  ✅ charts/charts_for_report/modeling/feature_importance_*.png"
echo "  ✅ charts/charts_for_report/modeling/1_roc_curves.png"
echo "  ✅ charts/charts_for_report/modeling/2_model_comparison.png"
echo "  ✅ charts/charts_for_report/modeling/3_confusion_matrix.png"
echo "  ✅ charts/charts_for_report/modeling/4_feature_importance.png"
echo "  ✅ charts/charts_for_report/modeling/5_precision_recall_tradeoff.png"
echo "  ✅ charts/charts_for_report/modeling/6_training_time_vs_performance.png"
echo "  ✅ charts/charts_for_report/modeling/7_bias_variance_tradeoff.png"
echo "  ✅ charts/charts_for_report/modeling/8_overfitting_analysis.png"
echo "  ✅ charts/charts_for_report/modeling/9_catboost_fitting_graph.png (if CSV exists)"
echo ""
echo "📖 Documentation:"
echo "  📄 src/modeling_evaluation_for_report/建模与评估报告.md"
echo ""
echo "================================================================================"
echo "Next Steps:"
echo "  1. Review MODELING_REPORT.md for comprehensive documentation"
echo "  2. Check charts/model/ for all visualizations"
echo "  3. Use these materials for your CRISP-DM report"
echo "================================================================================"
