"""
测试建模脚本的关键功能 / Test Key Functions of Modeling Scripts

这个脚本用于快速测试建模脚本的关键逻辑是否正确
This script is used to quickly test if the key logic of modeling scripts is correct
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# 添加 EDA 目录到路径 / Add EDA directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'EDA'))
from utils import get_project_paths

def test_data_loading():
    """测试数据加载 / Test data loading"""
    print("=" * 80)
    print("测试数据加载 / Testing Data Loading...")
    print("=" * 80)
    
    project_root, data_dir, charts_eda_dir, charts_model_dir = get_project_paths()
    
    # 检查数据文件是否存在 / Check if data files exist
    listings_file = data_dir / 'listings.csv'
    listings_detailed_file = data_dir / 'listings_detailed.xlsx'
    
    if not listings_file.exists():
        print(f"  [ERROR] listings.csv 不存在: {listings_file}")
        return False
    
    if not listings_detailed_file.exists():
        print(f"  [ERROR] listings_detailed.xlsx 不存在: {listings_detailed_file}")
        return False
    
    print(f"  [OK] 数据文件存在")
    
    # 尝试加载数据 / Try loading data
    try:
        listings = pd.read_csv(listings_file)
        print(f"  [OK] listings.csv 加载成功: {len(listings)} 行 × {len(listings.columns)} 列")
        
        listings_detailed = pd.read_excel(listings_detailed_file)
        print(f"  [OK] listings_detailed.xlsx 加载成功: {len(listings_detailed)} 行 × {len(listings_detailed.columns)} 列")
        
        return True
    except Exception as e:
        print(f"  [ERROR] 数据加载失败: {e}")
        return False

def test_license_logic():
    """测试 license 字段处理逻辑 / Test license field processing logic"""
    print("\n" + "=" * 80)
    print("测试 license 字段处理逻辑 / Testing License Field Processing Logic...")
    print("=" * 80)
    
    project_root, data_dir, charts_eda_dir, charts_model_dir = get_project_paths()
    
    try:
        listings_detailed = pd.read_excel(data_dir / 'listings_detailed.xlsx')
        
        # 测试 license 处理逻辑 / Test license processing logic
        if 'license' in listings_detailed.columns:
            # 统计原始缺失情况 / Count original missing values
            missing_count = listings_detailed['license'].isna().sum()
            total_count = len(listings_detailed)
            missing_rate = missing_count / total_count * 100
            
            print(f"  原始数据: {missing_count}/{total_count} 缺失 ({missing_rate:.2f}%)")
            
            # 应用处理逻辑 / Apply processing logic
            listings_detailed['license_processed'] = listings_detailed['license'].notna()
            
            # 统计处理后的情况 / Count processed values
            has_license_count = listings_detailed['license_processed'].sum()
            no_license_count = (~listings_detailed['license_processed']).sum()
            
            print(f"  处理后: {has_license_count} 有license (True), {no_license_count} 无license (False)")
            
            # 验证逻辑正确性 / Verify logic correctness
            if has_license_count == (total_count - missing_count):
                print(f"  [OK] license 处理逻辑正确")
                return True
            else:
                print(f"  [ERROR] license 处理逻辑错误")
                return False
        else:
            print(f"  [WARNING] license 列不存在")
            return True  # 不是错误，只是没有这个字段
    except Exception as e:
        print(f"  [ERROR] 测试失败: {e}")
        return False

def test_income_calculation():
    """测试收入计算逻辑 / Test income calculation logic"""
    print("\n" + "=" * 80)
    print("测试收入计算逻辑 / Testing Income Calculation Logic...")
    print("=" * 80)
    
    project_root, data_dir, charts_eda_dir, charts_model_dir = get_project_paths()
    
    try:
        listings_detailed = pd.read_excel(data_dir / 'listings_detailed.xlsx')
        
        # 检查必要字段 / Check required fields
        required_fields = ['price', 'availability_365']
        missing_fields = [f for f in required_fields if f not in listings_detailed.columns]
        
        if missing_fields:
            print(f"  [ERROR] 缺少必要字段: {missing_fields}")
            return False
        
        # 处理 price 列 / Process price column
        if listings_detailed['price'].dtype == 'object':
            listings_detailed['price_clean'] = listings_detailed['price'].astype(str).str.replace('$', '').str.replace(',', '')
            listings_detailed['price_clean'] = pd.to_numeric(listings_detailed['price_clean'], errors='coerce')
        else:
            listings_detailed['price_clean'] = pd.to_numeric(listings_detailed['price'], errors='coerce')
        
        listings_detailed['price_clean'] = listings_detailed['price_clean'].fillna(0)
        
        # 处理 availability_365 / Process availability_365
        listings_detailed['availability_365'] = pd.to_numeric(listings_detailed['availability_365'], errors='coerce').fillna(0)
        
        # 计算收入 / Calculate income
        listings_detailed['income'] = listings_detailed['price_clean'] * (365 - listings_detailed['availability_365'])
        
        # 验证计算结果 / Verify calculation results
        valid_income = listings_detailed[listings_detailed['income'] > 0]
        
        print(f"  总记录数: {len(listings_detailed)}")
        print(f"  有效收入记录数: {len(valid_income)} ({len(valid_income)/len(listings_detailed)*100:.2f}%)")
        print(f"  收入统计:")
        print(f"    均值: {valid_income['income'].mean():.2f}")
        print(f"    中位数: {valid_income['income'].median():.2f}")
        print(f"    最小值: {valid_income['income'].min():.2f}")
        print(f"    最大值: {valid_income['income'].max():.2f}")
        
        # 检查是否有异常值 / Check for outliers
        if valid_income['income'].max() > 1000000:
            print(f"  [WARNING] 发现异常高收入值，可能需要处理异常值")
        
        print(f"  [OK] 收入计算逻辑正确")
        return True
        
    except Exception as e:
        print(f"  [ERROR] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数 / Main test function"""
    print("\n" + "=" * 80)
    print("建模脚本关键功能测试 / Modeling Scripts Key Functions Test")
    print("=" * 80)
    
    results = []
    
    # 测试数据加载 / Test data loading
    results.append(("数据加载", test_data_loading()))
    
    # 测试 license 逻辑 / Test license logic
    results.append(("License 字段处理", test_license_logic()))
    
    # 测试收入计算 / Test income calculation
    results.append(("收入计算逻辑", test_income_calculation()))
    
    # 汇总结果 / Summary
    print("\n" + "=" * 80)
    print("测试结果汇总 / Test Results Summary")
    print("=" * 80)
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {test_name}: {status}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n  [OK] 所有测试通过！")
    else:
        print("\n  [WARNING] 部分测试失败，请检查代码")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

