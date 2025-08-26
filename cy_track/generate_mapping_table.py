#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成匿名化映射表的脚本
"""

from periodic_table import generate_table, mask_table

def generate_mapping_table():
    """生成匿名化映射表"""
    print("=== 匿名化映射表 ===\n")
    
    # 生成原始数据
    df = generate_table()
    
    # TheGroup映射
    print("TheGroup映射:")
    print("原始值 -> 匿名化值")
    group_values = sorted(df['TheGroup'].unique())
    group_counter = 1
    for group in group_values:
        if group != 'unknow':
            print(f"{group:20} -> Group{group_counter}")
            group_counter += 1
        else:
            print(f"{group:20} -> unknow")
    
    print("\nElementalAffinity映射:")
    print("原始值 -> 匿名化值")
    affinity_values = sorted(df['ElementalAffinity'].unique())
    affinity_counter = 1
    for affinity in affinity_values:
        if affinity != 'unknow':
            print(f"{affinity:20} -> Affinity{affinity_counter}")
            affinity_counter += 1
        else:
            print(f"{affinity:20} -> unknow")
    
    print("\nPropertiesOfOxides映射:")
    print("原始值 -> 匿名化值")
    oxide_values = sorted(df['PropertiesOfOxides'].unique())
    oxide_counter = 1
    for oxide in oxide_values:
        if oxide != 'unknow':
            print(f"{oxide:30} -> Oxide{oxide_counter}")
            oxide_counter += 1
        else:
            print(f"{oxide:30} -> unknow")
    
    print("\n=== 数据统计 ===")
    print(f"总元素数: {len(df)}")
    print(f"TheGroup unknow数量: {(df['TheGroup'] == 'unknow').sum()}")
    print(f"ElementalAffinity unknow数量: {(df['ElementalAffinity'] == 'unknow').sum()}")
    print(f"PropertiesOfOxides unknow数量: {(df['PropertiesOfOxides'] == 'unknow').sum()}")
    
    # 测试匿名化
    print("\n=== 测试匿名化 ===")
    train_df, test_df = mask_table(df)
    print(f"训练集形状: {train_df.shape}")
    print(f"测试集形状: {test_df.shape}")
    print(f"列名: {list(train_df.columns)}")

if __name__ == "__main__":
    generate_mapping_table()
