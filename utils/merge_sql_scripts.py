#!/usr/bin/env python3
"""
脚本用于合并TPC-DS查询SQL文件，按照custom_sort的顺序
"""

import re
import os
import sys
from pathlib import Path
from utils.spark import custom_sort


def merge_sql_files(sql_dir, output_file):
    """
    合并SQL文件到单个文件
    
    Args:
        sql_dir: SQL文件目录路径
        output_file: 输出文件路径
    """
    sql_dir = Path(sql_dir)
    
    if not sql_dir.exists():
        print(f"错误: 目录 {sql_dir} 不存在")
        return False
    
    # 获取所有SQL文件
    sql_files = []
    for file_path in sql_dir.glob("q*.sql"):
        sql_files.append(file_path)
    
    if not sql_files:
        print(f"错误: 在目录 {sql_dir} 中没有找到q*.sql文件")
        return False
    
    # 按照custom_sort顺序排序
    sorted_files = sorted(sql_files, key=lambda x: custom_sort(x.stem))
    
    print(f"找到 {len(sorted_files)} 个SQL文件")
    print("排序后的文件顺序:")
    for i, file_path in enumerate(sorted_files, 1):
        print(f"{i:2d}. {file_path.name}")
    
    # 合并文件
    with open(output_file, 'w', encoding='utf-8') as out_f:
        out_f.write("-- 合并的TPC-DS查询SQL文件\n")
        out_f.write(f"-- 生成时间: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        out_f.write(f"-- 包含 {len(sorted_files)} 个查询\n")
        out_f.write("-- " + "="*60 + "\n\n")
        
        for i, file_path in enumerate(sorted_files, 1):
            out_f.write(f"-- 查询 {i}: {file_path.name}\n")
            out_f.write("-- " + "-"*50 + "\n")
            
            try:
                with open(file_path, 'r', encoding='utf-8') as in_f:
                    content = in_f.read().strip()
                    out_f.write(content)
                    
                    # 添加SQL查询分隔符
                    if not content.endswith(';'):
                        out_f.write(";")
                    out_f.write("\n\n")
                    
            except Exception as e:
                print(f"警告: 读取文件 {file_path} 时出错: {e}")
                out_f.write(f"-- 错误: 无法读取文件 {file_path}\n\n")
    
    print(f"\n合并完成! 输出文件: {output_file}")
    print(f"文件大小: {os.path.getsize(output_file) / 1024:.1f} KB")
    
    return True

def main():
    """主函数"""
    if len(sys.argv) != 3:
        print("用法: python merge_sql_scripts.py <SQL目录> <输出文件>")
        print("示例: python merge_sql_scripts.py /home/hive-testbench-hdp3/spark-queries-tpcds merged_queries.sql")
        sys.exit(1)
    
    sql_dir = sys.argv[1]
    output_file = sys.argv[2]
    
    success = merge_sql_files(sql_dir, output_file)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
