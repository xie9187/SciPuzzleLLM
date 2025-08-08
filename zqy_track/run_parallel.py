#!/usr/bin/env python3
"""

使用方法:
python run_parallel.py periodic_table.py --runs 5 --processes 2 --output ./my_output
"""

import os
import sys
import subprocess
import argparse
import time
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_single_instance(script_path, run_id):
    """运行单个实例"""
    start_time = time.time()
    
    
    # 设置环境变量 - 
    env = os.environ.copy()
    script_parent = Path(script_path).parent.resolve()
    current_pythonpath = env.get('PYTHONPATH', '')
    
    # 添加脚本所在目录到PYTHONPATH
    if current_pythonpath:
        env['PYTHONPATH'] = f"{script_parent}{os.pathsep}{current_pythonpath}"
    else:
        env['PYTHONPATH'] = str(script_parent)
    
    # 构建命令 - 使用绝对路径
    script_abs_path = Path(script_path).resolve()
    cmd = [sys.executable, str(script_abs_path)]
    

    
    try:
        logger.info(f"开始运行实例 {run_id}")
        
        # 运行脚本 - 使用脚本所在目录作为工作目录
        process = subprocess.Popen(
            cmd,
            cwd=str(script_parent),  # 使用脚本所在目录作为工作目录
            env=env,
            text=True
        )
        
        # 等待进程完成
        return_code = process.wait()
    
        end_time = time.time()
        duration = end_time - start_time
    
        success = return_code == 0
    
        if success:
            logger.info(f"✅ 实例 {run_id} 运行成功，耗时 {duration:.2f}秒")
        else:
            logger.warning(f"❌ 实例 {run_id} 运行失败，返回码: {return_code}")
    
        return {
            'run_id': run_id,
            'success': success,
            'duration': duration,
            'return_code': return_code
        }
        
    except Exception as e:
        logger.error(f"❌ 实例 {run_id} 运行异常: {e}")
        return {
            'run_id': run_id,
            'success': False,
            'error': str(e)
        }

def main():
    parser = argparse.ArgumentParser(description='并行运行periodic_table.py脚本')
    parser.add_argument('script', help='periodic_table.py脚本的路径')
    parser.add_argument('--runs', '-r', type=int, default=5, help='运行次数 (默认: 5)')
    parser.add_argument('--processes', '-p', type=int, default=2, help='并行进程数 (默认: 2)')
    
    args = parser.parse_args()
    
    # 验证脚本路径
    script_path = Path(args.script)
    if not script_path.exists():
        logger.error(f"❌ 脚本文件不存在: {script_path}")
        sys.exit(1)
    
    
    logger.info(f"🚀 开始并行运行 {args.runs} 个实例，并行数: {args.processes}")

    logger.info(f"📄 脚本路径: {script_path}")
    
    start_time = time.time()
    results = []
    
    with ProcessPoolExecutor(max_workers=args.processes) as executor:
        # 提交所有任务，每隔一秒提交一次
        futures = []
        for run_id in range(1, args.runs + 1):
            future = executor.submit(run_single_instance, str(script_path), run_id)
            futures.append(future)
            if run_id != args.runs:
                time.sleep(1)
        
        # 收集结果
        completed = 0
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1
            logger.info(f"📊 完成进度: {completed}/{args.runs}")
    
    end_time = time.time()
    total_duration = end_time - start_time
    
    # 统计结果
    successful_runs = [r for r in results if r.get('success', False)]
    failed_runs = [r for r in results if not r.get('success', False)]
    

    
    # 打印报告
    logger.info("=" * 60)
    logger.info("📈 运行报告")
    logger.info("=" * 60)
    logger.info(f"总运行次数: {len(results)}")
    logger.info(f"✅ 成功次数: {len(successful_runs)}")
    logger.info(f"❌ 失败次数: {len(failed_runs)}")
    logger.info(f"📊 成功率: {len(successful_runs)/len(results)*100:.1f}%")
    logger.info(f"⏱️  总耗时: {total_duration:.2f}秒")
    logger.info("=" * 60)
    
    if failed_runs:
        logger.warning("❌ 失败的实例:")
        for result in failed_runs:
            logger.warning(f"  实例 {result['run_id']}: {result.get('error', '未知错误')}")

if __name__ == '__main__':
    main() 