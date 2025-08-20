import os
import pandas as pd
from os.path import join
from data_utils import History
import copy
from typing import Any, Dict, Tuple, Optional
def get_folders(path):
    return [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]

def get_files(path):
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

def extract_match_rate(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        last_line = file.readlines()[-3]  # 获取最后一行
        # 提取数值
        match_rate = float(last_line.split(':')[-1].strip())
        return match_rate

def evaluate_iterations(exp_path):
    """
    分析每个日志文件夹的每轮iteration信息
    返回一个表格，横坐标是轮次，纵坐标是match rate和是否成功
    """
    result_folders = get_folders(exp_path)
    all_results = []
    
    for folder in result_folders:
        folder_path = join(exp_path, folder)
        log_file = join(folder_path, 'log.txt')
        
        # 检查是否有log.txt文件
        if not os.path.exists(log_file):
            print(f"Log file not found in {folder}")
            continue
            
        # 使用History类来加载记录
        history = History()
        records = history.load_records_from_log(folder_path)
        
        if not records:
            print(f"No records found in {folder}")
            continue
            
        # 分析每轮iteration
        for i, record in enumerate(records):
            match_rate = record.get('match_rate', 0.0)
            
            # 判断该轮是否成功（有match rate且大于0）
            success = 1 if match_rate > 0 else 0
            
            all_results.append({
                'folder': folder,
                'iteration': i + 1,
                'match_rate': match_rate if match_rate > 0 else 'NA',
                'success': success
            })
    
    return all_results

def create_iteration_table(results):
    """
    创建iteration表格，横坐标是轮次，纵坐标是match rate和是否成功
    """
    if not results:
        return pd.DataFrame()
    
    # 获取所有唯一的轮次
    all_iterations = sorted(set([r['iteration'] for r in results]))
    
    # 创建表格数据
    table_data = {}
    
    for iteration in all_iterations:
        iteration_results = [r for r in results if r['iteration'] == iteration]
        
        # 计算该轮次的统计信息
        match_rates = [r['match_rate'] for r in iteration_results if r['match_rate'] != 'NA']
        success_count = sum([r['success'] for r in iteration_results])
        total_count = len(iteration_results)
        
        # 计算平均match rate（排除NA值）
        avg_match_rate = sum(match_rates) / len(match_rates) if match_rates else 'NA'
        
        # 计算成功率
        success_rate = success_count / total_count if total_count > 0 else 0
        
        table_data[f'Iteration_{iteration}'] = {
            'avg_match_rate': avg_match_rate,
            'success_rate': success_rate,
            'total_runs': total_count,
            'successful_runs': success_count
        }
    
    return pd.DataFrame(table_data).T

def evaluate(exp_path):
    """
    主评估函数
    """
    print(f"Analyzing experiments in: {exp_path}")
    
    # 获取所有iteration结果
    results = evaluate_iterations(exp_path)
    
    if not results:
        print("No results found!")
        return
    
    # 创建iteration表格
    iteration_table = create_iteration_table(results)
    
    # 保存结果
    iteration_table.to_csv(join(exp_path, 'iteration_results.csv'))
    
    # 也保存详细结果
    detailed_df = pd.DataFrame(results)
    detailed_df.to_csv(join(exp_path, 'detailed_results.csv'), index=False)
    
    print(f"Results saved to:")
    print(f"  - {join(exp_path, 'iteration_results.csv')}")
    print(f"  - {join(exp_path, 'detailed_results.csv')}")
    
    # 打印表格
    print("\nIteration Results Table:")
    print(iteration_table)
    
    return iteration_table

class EarlyStopper:
    def __init__(self, patience: int = 3, min_delta: float = 0.0, min_iters: int = 0):
        self.patience = patience
        self.min_delta = min_delta
        self.min_iters = min_iters
        self.counter = 0
        self.best_score = -float("inf")
        self.best_payload = None  # 保存最优的完整状态

    def improved(self, score: float) -> bool:
        return score > self.best_score + self.min_delta

    def update(self, iter_idx: int, score: float, payload: Tuple[Any, Any, Any, Any]) -> bool:
        """
        返回值：是否应继续训练（True=继续, False=早停）
        """
        if self.improved(score):
            self.best_score = score
            # 深拷贝以确保后续迭代不破坏最优状态
            self.best_payload = copy.deepcopy(payload)
            self.counter = 0
            return True
        else:
            self.counter += 1
            if iter_idx + 1 <= self.min_iters:
                return True
            return self.counter < self.patience
        
if __name__ == '__main__':
    data_path = 'D:/data/SciPuzzleLLM'
    exp_path = join(data_path, 'logs', '100x2')
    evaluate(exp_path)
	