import os
import pandas as pd
from os.path import join
from data_utils import History
import copy
from typing import Any, Dict, Tuple, Optional
from io import StringIO
import numpy as np
import matplotlib.pyplot as plt

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
            eval_score = pd.read_fwf(StringIO(record.get('eval_score')), header=None, names = ['attribute', 'score'])
            
            # 判断该轮是否成功（有match rate且大于0）
            success = 1 if match_rate > 0 else 0
            
            all_results.append({
                'folder': folder,
                'iteration': i + 1,
                'match_rate': match_rate if match_rate > 0 else 'NA',
                'success': success,
                'eval_score': eval_score
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
        eval_scores = [r['eval_score'] for r in iteration_results if isinstance(r['eval_score'], pd.DataFrame)]
        success_count = sum([r['success'] for r in iteration_results])
        total_count = len(iteration_results)
        
        # 计算平均match rate（排除NA值）
        avg_match_rate = np.mean(match_rates)
        std_match_rate = np.std(match_rates)
        max_match_rate = np.max(match_rates)
        avg_attribute_match = pd.concat(eval_scores).groupby('attribute').mean()
        std_attribute_match = pd.concat(eval_scores).groupby('attribute').std()
        # 计算成功率
        success_rate = success_count / total_count if total_count > 0 else 0
        
        table_data[f'Iteration_{iteration}'] = {
            'avg_match_rate': avg_match_rate,
            'std_match_rate': std_match_rate,
            'max_match_rate': max_match_rate,
            'avg_attribute_match': avg_attribute_match,
            'success_rate': success_rate,
            'total_runs': total_count,
            'successful_runs': success_count
        }
    
    return pd.DataFrame(table_data).T


def plot_attr_rounds_heatmap(dfs, round_labels=None):
    """
    dfs: list[pd.DataFrame], each has index as attributes and a numeric column named 'score'
    round_labels: optional list of labels for x-axis (len == len(dfs)); defaults to 1..N
    """
    if not dfs:
        raise ValueError("dfs is empty")
        
    # Ensure every df exposes a 'score' Series with the same index order
    series_list = []
    for i, df in enumerate(dfs):
        if 'score' not in df.columns:
            raise ValueError(f"DataFrame at position {i} has no 'score' column")
        s = df['score'].copy()
        series_list.append(s.rename(i))  # temp name
    
    data = pd.concat(series_list, axis=1)
    # Columns are 0..N-1; rename to round labels
    if round_labels is None:
        round_labels = [f"R{i+1}" for i in range(len(dfs))]
    data.columns = round_labels
    
    # Plot
    fig, ax = plt.subplots(figsize=(max(6, len(dfs)*0.8), max(4, len(data)*0.4)))
    im = ax.imshow(data.values, aspect='auto', cmap='Greys')  # light small, dark large
    ax.set_xticks(np.arange(len(data.columns)))
    ax.set_xticklabels(data.columns, rotation=0)
    ax.set_yticks(np.arange(len(data.index)))
    ax.set_yticklabels(list(data.index))
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Attribute")
    
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("score")
    
    # Grid-like minor lines for readability
    ax.set_xticks(np.arange(-0.5, len(data.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(data.index), 1), minor=True)
    ax.grid(which='minor', linestyle=':', linewidth=0.5)
    ax.tick_params(which='minor', bottom=False, left=False)
    plt.savefig('attribute_heatmap.svg')

    plt.show()
    return data

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
    attribute_heatmap = plot_attr_rounds_heatmap(iteration_table['avg_attribute_match'].tolist())
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
    exp_path = join(data_path, 'logs', '30x10x5attribute')
    evaluate(exp_path)
	