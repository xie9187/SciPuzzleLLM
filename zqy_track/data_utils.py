import os
import sys
from datetime import datetime
from os.path import join
from viz_utils import create_periodic_table_plot

def print_and_enter(content):
    print(content)
    print(' ')

class Logger:
    def __init__(self, log_path):
        # 生成带时间戳的文件名
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.log_folder = join(log_path, timestamp)
        if not os.path.exists(self.log_folder):
            os.makedirs(self.log_folder)

        self.log_file = open(join(self.log_folder, 'log.txt'), 'w', encoding='utf-8')
        
        # 保存原始stdout
        self.original_stdout = sys.stdout
        
        # 设置同时输出到控制台和文件
        sys.stdout = self

    def write(self, message):
        self.original_stdout.write(message)  # 控制台输出
        self.log_file.write(message)         # 文件写入

    def flush(self):
        self.original_stdout.flush()
        self.log_file.flush()

    def close(self):
        sys.stdout = self.original_stdout
        self.log_file.close()

    def new_part(self, part_name):
        syb_len = 40
        line_str = "\n" + "="*syb_len + f" {part_name} " + "="*syb_len
        return line_str

    def log_table_as_csv(self, df):
        df.to_csv(join(self.log_folder, 'table.csv'))

    def log_table_as_img(self, df, iteration=None):
        for attribute in list(df.columns):
            if attribute in ['row', 'col']:
                continue
            key_idx = -1 if attribute == 'KnownAndMatched' else 0
            file_name = f'table_{attribute}.png' if iteration is None else f'table_{iteration}iter_{attribute}.png'
            create_periodic_table_plot(df, attribute, join(self.log_folder, file_name), key_idx=key_idx)

class History(object):
    def __init__(self):
        super(History, self).__init__()
        self.records = []

    def update_records(self, hypothesis, evaluation, match_rate, attribute=None, ascending=None):
        self.records.append({
            'hypothesis': hypothesis, 
            'evaluation': evaluation, 
            'match_rate': match_rate,
            'attribute': attribute,
            'ascending': ascending
        })

    def show_records(self):
        if len(self.records) == 0:
            return 'empty history'
        else:
            hist_str = ''
            for i in range(len(self.records)):
                hypothesis = self.records[i]['hypothesis']
                evaluation = self.records[i]['evaluation']
                match_rate = self.records[i]['match_rate']
                hist_str += f'Iteration #{i+1}\n'
                hist_str += f'Hypothesis:\n{hypothesis}\n\n'
                hist_str += f'Evaluation:\n{evaluation}\n\n'
                hist_str += f'Match Rate: {match_rate}\n\n'
            return hist_str

def execute_function(code_str, func_name, df):
    namespace = {}
    exec(code_str, namespace)
    function = namespace[func_name]
    result = function(df.copy()) 
    return result