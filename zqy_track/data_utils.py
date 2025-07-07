import os
import sys
import re
from datetime import datetime
import json
from os.path import join
from viz_utils import create_periodic_table_plot
from table_agents_v2 import RecordAgent

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
        self.records = ['']

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
        
    def load_records_from_log(self, log_path, iteration=-1):
        """
        Load records from a log file
        
        Args:
            log_path: Path to the log directory
            iteration: Specific iteration to load (-1 for all iterations)
        
        Returns:
            List of record dictionaries
        """
        log_file = join(log_path, 'log.txt')
        if not os.path.exists(log_file):
            print(f"Log file not found: {log_file}")
            return []
        
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split content by iteration sections
        iteration_sections = re.split(r'======================================== Iteration (\d+) ========================================', content)
        
        records = []
        for i in range(1, len(iteration_sections), 2):  # Skip empty first section
            iter_num = int(iteration_sections[i])
            iter_content = iteration_sections[i + 1] if i + 1 < len(iteration_sections) else ""
            
            # If specific iteration requested, skip others
            if iteration != -1 and iter_num != iteration:
                continue
            

            hypothesis_match = re.search(r'Hypothesis:\n(.*?)(?=\nCode:|$)', iter_content, re.DOTALL)
            
            hypothesis = hypothesis_match.group(1).strip() if hypothesis_match else ""
            
            # Extract evaluation from Induction Process
            evaluation_match = re.search(r'evaluation:\n(.*?)(?=\n\ndecision:|$)', iter_content, re.DOTALL)
            evaluation = evaluation_match.group(1).strip() if evaluation_match else ""
            
            # Extract match rate from Induction Process
            match_rate = 0.0
            match_rate_match = re.search(r'Match Rate: (\d+\.\d+)', iter_content)
            if match_rate_match:
                match_rate = float(match_rate_match.group(1))
            
            # Extract main attribute and ascending from Abduction Process
            main_attr_match = re.search(r'Main attribute:\n(.*?)(?=\n|$)', iter_content)
            attribute = ""
            ascending = True
            if main_attr_match:
                attr_line = main_attr_match.group(1).strip()
                attr_parts = attr_line.split()
                if len(attr_parts) >= 2:
                    attribute = attr_parts[0]
                    ascending = "ascending=True" in attr_line
            
            record = {
                'hypothesis': hypothesis,
                'evaluation': evaluation,
                'match_rate': match_rate,
                'attribute': attribute,
                'ascending': ascending,
            }
            records.append(record)
        self.records = records
        return records
    
    def select_record(self):
        if len(self.records) < 2:
            return json.dumps(self.records[-1], ensure_ascii=False)
        else:
            max_match_rate = max([record['match_rate'] for record in self.records[1:]])
            max_match_rate_record = [record for record in self.records[1:] if record['match_rate'] == max_match_rate]
            last_record = [self.records[-1]]
            merged_records = max_match_rate_record + last_record
            record_agent = RecordAgent()
            json_response = record_agent.merge_records(merged_records)
            return json_response
            


    

def execute_function(code_str, func_name, df):
    namespace = {}
    exec(code_str, namespace)
    function = namespace[func_name]
    result = function(df.copy()) 
    return result