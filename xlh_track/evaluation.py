import os
import pandas as pd
from os.path import join

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

def evaluate(exp_path):

	result_folders = get_folders(exp_path)
	results = {'folder': [], 'success_run':[], 'match_rate':[]}
	for folder in result_folders:
		files = get_files(join(exp_path, folder))
		if len(files) > 2:
			success_run = 1
			match_rate = extract_match_rate(join(exp_path, folder, 'log.txt'))
		else:
			success_run = 0
			match_rate = 0.
		results['folder'] .append(folder)
		results['success_run'] .append(success_run)
		results['match_rate'] .append(match_rate)

	df = pd.DataFrame(results)
	df.to_csv(join(exp_path, 'results.csv'), index=False)

if __name__ == '__main__':

	exp_path = r'D:\Data\SciLLM\logs\20250703'
	evaluate(exp_path)
	