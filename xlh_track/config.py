import os
import platform

# 根据操作系统自动选择路径
if platform.system() == 'Windows':
    data_path = r'D:\Data\SciLLM'
else:
    data_path = r'/Data/Linhai/SciLLM'

# 可选：添加路径存在性检查
if not os.path.exists(TCGA_data_path):
    raise FileNotFoundError(f"路径不存在: {TCGA_data_path}")
if not os.path.exists(save_path):
    os.makedirs(save_path, exist_ok=True)