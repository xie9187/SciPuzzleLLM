import os
import platform

# 根据操作系统自动选择路径
if platform.system() == 'Windows':
    data_path = r'D:\Data\SciLLM'
else:
    data_path = r'/data/linhai/SciLLM'