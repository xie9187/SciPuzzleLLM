FROM nvcr.io/nvidia/pytorch:24.07-py3

# 设置工作目录
WORKDIR /workspace

# 复制依赖文件并安装
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY . .

# 默认命令（可根据实际入口脚本调整）
CMD ["bash"]
