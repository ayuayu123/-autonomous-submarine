# SubmarineHunting Docker 镜像
# 用于离线部署的完整容器化解决方案

FROM python:3.13-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    # OpenGL 相关库
    libgl1-mesa-glx \
    libglib2.0-0 \
    # 音频支持（pygame）
    libsndfile1 \
    # 其他工具
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY lib/ /app/lib/
COPY src/ /app/src/
COPY torpedo.py /app/
COPY train.py /app/
COPY eval_sim.py /app/
COPY pygame_sim.py /app/
COPY run_human.py /app/
COPY requirements.txt /app/

# 复制依赖包
COPY packages/ /app/packages/

# 安装 Python 依赖
RUN pip install --no-index --find-links=/packages -r requirements.txt

# 创建输出目录
RUN mkdir -p /app/training_output /app/models

# 设置默认命令
CMD ["python", "train.py"]

# 暴露 TensorBoard 端口
EXPOSE 6006