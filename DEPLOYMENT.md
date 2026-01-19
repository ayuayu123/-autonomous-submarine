# SubmarineHunting 部署文档

## 环境要求

- **Python**: 3.13
- **操作系统**: Windows/Linux/macOS

## 快速部署

### 1. 准备 Conda 环境

在 conda 的 envs 目录中创建 `py313` 文件夹，将项目中的 `py313.tar.gz` 解压到该目录：

```bash
# 假设 conda 安装在默认位置
# Windows: C:\Users\<用户名>\anaconda3\envs\py313\
# Linux/Mac: ~/anaconda3/envs/py313/ 或 ~/miniconda3/envs/py313/
```

### 2. 启动环境

```bash
conda activate py313
```

### 3. 运行评估

```bash
python eval_model5.py
```

## 参数说明

`eval_model5.py` 默认使用以下参数：
- `--model`: `best_model.zip`
- `--vecnormalize`: `vecnormalizeNew.pkl`

## 控制说明

运行后的控制方式：
- **WASD**: 移动相机
- **鼠标右键拖拽**: 旋转相机
- **R**: 重置当前回合
- **空格**: 暂停/继续
- **ESC**: 退出

---

## 训练模型

### 从头开始训练

```bash
python train.py
```

训练会使用 `train.py` 中 `TrainingConfig` 类定义的默认参数。

### 恢复训练

从已有的检查点恢复训练并自定义超参数：

```bash
python train.py \
    --resume_model ./training_output/xxx/checkpoints/submarine_ppo_500000_steps.zip \
    --resume_normalize ./training_output/xxx/checkpoints/submarine_ppo_500000_steps_normalize.pkl \
    --resume_stage 5 \
    --learning_rate 1e-5 \
    --n_epochs 20 \
    --total_timesteps 5000000
```

### 监控训练进度

```bash
tensorboard --logdir ./training_output
```

### 训练参数说明

所有配置参数在 `train.py` 的 `TrainingConfig` 类中定义。修改参数需要编辑该类或使用命令行参数（恢复训练时）。

#### 基础训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| total_timesteps | 9,000,000 | 总训练步数 |
| n_envs | 15 | 并行环境数 |
| seed | 42 | 随机种子 |
| device | "cuda" | 训练设备 ("cuda" 或 "cpu") |

#### PPO 算法参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| learning_rate | 3e-5 | 学习率 |
| n_steps | 2048 | 每次更新收集的步数 |
| batch_size | 512 | 批次大小 |
| n_epochs | 10 | PPO 优化轮数 |
| gamma | 0.99 | 折扣因子 |
| gae_lambda | 0.99 | GAE λ 参数 |
| clip_range | 0.2 | PPO 裁剪范围 |
| ent_coef | 0.01 | 熵系数（探索） |

#### 神经网络架构参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| point_cloud_history_len | 16 | 点云历史序列长度 |
| lstm_hidden_size | 128 | LSTM 隐藏层维度 |
| lstm_num_layers | 1 | LSTM 层数 |
| bidirectional | False | 是否使用双向 LSTM |
| pc_encoder_dims | "256" | 点云编码器维度（逗号分隔） |
| state_feature_dim | 64 | 状态特征维度 |
| dropout | 0.1 | Dropout 概率 |
| use_layer_norm | True | 是否使用 Layer Normalization |
| pi_hidden_1 | 256 | Actor 网络第一层 |
| pi_hidden_2 | 128 | Actor 网络第二层 |
| vf_hidden_1 | 256 | Critic 网络第一层 |
| vf_hidden_2 | 128 | Critic 网络第二层 |

#### 环境参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| num_obstacles | 20 | 障碍物数量（阶段5） |

#### 保存与评估

| 参数 | 默认值 | 说明 |
|------|--------|------|
| output_dir | "./training_output" | 输出目录 |
| save_freq | 50,000 | 保存检查点频率（步） |
| eval_freq | 20,000 | 评估频率（步） |

### 恢复训练专用参数

| 参数 | 说明 |
|------|------|
| --resume_model | 模型路径 (.zip) |
| --resume_normalize | VecNormalize 路径 (.pkl) |
| --resume_stage | 恢复的课程阶段 (1-5) |
| --learning_rate | 覆盖学习率 |
| --n_epochs | 覆盖 PPO epoch 数 |
| --total_timesteps | 覆盖总训练步数 |

### 课程学习阶段

训练使用 5 阶段课程学习：

| 阶段 | 障碍物数量 | 说明 |
|------|-----------|------|
| 1 | 3 | 入门阶段 |
| 2 | 5 | 简单 |
| 3 | 7 | 中等 |
| 4 | 9 | 困难 |
| 5 | 15 | 专家（含渐进式训练） |