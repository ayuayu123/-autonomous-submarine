# 强化学习环境配置文档

## 1. 环境概述

### 1.1 环境名称
- **类名**: `SubmarineEnv`
- **接口**: Gymnasium (OpenAI Gym 的继任者)
- **任务**: 潜艇避障与导航
- **文件位置**: [src/rl/submarine_env.py](src/rl/submarine_env.py)

### 1.2 任务描述
潜艇需要在圆柱形通道中导航，从起点出发，避开随机分布的障碍物和通道边界，最终到达终点。环境基于物理仿真，使用 PPO 算法进行训练。

### 1.3 环境架构
- **容器**: `ProvingGround` (原 `CylinderTunnel`) - 管理障碍物和碰撞检测，支持多种形状的障碍物
- **观测**: 混合 LSTM 模式 - 点云序列 + 状态向量
- **动作**: 3 维连续动作 [pitch, yaw, thrust]
- **奖励**: 进度主导设计，增加每步惩罚和推演碰撞惩罚
- **归一化**: VecNormalize - 自动对观测和奖励进行归一化
- **课程学习**: 自动渐进式训练，从简单到复杂（5个阶段：3→5→7→9→15个障碍物）
- **渐进式训练**: Stage 5 采用固定布局→渐进扰动→完全随机的训练策略

---

## 2. 环境配置参数

### 2.1 通道配置 (TunnelConfig)
```python
TunnelConfig(
    radius=5.0,                 # 通道半径 (米)
    length=50.0,                # 通道长度 (米)
    start_x=0.0,                # 通道起点 X 坐标
    center_y=0.0,               # 通道中心 Y 坐标
    center_z=100.0,             # 通道中心 Z 坐标 (深度)
    num_obstacles=15,           # 障碍物数量 (训练初始值)
    obstacle_radius_min=0.7,    # 障碍物最小半径 (米)
    obstacle_radius_max=1.0,    # 障碍物最大半径 (米)
    obstacle_start_x=6.0,       # 障碍物起始 X 坐标 (给潜艇留出起步空间)
    submarine_radius=0.5,       # 潜艇碰撞半径
    safe_spawn_radius=5.0,      # 潜艇生成安全半径

    # 渐进式训练模式配置 (用于课程5)
    progressive_mode=False,     # 是否启用渐进式训练
    progressive_stage=0,        # 渐进式阶段: 0=固定, 1-10=渐进调整, 11+=完全随机
    fixed_obstacle_seed=None,   # 固定障碍物布局的种子
    perturbation_amount=0.5,    # 障碍物位置扰动量 (米)，每阶段递增
)
```

### 2.2 点云配置 (PointCloudConfig)
```python
PointCloudConfig(
    num_points=256,             # 点云采样点数
    obstacle_points_ratio=0.7,  # 障碍物采样点占比 (70% 障碍物, 30% 隧道壁)
    normalize_range=50.0        # 归一化范围 (米)
)
```

### 2.3 环境参数
```python
max_steps=1200                 # 最大步数
dt=0.05                        # 物理仿真时间步 (50ms)
point_cloud_history_len=16     # 点云历史长度 (默认，可配置)
max_angle=20 * pi/180          # 最大舵角 (20度)

# 奖励函数参数
w_progress = 7.0               # 进度奖励权重 (每前进1%给7分)
r_goal = 1000.0                # 到达终点奖励
r_collision = -550.0           # 碰撞惩罚
r_future_collision = -4.5      # 推演碰撞惩罚 (每步)
r_timeout = -550.0             # 超时惩罚
r_step = -0.5                  # 每步惩罚
prediction_time = 3.0          # 推演时间 (秒)
```

---

## 3. 观测空间

### 3.1 观测空间类型
**类型**: `Dict` (混合观测空间)

### 3.2 观测空间结构
```python
observation_space = {
    "state": spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(13,),
        dtype=np.float32
    ),
    "point_cloud_seq": spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(16, 768),  # (history_len, num_points * 3)
        dtype=np.float32
    )
}
```

### 3.3 状态向量 (13维)
状态向量包含以下信息（已归一化）：

| 维度 | 名称 | 范围 | 说明 |
|------|------|------|------|
| 0 | rel_y | [-1, 1] | 相对于通道中心的 Y 偏移 |
| 1 | rel_z | [-1, 1] | 相对于通道中心的 Z 偏移 |
| 2-7 | nu_normalized | 各异 | 归一化速度向量 [u, v, w, p, q, r] |
| 8-10 | orientation | 各异 | 姿态角 [roll, pitch, yaw] (弧度) |
| 11 | dist_normalized | [0, 1] | 到边界的归一化距离 |
| 12 | progress | [0, 1] | 通道进度 (0-1) |

#### 速度归一化因子
```python
nu_normalized = nu / [5.0, 2.0, 2.0, 1.0, 1.0, 1.0]
```

### 3.4 点云序列 (768维/帧，共16帧)
- **形状**: `(16, 768)` = `(history_len, num_points * 3)`
- **数据格式**: 扁平化的 XYZ 坐标
- **归一化**: 除以 `normalize_range=50.0`
- **坐标系**: 机体坐标系 (相对于潜艇)
- **历史窗口**: 16帧点云序列，使用滑动窗口更新

#### 点云采样策略
1. **障碍物采样** (70%的点):
   - 按障碍物表面积比例分配采样点
   - 在球体表面均匀采样
   - 世界坐标系采样后转换到机体坐标系

2. **隧道壁采样** (30%的点):
   - 在潜艇前后 30% 隧道长度范围内采样
   - 在圆柱面上均匀采样
   - 围绕潜艇当前 X 位置采样

3. **坐标转换**:
   - 世界坐标系 → 相对世界坐标 (减去潜艇位置)
   - 相对世界坐标 → 机体坐标系 (乘以旋转矩阵的逆)
   - 归一化到 [-1, 1] 范围 (除以 normalize_range=50.0)

---

## 4. 动作空间

### 4.1 动作空间类型
**类型**: `Box` (连续动作空间)

### 4.2 动作空间定义
```python
action_space = spaces.Box(
    low=np.array([-1.0, -1.0, -1.0]),
    high=np.array([1.0, 1.0, 1.0]),
    dtype=np.float32
)
```

### 4.3 动作含义
动作向量包含 3 个控制量：

| 索引 | 名称 | 范围 | 映射到物理量 | 说明 |
|------|------|------|--------------|------|
| 0 | pitch_cmd | [-1, 1] | [-20°, 20°] | 艉舵角 (俯仰控制) |
| 1 | yaw_cmd | [-1, 1] | [-20°, 20°] | 方向舵角 (偏航控制) |
| 2 | thrust_cmd | [-1, 1] | [0, 1525] RPM | 推进器转速 |

**注意**：推力命令通过 `(thrust_cmd + 1) / 2` 映射到 [0, 1]，然后再乘以 1525 RPM。这样 0 在输出范围的中间，便于网络学习。

### 4.4 动作到控制输入的映射
```python
u_control[0] = -yaw_cmd                      # 上方方向舵
u_control[1] = yaw_cmd                       # 下方方向舵
u_control[2] = -pitch_cmd                    # 右舷艉舵
u_control[3] = pitch_cmd                     # 左舷艉舵
u_control[4] = ((thrust_cmd + 1) / 2) * 1525  # 推进器转速 (映射到 [0, 1525])
```

---

## 5. 奖励函数 (进度主导设计)

### 5.1 强化学习目标

**核心目标**：训练一个智能体，使其能够在圆柱形水下隧道中，从起点出发，**避开分布随机的障碍物和隧道边界**，**安全、高效地到达终点**。

### 任务分解
1. **导航任务**：在长 50 米、半径 5 米的圆柱形隧道中前进
2. **避障任务**：躲避随机生成的球体障碍物（3-15个，取决于课程阶段）
3. **安全约束**：不与隧道边界碰撞
4. **时间约束**：在 1200 步（60 秒仿真时间）内完成

### 5.2 奖励函数实现

```python
def _compute_reward(self, collision: bool, collision_reason: str, goal_reached: bool) -> float:
    reward = 0.0

    # 1. 超时惩罚（优先最高级）
    if self.steps >= self.max_steps:
        return self.r_timeout  # -550.0

    # 2. 碰撞惩罚
    if collision:
        return self.r_collision  # -550.0

    # 3. 到达终点奖励
    if goal_reached:
        return self.r_goal  # +1000.0

    progress = self.tunnel.get_progress(self.eta[0:3])

    # 4. 进度奖励（核心）
    if progress > self.max_progress:
        progress_delta = progress - self.max_progress
        reward += self.w_progress * progress_delta * 100.0
        self.max_progress = progress

    # 5. 推演碰撞惩罚（每步）
    if self._predict_collision():
        reward += self.r_future_collision  # -4.5

    # 6. 每步惩罚
    reward += self.r_step  # -0.5

    return reward
```

### 5.3 奖励参数配置
```python
# 奖励权重配置
w_progress = 7.0          # 进度奖励权重 (每前进1%给7分)
r_goal = 1000.0          # 到达终点奖励
r_collision = -550.0      # 碰撞惩罚 (撞壁或撞障碍物)
r_future_collision = -4.5 # 推演碰撞惩罚 (每步)
r_timeout = -550.0       # 超时惩罚
r_step = -0.5            # 每步惩罚 (鼓励快速完成)
prediction_time = 3.0     # 推演时间 (秒)
```

### 5.4 奖励总结
| 奖励项 | 值范围 | 目的 | 优先级 |
|--------|--------|------|--------|
| 到达终点 | +1000 | 鼓励完成目标 | 最高 |
| 碰撞惩罚 | -550 | 避免碰撞（严厉惩罚） | 最高 |
| 超时惩罚 | -550 | 避免拖延（比碰撞更严厉） | 最高 |
| 进度奖励 | [0, +700] | 【核心】鼓励推进进度（每1%给7分） | 最高 |
| 推演碰撞惩罚 | [-4.5, 0] | 前瞻性避障（每步） | 中 |
| 每步惩罚 | [-0.5, 0] | 鼓励快速完成任务（每步） | 低 |

### 5.5 设计理念

1. **进度优先**：进度增量奖励（每 1% 进度给 7 分）是主要的正收益来源，鼓励智能体高效完成隧道穿越

2. **严厉惩罚**：碰撞惩罚（-550）和超时惩罚（-550）都很严厉，迫使智能体学会安全避障和快速前进

3. **前瞻性**：推演碰撞惩罚基于未来 3 秒轨迹预测，通过 `attitudeEuler` 模拟未来位置，如果预测会碰撞则每步给予 -4.5 分

4. **防止刷分**：进度奖励基于 `max_progress` 跟踪，只有超过历史最大进度才给奖励，防止通过来回移动刷分

5. **简洁设计**：核心设计简洁，主要信号来自进度奖励，辅以推演碰撞和每步惩罚

### 5.6 推演碰撞检测

```python
def _predict_collision(self) -> bool:
    """
    推演未来是否会碰撞

    基于当前的位置、姿态和速度，推演 3 秒后的轨迹
    检查是否会碰撞到障碍物或墙壁
    """
    pred_eta = self.eta.copy()
    pred_nu = self.nu.copy()

    num_steps = int(self.prediction_time / self.dt)

    for _ in range(num_steps):
        pred_eta = attitudeEuler(pred_eta, pred_nu, self.dt)

        collision, _ = self.tunnel.check_collision(
            pred_eta[0:3],
            self.tunnel_config.submarine_radius
        )

        if collision:
            return True

    return False
```

---

## 6. 终止条件

### 6.1 终止触发
```python
terminated = collision or goal_reached
truncated = steps >= max_steps
```

### 6.2 终止原因
- **collision**: 潜艇与障碍物或通道边界碰撞
  - `collision_reason`: "boundary" 或 "obstacle"
- **goal_reached**: 潜艇到达通道终点 (X >= start_x + length)
- **timeout**: 超过最大步数 (1200步)

### 6.3 障碍物随机化机制

为了**防止训练过拟合**，每次环境 `reset()` 时：

1. **障碍物完全重新随机生成**
2. 潜艇的生成位置也会有随机偏移
3. 如果传入 `seed` 参数，则使用该种子生成可复现的布局

```python
# 每次 reset 都会重新生成障碍物
obs, info = env.reset()        # 随机障碍物布局
obs, info = env.reset(seed=42) # 固定障碍物布局 (可复现)
```

---

## 7. 物理模型

### 7.1 潜艇动力学
- **文件**: [torpedo.py](torpedo.py)
- **模型**: 6 自由度水下航行器
- **状态变量**:
  - `eta`: 位置和姿态 [x, y, z, φ, θ, ψ]
  - `nu`: 线速度和角速度 [u, v, w, p, q, r]
  - `u_actual`: 实际控制输入
  - `u_control`: 控制指令

### 7.2 物理仿真
```python
nu, u_actual = vehicle.dynamics(eta, nu, u_actual, u_control, dt)
eta = attitudeEuler(eta, nu, dt)
```

### 7.3 执行器动力学（渐进变化）

舵角和推进器转速**不是瞬时变化**的，而是通过**一阶惯性系统**渐进逼近目标值：

$$\dot{\delta} = \frac{\delta_{cmd} - \delta_{actual}}{T}$$

| 执行器 | 时间常数 T | 最大值 | 说明 |
|--------|------------|--------|------|
| 舵面 (Fin) | 0.1 秒 | ±15° | 需要约 0.3-0.5 秒达到目标值 |
| 推进器 (Thruster) | 0.1 秒 | 1525 RPM | 需要约 0.3-0.5 秒达到目标值 |

**响应特性** (阶跃响应):
| 时间 | 达到目标值的百分比 |
|------|--------------------|
| 0.1s (1T) | 63.2% |
| 0.2s (2T) | 86.5% |
| 0.3s (3T) | 95.0% |
| 0.5s (5T) | 99.3% |

> **训练影响**: 智能体需要学会"预判"，因为动作不会立即生效。这更接近真实物理系统。

### 7.4 控制输入映射

RL 智能体的动作需要映射到物理控制输入：

```python
# 动作空间: [pitch_cmd, yaw_cmd, thrust_cmd] ∈ [-1, 1]^3

# 1. 舵角控制 (pitch_cmd, yaw_cmd ∈ [-1, 1])
pitch_cmd = action[0] * max_angle  # max_angle = 20°
yaw_cmd = action[1] * max_angle    # max_angle = 20°

# 2. 推力控制 (thrust_cmd ∈ [-1, 1])
raw_thrust = action[2]
thrust_ratio = (raw_thrust + 1.0) / 2.0  # 映射到 [0, 1]
thrust_cmd = thrust_ratio * 1525  # RPM

# 3. 映射到 5 个执行器
u_control[0] = -yaw_cmd         # 上方方向舵
u_control[1] = yaw_cmd          # 下方方向舵
u_control[2] = -pitch_cmd      # 右舷艉舵
u_control[3] = pitch_cmd       # 左舷艉舵
u_control[4] = thrust_cmd      # 推进器转速
```

> **注意**: 推力命令通过 `(thrust_cmd + 1) / 2` 映射到 [0, 1]，这样 0 就在输出范围的中间，便于网络学习。

---

## 8. 神经网络架构

### 8.1 特征提取器
**类名**: `PointCloudLSTMExtractor` (优化版)
**文件**: [src/rl/custom_extractor.py](src/rl/custom_extractor.py)

### 8.2 网络结构

#### 点云编码器
```
输入: (batch, 16, 768)
↓
展开: (batch*16, 768)
↓
Linear(768, 256)
↓
LayerNorm
↓
ReLU
↓
恢复形状: (batch, 16, 256)
↓
LSTM(256, hidden_size=128)
↓
LayerNorm
→ 点云特征: (batch, 128)
```

#### 状态编码器
```
输入: (batch, 13)
↓
Linear(13, 128)
↓
LayerNorm
↓
ReLU
↓
Dropout(0.1)
↓
Linear(128, 64)
↓
LayerNorm
↓
ReLU
→ 状态特征: (batch, 64)
```

#### 特征融合
```
点云特征: (batch, 128)
状态特征: (batch, 64)
↓
Concat: (batch, 192)
→ 最终特征: (batch, 192)
```

### 8.3 PPO 策略网络

#### Actor 网络
```
输入特征: (batch, 192)
↓
Linear(192, 256)
↓
Linear(256, 128)
↓
Linear(128, 3)  # 3维动作
→ 动作均值和标准差
```

#### Critic 网络
```
输入特征: (batch, 192)
↓
Linear(192, 256)
↓
Linear(256, 128)
↓
Linear(128, 1)
→ 状态值估计
```

### 8.4 正则化技术
- **Dropout**: 0.1 (防止过拟合)
- **Layer Normalization**: 稳定训练
- **正交初始化**: LSTM 权重正交初始化，改善梯度流
- **遗忘门偏置**: 初始化为 1.0 (帮助长期依赖学习)

---

## 9. 训练配置

### 9.1 算法
- **算法**: PPO (Proximal Policy Optimization)
- **策略**: MultiInputPolicy (支持 Dict 观测空间)
- **特征提取器**: PointCloudLSTMExtractor (混合 LSTM 架构优化版)
- **并行环境**: SubprocVecEnv (多进程)
- **归一化**: VecNormalize (自动归一化观测和奖励)
- **课程学习**: CurriculumCallback (自动渐进式训练)

### 9.2 课程学习 (Curriculum Learning)

#### 9.2.1 课程阶段配置

课程学习采用从简单到复杂的渐进式训练策略，定义在 `train.py` 中：

```python
CURRICULUM_STAGES = {
    1: CurriculumConfig(stage=1, num_obstacles=3, w_velocity=0.2),
    2: CurriculumConfig(stage=2, num_obstacles=5, w_velocity=0.2),
    3: CurriculumConfig(stage=3, num_obstacles=7, w_velocity=0.2),
    4: CurriculumConfig(stage=4, num_obstacles=9, w_velocity=0.2),
    5: CurriculumConfig(stage=5, num_obstacles=15, w_velocity=0.2),
}
```

#### 9.2.2 课程阶段详解

| 阶段 | 障碍物数量 | 成功阈值 | 训练目标 |
|------|-----------|-------------|---------|
| **Stage 1** | 3 | 80% | 学会快速前进，掌握基本避障能力 |
| **Stage 2** | 5 | 80% | 增加障碍物，保持策略稳定性 |
| **Stage 3** | 7 | 80% | 进一步增加难度，提升避障精度 |
| **Stage 4** | 9 | 80% | 接近最终难度，优化策略 |
| **Stage 5** | 15 | 80% | 最终复杂环境，达到最优策略 |

#### 9.2.3 Stage 5 渐进式训练 (Progressive Training)

Stage 5 采用独特的渐进式训练策略，从固定布局逐步过渡到完全随机：

| 渐进阶段 | 模式 | 说明 |
|---------|------|------|
| **Stage 0** | 固定布局 | 使用固定种子 (seed=42) 生成障碍物，成功率≥70%后进入下一阶段 |
| **Stage 1-10** | 渐进扰动 | 在固定布局基础上添加随机扰动，扰动量从 ±0.5m 递增到 ±5.0m |
| **Stage 11+** | 完全随机 | 障碍物位置完全随机生成 |

**切换机制**:
- 每次评估（10个回合）后检查最近 10 次的成功率
- 如果成功率≥70%，自动进入下一渐进阶段
- 每个阶段增加 0.5m 的扰动量

**实现细节**:
- 使用 `Stage5ProgressiveManager` 管理渐进式训练进度
- 课程状态通过 `multiprocessing.Manager().dict()` 在多进程间共享
- `tunnel_config.progressive_mode` 和 `tunnel.config.progressive_stage` 同步更新确保配置一致性
- 使用 `deepcopy()` 为每个环境创建独立的配置副本，避免共享引用问题

#### 9.2.4 自动切换机制

`CurriculumCallback` 负责监控训练进度并自动切换课程：

- **评估频率**: 每 20000 步评估一次
- **评估回合数**: 10 个回合
- **评估指标**: 成功率（到达终点的回合数 / 总回合数）
- **切换条件**: 成功率达到该阶段的阈值
- **最小训练步数**: 50000 步（避免初期随机运气影响）

```python
# 课程切换逻辑示例
if success_rate >= threshold and num_timesteps >= 50000:
    advance_to_next_stage()
    # 调整障碍物数量
```

#### 9.2.5 自定义课程

可以通过修改 `train.py` 中的以下内容来自定义课程：

```python
# 1. 修改课程阶段配置
CURRICULUM_STAGES = {
    1: CurriculumConfig(stage=1, num_obstacles=2),
    2: CurriculumConfig(stage=2, num_obstacles=6),
    3: CurriculumConfig(stage=3, num_obstacles=12),
}

# 2. 修改成功阈值
curriculum_manager = CurriculumManager(
    eval_episodes=50,        # 增加评估回合数
    success_threshold=0.85   # 提高阈值
)

# 3. 修改评估频率
curriculum_callback = CurriculumCallback(
    eval_freq=10000,  # 降低评估频率，减少开销
    verbose=1
)
```

### 9.3 超参数

#### TrainingConfig 默认参数 (train.py)
```python
# 基础训练参数
total_timesteps = 9_000_000  # 总训练步数
n_envs = 15                  # 并行环境数量
seed = 42                    # 随机种子
output_dir = "./training_output"  # 输出目录

# PPO 超参数
learning_rate = 3e-5         # 学习率
n_steps = 2048               # 每个环境每更新的步数
batch_size = 512             # 小批量大小
n_epochs = 10                # 每次更新的轮数
gamma = 0.99                 # 折扣因子
gae_lambda = 0.99            # GAE lambda 参数
clip_range = 0.2            # PPO 裁剪范围
ent_coef = 0.01              # 熵系数 (探索)

# 环境配置 (初始值，会被课程学习覆盖)
num_obstacles = 20           # 初始障碍物数量

# 神经网络架构参数
point_cloud_history_len = 16 # 点云历史长度
lstm_hidden_size = 128       # LSTM 隐藏层大小
lstm_num_layers = 1          # LSTM 层数
bidirectional = False        # 是否使用双向 LSTM
pc_encoder_dims = "256"      # 点云编码器维度
state_feature_dim = 64       # 状态特征维度
dropout = 0.1                # Dropout 概率
use_layer_norm = True        # 是否使用 Layer Normalization

# PPO 网络架构
pi_hidden_1 = 256            # Actor 第一层
pi_hidden_2 = 128            # Actor 第二层
vf_hidden_1 = 256            # Critic 第一层
vf_hidden_2 = 128            # Critic 第二层

# 设备配置
device = "cuda"              # 训练设备
```

#### VecNormalize 归一化参数
```python
norm_obs = True              # 归一化观测
norm_reward = True           # 归一化奖励 (训练时)
clip_obs = 2                 # 裁剪观测到 [-2, 2]
clip_reward = 2              # 裁剪奖励到 [-2, 2]
```

### 9.4 训练命令示例
```bash
# 完整训练（使用默认配置）
python train.py

# 自定义总训练步数
python train.py --total_timesteps 5000000

# 指定设备
python train.py --device cuda

# 从指定阶段恢复训练
python train.py --resume_model ./training_output/xxx/checkpoints/submarine_ppo_500000_steps.zip \
                --resume_normalize ./training_output/xxx/checkpoints/submarine_ppo_500000_steps_normalize.pkl \
                --resume_stage 5 \
                --learning_rate 1e-5 \
                --n_epochs 20 \
                --batch_size 256
```

---

## 10. 课程学习实现细节

### 10.1 核心类

#### 10.1.1 CurriculumConfig
课程配置数据类，定义每个训练阶段的参数：

```python
class CurriculumConfig:
    """课程配置"""
    def __init__(self, stage: int, num_obstacles: int,
                 w_velocity: float = 1.0):
        self.stage = stage
        self.num_obstacles = num_obstacles
        self.w_velocity = w_velocity  # 保留用于未来扩展
```

#### 10.1.2 Stage5ProgressiveManager
课程5的渐进式训练管理器：

```python
class Stage5ProgressiveManager:
    """课程5的渐进式训练管理器

    训练策略:
    - Stage 0: 固定障碍物位置，成功率≥70%代表学会
    - Stage 1-10: 每次调整障碍物位置，调整量逐渐增大
    - Stage 11+: 完全随机化训练
    """

    def __init__(self, min_eval_episodes: int = 10, success_threshold: float = 0.7):
        self.progressive_stage = 0  # 0=固定, 1-10=渐进, 11+=随机
        self.min_eval_episodes = min_eval_episodes  # 最少评估episode数
        self.success_threshold = success_threshold  # 成功率阈值 (70%)
        self.episode_results = []  # 存储最近的成功/失败结果
        self.max_perturbation_stage = 10  # 最大渐进调整次数
```

#### 10.1.3 CurriculumManager
课程管理器，负责跟踪训练进度并决定是否切换课程：

```python
class CurriculumManager:
    def __init__(self, eval_episodes: int = 10, success_threshold: float = 0.8, start_stage: int = 1):
        self.current_stage = start_stage
        self.eval_episodes = eval_episodes        # 累积多少回合后评估
        self.success_threshold = success_threshold
        self.episode_results = []               # 存储最近的评估结果
        self.stage_thresholds = {               # 各阶段成功阈值
            1: 0.8,
            2: 0.8,
            3: 0.8,
            4: 0.8,
            5: 0.8,
        }

        # 课程5的渐进式训练管理器
        self.stage5_progressive = Stage5ProgressiveManager(consecutive_success_threshold=3)
```

#### 10.1.4 CurriculumCallback
课程切换回调函数，在训练过程中定期评估并切换课程：

```python
class CurriculumCallback(BaseCallback):
    def __init__(self, curriculum_manager: CurriculumManager,
                 eval_env, curriculum_state, eval_freq: int = 20000,
                 verbose: int = 0):
        self.curriculum_manager = curriculum_manager
        self.eval_env = eval_env
        self.curriculum_state = curriculum_state  # 共享状态字典
        self.eval_freq = eval_freq
        self.min_timesteps_before_eval = 50000  # 最小训练步数
```

### 10.2 共享课程状态

使用 `multiprocessing.Manager` 创建的共享字典，在所有并行环境间同步课程状态：

```python
from multiprocessing import Manager

manager = Manager()
curriculum_state = manager.dict({
    'current_stage': 1,
    'num_obstacles': 3,
    'w_velocity': 0.2,
    'progressive_mode': False,
    'progressive_stage': 0,
})
```

### 10.3 环境集成

每个环境在创建时读取共享状态，动态调整配置：

```python
def make_env(rank: int, seed: int = 0,
             curriculum_state=None):
    def _init():
        if curriculum_state is not None:
            current_stage = curriculum_state.get('current_stage', 1)
            num_obstacles = curriculum_state.get('num_obstacles', 3)
            progressive_mode = curriculum_state.get('progressive_mode', False)
            progressive_stage = curriculum_state.get('progressive_stage', 0)

        # 使用 deepcopy 创建独立的 tunnel_config 副本
        # 避免多进程间的共享引用问题
        env_tunnel_config = deepcopy(tunnel_config) if tunnel_config else None
        if env_tunnel_config is not None:
            env_tunnel_config.num_obstacles = num_obstacles
            env_tunnel_config.progressive_mode = progressive_mode
            env_tunnel_config.progressive_stage = progressive_stage
            # 对于课程5，设置固定的障碍物种子
            if progressive_mode and progressive_stage == 0:
                env_tunnel_config.fixed_obstacle_seed = 42

        env = SubmarineEnv(tunnel_config=env_tunnel_config, ...)
        return env
    return _init
```

**多进程配置同步机制**:
- 使用 `deepcopy()` 为每个 worker 进程创建独立的 `tunnel_config` 副本
- 防止多进程间共享同一配置对象导致的竞态条件
- 在 `SubmarineEnv.reset()` 中同步 `tunnel_config` 和 `tunnel.config` 确保一致性
- 添加调试日志追踪 `progressive_stage` 在环境间的同步状态

### 10.4 训练流程

```
初始化课程: Stage 1 (3 obstacles)
    ↓
训练 20000 步
    ↓
评估 10 个回合 → 计算成功率
    ↓
成功率 ≥ 80% 且 训练步数 ≥ 50000?
    ↓ 是
切换到 Stage 2 (5 obstacles)
    ↓
训练 20000 步
    ↓
评估 10 个回合 → 计算成功率
    ↓
成功率 ≥ 80%?
    ↓ 是
切换到 Stage 3 (7 obstacles)
    ↓
...
    ↓
切换到 Stage 5 (15 obstacles) + 启用渐进式训练
    ↓
Stage 5-0: 固定布局 (seed=42)，成功率≥70%后进入下一阶段
    ↓
Stage 5-1: 渐进扰动 ±0.5m，成功率≥70%后进入下一阶段
    ↓
Stage 5-2: 渐进扰动 ±1.0m，成功率≥70%后进入下一阶段
    ↓
...
    ↓
Stage 5-10: 渐进扰动 ±5.0m，成功率≥70%后进入下一阶段
    ↓
Stage 5-11+: 完全随机障碍物位置
    ↓
继续训练直到完成或手动停止
```

---

## 11. 评估配置

### 11.1 评估脚本
**文件**: [eval_sim.py](eval_sim.py)

### 11.2 评估功能
- 加载训练好的模型进行可视化
- 支持手动控制模式
- 支持数据保存（点云、状态、动作、奖励、位置）
- 支持 CSV 格式数据导出
- 实时渲染 3D 环境
- 支持 VecNormalize 统计信息加载（用于正确归一化观测）

### 11.3 评估命令示例
```bash
# 基本评估（加载模型和 VecNormalize 统计）
python eval_sim.py \
    --model ./training_output/submarine_ppo_hybrid_lstm_xxx/best_model/best_model.zip \
    --vecnormalize ./training_output/submarine_ppo_hybrid_lstm_xxx/vecnormalize.pkl

# 手动控制模式
python eval_sim.py --manual

# 不保存数据
python eval_sim.py --model ./best_model2.zip --no-save-data
```

---

## 12. 数据保存格式

### 12.1 保存的数据文件
每个回合保存以下文件到 `eval_data/episode_XXXX/`:

| 文件名 | 形状 | 说明 |
|--------|------|------|
| `point_clouds.npy` | (T, 256, 3) | 点云数据 |
| `states.npy` | (T, 13) | 状态数据 |
| `actions.npy` | (T, 3) | 动作数据 |
| `rewards.npy` | (T,) | 奖励数据 |
| `positions.npy` | (T, 3) | 位置数据 |
| `metadata.json` | - | 元数据 |

### 12.2 CSV 数据格式
成功回合会保存 CSV 格式数据到 `eval_data/csv_data/`:

| 文件名 | 说明 |
|--------|------|
| `trajectory_XXXX.csv` | 轨迹数据（位置、速度、动作、奖励等） |
| `obstacles_XXXX.csv` | 障碍物数据 |
| `point_clouds_XXXX.csv` | 点云数据 |

### 12.3 元数据内容
```json
{
  "episode_id": 0,
  "success": false,
  "total_reward": 123.45,
  "num_steps": 150,
  "point_clouds_shape": [150, 256, 3],
  "states_shape": [150, 13],
  "actions_shape": [150, 3],
  "rewards_shape": [150],
  "positions_shape": [150, 3]
}
```

---

## 13. 可视化

### 13.1 渲染器
**文件**: [src/view/renderer.py](src/view/renderer.py)

### 13.2 渲染特性
- OpenGL 3D 渲染
- 潜艇模型
- 障碍物球体
- 隧道边界
- 点云可视化
- 相机控制 (WASD 移动, 鼠标旋转)

### 13.3 控制键
- `WASD`: 相机移动
- `鼠标右键 + 拖动`: 相机旋转
- `I/K`: 俯仰控制 (手动模式)
- `J/L`: 偏航控制 (手动模式)
- `上/下箭头`: 调整推力 (手动模式)
- `R`: 重置回合
- `空格`: 暂停/继续
- `ESC`: 退出

---

## 14. 关键文件列表

| 文件 | 说明 |
|------|------|
| [src/rl/submarine_env.py](src/rl/submarine_env.py) | 强化学习环境主文件 |
| [src/rl/tunnel.py](src/rl/tunnel.py) | 通道和障碍物生成 (ProvingGround 类) |
| [src/rl/point_cloud_sampler.py](src/rl/point_cloud_sampler.py) | 点云采样器 |
| [src/rl/custom_extractor.py](src/rl/custom_extractor.py) | 自定义特征提取器 (混合 LSTM 架构优化版) |
| [train.py](train.py) | 训练脚本 (支持混合 LSTM 模式) |
| [eval_sim.py](eval_sim.py) | 评估脚本 (支持数据保存) |
| [run_human.py](run_human.py) | 手动控制模式 |
| [pygame_sim.py](pygame_sim.py) | 3D 仿真入口 (独立渲染器) |
| [torpedo.py](torpedo.py) | 潜艇物理模型 |
| [lib/gnc.py](lib/gnc.py) | 导航制导控制 |
| [analyze_rewards.py](analyze_rewards.py) | 奖励分析脚本 |
| [evaluate_new_rewards.py](evaluate_new_rewards.py) | 奖励函数对比评估脚本 |

---

## 15. 参考信息

### 15.1 依赖库
- `stable-baselines3`: 强化学习算法库
- `gymnasium`: 环境接口
- `pygame`: 可视化
- `numpy`: 数值计算
- `torch`: 深度学习框架
- `pandas`: CSV 数据导出 (可选)

### 15.2 相关论文
- PPO: Schulman et al. (2017) "Proximal Policy Optimization Algorithms"
- LSTM: Hochreiter & Schmidhuber (1997) "Long Short-Term Memory"

---

**文档更新时间**: 2026-01-14
**项目路径**: `SubmarineHunting`

### 更新日志
- 2026-01-14: 更新渐进式训练机制（70%成功率阈值替代连续成功）、多进程配置同步（deepcopy独立副本）、调试日志改进
- 2026-01-14: 更新奖励函数参数、课程阶段配置（5阶段）、训练超参数、神经网络架构
- 2026-01-09: 添加课程学习实现细节、更新训练流程和评估配置
- 2026-01-07: 添加执行器动力学说明、障碍物随机化机制、更新默认参数
