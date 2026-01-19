# SubmarineHunting: 高保真自主水下航行器 (AUV) 仿真与强化学习平台

**SubmarineHunting** 是一个开源的、模块化的综合仿真平台，专为自主水下航行器（Autonomous Underwater Vehicle, AUV）的研究与开发设计。本项目深度融合了**高保真物理建模**、**实时 3D 可视化**以及**深度强化学习 (Deep RL)** 技术，旨在为研究人员和开发者提供一个从理论验证到智能体训练的全栈解决方案。

本项目不仅复现了经典的 REMUS 100 AUV 物理模型，还提供了一个符合 Gymnasium 标准的强化学习环境，支持在复杂的水下障碍物环境中训练智能避障策略。

**最新特性**：
- **混合 LSTM 架构**：点云序列经 LSTM 处理后与状态融合，具备时序记忆能力
- **课程学习**：5阶段自动渐进式训练，从3个障碍物到15个障碍物
- **Stage 5 渐进式训练**：固定布局→渐进扰动→完全随机的独特训练策略
- **表面点云采样**：直接对障碍物和隧道壁表面进行随机采样，生成精确的 XYZ 坐标点云
- **进度主导奖励**：简洁的奖励设计，核心信号来自进度奖励
- **推演碰撞检测**：基于未来3秒轨迹预测，鼓励智能体提前规避危险

---

## 📚 目录 (Table of Contents)

1. [项目概述与特性](#-项目概述与特性)
2. [快速入门](#-快速入门)
   - [环境依赖安装](#21-环境依赖安装)
   - [运行交互式仿真](#22-运行交互式仿真)
   - [训练强化学习模型](#23-训练强化学习模型)
   - [模型评估与回放](#24-模型评估与回放)
3. [系统架构详解](#-系统架构详解)
   - [文件结构树](#31-文件结构树)
   - [核心设计模式 (Actor-Stage)](#32-核心设计模式-actor-stage)
   - [渲染引擎 (OpenGL)](#33-渲染引擎-opengl)
4. [数学模型与物理引擎](#-数学模型与物理引擎)
   - [6-DOF 运动方程](#41-6-dof-运动方程)
   - [水动力学计算](#42-水动力学计算)
   - [控制算法 (PID & SMC)](#43-控制算法-pid--smc)
5. [强化学习环境 (SubmarineEnv)](#-强化学习环境-submarineenv)
   - [观测模式](#51-观测模式)
   - [动作空间 (Action)](#52-动作空间-action)
   - [奖励机制 (Reward)](#53-奖励机制-reward)
   - [环境配置](#54-环境配置)
6. [训练配置与使用](#-训练配置与使用)
   - [训练模式](#61-训练模式)
   - [课程学习](#62-课程学习)
   - [命令行参数详解](#63-命令行参数详解)
   - [TensorBoard 监控](#64-tensorboard-监控)
7. [高级配置与开发指南](#-高级配置与开发指南)
   - [自定义场景](#71-自定义场景)
   - [添加新实体](#72-添加新实体)
   - [自定义神经网络架构](#73-自定义神经网络架构)
8. [常见问题 (Troubleshooting)](#-常见问题-troubleshooting)
9. [参考文献与许可证](#-参考文献与许可证)

---

## 🌟 项目概述与特性

SubmarineHunting 的核心目标是填补简易 2D 仿真与昂贵商业软件（如 OrcaFlex）之间的空白，提供一个轻量级但物理精确的 Python 仿真环境。

### 核心特性

- **基于物理的 6-DOF 动力学**: 完整实现了 Fossen (2021) 提出的水下航行器运动方程，包含附加质量、科里奥利力、非线性阻力和恢复力。
- **高精度流体计算**: 使用 `Hoerner` 曲线计算交叉流阻力 (Cross-flow Drag)，使用 NACA 翼型数据计算舵面升阻力。
- **混合控制架构**:
  - **底层控制**: 内置 PID 和积分滑模控制 (Integral SMC) 用于姿态稳定。
  - **上层决策**: 支持 PPO (Proximal Policy Optimization) 和多种 LSTM 变体进行路径规划。
- **先进感知系统**:
  - **表面点云采样**: 直接对障碍物和隧道边界表面进行随机采样，生成精确的 XYZ 坐标点云，替代传统的射线投射方式。
  - **混合 LSTM 模式**: 结合点云采样与时序记忆网络 (LSTM)，实现对复杂水下环境的高效感知与状态融合。
- **实时 3D 渲染**: 基于 Pygame 和 PyOpenGL 的轻量级渲染器，支持动态视角切换、HUD 仪表盘和轨迹可视化。

---

## 🚀 快速入门

### 2.1 环境依赖安装

本项目支持 Windows, Linux 和 macOS。建议使用 Anaconda 管理环境。

**第一步：创建虚拟环境**
```bash
conda create -n sub_hunt python=3.10
conda activate sub_hunt
```

**第二步：安装依赖**
```bash
# 方式一：使用 requirements.txt
pip install -r requirements.txt

# 方式二：手动安装
# 1. 基础科学计算库
pip install numpy matplotlib

# 2. 仿真与渲染库
pip install pygame PyOpenGL PyOpenGL_accelerate

# 3. 强化学习库 (基于 Stable-Baselines3)
pip install gymnasium stable-baselines3 tensorboard shimmy

# 4. 深度学习框架 (混合 LSTM 模式需要)
pip install torch  # 建议使用 CUDA 版本加速训练
```

> **Windows 用户提示**: 如果安装 `PyOpenGL` 遇到问题，建议从 [UCI Python Binaries](https://www.lfd.uci.edu/~gohlke/pythonlibs/) 下载对应版本的 `.whl` 文件手动安装。

### 2.2 运行交互式仿真

本项目提供两种手动驾驶模式，供您理解 AUV 运动特性。

#### 方式一：独立 3D 仿真器 (pygame_sim.py)

启动 `pygame_sim.py` 即可进入独立的手动驾驶模式。

```bash
python pygame_sim.py
```

**键盘操作指南**:
- **W / S**: 调节艉舵 (Pitch) -> 控制下潜/上浮
- **A / D**: 调节方向舵 (Yaw) -> 控制左转/右转
- **↑ / ↓**: 调节螺旋桨转速 (RPM) -> 控制前进速度
- **鼠标右键拖拽**: 旋转 3D 摄像机视角
- **ESC**: 退出仿真

#### 方式二：基于 RL 环境的手动模式 (run_human.py)

启动 `run_human.py` 可以在强化学习环境中进行手动控制，体验与训练时相同的环境设置。

```bash
python run_human.py
```

**键盘操作指南**:
- **W / S**: 调节艉舵 (Pitch) -> 控制下潜/上浮
- **A / D**: 调节方向舵 (Yaw) -> 控制左转/右转
- **↑ / ↓**: 调节推力 (Thrust) -> 控制前进速度
- **ESC**: 退出仿真

### 2.3 训练强化学习模型

使用 PPO 算法训练潜艇在水下隧道中自动避障。训练脚本为 `train.py`。

```bash
# 启动训练 (混合 LSTM 模式，默认参数)
python train.py

# 自定义总训练步数
python train.py --total_timesteps 5000000

# 指定设备
python train.py --device cuda

# 从指定阶段恢复训练
python train.py --resume_model ./training_output/xxx/checkpoints/submarine_ppo_500000_steps.zip \
                --resume_normalize ./training_output/xxx/checkpoints/submarine_ppo_500000_steps_normalize.pkl \
                --resume_stage 3
```

**训练模式说明**：
- **混合 LSTM 模式**（默认）：使用点云序列经 LSTM 处理后与状态融合的混合架构
- **课程学习**（自动）：从简单到复杂的渐进式训练
  - **Stage 1**: 3个障碍物，学会快速前进（进度主导）
  - **Stage 2**: 5个障碍物，增加难度，保持策略
  - **Stage 3**: 7个障碍物，进一步增加难度
  - **Stage 4**: 9个障碍物，接近最终难度
  - **Stage 5**: 15个障碍物 + 渐进式训练（固定布局→渐进扰动→完全随机）
- **VecNormalize**：自动对观测和奖励进行归一化，稳定训练
- **多进程训练**：默认使用 15 个并行环境加速训练

训练日志默认保存在 `./training_output/` 目录下，可以使用 TensorBoard 实时监控：
```bash
tensorboard --logdir ./training_output
```

### 2.4 模型评估与回放

训练完成后，使用 `eval_sim.py` 加载模型并观察其在 3D 环境中的表现。

```bash
# 基本评估（加载模型和归一化统计）
python eval_sim.py \
    --model ./training_output/submarine_ppo_hybrid_lstm_xxx/best_model/best_model.zip \
    --vecnormalize ./training_output/submarine_ppo_hybrid_lstm_xxx/vecnormalize.pkl

# 手动控制模式
python eval_sim.py --manual

# 不保存数据
python eval_sim.py --model ./best_model2.zip --no-save-data
```

**评估控制键**：
- **WASD**: 相机移动
- **鼠标右键拖拽**: 旋转 3D 摄像机视角
- **I/K**: 俯仰控制（手动模式）
- **J/L**: 偏航控制（手动模式）
- **↑/↓**: 调整推力（手动模式）
- **Space**: 暂停/继续
- **R**: 重置环境
- **ESC**: 退出评估

**数据保存**：
- 默认保存每个回合的点云、状态、动作、奖励和位置数据
- 保存路径：`./eval_data/episode_XXXX/`
- 包含 `.npy` 数据文件和 `metadata.json` 元数据

---

## 🏗 系统架构详解

### 3.1 文件结构树

```text
SubmarineHunting/
├── lib/                            # [算法核心] 数学与控制库
│   ├── gnc.py                      # 制导、导航与控制 (GNC) 工具函数
│   ├── models.py                   # 动力学模型矩阵计算 (Clarke83, Fossen方程)
│   ├── control.py                  # 控制器实现 (PID, Integral SMC)
│   ├── guidance.py                 # 制导律实现 (Line-of-Sight)
│   ├── actuator.py                 # 执行机构模型 (舵面, 推进器)
│   ├── mainLoop.py                 # 主循环控制
│   └── plotTimeSeries.py           # 时间序列绘图工具
│
├── src/                            # [仿真引擎]
│   ├── core/                       # 核心架构
│   │   ├── actor.py                # Actor 基类
│   │   └── stage.py                # LogicStage (仿真主循环)
│   │
│   ├── physics/                    # 物理层
│   │   └── submarine_actor.py      # 潜艇实体 (连接 torpedo.py 与 Actor 系统)
│   │
│   ├── gameplay/                   # 游戏逻辑层
│   │   ├── obstacle_actor.py       # 障碍物
│   │   ├── tunnel_actor.py         # 隧道环境
│   │   └── target_point_actor.py   # 目标点
│   │
│   ├── rl/                         # [强化学习层] ⭐
│   │   ├── __init__.py
│   │   ├── submarine_env.py        # Gymnasium 环境封装 (混合 LSTM 模式)
│   │   ├── point_cloud_sampler.py  # 点云采样器 (障碍物和隧道壁表面采样)
│   │   ├── tunnel.py               # 隧道生成算法 (ProvingGround, TunnelConfig)
│   │   └── custom_extractor.py     # 自定义特征提取器 (LSTM + MLP 混合架构, 优化版)
│   │
│   └── view/                       # 视图层
│       └── renderer.py             # OpenGL 渲染器
│
├── torpedo.py                      # 物理模型入口 (TorpedoPhysics 类)
├── train.py                        # 训练入口 (混合 LSTM 模式)
├── eval_sim.py                     # 评估与可视化脚本 (支持数据保存)
├── eval_model5.py                  # 旧版评估脚本 (保留兼容)
├── pygame_sim.py                   # 3D 仿真入口 (独立渲染器)
├── run_human.py                    # 手动控制模式 (使用 SubmarineEnv)
├── requirements.txt                # 依赖列表
├── README.md                       # 项目主文档
└── RL环境配置文档.md               # RL 环境详细配置文档
```

### 3.2 核心设计模式 (Actor-Stage)

本项目采用了经典的 **Actor-Stage** 模式来解耦物理计算与场景管理：

- **Actor (`src/core/actor.py`)**: 仿真实体的最小单元。每个 Actor 维护自己的位置 `position`、姿态 `orientation` 以及 `update(dt)` 更新逻辑。
- **LogicStage (`src/core/stage.py`)**: 仿真世界的容器。它维护一个 Actor 列表，并在每个时间步调用所有 Actor 的 `update` 方法。

### 3.3 渲染引擎 (OpenGL)

渲染模块 `src/view/renderer.py` 是完全独立的。它不参与物理计算，只负责读取 LogicStage 中当前的状态并进行绘制。

- **Headless Mode**: 在强化学习训练时，只运行 LogicStage 而不初始化 Renderer，极大提高训练速度。
- **Visualization**: 在需要演示时，Renderer 以 60FPS 的帧率绘制场景。

---

## 📐 数学模型与物理引擎

### 4.1 6-DOF 运动方程

AUV 的运动描述采用了标准的 6 自由度方程：

$$ M \dot{\nu} + C(\nu)\nu + D(\nu)\nu + g(\eta) = \tau $$

其中：
- **$\eta = [x, y, z, \phi, \theta, \psi]^T$**: 北东地 (NED) 坐标系下的位置和欧拉角。
- **$\nu = [u, v, w, p, q, r]^T$**: 机体坐标系 (Body Frame) 下的线速度和角速度。
- **$M$**: 系统惯性矩阵，包含刚体质量 $M_{RB}$ 和附加质量 $M_A$。
- **$C(\nu)$**: 科里奥利-向心力矩阵。
- **$D(\nu)$**: 阻尼矩阵，包含线性阻尼 $D_L$ 和非线性二次阻尼 $D_{NL}(\nu)$。
- **$g(\eta)$**: 重力和浮力产生的恢复力向量。
- **$\tau$**: 控制输入向量。

### 4.2 水动力学计算

`torpedo.py` 中的 `TorpedoPhysics` 类负责具体的受力计算：

1. **附加质量 (Added Mass)**: 使用椭球体近似计算 REMUS 100 的附加质量系数。
2. **交叉流阻力 (Cross-flow Drag)**: 使用条带理论 (Strip Theory) 计算横向水流对细长艇体的积分阻力。
3. **升阻力特性**: 舵面的升力系数 $C_L$ 和阻力系数 $C_D$ 使用 NACA 翼型数据拟合。

### 4.3 控制算法 (PID & SMC)

`lib/control.py` 提供了两种控制实现：

1. **PID Pole Placement**: 用于深度控制，通过极点配置法计算 $K_p, K_i, K_d$。
2. **Integral SMC (积分滑模控制)**: 用于航向控制，具有极强的鲁棒性。

### 4.4 执行器动力学

舵角和推进器转速**不是瞬间变化**的，而是通过**一阶惯性系统**渐进逼近目标值：

| 执行器 | 时间常数 | 最大值 | 响应时间 |
|--------|----------|--------|----------|
| 舵面 (Fin) | 0.1s | ±15° | ~0.3-0.5s |
| 推进器 | 0.1s | 1525 RPM | ~0.3-0.5s |

这意味着 RL 智能体给出的动作是期望命令值，而物理系统使用的是**逐渐逼近的实际值**。智能体需要学会"预判"。

---

## 🧠 强化学习环境 (SubmarineEnv)

`src/rl/submarine_env.py` 定义了强化学习交互环境。

### 5.1 观测模式 (Hybrid LSTM)

SubmarineEnv 默认采用 **混合 LSTM 模式**，通过 Dict 观测空间提供多模态数据：

```python
env = SubmarineEnv(point_cloud_history_len=16)
```

- **观测空间**: `Dict`
  - `"state"`: `Box(shape=(13,))` - 状态向量
  - `"point_cloud_seq"`: `Box(shape=(16, 768))` - 点云序列历史 (16帧 × 256点 × 3坐标)

**点云采样说明**:
- 使用 `PointCloudSampler` 对障碍物和隧道壁表面进行随机采样
- 障碍物采样采用球面均匀采样算法，按表面积比例分配采样点
- 隧道壁采样在潜艇前后 30% 隧道长度范围内进行
- 采样点从世界坐标系转换到机体坐标系，并归一化到 [-1, 1] 范围
- 采样点直接落在物体表面，无感知范围限制

### 状态向量构成 (13 维)

| 维度 | 名称 | 说明 |
|------|------|------|
| 0-1 | 相对位置 | 潜艇相对于隧道中心的 $y, z$ 偏移量 (归一化) |
| 2-7 | 速度状态 | 归一化的机体速度 $[u, v, w, p, q, r]$ |
| 8-10 | 姿态角 | 滚转、俯仰、偏航角 (弧度) |
| 11 | 边界距离 | 到隧道边界的距离 (归一化) |
| 12 | 行程进度 | 当前行程进度 (0-1) |

> **注意**: 智能体通过点云历史信息和速度/姿态响应来学习执行器动力学，不再在观测中直接提供控制输入。

### 5.2 动作空间 (Action)

动作空间为 `Box(-1, 1, shape=(3,))`：

| 索引 | 名称 | 范围 | 物理映射 |
|------|------|------|----------|
| 0 | Pitch Command | [-1, 1] | 艉舵角 ±20° |
| 1 | Yaw Command | [-1, 1] | 方向舵角 ±20° |
| 2 | Thrust Command | [-1, 1] | 螺旋桨转速 0~1525 RPM |

**注意**：推力命令通过 `(thrust_cmd + 1) / 2` 映射到 [0, 1]，然后再乘以 1525 RPM。这样 0 在输出范围的中间，便于网络学习。

### 5.3 奖励机制 (Reward - 进度主导设计)

奖励函数采用简洁设计，核心思想是让智能体专注于实际前进进度，通过进度奖励引导行为。

| 奖励类型 | 数值 | 说明 |
|----------|------|------|
| 到达终点奖励 | +1000 | 到达终点，立即结束回合 |
| 进度奖励 | +0~+700×Δprogress | 【核心】行程进度增量奖励（每前进1%给7分） |
| 碰撞惩罚 | -550 | 撞击障碍物或墙壁，立即结束回合 |
| 超时惩罚 | -550 | 超过最大步数（比碰撞惩罚更严厉，鼓励快速完成） |
| 推演碰撞惩罚 | -4.5/步 | 基于未来3秒轨迹预测，预测会碰撞时给予惩罚 |
| 每步惩罚 | -0.5/步 | 鼓励快速完成任务 |

**设计理念**：
- **进度优先**：进度增量奖励（每1%进度给7分）是主要的正收益来源，鼓励智能体高效完成隧道穿越
- **严厉惩罚**：碰撞惩罚（-550）和超时惩罚（-550）都很严厉，迫使智能体学会安全避障和快速前进
- **前瞻性**：推演碰撞惩罚基于未来3秒轨迹预测，鼓励智能体提前规避危险
- **防止刷分**：进度奖励基于历史最大进度，只有超过历史进度才给奖励，防止通过来回移动刷分
- **简洁设计**：核心设计简洁，主要信号来自进度奖励，辅以推演碰撞和每步惩罚

### 5.4 环境配置

#### TunnelConfig (隧道配置)
```python
from src.rl.tunnel import TunnelConfig

tunnel_config = TunnelConfig(
    radius=5.0,               # 隧道半径 (m)
    length=50.0,              # 隧道长度 (m)
    center_z=100.0,           # 隧道中心 Z 坐标 (m)
    num_obstacles=15,         # 障碍物数量
    obstacle_radius_min=0.7,  # 障碍物最小半径 (m)
    obstacle_radius_max=1.0,  # 障碍物最大半径 (m)

    # 渐进式训练模式配置 (用于课程5)
    progressive_mode=False,   # 是否启用渐进式训练
    progressive_stage=0,      # 渐进式阶段: 0=固定, 1-10=渐进调整, 11+=完全随机
    fixed_obstacle_seed=None, # 固定障碍物布局的种子
    perturbation_amount=0.5,  # 障碍物位置扰动量 (米)，每阶段递增
)
```

**注意**：训练脚本中使用的是更紧凑的隧道配置，适合快速训练和评估。

#### PointCloudConfig (点云采样配置)
```python
from src.rl.point_cloud_sampler import PointCloudConfig

point_cloud_config = PointCloudConfig(
    num_points=256,           # 总采样点数
    obstacle_points_ratio=0.7, # 障碍物采样点占比 (剩余用于隧道壁采样)
    normalize_range=50.0,     # 归一化范围 (m)，用于将坐标映射到 [-1, 1]
)
```

**采样策略说明**:
- **障碍物采样**: 使用球面均匀采样算法，按障碍物表面积比例分配采样点数
- **隧道壁采样**: 在潜艇前后 30% 隧道长度范围内，对圆柱面进行随机采样
- **坐标转换**: 采样点从世界坐标系转换到机体坐标系，并归一化到 [-1, 1] 范围
- **无感知范围限制**: 采样点直接落在物体表面，不进行距离裁剪

### 5.5 障碍物随机化机制

为了**防止 RL 训练过拟合**，每次环境 `reset()` 时：

1. **障碍物完全重新随机生成**
2. 潜艇的生成位置也会有随机偏移
3. 如果传入 `seed` 参数，则使用该种子生成可复现的布局

```python
# 每次 reset 都会重新生成障碍物
obs, info = env.reset()        # 随机障碍物布局
obs, info = env.reset(seed=42) # 固定障碍物布局 (可复现)
```

---

## 🎛 训练配置与使用

### 6.1 训练模式

本项目目前采用 **混合 LSTM 模式 + 课程学习**，这是经过优化的架构设计：

| 模式 | 观测类型 | 策略网络 | 特点 |
|------|----------|----------|------|
| **混合 LSTM** 🌟 | Dict | MultiInputPolicy + PointCloudLSTMExtractor | 点云序列经 LSTM 处理后与状态融合，具备时序记忆能力 |

**架构优势**:
- 点云序列通过 LSTM 提取时序特征，捕捉环境变化趋势
- 状态数据通过 MLP 提取静态特征
- 两者拼接后输入 PPO 的 Actor/Critic 网络
- 支持双向 LSTM、Dropout、Layer Normalization 等正则化技术

### 6.2 课程学习 (Curriculum Learning)

训练采用自动课程学习策略，从简单场景逐渐过渡到复杂场景：

#### 课程阶段配置

| 阶段 | 障碍物数量 | 成功阈值 | 目标 |
|------|-----------|-------------|---------|
| **Stage 1** | 3 | 80% | 学会快速前进（进度主导） |
| **Stage 2** | 5 | 80% | 学会避障（增加障碍物，保持策略） |
| **Stage 3** | 7 | 80% | 进一步增加难度，提升避障精度 |
| **Stage 4** | 9 | 80% | 接近最终难度，优化策略 |
| **Stage 5** | 15 | 80% | 最终复杂环境，达到最优策略 |

#### Stage 5 渐进式训练 (Progressive Training)

Stage 5 采用独特的渐进式训练策略，从固定布局逐步过渡到完全随机：

| 渐进阶段 | 模式 | 说明 |
|---------|------|------|
| **Stage 0** | 固定布局 | 使用固定种子 (seed=42) 生成障碍物，成功率≥70%后进入下一阶段 |
| **Stage 1-10** | 渐进扰动 | 在固定布局基础上添加随机扰动，扰动量从 ±0.5m 递增到 ±5.0m |
| **Stage 11+** | 完全随机 | 障碍物位置完全随机生成 |

#### 自动切换机制

- 每 20000 步进行一次评估
- 评估 10 个回合，计算成功率
- 最小训练步数 50000 步（避免初期随机运气影响）
- 当成功率达到阈值时，自动切换到下一阶段
- Stage 5 中，当最近 10 次评估的成功率≥70%时，自动进入下一渐进阶段

**多进程配置同步**:
- 使用 `deepcopy()` 为每个 worker 进程创建独立的 `tunnel_config` 副本
- 防止多进程间共享同一配置对象导致的竞态条件
- 在 `SubmarineEnv.reset()` 中同步 `tunnel_config` 和 `tunnel.config` 确保一致性

#### 自定义课程

您可以在 `train.py` 中修改 `CURRICULUM_STAGES` 字典来自定义课程：

```python
CURRICULUM_STAGES = {
    1: CurriculumConfig(stage=1, num_obstacles=3, w_velocity=0.2),
    2: CurriculumConfig(stage=2, num_obstacles=5, w_velocity=0.2),
    3: CurriculumConfig(stage=3, num_obstacles=7, w_velocity=0.2),
    4: CurriculumConfig(stage=4, num_obstacles=9, w_velocity=0.2),
    5: CurriculumConfig(stage=5, num_obstacles=15, w_velocity=0.2),
    # 添加更多阶段...
}
```

### 6.3 命令行参数详解

```bash
python train.py [OPTIONS]
```

#### 训练基础参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--total-timesteps` | 9,000,000 | 总训练步数 |
| `--n-envs` | 15 | 并行环境数量 |
| `--seed` | 42 | 随机种子 |
| `--output-dir` | `./training_output` | 输出目录 |

#### 恢复训练参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--resume-model` | None | 模型路径 (用于恢复训练) |
| `--resume-normalize` | None | VecNormalize 路径 |
| `--resume-stage` | 1 | 从哪个课程阶段恢复 (1-5) |

#### PPO 超参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--learning-rate` | 3e-5 | 学习率 |
| `--n-steps` | 2048 | 每个环境每次更新的步数 |
| `--batch-size` | 512 | Mini-batch 大小 |
| `--n-epochs` | 10 | 每次更新的 epoch 数 |
| `--gamma` | 0.99 | 折扣因子 |
| `--gae-lambda` | 0.99 | GAE (Generalized Advantage Estimation) 参数 |
| `--clip-range` | 0.2 | PPO 裁剪范围 |
| `--ent-coef` | 0.01 | 熵系数，控制探索程度 |

#### 网络架构参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--pi-hidden-1` | 256 | Actor 网络第一层隐藏单元数 |
| `--pi-hidden-2` | 128 | Actor 网络第二层隐藏单元数 |
| `--vf-hidden-1` | 256 | Critic 网络第一层隐藏单元数 |
| `--vf-hidden-2` | 128 | Critic 网络第二层隐藏单元数 |

#### 混合 LSTM 特征提取器参数 (需修改 train.py 中的 TrainingConfig)
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--point-cloud-history-len` | 16 | 点云历史帧数 |
| `--lstm-hidden-size` | 128 | LSTM 隐藏层大小 |
| `--lstm-num-layers` | 1 | LSTM 层数 (1-3) |
| `--bidirectional` | False | 是否使用双向 LSTM（输出维度翻倍） |
| `--pc-encoder-dims` | "256" | 点云编码器隐藏层维度 (逗号分隔) |
| `--state-feature-dim` | 64 | 状态特征输出维度 |
| `--dropout` | 0.1 | Dropout 概率 (0 表示不使用，推荐 0.1-0.2) |
| `--use-layer-norm` | True | 是否使用 Layer Normalization（推荐开启） |

#### 保存与设备
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--save-freq` | 50000 | 检查点保存频率 |
| `--eval-freq` | 20000 | 评估频率 |
| `--device` | cuda | 设备 (auto/cuda/cpu) |

### 6.4 TensorBoard 监控

```bash
tensorboard --logdir ./training_output
```

在浏览器中打开 `http://localhost:6006` 查看：
- **rollout/ep_rew_mean**: 平均回合奖励
- **rollout/ep_len_mean**: 平均回合长度
- **time/fps**: 训练速度

---

## 🔧 高级配置与开发指南

### 7.1 自定义场景

如果你想创建一个不同的训练场景（例如开阔水域而不是隧道），可以修改 `src/rl/tunnel.py` 中的 `ProvingGround` 类。

1. 将 `check_boundary_collision` 中的逻辑移除或修改。
2. 在 `generate_obstacles` 中改变障碍物的分布规律。

### 7.2 添加新实体

要添加一个新的动态实体（例如一条巡逻的敌方潜艇）：

1. 在 `src/gameplay/` 下创建 `enemy_actor.py`。
2. 继承 `Actor` 类，并在 `update(dt)` 中实现简单的巡逻逻辑。
3. 在 `SubmarineEnv` 或 `pygame_sim.py` 中实例化该 Actor 并添加到 `LogicStage`。
4. (可选) 在 `Renderer` 中添加对应的绘制代码。

### 7.3 自定义神经网络架构

混合 LSTM 模式使用了优化的自定义特征提取器 `src/rl/custom_extractor.py`。

#### PointCloudLSTMExtractor 架构 (优化版)

```
观测输入 (Dict):
├── "state" (13维)
│         ↓
│   [Linear 13→128 → LayerNorm → ReLU → Dropout(0.1)]
│   [Linear 128→64 → LayerNorm → ReLU]
│         ↓
│    状态特征 (64维)
│
└── "point_cloud_seq" (16×768)
          ↓
    [Linear 768→256 → LayerNorm → ReLU] (逐帧)
          ↓
       [LSTM 256→128] (可选双向, 层间 Dropout)
          ↓
    [LayerNorm]
          ↓
     点云时序特征 (128维或256维, 取决于是否双向)
          ↓
     ┌─────┴─────┐
     │  Concat   │ → (192或320维) → PPO Actor/Critic
     └───────────┘
```

#### 架构特性

1. **正则化技术**:
   - **Layer Normalization**: 在线性层后、激活函数前应用，稳定训练
   - **Dropout**: 在激活函数后应用，防止过拟合 (默认 0.1)

2. **权重初始化**:
   - LSTM 输入到隐藏权重: Xavier 均匀初始化
   - LSTM 隐藏到隐藏权重: 正交初始化 (改善梯度流)
   - LSTM 遗忘门偏置: 初始化为 1 (帮助学习长期依赖)
   - 线性层: Xavier 均匀初始化

3. **动态维度计算**:
   - 状态编码器中间层动态计算: `max(64, state_feature_dim, state_dim * 2)`
   - 确保足够的表达能力

4. **双向 LSTM 支持**:
   - 可选开启双向模式，输出维度翻倍
   - 拼接正向和反向的最后隐藏状态

#### 修改特征提取器

```python
# src/rl/custom_extractor.py

class MyCustomExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, ...):
        # 自定义网络结构
        pass

    def forward(self, observations):
        # 自定义前向传播
        pass

# train.py 中使用
policy_kwargs = dict(
    features_extractor_class=MyCustomExtractor,
    features_extractor_kwargs=dict(...),
)
```

#### 评估与数据保存

使用 `eval_sim.py` 进行模型评估和数据保存：

```python
# 加载模型并评估
python eval_sim.py \
    --model ./training_output/submarine_ppo_hybrid_lstm_20260114/best_model/best_model.zip \
    --vecnormalize ./training_output/submarine_ppo_hybrid_lstm_20260114/vecnormalize.pkl \
    --save-data \
    --save-dir ./eval_data
```

**保存的数据格式**：
- `point_clouds.npy`: (T, 256, 3) 点云数据
- `states.npy`: (T, 13) 状态数据
- `actions.npy`: (T, 3) 动作数据
- `rewards.npy`: (T,) 奖励数据
- `positions.npy`: (T, 3) 位置数据
- `metadata.json`: 元数据（回合ID、成功状态、总奖励等）

---

## ❓ 常见问题 (Troubleshooting)

**Q1: 运行 `pygame_sim.py` 时报错 `OpenGL.error.NullFunctionError`?**

A: 这通常是因为 Windows 缺少 OpenGL 驱动或 PyOpenGL 安装不完整。请尝试：
1. 更新显卡驱动。
2. 卸载 `pip uninstall PyOpenGL PyOpenGL_accelerate`。
3. 从 [UCI](https://www.lfd.uci.edu/~gohlke/pythonlibs/) 下载对应 Python 版本的 `.whl` 文件重新安装。

**Q2: 训练时提示 PyTorch 相关错误?**

A: 混合 LSTM 模式需要 PyTorch。请运行：
```bash
pip install torch
```
建议使用 CUDA 版本以加速训练。

**Q3: 潜艇一直在原地打转?**

A: 检查 `torpedo.py` 中的 PID 参数。如果仿真步长 (`dt`) 改变，可能需要重新调整 $K_p, K_i, K_d$。默认参数是针对 $dt=0.05s$ 调优的。

**Q4: 训练速度很慢?**

A:
1. 使用 GPU (`--device cuda`) 加速 PyTorch 计算。
2. 增加 `--n-envs` 并行环境数量。
3. 减少 `--point-cloud-history-len` 可以加快速度。
4. 调整 `--batch-size` 和 `--n-steps` 以平衡内存使用和训练速度。

---

## 📚 参考文献与许可证

### 核心参考文献

- Fossen, T. I. (2021). *Handbook of Marine Craft Hydrodynamics and Motion Control*. 2nd Edition, Wiley.
- Prestero, T. (2001). *Verification of a six-degree of freedom simulation model for the REMUS AUV*. Master's thesis, MIT/WHOI.
- Schulman, J., et al. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347.

### 依赖项目

- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
- [Gymnasium](https://gymnasium.farama.org/)
- [PyOpenGL](http://pyopengl.sourceforge.net/)

### 许可证
本项目采用 [MIT License](LICENSE) 许可证。

Copyright (c) 2023-2026 SubmarineHunting Team

---

*Created with ❤️ by the SubmarineHunting Team.*
*Last updated: 2026-01-14* (文档同步最新代码状态)

### 更新日志
- **2026-01-14**:
  - 更新混合 LSTM 架构说明（带 Dropout、LayerNorm、正交初始化）
  - 更新课程学习配置（5阶段+Stage 5 渐进式训练，70%成功率阈值）
  - 更新渐进式训练机制（使用成功率阈值替代连续成功次数）
  - 更新多进程配置同步（deepcopy独立副本，避免共享引用问题）
  - 更新奖励函数参数（w_progress=7.0, r_future_collision=-4.5）
  - 更新训练超参数（total_timesteps=9M, n_envs=15）
  - 更新命令行参数说明（增加恢复训练参数）
- **2026-01-09**: 添加课程学习自动训练机制、更新训练流程和评估配置
- **2026-01-07**: 添加执行器动力学说明、障碍物随机化机制、更新文档
