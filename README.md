# SubmarineHunting

## 简介
SubmarineHunting 是一个基于 Python 的潜艇/鱼雷仿真项目。该项目包含 6 自由度（6-DOF）动力学建模、控制系统设计以及可视化仿真。用户可以通过数值仿真分析车辆性能，或通过 3D 实时仿真进行交互式控制体验。

## 功能特性
- **动力学仿真**: 基于物理的 6 自由度潜艇/鱼雷运动学与动力学模型。
- **控制系统**: 包含阶跃输入（Step Input）和深度/航向自动驾驶（Depth & Heading Autopilot）模式。
- **数据可视化**:
  - 使用 `matplotlib` 绘制位置、速度、姿态等状态变量随时间变化的曲线。
  - 生成 3D 轨迹图。
- **3D 实时仿真**: 基于 `pygame` 和 `OpenGL` 的交互式 3D 可视化。
- **仿真环境**:
  - 坐标单位：1 单位 = 1 米。
  - 初始状态：潜艇初始位置为 (0, 0, 400)。

## 数学模型
本项目采用了 Fossen (2021) 提出的标准水下航行器 6 自由度运动方程。

### 1. 运动学 (Kinematics)
描述车辆在惯性坐标系（NED Frame）下的位置和姿态变化：
$$ \dot{\eta} = J(\eta) \nu $$
其中：
- $\eta = [x, y, z, \phi, \theta, \psi]^T$：北东地（NED）坐标系下的位置和欧拉角。
- $\nu = [u, v, w, p, q, r]^T$：机体坐标系（Body Frame）下的线速度和角速度。
- $J(\eta)$：速度变换矩阵，包含旋转矩阵 $R_b^n(\Theta)$ 和角速度变换矩阵 $T_{\Theta}(\Theta)$。

### 2. 动力学 (Dynamics)
描述车辆在机体坐标系下的受力与运动关系（牛顿-欧拉方程）：
$$ M \dot{\nu} + C(\nu)\nu + D(\nu)\nu + g(\eta) = \tau $$
其中：
- $M = M_{RB} + M_A$：系统惯性矩阵（刚体质量 + 附加质量）。
- $C(\nu) = C_{RB}(\nu) + C_A(\nu)$：科里奥利和向心力矩阵。
- $D(\nu)$：阻尼矩阵（包含线性阻尼和非线性二次阻尼）。
- $g(\eta)$：恢复力向量（重力 $W$ 与浮力 $B$ 产生的力和力矩）。
- $\tau$：控制输入向量（由推进器和舵面产生的力和力矩）。

## 环境要求
本项目依赖以下 Python 库：
- `numpy`: 用于数值计算
- `matplotlib`: 用于绘图
- `pygame`: 用于 3D 实时可视化窗口管理
- `PyOpenGL`: 用于 3D 图形渲染

## 安装说明
1. 克隆或下载本项目到本地。
2. 确保已安装 Python 3.x。
3. 安装依赖库：
   ```bash
   pip install numpy matplotlib pygame PyOpenGL
   ```

## 使用说明

### 1. 运行数值仿真
运行 `main.py` 进行数值计算并生成图表：
```bash
python main.py
```
程序运行结束后，将显示车辆的状态曲线（位置、速度、控制输入）以及 3D 轨迹图。
*注意：默认配置可能为 `stepInput` 模式，可在 `main.py` 中修改 `vehicle = torpedo(controlSystem="...")` 来切换控制模式。*

### 2. 运行 3D 可视化 (交互模式)
运行 `pygame_sim.py` 启动实时 3D 演示：
```bash
python pygame_sim.py
```
**操作指南**:
- **W / S**: 控制俯仰（下潜 / 上浮）
- **A / D**: 控制偏航（左转 / 右转）
- **↑ / ↓ (方向键)**: 增加 / 减少推进器转速 (RPM)
- **1 / 2 / 3 / 4**: 微调四个舵面的偏移量 (按住 Shift 键反向调节)
- **鼠标右键拖拽**: 旋转摄像机视角

## 开发指南：如何构造并添加自定义 Actor

本项目使用 Actor-Stage 模式来管理仿真对象。如果需要添加新的实体（如障碍物、其他潜艇、目标等），请遵循以下步骤：

### 1. 继承 Actor 基类
在 `src/` 目录下（或合适的子目录）创建一个新文件，定义一个继承自 `src.core.actor.Actor` 的类。

```python
from src.core.actor import Actor
import numpy as np

class MyCustomActor(Actor):
    def __init__(self, name="MyActor", x=0, y=0, z=0):
        # 1. 初始化基类
        super().__init__(name=name, x=x, y=y, z=z)
        
        # 2. 自定义属性
        self.speed = 1.0 

    def update(self, dt):
        """
        每帧调用一次的逻辑更新
        dt: 距离上一帧的时间间隔 (秒)
        """
        # 示例：简单的直线运动
        self.position[0] += self.speed * dt  # 向北移动
        
        # 如果需要，也可以更新姿态 self.orientation (roll, pitch, yaw)
```

### 2. 将 Actor 添加到 Stage
在仿真主程序（如 `pygame_sim.py` 或自定义脚本）中，实例化你的 Actor 并将其添加到 `LogicStage`。

```python
from src.core.stage import LogicStage
# from path.to.your.file import MyCustomActor # 导入你的类

# 1. 创建舞台 (通常在 App 初始化时完成)
# stage = LogicStage()

# 2. 创建并添加 Actor
# my_actor = MyCustomActor(name="TestObject", x=10, y=20, z=50)
# stage.add_actor(my_actor)

# 3. 在主循环中，stage.update(dt) 会自动调用 my_actor.update(dt)
```

## 项目结构
- `main.py`: 数值仿真主程序，负责计算和绘图。
- `pygame_sim.py`: 基于 Pygame 和 OpenGL 的交互式 3D 可视化程序入口。
- `src/`: 源代码目录 (重构新增)
  - `core/`: 核心架构 (Stage, Actor)
  - `physics/`: 物理实体 (`SubmarineActor`)
  - `view/`: 渲染引擎 (`Renderer`)
- `torpedo.py`: 潜航器物理模型定义 (REMUS 100)。
- `lib/`: 核心算法库
  - `gnc.py`: GNC (制导、导航与控制) 算法。
  - `models.py`: 动力学方程。
  - `actuator.py`: 执行机构模型。
  - `plotTimeSeries.py`: 绘图工具。

## 许可证
[MIT License](LICENSE) (如有)
