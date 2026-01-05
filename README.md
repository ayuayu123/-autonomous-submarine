# SubmarineHunting

## 简介
SubmarineHunting 是一个基于 Python 的潜艇/鱼雷仿真项目。该项目包含 6 自由度（6-DOF）动力学建模、控制系统设计以及可视化仿真。用户可以通过数值仿真分析车辆性能，或通过 3D 实时仿真进行交互式控制体验。

## 功能特性
- **动力学仿真**: 基于物理的 6 自由度潜艇/鱼雷运动学与动力学模型。
- **控制系统**: 包含阶跃输入（Step Input）和深度/航向自动驾驶（Depth & Heading Autopilot）模式。
- **数据可视化**:
  - 使用 `matplotlib` 绘制位置、速度、姿态等状态变量随时间变化的曲线。
  - 生成 3D 轨迹图。
- **3D 实时仿真**: 基于 `pygame` 的交互式 3D 线框模型可视化。

## 环境要求
本项目依赖以下 Python 库：
- `numpy`: 用于数值计算
- `matplotlib`: 用于绘图
- `pygame`: 用于 3D 实时可视化

## 安装说明
1. 克隆或下载本项目到本地。
2. 确保已安装 Python 3.x。
3. 安装依赖库：
   ```bash
   pip install numpy matplotlib pygame
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
- **鼠标右键拖拽**: 旋转摄像机视角

## 项目结构
- `main.py`: 数值仿真主程序，负责计算和绘图。
- `pygame_sim.py`: 基于 Pygame 的交互式 3D 可视化程序。
- `torpedo.py`: 定义 `torpedo` 类，包含潜艇物理参数、动力学方程和控制逻辑。
- `lib/`: 核心库文件夹
  - `gnc.py`: 制导、导航与控制相关函数（如欧拉角姿态更新）。
  - `models.py`: 动力学模型实现。
  - `plotTimeSeries.py`: 绘图工具函数。
  - `actuator.py`, `control.py`, `guidance.py`: 辅助模块。

## 许可证
[MIT License](LICENSE) (如有)
