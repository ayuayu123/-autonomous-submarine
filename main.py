import numpy as np
import matplotlib.pyplot as plt
from torpedo import torpedo
from lib.gnc import attitudeEuler
from lib.plotTimeSeries import plotVehicleStates, plotControls, plot3D

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 1. 初始化车辆
# 模式: "stepInput" 或 "depthHeadingAutopilot"
vehicle = torpedo(controlSystem="stepInput", r_rpm=1500)

print(f"项目: {vehicle.name}")
print(f"控制模式: {vehicle.controlMode}")

# 2. 初始状态
eta = np.zeros(6)  # 位置和姿态 [x, y, z, phi, theta, psi]
nu = np.zeros(6)  # 速度 [u, v, w, p, q, r]
u_actual = np.zeros(vehicle.dimU)
u_control = np.zeros(vehicle.dimU)

# 3. 仿真参数
sampleTime = 0.05
t_final = 10
N = int(t_final / sampleTime)

# 存储历史数据以便绘图
# simData 结构: [eta (6), nu (6), u_control (dimU), u_actual (dimU)]
simData = np.zeros((N, 12 + 2 * vehicle.dimU))  
t_history = np.zeros(N)

# 4. 主循环
print("开始仿真...")
for i in range(N):
    t = i * sampleTime

    # 获取控制指令
    if vehicle.controlMode == "stepInput":
        u_control = vehicle.stepInput(t)
    else:
        u_control = vehicle.depthHeadingAutopilot(eta, nu, sampleTime)

    # 计算动力学
    nu, u_actual = vehicle.dynamics(eta, nu, u_actual, u_control, sampleTime)

    # 积分位置 (运动学)
    eta = attitudeEuler(eta, nu, sampleTime)

    # 存储
    simData[i, 0:6] = eta
    simData[i, 6:12] = nu
    simData[i, 12:12 + vehicle.dimU] = u_control
    simData[i, 12 + vehicle.dimU:] = u_actual
    t_history[i] = t

print("仿真结束。")

# 5. 绘图
plotVehicleStates(t_history, simData, 1)
plotControls(t_history, simData, vehicle, 2)

# 6. 3D 轨迹图
# numDataPoints: 动画帧数/采样点数
# FPS: 帧率
# filename: 保存文件名 (None 表示不保存)
plot3D(simData, numDataPoints=100, FPS=10, filename=None, figNo=3)

plt.show()
