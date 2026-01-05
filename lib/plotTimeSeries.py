# -*- coding: utf-8 -*-
"""
Simulator plotting functions:

plotVehicleStates(simTime, simData, figNo) 
plotControls(simTime, simData, vehicle, figNo)
def plot3D(simData, numDataPoints, FPS, filename, figNo)

Author:     Thor I. Fossen
"""

import math
import matplotlib.pyplot as plt
import numpy as np
from .gnc import ssa
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

legendSize = 10  # legend size
figSize1 = [25, 13]  # figure1 size in cm
figSize2 = [25, 13]  # figure2 size in cm
dpiValue = 150  # figure dpi value


def R2D(value):  # radians to degrees
    return value * 180 / math.pi


def cm2inch(value):  # inch to cm
    return value / 2.54


# plotVehicleStates(simTime, simData, figNo) plots the 6-DOF vehicle
# position/attitude and velocities versus time in figure no. figNo
def plotVehicleStates(simTime, simData, figNo):

    # Time vector
    t = simTime

    # State vectors
    x = simData[:, 0]
    y = simData[:, 1]
    z = simData[:, 2]
    phi = R2D(ssa(simData[:, 3]))
    theta = R2D(ssa(simData[:, 4]))
    psi = R2D(ssa(simData[:, 5]))
    u = simData[:, 6]
    v = simData[:, 7]
    w = simData[:, 8]
    p = R2D(simData[:, 9])
    q = R2D(simData[:, 10])
    r = R2D(simData[:, 11])

    # Speed
    U = np.sqrt(np.multiply(u, u) + np.multiply(v, v) + np.multiply(w, w))

    beta_c  = R2D(ssa(np.arctan2(v,u)))   # crab angle, beta_c    
    alpha_c = R2D(ssa(np.arctan2(w,u)))   # flight path angle
    chi = R2D(ssa(simData[:, 5] + np.arctan2(v, u)))  # course angle, chi=psi+beta_c

    # Plots
    plt.figure(
        figNo, figsize=(cm2inch(figSize1[0]), cm2inch(figSize1[1])), dpi=dpiValue
    )
    plt.grid()

    plt.subplot(3, 3, 1)
    plt.plot(y, x)
    plt.legend(["东北位置 (m)"], fontsize=legendSize)
    plt.grid()

    plt.subplot(3, 3, 2)
    plt.plot(t, z)
    plt.legend(["深度 (m)"], fontsize=legendSize)
    plt.grid()

    plt.title("航行器状态", fontsize=12)

    plt.subplot(3, 3, 3)
    plt.plot(t, phi, t, theta)
    plt.legend(["横滚角 (deg)", "俯仰角 (deg)"], fontsize=legendSize)
    plt.grid()

    plt.subplot(3, 3, 4)
    plt.plot(t, U)
    plt.legend(["速度 (m/s)"], fontsize=legendSize)
    plt.grid()

    plt.subplot(3, 3, 5)
    plt.plot(t, chi)
    plt.legend(["航向角 (deg)"], fontsize=legendSize)
    plt.grid()

    plt.subplot(3, 3, 6)
    plt.plot(t, theta, t, alpha_c)
    plt.legend(["俯仰角 (deg)", "飞行路径角 (deg)"], fontsize=legendSize)
    plt.grid()

    plt.subplot(3, 3, 7)
    plt.plot(t, u, t, v, t, w)
    plt.xlabel("时间 (s)", fontsize=12)
    plt.legend(
        ["纵向速度 (m/s)", "横向速度 (m/s)", "垂向速度 (m/s)"],
        fontsize=legendSize,
    )
    plt.grid()

    plt.subplot(3, 3, 8)
    plt.plot(t, p, t, q, t, r)
    plt.xlabel("时间 (s)", fontsize=12)
    plt.legend(
        ["横滚角速度 (deg/s)", "俯仰角速度 (deg/s)", "偏航角速度 (deg/s)"],
        fontsize=legendSize,
    )
    plt.grid()

    plt.subplot(3, 3, 9)
    plt.plot(t, psi, t, beta_c)
    plt.xlabel("时间 (s)", fontsize=12)
    plt.legend(["偏航角 (deg)", "侧滑角 (deg)"], fontsize=legendSize)
    plt.grid()


# plotControls(simTime, simData) plots the vehicle control inputs versus time
# in figure no. figNo
def plotControls(simTime, simData, vehicle, figNo):

    DOF = 6

    # Time vector
    t = simTime

    plt.figure(
        figNo, figsize=(cm2inch(figSize2[0]), cm2inch(figSize2[1])), dpi=dpiValue
    )

    # Columns and rows needed to plot vehicle.dimU control inputs
    col = 2
    row = int(math.ceil(vehicle.dimU / col))

    # Plot the vehicle.dimU active control inputs
    for i in range(0, vehicle.dimU):

        u_control = simData[:, 2 * DOF + i]  # control input, commands
        u_actual = simData[:, 2 * DOF + vehicle.dimU + i]  # actual control input

        if vehicle.controls[i].find("deg") != -1:  # convert angles to deg
            u_control = R2D(u_control)
            u_actual = R2D(u_actual)

        plt.subplot(row, col, i + 1)
        plt.plot(t, u_control, t, u_actual)
        plt.legend(
            [vehicle.controls[i] + ", 指令", vehicle.controls[i] + ", 实际"],
            fontsize=legendSize,
        )
        plt.xlabel("时间 (s)", fontsize=12)
        plt.grid()


# plot3D(simData,numDataPoints,FPS,filename,figNo) plots the vehicles position (x, y, z) in 3D
# in figure no. figNo
def plot3D(simData, numDataPoints, FPS, filename=None, figNo=3):
    
    # State vectors
    x = simData[:,0]
    y = simData[:,1]
    z = simData[:,2]
    
    # down-sampling the xyz data points
    # Ensure step is at least 1
    step = len(x) // numDataPoints
    if step < 1:
        step = 1
        
    N = y[::step]
    E = x[::step]
    D = z[::step]
    
    # Animation function
    def anim_function(num, dataSet, line):
        
        line.set_data(dataSet[0:2, :num])    
        line.set_3d_properties(dataSet[2, :num])    
        ax.view_init(elev=10.0, azim=-120.0)
        
        return line
    
    dataSet = np.array([N, E, -D])      # Down is negative z
    
    # Attaching 3D axis to the figure
    fig = plt.figure(figNo, figsize=(cm2inch(figSize1[0]),cm2inch(figSize1[1])), dpi=dpiValue)
    ax = p3.Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax) 
    
    # Line/trajectory plot
    line = plt.plot(dataSet[0], dataSet[1], dataSet[2], lw=2, c='b')[0] 

    # Setting the axes properties
    ax.set_xlabel('X / 东 (East)')
    ax.set_ylabel('Y / 北 (North)')
    ax.set_zlim3d([-100, 20])                   # default depth = -100 m
    
    if np.amax(z) > 100.0:
        ax.set_zlim3d([-np.amax(z), 20])
        
    ax.set_zlabel('-Z / 高度 (Altitude)')

    # Plot 2D surface for z = 0
    [x_min, x_max] = ax.get_xlim()
    [y_min, y_max] = ax.get_ylim()
    x_grid = np.arange(x_min-20, x_max+20)
    y_grid = np.arange(y_min-20, y_max+20)
    [xx, yy] = np.meshgrid(x_grid, y_grid)
    zz = 0 * xx
    ax.plot_surface(xx, yy, zz, alpha=0.3)
                    
    # Title of plot
    ax.set_title('3D 轨迹图 (北-东-下)', fontsize=12)
    
    # Create the animation object
    ani = animation.FuncAnimation(fig, 
                         anim_function, 
                         frames=len(N), 
                         fargs=(dataSet,line),
                         interval=200, 
                         blit=False,
                         repeat=True)
    
    # Save the 3D animation as a gif file if filename is provided
    if filename is not None:
        print(f"Saving 3D animation to {filename}...")
        try:
            ani.save(filename, writer=animation.PillowWriter(fps=FPS))
            print("Animation saved.")
        except Exception as e:
            print(f"Failed to save animation: {e}")  

