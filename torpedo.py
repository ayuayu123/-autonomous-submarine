#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
torpedo.py: 鱼雷形潜航器模型（独立版本 - 重构为三类架构）
"""
import numpy as np
import math
import sys

# 修改了这里的导入路径
from lib.control import integralSMC
from lib.gnc import crossFlowDrag, forceLiftDrag, Hmtrx, m2c, gvect, ssa
from lib.actuator import fin, thruster


# Class TorpedoPhysics
class TorpedoPhysics:
    """
    负责鱼雷的物理仿真、动力学和流体动力学计算
    """
    def __init__(self):
        # 常量
        self.rho = 1026  # 水的密度 (kg/m^3)
        g = 9.81  # 重力加速度 (m/s^2)

        self.L = 1.6  # 长度 (m)
        self.diam = 0.19  # 圆柱体直径 (m)

        # 推进器 (用于获取参数)
        self.prop = thruster(self.rho)

        # Hydrodynamics (Fossen 2021, Section 8.4.2)
        self.S = 0.7 * self.L * self.diam  # S = 矩形 L * diam 的 70%
        a = self.L / 2  # 半轴
        b = self.diam / 2
        self.r_bg = np.array([0, 0, 0.02], float)  # 重心相对于坐标原点
        self.r_bb = np.array([0, 0, 0], float)  # 浮心相对于坐标原点

        # 寄生阻力系数 CD_0
        Cd = 0.42  # 来自 Allen 等人 (2000)
        self.CD_0 = Cd * math.pi * b ** 2 / self.S

        # 在 CO 中表示的刚体质量矩阵
        m = 4 / 3 * math.pi * self.rho * a * b ** 2  # 椭球体质量
        Ix = (2 / 5) * m * b ** 2  # 转动惯量
        Iy = (1 / 5) * m * (a ** 2 + b ** 2)
        Iz = Iy
        MRB_CG = np.diag([m, m, m, Ix, Iy, Iz])  # 在 CG 中表示的 MRB
        H_rg = Hmtrx(self.r_bg)
        self.MRB = H_rg.T @ MRB_CG @ H_rg  # 在 CO 中表示的 MRB

        # 重力和浮力
        self.W = m * g
        self.B = self.W

        # 滚转附加转动惯量
        r44 = 0.3
        MA_44 = r44 * Ix

        # Lamb 的 k 因子
        e = math.sqrt(1 - (b / a) ** 2)
        alpha_0 = (2 * (1 - e ** 2) / pow(e, 3)) * (0.5 * math.log((1 + e) / (1 - e)) - e)
        beta_0 = 1 / (e ** 2) - (1 - e ** 2) / (2 * pow(e, 3)) * math.log((1 + e) / (1 - e))

        k1 = alpha_0 / (2 - alpha_0)
        k2 = beta_0 / (2 - beta_0)
        k_prime = pow(e, 4) * (beta_0 - alpha_0) / (
                (2 - e ** 2) * (2 * e ** 2 - (2 - e ** 2) * (beta_0 - alpha_0)))

        # 在 CO 中表示的附加质量系统矩阵
        self.MA = np.diag([m * k1, m * k2, m * k2, MA_44, k_prime * Iy, k_prime * Iy])

        # 包含附加质量的质量矩阵
        self.M = self.MRB + self.MA
        self.Minv = np.linalg.inv(self.M)

        # 滚转和俯仰的固有频率
        self.w_roll = math.sqrt(self.W * (self.r_bg[2] - self.r_bb[2]) /
                                self.M[3][3])
        self.w_pitch = math.sqrt(self.W * (self.r_bg[2] - self.r_bb[2]) /
                                 self.M[4][4])

        S_fin = 0.00665  # 舵翼面积
        CL_delta_r = 0.5  # 方向舵升力系数
        CL_delta_s = 0.7  # 艉舵升力系数

        portSternFin = fin(S_fin, CL_delta_s, -a, c=0.1, angle=0, rho=self.rho)
        bottomRudderFin = fin(S_fin, CL_delta_r, -a, c=0.1, angle=90, rho=self.rho)
        starSternFin = fin(S_fin, CL_delta_s, -a, c=0.1, angle=180, rho=self.rho)
        topRudderFin = fin(S_fin, CL_delta_r, -a, c=0.1, angle=270, rho=self.rho)
        self.actuators = [topRudderFin, bottomRudderFin, starSternFin, portSternFin, self.prop]
        self.dimU = len(self.actuators)

        # 低速线性阻尼矩阵参数
        self.T_surge = 20  # 纵向时间常数 (s)
        self.T_sway = 20  # 横向时间常数 (s)
        self.T_heave = self.T_sway  # 对于圆柱形 AUV 相等
        self.zeta_roll = 0.3  # 滚转相对阻尼比
        self.zeta_pitch = 0.8  # 俯仰相对阻尼比
        self.T_yaw = 1  # 偏航时间常数 (s)

    def dynamics(self, eta, nu, u_actual, u_control, sampleTime, V_c, beta_c):
        """
        Integrates the AUV equations of motion using Euler's method.
        使用欧拉法对 AUV 运动方程进行积分。
        """

        # 水流速度
        u_c = V_c * math.cos(beta_c - eta[5])  # 水流纵向速度
        v_c = V_c * math.sin(beta_c - eta[5])  # 水流横向速度

        nu_c = np.array([u_c, v_c, 0, 0, 0, 0], float)  # 水流速度
        Dnu_c = np.array([nu[5] * v_c, -nu[5] * u_c, 0, 0, 0, 0], float)  # 导数
        nu_r = nu - nu_c  # 相对速度
        alpha = math.atan2(nu_r[2], nu_r[0])  # 攻角
        U_r = math.sqrt(nu_r[0] ** 2 + nu_r[1] ** 2 + nu_r[2] ** 2)  # 相对速度

        # 在 CO 中表示的刚体/附加质量科里奥利/向心矩阵
        CRB = m2c(self.MRB, nu_r)
        CA = m2c(self.MA, nu_r)

        # 如果缺少二次旋转阻尼，滚转、俯仰和偏航中的 CA 项可能会使模型不稳定
        # 这些项假设为零
        CA[4][0] = 0  # 俯仰引起的二次速度项
        CA[0][4] = 0
        CA[4][2] = 0
        CA[2][4] = 0
        CA[5][0] = 0  # 偏航中的 Munk 力矩
        CA[0][5] = 0
        CA[5][1] = 0
        CA[1][5] = 0

        C = CRB + CA

        # 耗散力和力矩
        D = np.diag([
            self.M[0][0] / self.T_surge,
            self.M[1][1] / self.T_sway,
            self.M[2][2] / self.T_heave,
            self.M[3][3] * 2 * self.zeta_roll * self.w_roll,
            self.M[4][4] * 2 * self.zeta_pitch * self.w_pitch,
            self.M[5][5] / self.T_yaw
        ])

        # 线性纵向和横向阻尼
        D[0][0] = D[0][0] * math.exp(-3 * U_r)  # 在高速时消失，此时二次
        D[1][1] = D[1][1] * math.exp(-3 * U_r)  # 阻力和升力占主导地位

        tau_liftdrag = forceLiftDrag(self.diam, self.S, self.CD_0, alpha, U_r)
        tau_crossflow = crossFlowDrag(self.L, self.diam, self.diam, nu_r)

        # 恢复力和力矩
        g_force = gvect(self.W, self.B, eta[4], eta[3], self.r_bg, self.r_bb)

        # 一般力向量
        tau = np.zeros(6, float)
        for i in range(self.dimU):
            tau += self.actuators[i].tau(nu_r, nu)
            u_actual[i] = self.actuators[i].actuate(sampleTime, u_control[i])  # 执行器动力学

        # AUV 动力学
        tau_sum = tau + tau_liftdrag + tau_crossflow - np.matmul(C + D, nu_r) - g_force
        nu_dot = Dnu_c + np.matmul(self.Minv, tau_sum)

        # 前向欧拉积分 [k+1]
        nu += sampleTime * nu_dot

        return nu, u_actual


# Class TorpedoControl
class TorpedoControl:
    """
    负责控制逻辑、自动驾驶算法
    """
    def __init__(self, controlSystem, r_z, r_psi, r_rpm, physics_model):
        self.D2R = math.pi / 180
        self.ref_z = r_z
        self.ref_psi = r_psi
        self.ref_n = r_rpm
        self.controlMode = controlSystem

        # 检查输入参数
        if r_rpm < 0.0 or r_rpm > physics_model.prop.nMax:
            sys.exit("The RPM value should be in the interval 0-%s", (physics_model.prop.nMax))

        if r_z > 100.0 or r_z < 0.0:
            sys.exit('desired depth must be between 0-100 m')

        # 前馈增益（Nomoto 增益参数）
        self.K_nomoto = 5.0 / 20.0  # K_nomoto = r_max / delta_max
        self.T_nomoto = physics_model.T_yaw  # 偏航时间常数

        # 航向自动驾驶参考模型
        self.psi_d = 0  # 位置、速度和加速度状态
        self.r_d = 0
        self.a_d = 0
        self.wn_d = 0.1  # 期望固有频率
        self.zeta_d = 1  # 期望相对阻尼比
        self.r_max = 5.0 * math.pi / 180  # 最大偏航率

        # 航向自动驾驶
        self.lam = 0.1
        self.phi_b = 0.1  # 边界层厚度
        self.K_d = 0.5  # PID 增益
        self.K_sigma = 0.05  # SMC 切换增益

        self.e_psi_int = 0  # 偏航角误差积分状态

        # 深度自动驾驶
        self.wn_d_z = 0.02  # 期望固有频率，参考模型
        self.Kp_z = 0.1  # 垂向比例增益，外环
        self.T_z = 100.0  # 垂向积分增益，外环
        self.Kp_theta = 5.0  # 俯仰 PID 控制器
        self.Kd_theta = 2.0
        self.Ki_theta = 0.3
        self.K_w = 5.0  # 可选垂向速度反馈增益

        self.z_int = 0  # 垂向位置积分状态
        self.z_d = 0  # 期望位置，低通滤波器初始状态
        self.theta_int = 0  # 俯仰角积分状态

    def stepInput(self, t):
        """
        u_c = stepInput(t) generates step inputs.
        u_c = stepInput(t) 生成阶跃输入。
        """
        delta_r = 5 * self.D2R  # 舵角 (rad)
        delta_s = -5 * self.D2R  # 艉角 (rad)
        n = 1525  # 螺旋桨转速 (rpm)

        if t > 100:
            delta_r = 0

        if t > 50:
            delta_s = 0

        u_control = np.array([delta_r, -delta_r, -delta_s, delta_s, n], float)

        return u_control

    def depthHeadingAutopilot(self, eta, nu, sampleTime):
        """
        [delta_r, delta_s, n] = depthHeadingAutopilot(eta,nu,sampleTime)
        simultaneously control the heading and depth of the AUV.
        同时控制 AUV 的航向和深度。
        """
        z = eta[2]  # 垂向位置（深度）
        theta = eta[4]  # 俯仰角
        psi = eta[5]  # 偏航角
        w = nu[2]  # 垂向速度
        q = nu[4]  # 俯仰率
        r = nu[5]  # 偏航率
        e_psi = psi - self.psi_d  # 偏航角跟踪误差
        e_r = r - self.r_d  # 偏航率跟踪误差
        z_ref = self.ref_z  # 垂向位置（深度）设定点
        psi_ref = self.ref_psi * self.D2R  # 偏航角设定点

        # 螺旋桨指令
        n = self.ref_n

        # 深度自动驾驶（连续闭环）
        # 低通滤波后的期望深度指令
        self.z_d = math.exp(-sampleTime * self.wn_d_z) * self.z_d \
                   + (1 - math.exp(-sampleTime * self.wn_d_z)) * z_ref

        # PI 控制器
        theta_d = self.Kp_z * ((z - self.z_d) + (1 / self.T_z) * self.z_int)
        delta_s = -self.Kp_theta * ssa(theta - theta_d) - self.Kd_theta * q \
                  - self.Ki_theta * self.theta_int - self.K_w * w

        # 欧拉积分法 (k+1)
        self.z_int += sampleTime * (z - self.z_d)
        self.theta_int += sampleTime * ssa(theta - theta_d)

        # 航向自动驾驶（SMC 控制器）
        wn_d = self.wn_d  # 参考模型固有频率
        zeta_d = self.zeta_d  # 参考模型相对阻尼因子

        # 具有三阶参考模型的积分 SMC
        [delta_r, self.e_psi_int, self.psi_d, self.r_d, self.a_d] = \
            integralSMC(
                self.e_psi_int,
                e_psi, e_r,
                self.psi_d,
                self.r_d,
                self.a_d,
                self.T_nomoto,
                self.K_nomoto,
                wn_d,
                zeta_d,
                self.K_d,
                self.K_sigma,
                self.lam,
                self.phi_b,
                psi_ref,
                self.r_max,
                sampleTime
            )

        u_control = np.array([delta_r, -delta_r, delta_s, -delta_s, n], float)

        return u_control


# Class Torpedo (Entry Point)
class torpedo:
    """
    torpedo() - 入口类
        Rudder angle, stern plane and propeller revolution step inputs
        方向舵角、艉平面和螺旋桨转速阶跃输入

    torpedo('depthHeadingAutopilot',z_d,psi_d,n_d,V_c,beta_c)
        Depth and heading autopilots
        深度和航向自动驾驶仪
    """

    def __init__(
            self,
            controlSystem="stepInput",
            r_z=0,
            r_psi=0,
            r_rpm=0,
            V_current=0,
            beta_current=0,
    ):
        self.D2R = math.pi / 180
        self.V_c = V_current
        self.beta_c = beta_current * self.D2R
        
        # 初始化物理模型
        self.physics = TorpedoPhysics()
        
        # 初始化控制模型
        self.controller = TorpedoControl(controlSystem, r_z, r_psi, r_rpm, self.physics)

        if controlSystem == "depthHeadingAutopilot":
            self.controlDescription = (
                    "深度和航向自动驾驶仪, z_d = "
                    + str(r_z)
                    + ", psi_d = "
                    + str(r_psi)
                    + " deg"
            )

        else:
            self.controlDescription = (
                "艉舵、方向舵和螺旋桨的阶跃输入")
            controlSystem = "stepInput" # 确保一致性

        self.controlMode = controlSystem

        # 初始化 AUV 模型名称和参数
        self.name = "鱼雷形潜航器（独立版）"
        self.L = self.physics.L
        self.diam = self.physics.diam
        
        # 状态向量初始化
        self.nu = np.array([0, 0, 0, 0, 0, 0], float)  # 速度向量
        self.controls = [
            "上舵角 (deg)",
            "下舵角 (deg)",
            "右艉舵角 (deg)",
            "左艉舵角 (deg)",
            "螺旋桨转速 (rpm)"
        ]
        self.dimU = len(self.controls)
        self.u_actual = np.zeros(self.dimU, float)  # 控制输入向量

    def dynamics(self, eta, nu, u_actual, u_control, sampleTime):
        """
        Delegates to physics model
        """
        return self.physics.dynamics(eta, nu, u_actual, u_control, sampleTime, self.V_c, self.beta_c)

    def stepInput(self, t):
        """
        Delegates to controller
        """
        return self.controller.stepInput(t)

    def depthHeadingAutopilot(self, eta, nu, sampleTime):
        """
        Delegates to controller
        """
        return self.controller.depthHeadingAutopilot(eta, nu, sampleTime)
