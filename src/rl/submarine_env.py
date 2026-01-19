#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
submarine_env.py: 潜艇避障强化学习环境 (Gymnasium 接口)
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from torpedo import torpedo
from lib.gnc import attitudeEuler, Rzyx
from src.rl.tunnel import CylinderTunnel, TunnelConfig
from src.rl.point_cloud_sampler import PointCloudSampler, PointCloudConfig
from src.core.stage import LogicStage
from src.physics.submarine_actor import SubmarineActor
from src.gameplay.obstacle_actor import ObstacleActor
from src.view.renderer import Renderer
import pygame
from pygame.locals import *


class SubmarineEnv(gym.Env):
    """
    潜艇避障强化学习环境

    观测空间 (Hybrid LSTM 模式):
        - "state": 状态向量 (13维)
          * 相对位置: 2 (y, z offset)
          * 速度: 6 (u, v, w, p, q, r)
          * 姿态: 3 (roll, pitch, yaw)
          * 到边界距离: 1
          * 进度: 1
          * 注意: 移除了控制输入，智能体通过点云历史信息学习执行器动力学
        - "point_cloud_seq": 点云序列 (history_len, num_points*3)

    动作空间:
        - 艉舵角 (pitch control): [-1, 1] -> [-20°, 20°]
        - 方向舵角 (yaw control): [-1, 1] -> [-20°, 20°]
        - 推进器转速: [0, 1] -> [0, 1525 RPM]

    奖励:
        - 前进奖励
        - 居中奖励
        - 碰撞惩罚
        - 到达终点奖励
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self,
                 tunnel_config: Optional[TunnelConfig] = None,
                 point_cloud_config: Optional[PointCloudConfig] = None,
                 max_steps: int = 1200,  # 充足的容错空间，但不能太长否则模型会浮分
                 point_cloud_history_len: int = 8,  # 从40减少到8，降低LSTM负担
                 render_mode: Optional[str] = None):
        super().__init__()

        self.render_mode = render_mode
        self.max_steps = max_steps
        self.point_cloud_history_len = point_cloud_history_len

        # 点云历史缓冲区
        self.point_cloud_history = None

        # 物理仿真时间步
        self.dt = 0.05  # 50ms

        # 控制角度限制
        self.max_angle = 20 * np.pi / 180  # 20 degrees

        # 奖励函数参数 (v30 - 短距离和连续避障增强)
        # v30改进：
        #   1. 更细粒度的预测时间步（0.2s起步）
        #   2. 连续障碍物感知（前方多个障碍物的序列观测）
        #   3. 避障惯性机制（躲避后短暂降低居中引力）
        #   4. 紧急碰撞惩罚（碰撞不可避免时的强信号）
        self.w_goal = 500.0           # 到达终点奖励
        self.w_progress = 1.5         # 进度奖励
        self.p_collision = -100.0     # 碰撞惩罚
        self.c_step = -0.01           # 每步惩罚
        
        # 惯性相关
        self.v_safe = 1.8
        self.v_min = 0.5
        self.p_overspeed = -0.15
        self.p_stall = -1.5           # 完全静止惩罚
        self.p_low_speed = -0.8       # 低速区惩罚系数
        self.p_drift = -0.1
        
        # 边界危险惩罚
        self.d_danger_boundary = 2.5
        self.p_danger_boundary = -2.0
        
        # 障碍物危险惩罚
        self.d_danger_obstacle = 2.0
        self.p_danger_obstacle = -3.0
        self.num_danger_obstacles = 3
        
        # 【v30 增强】预测碰撞参数 - 更细粒度的时间步
        self.prediction_look_ahead = 2.5
        self.p_prediction = -1.0
        self.prediction_time_steps = [0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 2.5]  # 【v30新增】更细的时间步
        
        # 【v30 新增】紧急碰撞惩罚
        self.emergency_time_threshold = 0.5   # 碰撞时间<0.5s视为紧急
        self.p_emergency = -3.0               # 紧急惩罚（比普通prediction更重）
        
        # 主动避障奖励
        self.w_avoidance = 5.0
        self.d_avoidance_trigger = 2.5
        self.d_avoidance_safe = 3.0
        
        # 居中奖励
        self.w_centering = 0.15
        self.w_centering_reduced = 0.05
        
        # 持续偏移惩罚参数
        self.p_persistent_offset = -1.0
        self.offset_history_len = 25
        self.offset_history_check_start = 10
        self.offset_threshold = 0.15
        
        # 【v30 新增】避障惯性机制
        # 刚完成避障后，短暂降低居中引力，让动作更连贯
        self.avoidance_inertia_steps = 15     # 避障后惯性持续步数
        self.avoidance_inertia_decay = 0.9    # 惯性衰减率
        self._avoidance_inertia = 0.0         # 当前惯性强度 (0-1)
        self._last_avoiding = False           # 上一步是否在避障
        
        # 躲避方向多样性追踪
        self.avoidance_direction_history = []
        self.max_avoidance_history = 20
        self.w_direction_diversity = 0.3

        # 用于进度追踪的变量
        self.last_progress = 0.0
        
        # 【v15】用于主动避障奖励追踪
        self.tracked_obstacles = set()
        
        # 【v26】横向偏移历史（用于检测固定轨迹）
        self.offset_history_y = []  # Y方向偏移历史 (相对于中心)
        self.offset_history_z = []  # Z方向偏移历史
        
        # 【调试】奖励组件分解记录
        self._reward_components = {}
        self._debug_info = {}

        # 初始化通道
        if tunnel_config is None:
            # 如果没有提供配置，则使用默认值
            tunnel_config = TunnelConfig(radius=20.0, length=500.0)

        self.tunnel_config = tunnel_config
        self.tunnel = CylinderTunnel(self.tunnel_config)
        
        # 【新增】环境随机化相关变量
        self._rng = np.random.default_rng()

        # 初始化点云采样器
        self.point_cloud_config = point_cloud_config or PointCloudConfig()
        self.point_cloud_sampler = PointCloudSampler(self.point_cloud_config)

        # 初始化潜艇物理模型
        self.vehicle = torpedo(controlSystem="stepInput", r_rpm=0)

        # 定义动作空间: [pitch, yaw, thrust]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0]), # 注意第三个也是 -1
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )

        # 定义观测空间
        self._setup_observation_space()

        # 状态变量
        self.eta = None  # [x, y, z, phi, theta, psi]
        self.nu = None  # [u, v, w, p, q, r]
        self.u_actual = None
        self.u_control = None
        self.steps = 0
        self.last_distances = None

        # 重置上一步动作命令
        self.last_pitch_cmd = 0.0
        self.last_yaw_cmd = 0.0

        # 用于渲染
        self._renderer = None
        self._logic_stage = None
        self._submarine_actor = None
        self._tunnel_actor = None

    def _setup_observation_space(self):
        """设置观测空间"""
        # 基础状态维度 (v30 - 加入连续障碍物序列信息)
        # - 相对位置 (到通道中心): 2 (y, z offset)
        # - 速度: 6 (nu)
        # - 姿态: 3 (roll, pitch, yaw)
        # - 到边界距离: 1
        # - 进度: 1
        # - 最近3个障碍物信息: 3 * 3 = 9 (每个障碍物: y_dir, z_dir, urgency)
        # - 【v30新增】前方障碍物序列: 3 * 4 = 12 (接下来3段通道的障碍物密度和方向)
        # - 【v30新增】避障惯性状态: 1 (当前是否在避障惯性期)
        self.num_obstacle_info = 3  # 追踪最近3个障碍物
        self.num_sequence_segments = 3  # 【v30】前方分成3段分析
        self.base_obs_dim = 2 + 6 + 3 + 1 + 1 + self.num_obstacle_info * 3 + self.num_sequence_segments * 4 + 1  # = 35

        # 混合 LSTM 模式：使用 Dict 观测空间
        point_cloud_dim = self.point_cloud_sampler.output_dim  # num_points * 3

        self.observation_space = spaces.Dict({
            "state": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.base_obs_dim,),
                dtype=np.float32
            ),
            "point_cloud_seq": spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.point_cloud_history_len, point_cloud_dim),
                dtype=np.float32
            ),
        })
        print(f"[SubmarineEnv] 混合 LSTM 模式 (v30 - 点云 + 障碍物序列):")
        print(f"  - 状态维度: {self.base_obs_dim} (含{self.num_obstacle_info}个障碍物方向 + {self.num_sequence_segments}段序列)")
        print(f"  - 点云序列: ({self.point_cloud_history_len}, {point_cloud_dim})")
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """重置环境"""
        super().reset(seed=seed)
        
        # 创建随机数生成器
        self._rng = np.random.default_rng(seed)
        
        # 【新增】应用环境随机化
        if self.tunnel_config.enable_randomization:
            # 获取随机化后的配置
            randomized_config = self.tunnel_config.get_randomized_copy(self._rng)
            # 重新创建隧道（使用随机化配置）
            self.tunnel = CylinderTunnel(randomized_config, seed=seed)
        else:
            # 每次 reset 都重新生成障碍物，防止过拟合
            self.tunnel.reset(seed=seed)

        # 获取生成位置
        spawn_pos = self.tunnel.get_submarine_spawn_position()

        # 初始化潜艇状态
        self.eta = np.zeros(6)
        self.eta[0:3] = spawn_pos
        self.eta[5] = 0.0  # 初始朝向 X 正方向

        self.nu = np.zeros(6)
        self.u_actual = np.zeros(self.vehicle.dimU)
        self.u_control = np.zeros(self.vehicle.dimU)

        self.steps = 0
        self.last_distances = None
        self.last_progress = 0.0  # 重置进度追踪
        self.tracked_obstacles = set()  # 【v15】重置障碍物追踪
        self._was_in_danger = False  # 【v15b】重置危险状态追踪
        
        # 【v26】重置偏移历史
        self.offset_history_y = []
        self.offset_history_z = []
        
        # 【v30】重置避障惯性
        self._avoidance_inertia = 0.0
        self._last_avoiding = False
        self.avoidance_direction_history = []

        # 重置点云历史缓冲区 (hybrid_lstm 模式)
        point_cloud_dim = self.point_cloud_sampler.output_dim
        self.point_cloud_history = np.zeros(
            (self.point_cloud_history_len, point_cloud_dim),
            dtype=np.float32
        )

        # 获取初始观测
        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        执行一步仿真

        Args:
            action: [pitch_cmd, yaw_cmd, thrust_cmd] 范围 [-1,1], [-1,1], [0,1]

        Returns:
            observation, reward, terminated, truncated, info
        """
        self.steps += 1

        # 解析动作
        pitch_cmd = float(action[0]) * self.max_angle  # stern angle
        yaw_cmd = float(action[1]) * self.max_angle  # rudder angle
        # 【关键修改】将 [-1, 1] 的网络输出映射到 [0, 1] 的物理推力
        # 这样网络更容易学习，因为 0 就在中间
        raw_thrust = float(action[2]) 
        thrust_ratio = (raw_thrust + 1.0) / 2.0  # 映射到 [0, 1]
        thrust_cmd = thrust_ratio * 1525  # RPM

        # 记录上一步命令用于平滑惩罚
        self.last_pitch_cmd = pitch_cmd
        self.last_yaw_cmd = yaw_cmd

        # 映射到控制输入 (与 submarine_actor.py 一致)
        # u_control = [delta_s1, delta_s2, delta_r1, delta_r2, n]
        self.u_control[0] = -yaw_cmd  # Top rudder
        self.u_control[1] = yaw_cmd  # Bottom rudder
        self.u_control[2] = -pitch_cmd  # Starboard stern
        self.u_control[3] = pitch_cmd  # Port stern
        self.u_control[4] = thrust_cmd

        # 执行物理仿真
        self.nu, self.u_actual = self.vehicle.dynamics(
            self.eta, self.nu, self.u_actual, self.u_control, self.dt
        )
        self.eta = attitudeEuler(self.eta, self.nu, self.dt)

        # 碰撞检测
        collision, collision_reason = self.tunnel.check_collision(
            self.eta[0:3],
            self.tunnel_config.submarine_radius
        )

        # 检查是否到达终点
        goal_reached = self.tunnel.is_goal_reached(self.eta[0:3])

        # 【v15b】先采样点云，用于观测和奖励计算
        # 这确保奖励函数基于点云而非"上帝视角"坐标
        R = Rzyx(self.eta[3], self.eta[4], self.eta[5])
        current_points = self.point_cloud_sampler.sample(self.eta[0:3], R, self.tunnel)
        
        # 计算奖励（传入点云数据）
        reward = self._compute_reward(collision, collision_reason, goal_reached, current_points)

        # 终止条件
        terminated = collision or goal_reached
        truncated = self.steps >= self.max_steps

        # 获取观测（复用已采样的点云）
        obs = self._get_observation_with_points(current_points)
        info = self._get_info()
        info['collision'] = collision
        info['collision_reason'] = collision_reason
        info['goal_reached'] = goal_reached
        
        # 【新增】将奖励组件放入 info，供回调使用（兼容 SubprocVecEnv）
        info['reward_components'] = self._reward_components.copy()

        return obs, reward, terminated, truncated, info

    def _get_observation(self):
        """获取观测值 (重新采样点云)"""
        # 采样新的点云
        R = Rzyx(self.eta[3], self.eta[4], self.eta[5])
        points = self.point_cloud_sampler.sample(self.eta[0:3], R, self.tunnel)
        return self._get_observation_with_points(points)
    
    def _get_observation_with_points(self, points: np.ndarray):
        """
        获取观测值 (v30 - 增加连续障碍物序列信息)
        
        Args:
            points: 已采样的点云数据 (num_points, 3)
        """
        cfg = self.tunnel_config

        # 相对于通道中心的偏移 (归一化)
        rel_y = (self.eta[1] - self.tunnel.axis_y) / cfg.radius
        rel_z = (self.eta[2] - self.tunnel.axis_z) / cfg.radius

        # 速度 (归一化)
        nu_normalized = self.nu / np.array([5.0, 2.0, 2.0, 1.0, 1.0, 1.0])

        # 姿态 (已经是弧度)
        orientation = self.eta[3:6]

        # 到边界距离 (归一化)
        dist_to_boundary = self.tunnel.get_distance_to_boundary(self.eta[0:3])
        dist_normalized = dist_to_boundary / cfg.radius

        # 进度
        progress = self.tunnel.get_progress(self.eta[0:3])

        # 【v14b】从点云计算危险方向（不依赖障碍物精确位置）
        danger_info = self._get_danger_direction_from_pointcloud(points)
        
        # 【v30新增】前方障碍物序列信息
        sequence_info = self._get_obstacle_sequence_info(points)
        
        # 【v30新增】避障惯性状态
        avoidance_inertia = np.array([self._avoidance_inertia], dtype=np.float32)

        # 组合基础状态观测 (v30 - 增加序列信息和惯性状态)
        base_obs = np.concatenate([
            [rel_y, rel_z],
            nu_normalized,
            orientation,
            [dist_normalized],
            [progress],
            danger_info,      # 最近障碍物方向 (9维)
            sequence_info,    # 【v30】前方序列信息 (12维)
            avoidance_inertia # 【v30】避障惯性状态 (1维)
        ]).astype(np.float32)

        # 扁平化点云用于LSTM
        current_point_cloud = self.point_cloud_sampler.get_flattened_points(points)

        # 更新历史缓冲区 (滑动窗口)
        self.point_cloud_history = np.roll(self.point_cloud_history, shift=-1, axis=0)
        self.point_cloud_history[-1] = current_point_cloud

        return {
            "state": base_obs,
            "point_cloud_seq": self.point_cloud_history.copy(),
        }
    
    def _get_danger_direction_from_pointcloud(self, points: np.ndarray) -> np.ndarray:
        """
        从点云计算危险方向 (v14b - 可部署版本)
        
        这是真实部署时也能使用的方法！
        不依赖障碍物的精确位置，只使用点云数据。
        
        方法：
        1. 找到点云中最近的N个点（前方的）
        2. 计算这些点的加权平均方向
        3. 作为"危险方向"信号
        
        Args:
            points: (num_points, 3) 点云坐标（机体坐标系）
            
        返回: (num_obstacle_info * 3,) 数组
        每组包含: [y_direction, z_direction, urgency]
        - y_direction: 最近点在Y方向的位置 (-1到1，负=左，正=右)
        - z_direction: 最近点在Z方向的位置 (-1到1，负=下，正=上)
        - urgency: 紧迫度 (0=安全, 1=非常近)
        """
        cfg = self.tunnel_config
        result = []
        
        # 过滤：只保留前方的点（X > 0，表示在潜艇前方）
        forward_mask = points[:, 0] > 0.5  # X > 0.5m 才算前方
        forward_points = points[forward_mask]
        
        if len(forward_points) == 0:
            # 没有前方的点，返回安全信号
            return np.array([0.0, 0.0, 0.0] * self.num_obstacle_info, dtype=np.float32)
        
        # 计算到每个点的距离
        distances = np.linalg.norm(forward_points, axis=1)
        
        # 按距离排序，取最近的几组点
        sorted_indices = np.argsort(distances)
        
        # 每组取若干个最近的点，计算其平均方向
        points_per_group = 20  # 每组取20个点做平均（更稳定）
        
        for i in range(self.num_obstacle_info):
            start_idx = i * points_per_group
            end_idx = min(start_idx + points_per_group, len(sorted_indices))
            
            if start_idx < len(sorted_indices):
                # 取这组点
                group_indices = sorted_indices[start_idx:end_idx]
                group_points = forward_points[group_indices]
                group_distances = distances[group_indices]
                
                # 距离加权平均（越近的点权重越大）
                weights = 1.0 / (group_distances + 0.1)  # 避免除零
                weights = weights / weights.sum()
                
                # 计算加权平均方向
                avg_point = np.average(group_points, axis=0, weights=weights)
                avg_distance = np.average(group_distances, weights=weights)
                
                # Y方向：归一化到 [-1, 1]
                y_dir = np.clip(avg_point[1] / cfg.radius, -1.0, 1.0)
                
                # Z方向：归一化到 [-1, 1]
                z_dir = np.clip(avg_point[2] / cfg.radius, -1.0, 1.0)
                
                # 紧迫度：距离越近越接近1（5m范围内开始预警，提前反应）
                urgency = np.clip(1.0 - avg_distance / 5.0, 0.0, 1.0)
                
                result.extend([y_dir, z_dir, urgency])
            else:
                # 没有足够的点，填充安全值
                result.extend([0.0, 0.0, 0.0])
        
        return np.array(result, dtype=np.float32)
    
    def _get_obstacle_sequence_info(self, points: np.ndarray) -> np.ndarray:
        """
        【v30新增】获取前方障碍物序列信息
        
        将前方通道分成多段，分析每段的障碍物分布，帮助Agent做更远的规划。
        
        Args:
            points: 机体坐标系下的点云 (N, 3)
            
        Returns:
            sequence_info: (num_sequence_segments * 4,) 数组
            每段包含: [density, avg_y, avg_z, urgency]
            - density: 该段障碍物密度 (0-1)
            - avg_y: 障碍物平均Y位置 (-1到1)
            - avg_z: 障碍物平均Z位置 (-1到1)
            - urgency: 该段紧迫度 (0-1，越近越紧迫)
        """
        cfg = self.tunnel_config
        result = []
        
        # 分段范围：0-4m, 4-8m, 8-12m
        segment_ranges = [(0.5, 4.0), (4.0, 8.0), (8.0, 12.0)]
        collision_width = 2.0  # 碰撞通道宽度
        
        for seg_idx, (x_min, x_max) in enumerate(segment_ranges):
            # 筛选该段的点
            segment_mask = (points[:, 0] > x_min) & (points[:, 0] <= x_max)
            segment_points = points[segment_mask]
            
            if len(segment_points) == 0:
                # 该段无障碍物
                result.extend([0.0, 0.0, 0.0, 0.0])
                continue
            
            # 检查正前方碰撞通道内的点
            lateral_dist = np.sqrt(segment_points[:, 1]**2 + segment_points[:, 2]**2)
            in_path = lateral_dist < collision_width
            blocking_points = segment_points[in_path]
            
            if len(blocking_points) == 0:
                # 该段正前方无障碍
                result.extend([0.0, 0.0, 0.0, 0.0])
                continue
            
            # 计算密度（归一化：假设最多20个点算满）
            density = min(len(blocking_points) / 20.0, 1.0)
            
            # 计算平均位置
            avg_y = np.clip(np.mean(blocking_points[:, 1]) / cfg.radius, -1.0, 1.0)
            avg_z = np.clip(np.mean(blocking_points[:, 2]) / cfg.radius, -1.0, 1.0)
            
            # 紧迫度：段越靠前越紧迫（第一段=1.0，第三段=0.33）
            urgency = 1.0 - seg_idx / self.num_sequence_segments
            
            result.extend([density, avg_y, avg_z, urgency])
        
        return np.array(result, dtype=np.float32)
    
    def _compute_path_clearance_reward(self, points: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        【v22 综合优化版】路径通畅度奖励
        
        改进：
        1. 基于舵角/动作方向判断躲避意图（而非只看横向速度）
        2. 移除 proximity_penalty（与 prediction_penalty 重复）
        3. 降低惩罚力度，提高正向奖励触发率
        
        Args:
            points: 机体坐标系下的点云 (N, 3)
            
        Returns:
            reward: 路径通畅度奖励/惩罚
            best_direction: 最佳规避方向 [y, z]（用于调试）
        """
        check_distance = 8.0   # v22.1: 从6m增加到8m，给更多反应时间
        collision_width = 1.8  # 碰撞通道宽度
        
        # 只看前方的点（X > 0.5m 且 < check_distance）
        forward_mask = (points[:, 0] > 0.5) & (points[:, 0] < check_distance)
        forward_points = points[forward_mask]
        
        if len(forward_points) == 0:
            return 0.0, np.array([0.0, 0.0])
        
        # 检查正前方碰撞通道是否有障碍
        lateral_dist = np.sqrt(forward_points[:, 1]**2 + forward_points[:, 2]**2)
        in_path = lateral_dist < collision_width
        path_blocked = np.any(in_path)
        
        if not path_blocked:
            # 正前方通道畅通 → 小奖励鼓励保持畅通
            return 0.05, np.array([0.0, 0.0])
        
        # === 正前方有障碍，需要评估侧向空隙 ===
        blocking_points = forward_points[in_path]
        
        # 计算权重：X越小（越近）权重越大
        weights = 1.0 / (blocking_points[:, 0] + 0.1)
        weights = weights / weights.sum()
        
        # 加权平均障碍物位置
        avg_obstacle_y = np.average(blocking_points[:, 1], weights=weights)
        avg_obstacle_z = np.average(blocking_points[:, 2], weights=weights)
        
        # 最佳规避方向
        best_avoid_y = -np.sign(avg_obstacle_y) if abs(avg_obstacle_y) > 0.3 else 0.0
        best_avoid_z = -np.sign(avg_obstacle_z) if abs(avg_obstacle_z) > 0.3 else 0.0
        
        if best_avoid_y == 0 and best_avoid_z == 0:
            left_clear = np.sum(forward_points[:, 1] < -collision_width)
            right_clear = np.sum(forward_points[:, 1] > collision_width)
            best_avoid_y = -1.0 if left_clear > right_clear else 1.0
        
        best_direction = np.array([best_avoid_y, best_avoid_z])
        dir_norm = np.linalg.norm(best_direction)
        if dir_norm > 0:
            best_direction = best_direction / dir_norm
        
        # 计算紧迫度
        min_obstacle_x = np.min(blocking_points[:, 0])
        urgency = 1.0 - (min_obstacle_x / check_distance)
        urgency = np.clip(urgency, 0.0, 1.0)
        
        # ============ 【v22 核心改进】基于舵角判断躲避意图 ============
        # 不仅看横向速度，还看舵角方向（舵角响应比速度快）
        
        # 1. 横向速度一致性
        current_lateral = np.array([self.nu[1], self.nu[2]])
        velocity_alignment = np.dot(current_lateral, best_direction)
        
        # 2. 舵角一致性（新增）
        # last_yaw_cmd > 0 表示向右转，best_avoid_y > 0 也表示向右躲
        yaw_alignment = self.last_yaw_cmd * best_direction[0]  # Y方向
        pitch_alignment = self.last_pitch_cmd * best_direction[1]  # Z方向
        action_alignment = yaw_alignment + pitch_alignment
        
        # 综合评分：速度(0.4权重) + 动作(0.6权重)
        # 动作权重更高，因为动作是意图的直接体现
        combined_alignment = 0.4 * velocity_alignment + 0.6 * action_alignment
        
        # ============ 奖励计算（v23调整版）============
        # 移除对"错误方向"的强惩罚，主要奖励正确行为，减少对可能合理策略的干扰
        if combined_alignment > 0.05:
            # 正在向正确方向躲避 → 正向奖励
            # 奖励力度适中
            reward = 0.8 * urgency * min(combined_alignment, 1.0)
        elif combined_alignment < -0.1:
            # 向明显错误方向移动 (只在非常错误时惩罚, 且力度减半)
            reward = -0.4 * urgency * min(-combined_alignment, 1.0)
        else:
            # 中性/不确定 -> 无奖励无惩罚
            reward = 0.0
        
        # 【移除 proximity_penalty】因为与 prediction_penalty 重复
        # proximity_penalty = 0  
        
        return reward, best_direction

    def _get_info(self) -> Dict[str, Any]:
        """获取附加信息"""
        return {
            'position': self.eta[0:3].copy(),
            'velocity': self.nu[0:3].copy(),
            'orientation': self.eta[3:6].copy(),
            'progress': self.tunnel.get_progress(self.eta[0:3]),
            'distance_to_boundary': self.tunnel.get_distance_to_boundary(self.eta[0:3]),
            'steps': self.steps,
            'forward_speed': self.nu[0],
        }

    def _predict_collision(self, look_ahead_time: float = 2.5) -> Tuple[bool, float]:
        """
        【v30增强】预测未来轨迹是否会碰撞 - 更细粒度的时间步
        
        基于当前位置和速度，线性预测未来位置，检查是否会碰撞
        
        v30改进：
        - 使用更细的时间步（从0.2s开始），更早检测到短距离碰撞
        - 时间步可配置
        
        Args:
            look_ahead_time: 最大预测时间（秒）
        
        Returns:
            will_collide: 是否预测会碰撞
            time_to_collision: 预测碰撞时间（秒），无碰撞返回 inf
        """
        pos = self.eta[0:3].copy()
        vel = self.nu[0:3].copy()
        
        # 【v30改进】使用更细粒度的时间步
        # 原时间步: [0.5, 1.0, 1.5, 2.0] - 无法检测<0.5s的碰撞
        # 新时间步: [0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 2.5] - 能检测0.2s的碰撞
        for t in self.prediction_time_steps:
            if t > look_ahead_time:
                break
            
            # 预测未来位置（简单线性）
            future_pos = pos + vel * t
            
            # 检查是否碰撞
            collision, _ = self.tunnel.check_collision(
                future_pos, 
                self.tunnel_config.submarine_radius
            )
            
            if collision:
                return True, t
        
        return False, float('inf')

    def _compute_reward(self, collision: bool, collision_reason: str, goal_reached: bool, 
                         points: np.ndarray = None) -> float:
        """
        计算奖励 (v21h - 预测碰撞版 + 调试记录)
        
        核心改进：
        1. 新增预测碰撞惩罚：预测到未来会碰撞时提前惩罚
        2. 降低障碍物警戒区惩罚：防止躲避过激
        3. 平衡各项奖励值
        4. 【新增】记录各项奖励组件用于调试
        """
        # 初始化奖励组件记录 (v30 - 增加紧急碰撞和避障惯性)
        self._reward_components = {
            'progress': 0.0,
            'speed_penalty': 0.0,
            'path_clearance': 0.0,
            'proximity_penalty': 0.0,
            'boundary_penalty': 0.0,
            'boundary_velocity_penalty': 0.0,
            'prediction_penalty': 0.0,
            'emergency_penalty': 0.0,              # 【v30新增】
            'persistent_offset_penalty': 0.0,
            'centering_reward': 0.0,
            'collision': 0.0,
            'goal': 0.0,
            'total': 0.0
        }
        
        # 初始化调试信息
        self._debug_info = {
            'forward_speed': self.nu[0],
            'progress': 0.0,
            'dist_to_boundary': 0.0,
            'path_blocked': False,
            'urgency': 0.0,
            'best_avoid_direction': [0.0, 0.0],
            'will_collide': False,
            'time_to_collision': float('inf')
        }
        
        # ========== 终止条件 ==========
        
        if collision:
            self._reward_components['collision'] = self.p_collision
            self._reward_components['total'] = self.p_collision
            self._debug_info['collision_reason'] = collision_reason
            return self.p_collision  # -100
        
        if goal_reached:
            # 到达终点：基础奖励 + 时间效率奖励
            time_bonus = max(0, (self.max_steps - self.steps) / self.max_steps) * 100.0
            total_goal = self.w_goal + time_bonus
            self._reward_components['goal'] = total_goal
            self._reward_components['total'] = total_goal
            return total_goal  # 500 + bonus
        
        # ========== 每步奖励计算 ==========
        reward = 0.0
        
        # 获取基本信息
        progress = self.tunnel.get_progress(self.eta[0:3])
        forward_speed = self.nu[0]  # 前进速度 (u)
        self._debug_info['progress'] = progress
        self._debug_info['forward_speed'] = forward_speed
        
        # 1. 进度奖励（v23增强版 - 使用参数）
        progress_delta = progress - self.last_progress
        progress_reward = self.w_progress * progress_delta * 100  # 使用 self.w_progress
        reward += progress_reward
        self._reward_components['progress'] = progress_reward
        self.last_progress = progress
        
        # 2. 【v28 增强】速度惩罚 - 更强的低速/停滞惩罚
        # 问题诊断：agent学会"前方有障碍就停下来"，因为低速惩罚太弱
        # 解决方案：大幅增强停滞惩罚，使"停下来"比"前进避障"更差
        speed_penalty = 0.0
        if forward_speed < 0.1:
            # 完全静止：重惩罚
            speed_penalty = self.p_stall  # -1.5
        elif forward_speed < self.v_min:  # 使用参数化的最低速度 (0.5)
            # 低速区（0.1-0.5m/s）：渐进惩罚，更强
            low_speed_ratio = (self.v_min - forward_speed) / (self.v_min - 0.1)
            speed_penalty = self.p_low_speed * low_speed_ratio  # -0.8 * ratio
        elif forward_speed > self.v_safe:  # 超速阈值1.8
            speed_penalty = self.p_overspeed * (forward_speed - self.v_safe)  # 超速惩罚
        # 注意：0.5 < speed < 1.8 区间内不惩罚，给agent更多自由
        reward += speed_penalty
        self._reward_components['speed_penalty'] = speed_penalty
        
        # 3. 【v22 调整】路径通畅度奖励（降低权重避免过度惩罚）
        path_reward = 0.0
        if points is not None and len(points) > 0:
            raw_path_reward, best_direction = self._compute_path_clearance_reward(points)
            path_reward = raw_path_reward * 1.2  # 从2.0降低到1.2
            reward += path_reward
            self._debug_info['best_avoid_direction'] = best_direction.tolist()
        self._reward_components['path_clearance'] = path_reward
        

        # 3.5. 【v23恢复】障碍物距离惩罚 (基于真实距离)
        # 即使没有碰撞，只要进入障碍物危险区与其过近，就给予惩罚
        proximity_penalty = 0.0
        nearby_obstacles = self.tunnel.get_nearby_obstacles(
            self.eta[0:3], 
            max_distance=self.d_danger_obstacle, 
            max_count=1
        )
        
        if len(nearby_obstacles) > 0:
            dist, _, _, _ = nearby_obstacles[0]
            # 距离小于阈值 (get_nearby_obstacles已筛选，但再次确认更安全)
            if dist < self.d_danger_obstacle:
                # 越近惩罚越大 (二次曲线)
                danger_ratio = 1.0 - (dist / self.d_danger_obstacle)
                proximity_penalty = self.p_danger_obstacle * (danger_ratio ** 2)
                reward += proximity_penalty
                
                # 记录最近障碍物距离用于调试
                self._debug_info['nearest_obstacle_dist'] = dist
                
        self._reward_components['proximity_penalty'] = proximity_penalty

        # 4. 【v21d 新增】边界距离惩罚
        cfg = self.tunnel_config
        dist_from_center = np.sqrt(
            (self.eta[1] - self.tunnel.axis_y) ** 2 + 
            (self.eta[2] - self.tunnel.axis_z) ** 2
        )
        dist_to_boundary = cfg.radius - dist_from_center
        self._debug_info['dist_to_boundary'] = dist_to_boundary
        
        # 边界预警距离
        boundary_warning_dist = max(cfg.radius * 0.4, cfg.submarine_radius + 1.0)
        
        boundary_penalty = 0.0
        boundary_velocity_penalty = 0.0
        if dist_to_boundary < boundary_warning_dist:
            # 越靠近边界惩罚越大（二次曲线）
            boundary_ratio = 1.0 - (dist_to_boundary / boundary_warning_dist)
            boundary_penalty = self.p_danger_boundary * (boundary_ratio ** 2)  # 使用参数 (max -0.8)
            reward += boundary_penalty
            
            # 如果正在向边界移动，额外惩罚
            center_dir = np.array([0, self.tunnel.axis_y - self.eta[1], self.tunnel.axis_z - self.eta[2]])
            center_dist = np.linalg.norm(center_dir[1:3])
            if center_dist > 0.1:
                center_dir_normalized = center_dir[1:3] / center_dist
                lateral_vel = np.array([self.nu[1], self.nu[2]])
                toward_center = np.dot(lateral_vel, center_dir_normalized)
                if toward_center < -0.1:
                    boundary_velocity_penalty = -0.5 * boundary_ratio * min(-toward_center, 1.0)
                    reward += boundary_velocity_penalty
        
        self._reward_components['boundary_penalty'] = boundary_penalty
        self._reward_components['boundary_velocity_penalty'] = boundary_velocity_penalty
        
        # 5. 【v30 增强】预测碰撞惩罚 + 紧急碰撞惩罚
        # v30改进：更细粒度的时间步 + 紧急情况额外惩罚
        will_collide, time_to_collision = self._predict_collision(
            look_ahead_time=self.prediction_look_ahead
        )
        self._debug_info['will_collide'] = will_collide
        self._debug_info['time_to_collision'] = time_to_collision
        
        prediction_penalty = 0.0
        emergency_penalty = 0.0
        
        if will_collide:
            # 基础预测惩罚：时间越近惩罚越大
            prediction_penalty = self.p_prediction * (
                self.prediction_look_ahead - time_to_collision
            ) / self.prediction_look_ahead
            reward += prediction_penalty
            
            # 【v30新增】紧急碰撞惩罚：当碰撞时间 < 0.5s 时额外惩罚
            # 目的：让Agent学会"不要走到这一步"，提前避障
            if time_to_collision < self.emergency_time_threshold:
                # 时间越短惩罚越重（0s=-3.0，0.5s=0）
                emergency_penalty = self.p_emergency * (
                    1.0 - time_to_collision / self.emergency_time_threshold
                )
                reward += emergency_penalty
                self._debug_info['emergency'] = True
        
        self._reward_components['prediction_penalty'] = prediction_penalty
        self._reward_components['emergency_penalty'] = emergency_penalty
        
        # 【v30新增】更新避障惯性状态
        # 如果当前正在躲避障碍物，设置惯性=1.0
        currently_avoiding = will_collide or (path_reward > 0.1)  # 正在避障
        if currently_avoiding and not self._last_avoiding:
            # 刚刚开始避障，设置惯性
            self._avoidance_inertia = 1.0
        elif not currently_avoiding and self._avoidance_inertia > 0:
            # 不再避障，惯性衰减
            self._avoidance_inertia *= self.avoidance_inertia_decay
            if self._avoidance_inertia < 0.1:
                self._avoidance_inertia = 0.0
        
        self._last_avoiding = currently_avoiding
        self._debug_info['avoidance_inertia'] = self._avoidance_inertia
        
        # 6. 【v26 新增】持续偏移惩罚 - 防止固定轨迹避障
        # 追踪横向位置历史，如果持续偏向某一侧则惩罚
        rel_y = (self.eta[1] - self.tunnel.axis_y) / cfg.radius  # 归一化Y偏移 [-1, 1]
        rel_z = (self.eta[2] - self.tunnel.axis_z) / cfg.radius  # 归一化Z偏移 [-1, 1]
        
        # 记录偏移历史
        self.offset_history_y.append(rel_y)
        self.offset_history_z.append(rel_z)
        
        # 保持历史长度
        if len(self.offset_history_y) > self.offset_history_len:
            self.offset_history_y.pop(0)
            self.offset_history_z.pop(0)
        
        # 【v28 增强】计算持续偏移惩罚
        persistent_offset_penalty = 0.0
        check_start = getattr(self, 'offset_history_check_start', 15)  # v28: 15步开始
        if len(self.offset_history_y) >= check_start:
            # 计算平均偏移
            avg_offset_y = np.mean(self.offset_history_y)
            avg_offset_z = np.mean(self.offset_history_z)
            
            # 检查是否持续偏向某一侧
            # 如果平均偏移超过阈值，说明agent在固定走某条路线
            offset_magnitude = np.sqrt(avg_offset_y**2 + avg_offset_z**2)
            
            if offset_magnitude > self.offset_threshold:
                # 【v28】使用二次曲线增强惩罚：偏移越大惩罚增长越快
                excess = offset_magnitude - self.offset_threshold
                persistent_offset_penalty = self.p_persistent_offset * (excess ** 1.5)  # 1.5次方增强
                reward += persistent_offset_penalty
                self._debug_info['persistent_offset'] = offset_magnitude
                self._debug_info['avg_offset_y'] = avg_offset_y
                self._debug_info['avg_offset_z'] = avg_offset_z
        
        self._reward_components['persistent_offset_penalty'] = persistent_offset_penalty
        
        # 7. 【v29 改进】居中奖励 - 始终提供，只是强度不同
        # 核心改进：无论前方是否有障碍物，都给予居中奖励信号
        # 这确保 Agent 始终收到"回到中心"的激励
        centering_reward = 0.0
        
        # 计算基础居中奖励
        base_centering = 1.0 - min(dist_from_center / cfg.radius, 1.0)
        
        # 检测前方是否有近距离障碍物（放宽条件）
        front_clear = True
        if points is not None and len(points) > 0:
            forward_mask = (points[:, 0] > 0.5) & (points[:, 0] < 3.0)  # 【v29】进一步缩短到3m
            if np.any(forward_mask):
                forward_points = points[forward_mask]
                lateral_dist = np.sqrt(forward_points[:, 1]**2 + forward_points[:, 2]**2)
                front_clear = not np.any(lateral_dist < 1.2)  # 【v29】进一步缩窄到1.2m
        
        if front_clear:
            # 前方畅通：给予满额居中奖励
            centering_reward = self.w_centering * base_centering
        else:
            # 前方有障碍物：仍给予降低的居中奖励
            centering_reward = self.w_centering_reduced * base_centering
        
        # 【v30新增】避障惯性机制：刚完成避障后，降低居中奖励
        # 目的：让连续避障动作更连贯，不要刚躲完就被"拉回中心"
        if self._avoidance_inertia > 0:
            # 惯性期内，居中奖励减弱（保留原始值的30%）
            inertia_factor = 0.3 + 0.7 * (1.0 - self._avoidance_inertia)
            centering_reward *= inertia_factor
            self._debug_info['centering_reduced_by_inertia'] = True
        
        reward += centering_reward
        
        self._reward_components['centering_reward'] = centering_reward
        self._debug_info['front_clear'] = front_clear
        self._debug_info['rel_position'] = [rel_y, rel_z]
        
        # 记录总奖励
        self._reward_components['total'] = reward
        
        return reward
    
    def get_reward_components(self) -> Dict[str, float]:
        """获取上一步的奖励组件分解（用于调试）"""
        return self._reward_components.copy()
    
    def get_debug_info(self) -> Dict[str, Any]:
        """获取上一步的调试信息（用于调试）"""
        return self._debug_info.copy()

    def render(self):
        """渲染环境"""
        if self.render_mode != "human":
            return None

        from src.gameplay.tunnel_actor import TunnelActor

        if self._renderer is None:
            # Init pygame
            pygame.init()
            self.width, self.height = 1280, 900
            pygame.display.set_mode((self.width, self.height), DOUBLEBUF | OPENGL | RESIZABLE)
            pygame.display.set_caption("Submarine Env (Human Mode)")

            self._renderer = Renderer(self.width, self.height)
            self._logic_stage = LogicStage()

            # Create actors
            self._submarine_actor = SubmarineActor("HeroSub")
            self._logic_stage.add_actor(self._submarine_actor)

            self._tunnel_actor = TunnelActor(config=self.tunnel_config)
            self._logic_stage.add_actor(self._tunnel_actor)

            # Sync obstacles
            for i, obs in enumerate(self.tunnel.obstacles):
                obs_actor = ObstacleActor(
                    name=f"Obstacle_{i}",
                    x=obs.position[0],
                    y=obs.position[1],
                    z=obs.position[2],
                    radius=obs.radius
                )
                self._logic_stage.add_actor(obs_actor)

            # Set renderer tunnel info
            self._renderer.tunnel = self.tunnel
            self._renderer.tunnel_config = self.tunnel_config

        # Sync state
        self._submarine_actor.eta = self.eta
        self._submarine_actor.nu = self.nu
        self._submarine_actor.u_actual = self.u_actual
        self._submarine_actor._sync_state()

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
            elif event.type == pygame.VIDEORESIZE:
                self.width, self.height = event.w, event.h
                pygame.display.set_mode((self.width, self.height), DOUBLEBUF | OPENGL | RESIZABLE)
                self._renderer.resize(self.width, self.height)

        self._renderer.render(self._logic_stage)

    def close(self):
        """关闭环境"""
        if self._renderer is not None:
            pygame.quit()
            self._renderer = None

    def get_tunnel(self) -> CylinderTunnel:
        """获取通道对象 (用于外部可视化)"""
        return self.tunnel

    def get_state(self) -> Dict[str, np.ndarray]:
        """获取完整状态 (用于外部可视化)"""
        return {
            'eta': self.eta.copy(),
            'nu': self.nu.copy(),
            'u_actual': self.u_actual.copy(),
            'u_control': self.u_control.copy(),
        }


# 注册环境
def make_submarine_env(**kwargs):
    """创建环境的工厂函数"""
    return SubmarineEnv(**kwargs)


if __name__ == "__main__":
    # 测试环境
    print("Testing SubmarineEnv...")

    env = SubmarineEnv()
    obs, info = env.reset(seed=42)

    print(f"Observation keys: {obs.keys()}")
    print(f"State shape: {obs['state'].shape}")
    print(f"Point cloud sequence shape: {obs['point_cloud_seq'].shape}")
    print(f"Action space: {env.action_space}")
    print(f"Initial position: {info['position']}")
    print(f"Initial progress: {info['progress']:.2%}")

    # 运行几步
    total_reward = 0
    for i in range(100):
        action = env.action_space.sample()
        action[2] = 0.5  # 固定推力
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if (i + 1) % 20 == 0:
            print(f"Step {i + 1}: pos={info['position']}, progress={info['progress']:.2%}, reward={reward:.2f}")

        if terminated or truncated:
            print(f"Episode ended: collision={info.get('collision')}, goal={info.get('goal_reached')}")
            break

    print(f"Total reward: {total_reward:.2f}")
    env.close()
    print("Test complete!")