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
                 max_steps: int = 6000,
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

        # 奖励函数参数 (v9 - 简化聚焦)
        # 核心思想：只关注 到达终点 + 避障 + 惯性控制
        self.w_goal = 200.0           # 到达终点奖励
        self.w_progress = 10.0        # 进度奖励 (每1%进度 = +10)
        self.p_collision = -50.0      # 碰撞惩罚 (终止条件)
        self.c_step = -0.05           # 每步小惩罚 (鼓励效率)
        
        # 惯性相关 - 解决物理模型高速左偏问题
        self.v_safe = 0.8             # 安全速度阈值 (m/s) - 再降低让潜艇更慢
        self.p_overspeed = -0.5       # 超速惩罚 (速度越快惩罚越大)
        self.p_drift = -0.3           # 横向漂移惩罚
        
        # 障碍物感知 - 渐进式危险警告
        self.d_danger = 3.0           # 危险距离阈值 (米) - 增大提前预警
        self.p_danger = -0.2          # 危险区惩罚 (渐进式)

        # 用于进度追踪的变量
        self.last_progress = 0.0

        # 初始化通道
        if tunnel_config is None:
            # 如果没有提供配置，则使用默认值
            tunnel_config = TunnelConfig(radius=20.0, length=500.0)

        self.tunnel_config = tunnel_config
        self.tunnel = CylinderTunnel(self.tunnel_config)

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
        # 基础状态维度
        # - 相对位置 (到通道中心): 2 (y, z offset)
        # - 速度: 6 (nu)
        # - 姿态: 3 (roll, pitch, yaw)
        # - 到边界距离: 1
        # - 进度: 1
        # - 【新增】最近障碍物信息: 3个障碍物 x 4 (距离 + xyz方向) = 12
        self.num_nearest_obstacles = 3  # 用于观测的最近障碍物数量
        self.base_obs_dim = 2 + 6 + 3 + 1 + 1 + self.num_nearest_obstacles * 4  # = 25

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
        print(f"[SubmarineEnv] 混合 LSTM 模式 (v2):")
        print(f"  - 状态维度: {self.base_obs_dim} (含{self.num_nearest_obstacles}个最近障碍物信息)")
        print(f"  - 点云序列: ({self.point_cloud_history_len}, {point_cloud_dim})")
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """重置环境"""
        super().reset(seed=seed)

        # 每次 reset 都重新生成障碍物，防止过拟合
        # 如果传入 seed 则使用该 seed，否则使用 None (随机)
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

        # 计算奖励
        reward = self._compute_reward(collision, collision_reason, goal_reached)

        # 终止条件
        terminated = collision or goal_reached
        truncated = self.steps >= self.max_steps

        # 获取观测和信息
        obs = self._get_observation()
        info = self._get_info()
        info['collision'] = collision
        info['collision_reason'] = collision_reason
        info['goal_reached'] = goal_reached

        return obs, reward, terminated, truncated, info

    def _get_observation(self):
        """获取观测值"""
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
        
        # 【新增】最近障碍物信息 (距离 + 归一化方向)
        nearby_obstacles = self.tunnel.get_nearby_obstacles(
            self.eta[0:3], 
            max_distance=self.point_cloud_config.sample_distance,
            max_count=self.num_nearest_obstacles
        )
        
        obstacle_info = []
        for i in range(self.num_nearest_obstacles):
            if i < len(nearby_obstacles):
                dist, rel_pos, obs_radius = nearby_obstacles[i]
                # 归一化距离 (0=接触, 1=最远)
                norm_dist = min(dist / self.point_cloud_config.sample_distance, 1.0)
                # 归一化方向向量
                dir_norm = rel_pos / (np.linalg.norm(rel_pos) + 1e-6)
                obstacle_info.extend([norm_dist, dir_norm[0], dir_norm[1], dir_norm[2]])
            else:
                # 如果没有足够的障碍物，用最大距离和零向量填充
                obstacle_info.extend([1.0, 0.0, 0.0, 0.0])

        # 组合基础状态观测
        base_obs = np.concatenate([
            [rel_y, rel_z],
            nu_normalized,
            orientation,
            [dist_normalized],
            [progress],
            obstacle_info,  # 新增的障碍物信息
        ]).astype(np.float32)

        # 混合 LSTM 模式：返回 Dict 观测
        # 获取当前点云数据
        R = Rzyx(self.eta[3], self.eta[4], self.eta[5])
        points = self.point_cloud_sampler.sample(self.eta[0:3], R, self.tunnel)
        current_point_cloud = self.point_cloud_sampler.get_flattened_points(points)

        # 更新历史缓冲区 (滑动窗口)
        self.point_cloud_history = np.roll(self.point_cloud_history, shift=-1, axis=0)
        self.point_cloud_history[-1] = current_point_cloud

        return {
            "state": base_obs,
            "point_cloud_seq": self.point_cloud_history.copy(),
        }

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

    def _compute_reward(self, collision: bool, collision_reason: str, goal_reached: bool) -> float:
        """
        计算奖励 (v9 - 简化聚焦)
        
        核心思想：只关注 到达终点 + 避障 + 惯性控制
        只有4个核心奖励项：
        1. 进度奖励 - 前进 = 正奖励
        2. 碰撞惩罚 - 终止条件
        3. 超速/漂移惩罚 - 控制惯性，解决物理高速左偏问题
        4. 危险区渐进惩罚 - 靠近障碍物/边界时预警
        """
        # ========== 终止条件 ==========
        
        if collision:
            return self.p_collision  # -50
        
        if goal_reached:
            # 到达终点：基础奖励 + 时间效率奖励
            time_bonus = max(0, (self.max_steps - self.steps) / self.max_steps) * 50.0
            return self.w_goal + time_bonus  # 200 + bonus
        
        # ========== 每步奖励计算 ==========
        reward = 0.0
        
        # 获取基本信息
        progress = self.tunnel.get_progress(self.eta[0:3])
        forward_speed = self.nu[0]  # 前进速度 (u)
        lateral_speed = abs(self.nu[1])  # 横向速度 (v) - 左右漂移
        
        # 1. 【核心】进度奖励 - 鼓励前进
        progress_delta = progress - self.last_progress
        reward += self.w_progress * progress_delta * 100  # 每1%进度 = +10
        self.last_progress = progress
        
        # 2. 每步小惩罚 - 鼓励效率
        reward += self.c_step  # -0.05
        
        # 3. 【惯性控制】超速惩罚 - 速度过快会导致物理模型左偏
        if forward_speed > self.v_safe:
            overspeed = forward_speed - self.v_safe
            reward += self.p_overspeed * overspeed  # -0.5 * 超速量
        
        # 4. 【惯性控制】横向漂移惩罚 - 惩罚左右偏移
        if lateral_speed > 0.1:
            reward += self.p_drift * lateral_speed  # -0.3 * 漂移速度
        
        # 5. 【安全】渐进式危险惩罚 - 靠近边界/障碍物
        dist_to_boundary = self.tunnel.get_distance_to_boundary(self.eta[0:3])
        if dist_to_boundary < self.d_danger:
            # 距离越近惩罚越大 (二次增长)
            danger_ratio = 1.0 - (dist_to_boundary / self.d_danger)
            reward += self.p_danger * (danger_ratio ** 2)  # -0.2 * ratio^2
        
        # 检查附近障碍物
        nearby = self.tunnel.get_nearby_obstacles(self.eta[0:3], max_distance=self.d_danger, max_count=1)
        if len(nearby) > 0:
            obs_dist = nearby[0][0]  # 到最近障碍物表面的距离
            if obs_dist < self.d_danger:
                obs_danger_ratio = 1.0 - (obs_dist / self.d_danger)
                reward += self.p_danger * (obs_danger_ratio ** 2)
        
        return reward

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