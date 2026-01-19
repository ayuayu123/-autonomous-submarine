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
                 max_steps: int = 1200,
                 point_cloud_history_len: int = 8,
                 render_mode: Optional[str] = None,
                 curriculum_state=None):
        super().__init__()

        self.render_mode = render_mode
        self.max_steps = max_steps
        self.point_cloud_history_len = point_cloud_history_len

        # 课程学习状态 (共享字典，用于支持动态课程切换)
        self.curriculum_state = curriculum_state
        self._last_curriculum_stage = None  # 追踪上次课程阶段，用于检测变化

        # 点云历史缓冲区
        self.point_cloud_history = None

        # 物理仿真时间步
        self.dt = 0.05  # 50ms

        # 控制角度限制
        self.max_angle = 20 * np.pi / 180  # 20 degrees

        # 奖励函数参数
        self.w_progress = 7.0  # 进度奖励权重 (每前进1%给1分)
        self.r_goal = 1000.0  # 到达终点奖励
        self.r_collision = -700.0  # 碰撞惩罚 (撞壁或撞障碍物)
        self.r_future_collision = -3.5  # 推演碰撞惩罚 (每步)
        self.r_timeout = -700.0  # 超时惩罚
        self.r_step = -0.5  # 每步惩罚
        self.prediction_time = 3.0  # 推演时间 (秒)

        # 用于动作平滑惩罚和进度追踪的变量
        self.last_pitch_cmd = 0.0
        self.last_yaw_cmd = 0.0
        self.last_progress = 0.0  # 上一步的进度
        self.max_progress = 0.0  # 历史最大进度 (用于防止刷分)

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
            low=np.array([-1.0, -1.0, -1.0]),  # 注意第三个也是 -1
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
        # 移除了实际控制量，依靠点云历史信息学习执行器动力学
        self.base_obs_dim = 2 + 6 + 3 + 1 + 1  # = 13

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
        print(f"[SubmarineEnv] 混合 LSTM 模式:")
        print(f"  - 状态维度: {self.base_obs_dim}")
        print(f"  - 点云序列: ({self.point_cloud_history_len}, {point_cloud_dim})")

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """重置环境"""
        super().reset(seed=seed)

        # 检查课程状态是否更新，动态同步配置
        if self.curriculum_state is not None:
            current_stage = self.curriculum_state.get('current_stage', 1)
            progressive_mode = self.curriculum_state.get('progressive_mode', False)
            progressive_stage_from_state = self.curriculum_state.get('progressive_stage', 0)

            # 调试日志 - 在课程5时总是打印
            if progressive_mode and current_stage == 5:
                is_same = id(self.tunnel_config) == id(self.tunnel.config)
                print(f"[SubmarineEnv.reset()] DEBUG: current_stage={current_stage}, _last_curriculum_stage={self._last_curriculum_stage}")
                print(f"[SubmarineEnv.reset()] DEBUG: progressive_mode={progressive_mode}, progressive_stage_from_state={progressive_stage_from_state}")
                print(f"[SubmarineEnv.reset()] DEBUG: tunnel_config.progressive_stage={self.tunnel_config.progressive_stage}, tunnel.config.progressive_stage={self.tunnel.config.progressive_stage}")
                print(f"[SubmarineEnv.reset()] DEBUG: id(tunnel_config)={id(self.tunnel_config)}, id(tunnel.config)={id(self.tunnel.config)}, same={is_same}")

            # 检测课程阶段是否变化
            if current_stage != self._last_curriculum_stage:
                num_obstacles = self.curriculum_state.get('num_obstacles', 3)
                w_velocity = self.curriculum_state.get('w_velocity', 0.5)
                progressive_mode = self.curriculum_state.get('progressive_mode', False)
                progressive_stage = self.curriculum_state.get('progressive_stage', 0)

                # 更新隧道配置
                self.tunnel_config.num_obstacles = num_obstacles
                self.tunnel.config.num_obstacles = num_obstacles
                self.tunnel_config.progressive_mode = progressive_mode
                self.tunnel.config.progressive_mode = progressive_mode
                self.tunnel_config.progressive_stage = progressive_stage
                self.tunnel.config.progressive_stage = progressive_stage

                # 对于课程5的渐进式训练，设置固定障碍物种子
                if progressive_mode and progressive_stage == 0 and self.tunnel_config.fixed_obstacle_seed is None:
                    self.tunnel_config.fixed_obstacle_seed = 42
                    self.tunnel.config.fixed_obstacle_seed = 42

                print(f"[SubmarineEnv] 课程阶段更新: Stage {self._last_curriculum_stage} → Stage {current_stage}")
                print(f"  - 障碍物数量: {num_obstacles}")
                print(f"  - 速度奖励权重: {w_velocity}")
                if progressive_mode:
                    print(f"  - 渐进式训练: Stage {progressive_stage}")

                self._last_curriculum_stage = current_stage
            else:
                # 即使课程阶段没有变化，也要检查渐进式训练阶段是否变化
                if self.curriculum_state.get('progressive_mode', False):
                    progressive_stage = self.curriculum_state.get('progressive_stage', 0)
                    print(f"[SubmarineEnv.reset() else] DEBUG: progressive_stage_from_state={progressive_stage}, tunnel_config.progressive_stage={self.tunnel_config.progressive_stage}, tunnel.config.progressive_stage={self.tunnel.config.progressive_stage}")
                    print(f"[SubmarineEnv.reset() else] DEBUG: id(tunnel_config)={id(self.tunnel_config)}, id(tunnel.config)={id(self.tunnel.config)}, same={id(self.tunnel_config)==id(self.tunnel.config)}")
                    if progressive_stage != self.tunnel_config.progressive_stage:
                        print(f"[SubmarineEnv] Progressive stage updated: {self.tunnel_config.progressive_stage} -> {progressive_stage}")
                    self.tunnel_config.progressive_stage = progressive_stage
                    self.tunnel.config.progressive_stage = progressive_stage  # 同步更新 tunnel.config
                    print(f"[SubmarineEnv.reset() else] DEBUG: AFTER UPDATE - tunnel_config.progressive_stage={self.tunnel_config.progressive_stage}, tunnel.config.progressive_stage={self.tunnel.config.progressive_stage}")

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
        self.max_progress = 0.0  # 重置历史最大进度

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

        # 组合基础状态观测 (不包含控制输入)
        # 智能体通过点云历史信息和速度/姿态响应来学习执行器动力学
        base_obs = np.concatenate([
            [rel_y, rel_z],
            nu_normalized,
            orientation,
            [dist_normalized],
            [progress],
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

    def _predict_collision(self) -> bool:
        """
        推演未来是否会碰撞

        基于当前的位置、姿态和速度，推演2秒后的轨迹
        检查是否会碰撞到障碍物或墙壁

        Returns:
            bool: 如果推演2秒后会碰撞则返回 True
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

    def _compute_reward(self, collision: bool, collision_reason: str, goal_reached: bool) -> float:
        """
        计算奖励

        进度奖励：基于历史最大进度，防止来回刷分
        碰撞惩罚：撞壁或撞障碍物扣除500分
        终点奖励：到达终点加1000分
        超时惩罚：超时扣除550分
        推演碰撞惩罚：未来2秒会碰撞则每步扣10分
        每步惩罚：每步扣除0.5分，鼓励快速完成
        """
        reward = 0.0

        # 超时惩罚
        if self.steps >= self.max_steps:
            return self.r_timeout

        # 碰撞惩罚
        if collision:
            return self.r_collision

        # 终点奖励
        if goal_reached:
            return self.r_goal

        progress = self.tunnel.get_progress(self.eta[0:3])

        # 进度奖励：只有当进度超过历史最大值时才给予奖励
        # 这样可以防止 agent 通过来回移动刷分
        if progress > self.max_progress:
            progress_delta = progress - self.max_progress
            reward += self.w_progress * progress_delta * 100.0  # 乘以100将0-1的进度转为百分比
            self.max_progress = progress

        self.last_progress = progress

        # 推演碰撞惩罚
        if self._predict_collision():
            reward += self.r_future_collision

        # 每步惩罚
        reward += self.r_step

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