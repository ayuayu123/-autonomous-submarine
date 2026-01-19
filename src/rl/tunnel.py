#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tunnel.py: 圆柱通道环境，包含障碍物生成
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


from enum import Enum, auto

class ShapeType(Enum):
    """障碍物形状枚举"""
    SPHERE = auto()
    BOX = auto()
    # CYLINDER = auto() # 后续可扩展

@dataclass
class Obstacle:
    """障碍物/Actor 定义"""
    position: np.ndarray  # [x, y, z] 中心位置
    radius: float = 1.0   # 碰撞/包围半径 (球体半径 或 Box的包围球半径)
    shape: ShapeType = ShapeType.SPHERE
    size: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 1.0])) # [Lx, Ly, Lz] 仅对 Box 有效
    orientation: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0])) # [roll, pitch, yaw]
    
    def __post_init__(self):
        self.position = np.array(self.position, dtype=float)
        self.size = np.array(self.size, dtype=float)
        self.orientation = np.array(self.orientation, dtype=float)
        
        # 如果是 Box，自动计算包围半径
        if self.shape == ShapeType.BOX:
            self.radius = np.linalg.norm(self.size) / 2.0


@dataclass
class TunnelConfig:
    """圆柱通道配置"""
    # 通道参数
    radius: float = field(default=5.0)    # 通道半径 (米)
    length: float = field(default=50.0)    # 通道长度 (米)

    # 通道起点 (潜艇初始位置附近)
    start_x: float = 0.0
    center_y: float = 0.0          # 圆柱中心 Y 坐标
    center_z: float = 100.0        # 圆柱中心 Z 坐标 (深度)

    # 障碍物参数
    num_obstacles: int = 10        # 障碍物数量
    obstacle_radius_min: float = 0.7 # 障碍物最小半径 (米)
    obstacle_radius_max: float = 1 # 障碍物最大半径 (米)
    obstacle_start_x: float = 10.0  # 障碍物从 X=10 开始生成，给潜艇留出起步空间

    # 安全参数
    submarine_radius: float = 0.5   # 潜艇碰撞半径
    safe_spawn_radius: float = 5.0  # 潜艇生成位置周围的安全区域

    # 渐进式训练模式配置 (用于课程5)
    progressive_mode: bool = False  # 是否启用渐进式训练
    progressive_stage: int = 0      # 渐进式阶段: 0=固定, 1-10=渐进调整, 11+=完全随机
    fixed_obstacle_seed: Optional[int] = None  # 固定障碍物布局的种子
    perturbation_amount: float = 0.5  # 障碍物位置扰动量 (米)，每阶段递增


class ProvingGround:
    """
    试验场环境 (原 CylinderTunnel)
    
    支持自定义 Actor 的位置和形状。
    保留圆柱形边界作为可选限制。
    """

    def __init__(self, config: Optional[TunnelConfig] = None, seed: Optional[int] = None):
        self.config = config or TunnelConfig()
        self.rng = np.random.default_rng(seed)

        # 通道轴心线 (YZ 平面的圆心)
        self.axis_y = self.config.center_y
        self.axis_z = self.config.center_z

        # 障碍物/Actor 列表
        self.obstacles: List[Obstacle] = []

        # 用于渐进式训练的基础障碍物布局 (固定位置)
        self._base_obstacles: List[Obstacle] = []

        # 生成默认障碍物 (如果配置了数量)
        if self.config.num_obstacles > 0:
            if self.config.radius is not None and self.config.length is not None:
                self.generate_obstacles()
            else:
                # 如果没有提供尺寸，我们就不自动生成障碍物，避免崩溃
                # 这允许 ProvingGround 作为纯粹的容器使用
                pass
            
    def add_actor(self, 
                  position: List[float], 
                  shape: ShapeType = ShapeType.SPHERE, 
                  radius: float = 1.0, 
                  size: List[float] = None, 
                  orientation: List[float] = None):
        """
        手动添加 Actor 到试验场
        
        Args:
            position: [x, y, z]
            shape: ShapeType.SPHERE or ShapeType.BOX
            radius: 半径 (仅 Sphere 有效)
            size: [Lx, Ly, Lz] (仅 Box 有效)
            orientation: [roll, pitch, yaw] (仅 Box 有效)
        """
        if size is None:
            size = [1.0, 1.0, 1.0]
        if orientation is None:
            orientation = [0.0, 0.0, 0.0]
            
        obs = Obstacle(
            position=np.array(position),
            radius=radius,
            shape=shape,
            size=np.array(size),
            orientation=np.array(orientation)
        )
        self.obstacles.append(obs)
        
    def clear_actors(self):
        """清除所有 Actor"""
        self.obstacles = []

    def generate_obstacles(self):
        """随机生成障碍物 (兼容旧模式)"""
        # self.obstacles = [] # 不强制清空，允许混合

        cfg = self.config
        attempts = 0
        max_attempts = cfg.num_obstacles * 10
        count = 0

        while count < cfg.num_obstacles and attempts < max_attempts:
            attempts += 1

            # 随机位置 (在通道内)
            obstacle_end_x = cfg.start_x + cfg.length
            x = self.rng.uniform(cfg.obstacle_start_x, obstacle_end_x)

            # 在圆形截面内随机采样 (极坐标)
            r = self.rng.uniform(0, cfg.radius + cfg.obstacle_radius_max * 0.5)
            theta = self.rng.uniform(0, 2 * np.pi)

            y = self.axis_y + r * np.cos(theta)
            z = self.axis_z + r * np.sin(theta)

            # 随机半径
            obs_radius = self.rng.uniform(cfg.obstacle_radius_min, cfg.obstacle_radius_max)

            # 检查是否与已有障碍物重叠太多
            pos = np.array([x, y, z])
            too_close = False
            for obs in self.obstacles:
                dist = np.linalg.norm(pos - obs.position)
                if dist < (obs_radius + obs.radius) * 0.3:
                    too_close = True
                    break

            if not too_close:
                self.obstacles.append(Obstacle(position=pos, radius=obs_radius))
                count += 1

        print(f"[ProvingGround] Generated {count} random obstacles (Total: {len(self.obstacles)})")

    def generate_fixed_obstacles(self, seed: Optional[int] = None):
        """生成固定位置的障碍物 (使用指定种子)"""
        # 保存当前随机数生成器状态
        old_rng = self.rng
        self.rng = np.random.default_rng(seed)

        # 清空障碍物列表
        self.obstacles = []

        cfg = self.config
        attempts = 0
        max_attempts = cfg.num_obstacles * 10
        count = 0

        while count < cfg.num_obstacles and attempts < max_attempts:
            attempts += 1

            # 随机位置 (在通道内)
            obstacle_end_x = cfg.start_x + cfg.length
            x = self.rng.uniform(cfg.obstacle_start_x, obstacle_end_x)

            # 在圆形截面内随机采样 (极坐标)
            r = self.rng.uniform(0, cfg.radius + cfg.obstacle_radius_max * 0.5)
            theta = self.rng.uniform(0, 2 * np.pi)

            y = self.axis_y + r * np.cos(theta)
            z = self.axis_z + r * np.sin(theta)

            # 随机半径
            obs_radius = self.rng.uniform(cfg.obstacle_radius_min, cfg.obstacle_radius_max)

            # 检查是否与已有障碍物重叠太多
            pos = np.array([x, y, z])
            too_close = False
            for obs in self.obstacles:
                dist = np.linalg.norm(pos - obs.position)
                if dist < (obs_radius + obs.radius) * 0.3:
                    too_close = True
                    break

            if not too_close:
                self.obstacles.append(Obstacle(position=pos, radius=obs_radius))
                count += 1

        # 保存基础障碍物布局
        self._base_obstacles = [Obstacle(position=obs.position.copy(), radius=obs.radius) for obs in self.obstacles]

        # 恢复随机数生成器
        self.rng = old_rng

        print(f"[ProvingGround] Generated {count} fixed obstacles with seed {seed} (Total: {len(self.obstacles)})")

    def generate_perturbed_obstacles(self, stage: int):
        """
        基于固定布局生成带扰动的障碍物

        Args:
            stage: 渐进式训练阶段 (0=固定, 1-10=渐进调整, 11+=完全随机)
        """
        # 清空当前障碍物
        self.obstacles = []

        if stage == 0:
            # 固定模式: 直接使用基础布局
            for base_obs in self._base_obstacles:
                self.obstacles.append(Obstacle(position=base_obs.position.copy(), radius=base_obs.radius))
            print(f"[ProvingGround] Using fixed obstacle layout (stage={stage})")
        elif 1 <= stage <= 10:
            # 渐进调整模式: 在基础布局上添加扰动
            perturbation = stage * self.config.perturbation_amount
            for base_obs in self._base_obstacles:
                # 添加随机扰动
                noise = self.rng.uniform(-perturbation, perturbation, size=3)
                new_pos = base_obs.position + noise
                self.obstacles.append(Obstacle(position=new_pos, radius=base_obs.radius))
            print(f"[ProvingGround] Generated perturbed obstacles (stage={stage}, perturbation={perturbation:.2f}m)")
        else:
            # 完全随机模式
            self.generate_obstacles()
            print(f"[ProvingGround] Generated completely random obstacles (stage={stage})")

    def reset(self, seed: Optional[int] = None):
        """重置环境，重新生成障碍物"""
        # 每次 reset 都更新随机数生成器
        # 如果传入 seed 则使用该 seed (可复现)
        # 如果 seed=None 则使用新的随机种子 (每次不同)
        self.rng = np.random.default_rng(seed)

        # 清除所有障碍物并重新生成
        # 这样可以防止 RL 训练过拟合到特定障碍物布局
        self.obstacles = []

        if self.config.progressive_mode:
            # 调试日志 - 在渐进式模式时总是打印
            print(f"[ProvingGround] reset() called: progressive_stage={self.config.progressive_stage}, _base_obstacles={len(self._base_obstacles)}")
            # 渐进式训练模式
            if not self._base_obstacles and self.config.fixed_obstacle_seed is not None:
                # 第一次: 生成固定布局
                self.generate_fixed_obstacles(self.config.fixed_obstacle_seed)
            else:
                # 使用基础布局生成带扰动的障碍物
                self.generate_perturbed_obstacles(self.config.progressive_stage)
        else:
            # 普通模式: 随机生成
            if self.config.num_obstacles > 0:
                self.generate_obstacles()
    
    def get_submarine_spawn_position(self) -> np.ndarray:
        """获取潜艇的安全生成位置"""
        cfg = self.config
        
        # 在通道中心附近生成，带一点随机偏移
        offset_r = self.rng.uniform(0, cfg.safe_spawn_radius)
        offset_theta = self.rng.uniform(0, 2 * np.pi)
        
        x = cfg.start_x + 5.0  # 稍微前移
        y = self.axis_y + offset_r * np.cos(offset_theta)
        z = self.axis_z + offset_r * np.sin(offset_theta)
        
        return np.array([x, y, z])
    
    def check_boundary_collision(self, position: np.ndarray, radius: float = 1.0) -> bool:
        """
        检查是否与通道边界碰撞
        
        Args:
            position: [x, y, z] 潜艇位置
            radius: 潜艇碰撞半径
            
        Returns:
            True if collision
        """
        cfg = self.config
        
        # 计算到通道轴心的距离 (在 YZ 平面)
        dy = position[1] - self.axis_y
        dz = position[2] - self.axis_z
        r = np.sqrt(dy**2 + dz**2)
        
        # 碰撞条件：潜艇外边缘超出通道
        if r + radius > cfg.radius:
            return True
        
        return False
    
    def check_obstacle_collision(self, position: np.ndarray, radius: float = 1.0) -> Tuple[bool, Optional[Obstacle]]:
        """
        检查是否与障碍物碰撞
        
        Args:
            position: [x, y, z] 潜艇位置
            radius: 潜艇碰撞半径
            
        Returns:
            (collision, obstacle) - 如果碰撞返回对应障碍物
        """
        for obs in self.obstacles:
            dist = np.linalg.norm(position - obs.position)
            if dist < radius + obs.radius:
                return True, obs
        
        return False, None
    
    def check_collision(self, position: np.ndarray, radius: float = 1.0) -> Tuple[bool, str]:
        """
        综合碰撞检测
        
        Returns:
            (collision, reason) - reason: "boundary", "obstacle", or ""
        """
        if self.check_boundary_collision(position, radius):
            return True, "boundary"
        
        obs_collision, _ = self.check_obstacle_collision(position, radius)
        if obs_collision:
            return True, "obstacle"
        
        return False, ""
    
    def get_progress(self, position: np.ndarray) -> float:
        """
        获取通道进度 (0 ~ 1)
        
        Args:
            position: [x, y, z] 潜艇位置
            
        Returns:
            progress: 0.0 (起点) to 1.0 (终点)
        """
        cfg = self.config
        progress = (position[0] - cfg.start_x) / cfg.length
        return np.clip(progress, 0.0, 1.0)
    
    def is_goal_reached(self, position: np.ndarray) -> bool:
        """检查是否到达终点"""
        cfg = self.config
        return position[0] >= cfg.start_x + cfg.length
    
    def get_distance_to_boundary(self, position: np.ndarray) -> float:
        """获取到通道边界的距离"""
        dy = position[1] - self.axis_y
        dz = position[2] - self.axis_z
        r = np.sqrt(dy**2 + dz**2)
        return self.config.radius - r
    
    def get_nearby_obstacles(self, position: np.ndarray, max_distance: float = 50.0, max_count: int = 10) -> List[Tuple[float, np.ndarray]]:
        """
        获取附近的障碍物
        
        Returns:
            List of (distance, relative_position) tuples, sorted by distance
        """
        nearby = []
        for obs in self.obstacles:
            rel_pos = obs.position - position
            dist = np.linalg.norm(rel_pos) - obs.radius  # 到障碍物表面的距离
            if dist < max_distance:
                nearby.append((dist, rel_pos, obs.radius))
        
        nearby.sort(key=lambda x: x[0])
        return nearby[:max_count]
    
    def get_obstacles_in_range(self, x_min: float, x_max: float) -> List[Obstacle]:
        """获取指定 X 范围内的障碍物"""
        return [obs for obs in self.obstacles if x_min <= obs.position[0] <= x_max]


# 兼容旧代码引用
CylinderTunnel = ProvingGround
