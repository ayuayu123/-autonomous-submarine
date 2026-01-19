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
    num_obstacles: int = 20        # 障碍物数量
    obstacle_radius_min: float = 0.7 # 障碍物最小半径 (米)
    obstacle_radius_max: float = 1 # 障碍物最大半径 (米)
    obstacle_start_x: float = 12.0  # 障碍物从 X=12 开始生成，给潜艇更多起步空间
    obstacle_min_spacing: float = 1.5  # 障碍物之间的最小间距（米），潜艇半径约0.5m
    
    # 【课程学习新增】中心区域障碍物比例
    # 这个比例的障碍物会强制生成在通道中心区域，防止"直通路"
    center_obstacle_ratio: float = 0.4  # 40%障碍物在中心
    
    # 安全参数
    submarine_radius: float = 0.5   # 潜艇碰撞半径
    safe_spawn_radius: float = 1.0  # 潜艇生成位置随机偏移 (减小以保持居中)
    
    # 【新增】环境随机化参数 (Domain Randomization)
    enable_randomization: bool = False  # 是否启用随机化
    radius_variation: float = 0.0       # 隧道半径随机范围 (±米)
    obstacle_count_variation: int = 0   # 障碍物数量随机范围 (±个)
    obstacle_size_variation: float = 0.0  # 障碍物大小随机比例 (±%)
    spawn_offset_variation: float = 0.0   # 起始位置随机偏移 (±米)
    
    def get_randomized_copy(self, rng: np.random.Generator) -> 'TunnelConfig':
        """
        获取随机化后的配置副本
        
        Args:
            rng: 随机数生成器
            
        Returns:
            随机化后的TunnelConfig副本
        """
        if not self.enable_randomization:
            return self
        
        # 随机化隧道半径
        new_radius = self.radius + rng.uniform(-self.radius_variation, self.radius_variation)
        new_radius = max(4.0, new_radius)  # 最小4米
        
        # 随机化障碍物数量
        new_num_obstacles = self.num_obstacles + rng.integers(-self.obstacle_count_variation, self.obstacle_count_variation + 1)
        new_num_obstacles = max(0, new_num_obstacles)
        
        # 随机化障碍物大小参数
        size_scale = 1.0 + rng.uniform(-self.obstacle_size_variation, self.obstacle_size_variation)
        new_obs_min = self.obstacle_radius_min * size_scale
        new_obs_max = self.obstacle_radius_max * size_scale
        
        # 随机化起始位置偏移（影响safe_spawn_radius）
        new_spawn_radius = self.safe_spawn_radius + rng.uniform(0, self.spawn_offset_variation)
        
        return TunnelConfig(
            radius=new_radius,
            length=self.length,
            start_x=self.start_x,
            center_y=self.center_y,
            center_z=self.center_z,
            num_obstacles=new_num_obstacles,
            obstacle_radius_min=new_obs_min,
            obstacle_radius_max=new_obs_max,
            obstacle_start_x=self.obstacle_start_x,
            obstacle_min_spacing=self.obstacle_min_spacing,
            center_obstacle_ratio=self.center_obstacle_ratio,
            submarine_radius=self.submarine_radius,
            safe_spawn_radius=new_spawn_radius,
            enable_randomization=False,  # 副本不再需要随机化
        )



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
        """
        随机生成障碍物 (v5 - 角度区域均匀分布 + 强制外围)
        
        核心改进（防止固定轨迹策略）:
        1. 将角度空间分成8个区域，强制每个区域都有障碍物
        2. 增加中心区域障碍物比例到50%
        3. 某些X区段强制在外围生成障碍物，迫使Agent穿越中心
        4. 障碍物之间保持最小间距
        """
        cfg = self.config
        
        # 障碍物最小间距
        min_spacing = cfg.obstacle_min_spacing
        
        # 计算障碍物区域
        obstacle_zone_start = cfg.obstacle_start_x
        obstacle_zone_end = cfg.start_x + cfg.length
        obstacle_zone_length = obstacle_zone_end - obstacle_zone_start
        
        # 【v5】将隧道分成多个X区段
        num_segments = min(cfg.num_obstacles, 10)
        obstacles_per_segment = cfg.num_obstacles // num_segments
        extra_obstacles = cfg.num_obstacles % num_segments
        segment_length = obstacle_zone_length / num_segments
        
        # 【v5 新增】角度区域追踪，确保障碍物覆盖各个方向
        # 将圆周分成8个扇区（每个45度）
        num_angle_sectors = 8
        angle_sector_size = 2 * np.pi / num_angle_sectors
        angle_sector_counts = [0] * num_angle_sectors  # 每个扇区的障碍物计数
        
        max_r = cfg.radius - cfg.obstacle_radius_max - cfg.submarine_radius
        max_r = max(0.1, max_r)
        
        # 中心区域比例（从0.4提高到0.5）
        center_ratio = getattr(cfg, 'center_obstacle_ratio', 0.5)
        
        count = 0
        for seg_idx in range(num_segments):
            seg_start = obstacle_zone_start + seg_idx * segment_length
            seg_end = seg_start + segment_length
            
            num_in_segment = obstacles_per_segment + (1 if seg_idx < extra_obstacles else 0)
            
            # 【v5 新增】某些区段强制外围障碍物
            # 每隔3个区段，强制在外围生成至少1个障碍物
            force_outer = (seg_idx % 3 == 1) and num_in_segment > 1
            outer_generated = False
            
            attempts = 0
            max_attempts_per_segment = num_in_segment * 50  # 增加尝试次数
            segment_count = 0
            
            while segment_count < num_in_segment and attempts < max_attempts_per_segment:
                attempts += 1
                
                x = self.rng.uniform(seg_start, seg_end)
                
                # 【v5 改进】YZ平面位置：考虑角度覆盖
                # 找出当前最少障碍物的扇区
                min_sector_count = min(angle_sector_counts)
                sparse_sectors = [i for i, c in enumerate(angle_sector_counts) if c == min_sector_count]
                target_sector = self.rng.choice(sparse_sectors)
                
                # 在目标扇区内生成角度
                sector_start = target_sector * angle_sector_size
                theta = self.rng.uniform(sector_start, sector_start + angle_sector_size)
                
                # 【v5】决定半径：考虑强制外围
                if force_outer and not outer_generated:
                    # 强制外围：半径在 [max_r*0.6, max_r] 范围
                    r = self.rng.uniform(max_r * 0.6, max_r)
                elif self.rng.random() < center_ratio:
                    # 中心区域：r 在 [0, max_r*0.35] 范围（更集中于中心）
                    r = self.rng.uniform(0, max_r * 0.35)
                else:
                    # 外围区域：sqrt 采样
                    r = np.sqrt(self.rng.uniform(0.12, 1)) * max_r
                
                y = self.axis_y + r * np.cos(theta)
                z = self.axis_z + r * np.sin(theta)
                
                obs_radius = self.rng.uniform(cfg.obstacle_radius_min, cfg.obstacle_radius_max)
                
                # 检查与已有障碍物的间距
                pos = np.array([x, y, z])
                too_close = False
                for obs in self.obstacles:
                    dist = np.linalg.norm(pos - obs.position)
                    surface_dist = dist - obs_radius - obs.radius
                    if surface_dist < min_spacing:
                        too_close = True
                        break
                
                if not too_close:
                    self.obstacles.append(Obstacle(position=pos, radius=obs_radius))
                    segment_count += 1
                    count += 1
                    
                    # 更新扇区计数
                    angle_sector_counts[target_sector] += 1
                    
                    # 标记外围障碍物已生成
                    if force_outer and r > max_r * 0.5:
                        outer_generated = True
        
        # 打印角度分布统计
        if count >= cfg.num_obstacles:
            print(f"[ProvingGround v5] Generated {count} obstacles")
            print(f"  Angle distribution: {angle_sector_counts} (8 sectors)")
        else:
            print(f"[ProvingGround v5] Warning: Only generated {count}/{cfg.num_obstacles} obstacles")
    
    def reset(self, seed: Optional[int] = None):
        """重置环境，重新生成障碍物"""
        # 每次 reset 都更新随机数生成器
        # 如果传入 seed 则使用该 seed (可复现)
        # 如果 seed=None 则使用新的随机种子 (每次不同)
        self.rng = np.random.default_rng(seed)
        
        # 清除所有障碍物并重新生成
        # 这样可以防止 RL 训练过拟合到特定障碍物布局
        self.obstacles = []
        if self.config.num_obstacles > 0:
            self.generate_obstacles()
    
    def get_submarine_spawn_position(self) -> np.ndarray:
        """获取潜艇的安全生成位置"""
        cfg = self.config
        
        # 在通道中心附近生成，带一点随机偏移
        offset_r = self.rng.uniform(0, cfg.safe_spawn_radius)
        offset_theta = self.rng.uniform(0, 2 * np.pi)
        
        # 潜艇生成在 X=2.0 位置，距离障碍物开始(X=10.0)有约8m安全距离
        x = cfg.start_x + 2.0
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
    
    def get_nearby_obstacles(self, position: np.ndarray, max_distance: float = 50.0, max_count: int = 10) -> List[Tuple[float, int, np.ndarray, float]]:
        """
        获取附近的障碍物
        
        Args:
            position: [x, y, z] 当前位置
            max_distance: 最大搜索距离
            max_count: 最大返回数量
        
        Returns:
            List of (distance, obstacle_index, relative_position, radius) tuples, sorted by distance
            - distance: 到障碍物表面的距离
            - obstacle_index: 障碍物在列表中的索引（用于追踪）
            - relative_position: 障碍物相对于当前位置的向量
            - radius: 障碍物半径
        """
        nearby = []
        for idx, obs in enumerate(self.obstacles):
            rel_pos = obs.position - position
            dist = np.linalg.norm(rel_pos) - obs.radius  # 到障碍物表面的距离
            if dist < max_distance:
                nearby.append((dist, idx, rel_pos, obs.radius))
        
        nearby.sort(key=lambda x: x[0])
        return nearby[:max_count]
    
    def get_obstacles_in_range(self, x_min: float, x_max: float) -> List[Obstacle]:
        """获取指定 X 范围内的障碍物"""
        return [obs for obs in self.obstacles if x_min <= obs.position[0] <= x_max]


# 兼容旧代码引用
CylinderTunnel = ProvingGround
