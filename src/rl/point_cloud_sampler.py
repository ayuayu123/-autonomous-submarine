#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
point_cloud_sampler.py: 点云采样器 (固定采样版本)

使用确定性采样模式：
- 障碍物：Fibonacci 球面均匀采样 (每次采样位置固定)
- 隧道壁：均匀网格采样 (每次采样位置固定)

这样可以让 LSTM 更好地追踪障碍物的运动轨迹。
"""
import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class PointCloudConfig:
    """点云采样配置"""
    # 采样参数
    num_points: int = 512             # 总采样点数 (从384增加到512，提高精度)
    obstacle_points_ratio: float = 0.5  # 障碍物采样点占比 (50%给障碍物，50%给隧道壁)
    
    # 距离过滤参数 (视野范围)
    sample_distance: float = 25.0     # 采样距离 (米)
    
    # 归一化参数
    normalize_range: float = 25.0     # 归一化范围 (与采样距离匹配)
    
    # 每个障碍物的固定采样点数
    points_per_obstacle: int = 32     # 每个障碍物采样32个固定点


class PointCloudSampler:
    """
    点云采样器 (固定采样版本)
    
    使用确定性采样，生成时序一致的点云，便于 LSTM 学习。
    """
    
    def __init__(self, config: Optional[PointCloudConfig] = None):
        self.config = config or PointCloudConfig()
        
        # 计算障碍物和隧道壁的采样点数
        self.num_obstacle_points = int(self.config.num_points * self.config.obstacle_points_ratio)
        self.num_tunnel_points = self.config.num_points - self.num_obstacle_points
        
        # 预计算 Fibonacci 球面采样的固定方向 (单位球面上的点)
        self._fibonacci_directions = self._generate_fibonacci_sphere(self.config.points_per_obstacle)
        
        # 预计算隧道壁采样的固定角度
        self._tunnel_angles, self._tunnel_x_offsets = self._generate_tunnel_grid()
    
    def _generate_fibonacci_sphere(self, n_points: int) -> np.ndarray:
        """
        生成 Fibonacci 球面上的均匀分布点 (单位方向向量)
        
        这是一种在球面上均匀分布点的经典算法。
        """
        directions = []
        phi = np.pi * (3.0 - np.sqrt(5.0))  # 黄金角
        
        for i in range(n_points):
            y = 1 - (i / float(n_points - 1)) * 2  # y from 1 to -1
            radius = np.sqrt(1 - y * y)
            theta = phi * i
            
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            
            directions.append(np.array([x, y, z]))
        
        return np.array(directions)
    
    def _generate_tunnel_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成隧道壁采样的固定网格
        
        返回：(角度数组, X偏移数组)
        """
        # 在圆周上均匀分布角度
        n_angular = 16  # 圆周方向16个采样点
        n_axial = self.num_tunnel_points // n_angular  # 轴向方向的采样点数
        
        angles = np.linspace(0, 2 * np.pi, n_angular, endpoint=False)
        
        # X 方向的偏移 (相对于潜艇位置)
        # 后方30%，前方70%
        x_offsets = np.linspace(-self.config.sample_distance * 0.3, 
                                 self.config.sample_distance * 0.7, 
                                 n_axial)
        
        return angles, x_offsets
    
    def sample(self, 
               submarine_position: np.ndarray,
               rotation_matrix: np.ndarray,
               tunnel) -> np.ndarray:
        """
        执行点云采样 (固定模式)
        
        Args:
            submarine_position: [x, y, z] 潜艇位置 (世界坐标系)
            rotation_matrix: 3x3 旋转矩阵 (机体到世界)
            tunnel: ProvingGround/CylinderTunnel 对象
            
        Returns:
            relative_points: (num_points, 3) 相对于潜艇的点云坐标 (机体坐标系)
        """
        points_world = []
        
        # 1. 采样障碍物表面 (固定采样)
        obstacle_points = self._sample_obstacles_fixed(
            tunnel.obstacles, 
            submarine_position,
            self.num_obstacle_points
        )
        points_world.extend(obstacle_points)
        
        # 如果附近没有足够的障碍物，用隧道壁补充
        actual_obstacle_points = len(obstacle_points)
        extra_tunnel_points = self.num_obstacle_points - actual_obstacle_points
        
        # 2. 采样隧道壁 (固定采样)
        tunnel_points = self._sample_tunnel_wall_fixed(
            tunnel.config, 
            tunnel.axis_y, 
            tunnel.axis_z,
            submarine_position[0],
            self.num_tunnel_points + extra_tunnel_points
        )
        points_world.extend(tunnel_points)
        
        # 确保点数正确
        if len(points_world) < self.config.num_points:
            # 用零点填充 (表示没有检测到)
            for _ in range(self.config.num_points - len(points_world)):
                points_world.append(submarine_position + np.array([self.config.sample_distance, 0, 0]))
        elif len(points_world) > self.config.num_points:
            # 截断
            points_world = points_world[:self.config.num_points]
        
        # 转换为 numpy 数组
        points_world = np.array(points_world)  # (N, 3)
        
        # 3. 计算相对坐标 (世界坐标系)
        relative_world = points_world - submarine_position  # (N, 3)
        
        # 4. 转换到机体坐标系 (乘以旋转矩阵的逆，即转置)
        R_inv = rotation_matrix.T
        relative_body = (R_inv @ relative_world.T).T  # (N, 3)
        
        return relative_body
    
    def _sample_obstacles_fixed(self, obstacles: List, submarine_position: np.ndarray, 
                                 num_points: int) -> List[np.ndarray]:
        """
        对距离范围内的障碍物表面进行固定采样
        
        使用预计算的 Fibonacci 球面方向，确保每帧采样位置一致。
        """
        # 过滤出距离范围内的障碍物
        nearby_obstacles = []
        for obs in obstacles:
            dist = np.linalg.norm(obs.position - submarine_position) - obs.radius
            if dist < self.config.sample_distance:
                nearby_obstacles.append((dist, obs))
        
        # 按距离排序（近的优先）
        nearby_obstacles.sort(key=lambda x: x[0])
        
        if len(nearby_obstacles) == 0:
            return []
        
        points = []
        
        # 计算每个障碍物应分配的采样点数
        weights = []
        for dist, obs in nearby_obstacles:
            dist_weight = 1.0 / (dist + 1.0)
            area_weight = obs.radius ** 2
            weights.append(dist_weight * area_weight)
        
        total_weight = sum(weights)
        
        for i, (dist, obs) in enumerate(nearby_obstacles):
            weight_ratio = weights[i] / total_weight
            n_samples = max(1, int(num_points * weight_ratio))
            n_samples = min(n_samples, self.config.points_per_obstacle)
            
            # 使用固定的 Fibonacci 方向采样
            for j in range(n_samples):
                if j < len(self._fibonacci_directions):
                    direction = self._fibonacci_directions[j]
                    point = obs.position + direction * obs.radius
                    points.append(point)
        
        # 如果采样点数过多，优先保留近距离的点
        if len(points) > num_points:
            points_with_dist = [(np.linalg.norm(p - submarine_position), p) for p in points]
            points_with_dist.sort(key=lambda x: x[0])
            points = [p for _, p in points_with_dist[:num_points]]
        
        return points
    
    def _sample_tunnel_wall_fixed(self, config, axis_y: float, axis_z: float,
                                   submarine_x: float, num_points: int) -> List[np.ndarray]:
        """
        对隧道壁 (圆柱面) 进行固定网格采样
        
        使用预计算的角度和X偏移，确保每帧采样位置一致。
        """
        points = []
        
        # 计算有效 X 范围
        x_min = max(config.start_x, submarine_x - self.config.sample_distance * 0.3)
        x_max = min(config.start_x + config.length, submarine_x + self.config.sample_distance * 0.7)
        
        # 使用固定网格采样
        count = 0
        for x_offset in self._tunnel_x_offsets:
            x = submarine_x + x_offset
            
            # 检查 X 是否在有效范围内
            if x < x_min or x > x_max:
                continue
            
            for theta in self._tunnel_angles:
                if count >= num_points:
                    break
                    
                y = axis_y + config.radius * np.cos(theta)
                z = axis_z + config.radius * np.sin(theta)
                points.append(np.array([x, y, z]))
                count += 1
            
            if count >= num_points:
                break
        
        # 如果点数不足，用最后一个角度的点填充
        while len(points) < num_points:
            theta = self._tunnel_angles[len(points) % len(self._tunnel_angles)]
            x = submarine_x + self._tunnel_x_offsets[-1]
            x = min(max(x, x_min), x_max)
            y = axis_y + config.radius * np.cos(theta)
            z = axis_z + config.radius * np.sin(theta)
            points.append(np.array([x, y, z]))
        
        return points
    
    def get_flattened_points(self, points: np.ndarray) -> np.ndarray:
        """
        获取扁平化的点云数据，用于神经网络输入
        
        Args:
            points: (num_points, 3) 点云坐标
            
        Returns:
            flattened: (num_points * 3,) 扁平化并归一化的坐标
        """
        # 归一化到 [-1, 1] 范围
        normalized = points / self.config.normalize_range
        return normalized.flatten()
    
    @property
    def num_points(self) -> int:
        """获取采样点数"""
        return self.config.num_points
    
    @property
    def output_dim(self) -> int:
        """获取输出维度 (用于神经网络输入层)"""
        return self.config.num_points * 3
