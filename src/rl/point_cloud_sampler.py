#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
point_cloud_sampler.py: 点云采样器，对障碍物和隧道边界表面进行随机采样

替代原有的 lidar_sensor.py 射线投射方式，直接生成 XYZ 坐标点云。
"""
import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class PointCloudConfig:
    """点云采样配置"""
    # 采样参数
    num_points: int = 64              # 总采样点数 (从256减少到64，降低LSTM负担)
    obstacle_points_ratio: float = 0.5  # 障碍物采样点占比 (管道壁占比提高到50%)
    
    # 距离过滤参数 (视野范围)
    sample_distance: float = 25.0     # 采样距离 (米) - 再增加一点视野
    
    # 归一化参数
    normalize_range: float = 25.0     # 归一化范围 (与采样距离匹配)


class PointCloudSampler:
    """
    点云采样器
    
    对障碍物表面和隧道边界进行随机采样，生成相对于潜艇的 XYZ 坐标点云。
    """
    
    def __init__(self, config: Optional[PointCloudConfig] = None):
        self.config = config or PointCloudConfig()
        
        # 计算障碍物和隧道壁的采样点数
        self.num_obstacle_points = int(self.config.num_points * self.config.obstacle_points_ratio)
        self.num_tunnel_points = self.config.num_points - self.num_obstacle_points
    
    def sample(self, 
               submarine_position: np.ndarray,
               rotation_matrix: np.ndarray,
               tunnel) -> np.ndarray:
        """
        执行点云采样
        
        Args:
            submarine_position: [x, y, z] 潜艇位置 (世界坐标系)
            rotation_matrix: 3x3 旋转矩阵 (机体到世界)，用于将点云转换到机体坐标系
            tunnel: ProvingGround/CylinderTunnel 对象
            
        Returns:
            relative_points: (num_points, 3) 相对于潜艇的点云坐标 (机体坐标系)
        """
        points_world = []
        
        # 1. 采样障碍物表面 (只采样距离内的障碍物!)
        obstacle_points = self._sample_obstacles(
            tunnel.obstacles, 
            submarine_position,  # 新增：传入潜艇位置用于距离过滤
            self.num_obstacle_points
        )
        points_world.extend(obstacle_points)
        
        # 如果附近没有足够的障碍物，用隧道壁补充
        actual_obstacle_points = len(obstacle_points)
        extra_tunnel_points = self.num_obstacle_points - actual_obstacle_points
        
        # 2. 采样隧道壁 (增加采样点数以补充缺失的障碍物点)
        tunnel_points = self._sample_tunnel_wall(
            tunnel.config, 
            tunnel.axis_y, 
            tunnel.axis_z,
            submarine_position[0],  # 围绕潜艇当前 X 位置采样
            self.num_tunnel_points + extra_tunnel_points
        )
        points_world.extend(tunnel_points)
        
        # 转换为 numpy 数组
        points_world = np.array(points_world)  # (N, 3)
        
        # 3. 计算相对坐标 (世界坐标系)
        relative_world = points_world - submarine_position  # (N, 3)
        
        # 4. 转换到机体坐标系 (乘以旋转矩阵的逆，即转置)
        R_inv = rotation_matrix.T
        relative_body = (R_inv @ relative_world.T).T  # (N, 3)
        
        return relative_body
    
    def _sample_obstacles(self, obstacles: List, submarine_position: np.ndarray, 
                          num_points: int) -> List[np.ndarray]:
        """
        对距离范围内的障碍物表面进行采样
        
        Args:
            obstacles: 障碍物列表
            submarine_position: [x, y, z] 潜艇位置
            num_points: 总采样点数
            
        Returns:
            points: 采样点列表 (世界坐标系)
        """
        # 过滤出距离范围内的障碍物
        nearby_obstacles = []
        for obs in obstacles:
            dist = np.linalg.norm(obs.position - submarine_position) - obs.radius
            if dist < self.config.sample_distance:
                nearby_obstacles.append((dist, obs))
        
        # 按距离排序（近的优先采样更多点）
        nearby_obstacles.sort(key=lambda x: x[0])
        
        if len(nearby_obstacles) == 0:
            # 如果附近没有障碍物，返回空列表，后面会用隧道壁填充
            return []
        
        points = []
        
        # 计算每个障碍物应分配的采样点数
        # 使用距离倒数加权：越近的障碍物采样点越多
        weights = []
        for dist, obs in nearby_obstacles:
            # 距离权重：距离越近权重越高
            dist_weight = 1.0 / (dist + 1.0)  # +1防止除零
            # 面积权重
            area_weight = obs.radius ** 2
            weights.append(dist_weight * area_weight)
        
        total_weight = sum(weights)
        
        for i, (dist, obs) in enumerate(nearby_obstacles):
            # 按权重分配采样点数
            weight_ratio = weights[i] / total_weight
            n_samples = max(1, int(num_points * weight_ratio))
            
            # 在球体表面均匀采样
            sphere_points = self._sample_sphere_surface(
                center=obs.position,
                radius=obs.radius,
                num_points=n_samples
            )
            points.extend(sphere_points)
        
        # 如果采样点数不足，从最近的障碍物补充
        while len(points) < num_points and len(nearby_obstacles) > 0:
            _, obs = nearby_obstacles[0]  # 最近的障碍物
            extra_point = self._sample_sphere_surface(obs.position, obs.radius, 1)
            points.extend(extra_point)
        
        # 如果采样点数过多，优先保留近距离的点
        if len(points) > num_points:
            # 按距离排序点云
            points_with_dist = [(np.linalg.norm(p - submarine_position), p) for p in points]
            points_with_dist.sort(key=lambda x: x[0])
            points = [p for _, p in points_with_dist[:num_points]]
        
        return points
    
    def _sample_sphere_surface(self, center: np.ndarray, radius: float, 
                                num_points: int) -> List[np.ndarray]:
        """
        在球体表面均匀随机采样
        
        使用标准的球面均匀采样算法
        """
        points = []
        
        for _ in range(num_points):
            # 均匀球面采样
            phi = np.random.uniform(0, 2 * np.pi)
            cos_theta = np.random.uniform(-1, 1)
            sin_theta = np.sqrt(1 - cos_theta ** 2)
            
            x = center[0] + radius * sin_theta * np.cos(phi)
            y = center[1] + radius * sin_theta * np.sin(phi)
            z = center[2] + radius * cos_theta
            
            points.append(np.array([x, y, z]))
        
        return points
    
    def _sample_tunnel_wall(self, config, axis_y: float, axis_z: float,
                            submarine_x: float, num_points: int) -> List[np.ndarray]:
        """
        对隧道壁 (圆柱面) 进行采样
        
        Args:
            config: TunnelConfig 对象
            axis_y, axis_z: 隧道中心轴坐标
            submarine_x: 潜艇当前 X 坐标
            num_points: 采样点数
            
        Returns:
            points: 采样点列表 (世界坐标系)
        """
        points = []
        
        # 在潜艇前后一定范围内采样隧道壁 (使用采样距离配置)
        x_range = self.config.sample_distance  # 使用配置的采样距离 (10m)
        x_min = max(config.start_x, submarine_x - x_range * 0.3)  # 后方30%
        x_max = min(config.start_x + config.length, submarine_x + x_range * 0.7)  # 前方70%
        
        for _ in range(num_points):
            # 随机 X 坐标
            x = np.random.uniform(x_min, x_max)
            
            # 在圆周上随机采样角度
            theta = np.random.uniform(0, 2 * np.pi)
            
            # 计算圆柱面上的点
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
        # 裁剪极端值 - 用户要求取消感知范围限制，让抽样正好落在物体上
        # normalized = np.clip(normalized, -1.0, 1.0)
        return normalized.flatten()
    
    @property
    def num_points(self) -> int:
        """获取采样点数"""
        return self.config.num_points
    
    @property
    def output_dim(self) -> int:
        """获取输出维度 (用于神经网络输入层)"""
        return self.config.num_points * 3
