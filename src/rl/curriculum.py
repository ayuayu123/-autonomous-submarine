#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
curriculum.py: 课程学习配置与管理 (v2 - 更平滑的难度梯度)

设计思想：
1. 物理惯性考虑：
   - 潜艇半径 0.5m，安全速度 1.2m/s
   - 转弯半径约 10-15m（20°舵角）
   - 从检测到反应需要 2-3 秒（2.5-3.5m 的反应距离）

2. 障碍物密度设计：
   - 通道半径 6m（紧凑，迫使避障）
   - 障碍物半径 0.6-0.9m（适中）
   - 强制中心区域生成障碍物（40%），防止"直通路"

3. 点云设计：
   - 采样距离 15m（足够的预警距离）
   - 点数 384（平衡精度和计算量）

v2 改进：
- Stage 3-4 提高成功率要求，防止过得太快
- 新增 Stage 5 过渡阶段，平滑 10→18 障碍物的跳跃
- 添加 Stage 8-10 高级阶段，极端场景泛化训练
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class CurriculumStage:
    """课程学习的单个阶段配置"""
    name: str
    description: str
    
    # 通道参数
    tunnel_radius: float        # 通道半径 (m)
    tunnel_length: float        # 通道长度 (m)
    
    # 障碍物参数
    num_obstacles: int          # 障碍物数量
    obstacle_radius_min: float  # 最小障碍物半径
    obstacle_radius_max: float  # 最大障碍物半径
    obstacle_min_spacing: float # 障碍物最小间距
    obstacle_start_x: float     # 障碍物开始X位置
    center_ratio: float         # 中心区域障碍物比例 (0-1)
    
    # 点云参数
    num_points: int             # 点云采样点数
    sample_distance: float      # 采样最大距离
    
    # 晋级条件
    success_rate_threshold: float  # 晋级所需成功率
    min_episodes: int              # 最少评估episode数
    
    # 训练参数
    timesteps: int              # 本阶段最大训练步数
    min_timesteps_to_advance: int = 50000  # 达标后至少训练这么多步才能晋级
    
    # 【新增】环境随机化参数 (Domain Randomization)
    # 设置为 True 后，每个 episode 会随机化以下参数
    enable_randomization: bool = False  # 是否启用随机化
    radius_variation: float = 0.0       # 隧道半径随机范围 (±米)
    obstacle_count_variation: int = 0   # 障碍物数量随机范围 (±个)
    obstacle_size_variation: float = 0.0  # 障碍物大小随机比例 (±%)
    spawn_offset_variation: float = 0.0   # 起始位置随机偏移 (±米)


# ============================================================
# 课程阶段定义 (v2 - 更平滑的难度梯度 + 高级阶段)
# ============================================================
# 
# 改进要点：
# 1. Stage 2 成功率保持75%不变（避免过拟合）
# 2. Stage 3-4 提高成功率要求至70%/60%，防止过得太快
# 3. 新增 Stage 5 过渡阶段：10→13个障碍物，6.5→6.2m通道
# 4. 添加 Stage 8-10 高级阶段：极密、超长、终极挑战
# ============================================================

CURRICULUM_STAGES = [
    # ========== 基础阶段 ==========
    CurriculumStage(
        name="Stage 1: 基础导航",
        description="学习直线前进和居中，无障碍物",
        tunnel_radius=6.0,
        tunnel_length=30.0,
        num_obstacles=0,
        obstacle_radius_min=0.6,
        obstacle_radius_max=0.9,
        obstacle_min_spacing=1.5,
        obstacle_start_x=10.0,
        center_ratio=0.4,
        num_points=512,
        sample_distance=20.0,
        success_rate_threshold=0.90,
        min_episodes=50,
        timesteps=800_000,
        min_timesteps_to_advance=80_000,
    ),
    
    CurriculumStage(
        name="Stage 2: 简单避障",
        description="学习基本绕行，少量障碍物，宽通道",
        tunnel_radius=7.0,
        tunnel_length=40.0,
        num_obstacles=6,
        obstacle_radius_min=0.6,
        obstacle_radius_max=0.85,
        obstacle_min_spacing=2.0,
        obstacle_start_x=12.0,
        center_ratio=0.5,
        num_points=512,
        sample_distance=20.0,
        success_rate_threshold=0.75,  # 保持不变，避免过拟合
        min_episodes=100,
        timesteps=2_500_000,
        min_timesteps_to_advance=200_000,
    ),
    
    # ========== 进阶阶段（提高成功率要求）==========
    CurriculumStage(
        name="Stage 3: 初级密集",
        description="缩小通道，增加障碍物",
        tunnel_radius=6.5,
        tunnel_length=42.0,
        num_obstacles=8,
        obstacle_radius_min=0.6,
        obstacle_radius_max=0.85,
        obstacle_min_spacing=1.8,
        obstacle_start_x=11.0,
        center_ratio=0.45,
        num_points=512,
        sample_distance=20.0,
        success_rate_threshold=0.70,  # 【提高】从65%提高到70%
        min_episodes=100,
        timesteps=2_500_000,
        min_timesteps_to_advance=300_000,
    ),
    
    CurriculumStage(
        name="Stage 4: 中等避障",
        description="增加障碍物密度，学习连续躲避",
        tunnel_radius=6.5,
        tunnel_length=45.0,
        num_obstacles=10,
        obstacle_radius_min=0.6,
        obstacle_radius_max=0.9,
        obstacle_min_spacing=1.8,
        obstacle_start_x=10.0,
        center_ratio=0.45,
        num_points=512,
        sample_distance=20.0,
        success_rate_threshold=0.60,  # 【提高】从55%提高到60%
        min_episodes=100,
        timesteps=3_000_000,
        min_timesteps_to_advance=400_000,
    ),
    
    # ========== 【新增】过渡阶段（平滑 Stage 4→6 的跳跃）==========
    CurriculumStage(
        name="Stage 5: 过渡密集",
        description="13个障碍物，通道缩小到6.2m，平滑过渡",
        tunnel_radius=6.2,          # 【过渡】6.5 → 6.2 → 6.0
        tunnel_length=45.0,
        num_obstacles=13,           # 【过渡】10 → 13 → 18
        obstacle_radius_min=0.6,
        obstacle_radius_max=0.9,
        obstacle_min_spacing=1.6,   # 【过渡】1.8 → 1.6 → 1.5
        obstacle_start_x=10.0,
        center_ratio=0.45,
        num_points=512,
        sample_distance=20.0,
        success_rate_threshold=0.50,  # 50%达标
        min_episodes=100,
        timesteps=3_000_000,
        min_timesteps_to_advance=400_000,
    ),
    
    CurriculumStage(
        name="Stage 6: 密集避障",
        description="高密度障碍物，紧凑通道",
        tunnel_radius=6.0,
        tunnel_length=45.0,
        num_obstacles=18,
        obstacle_radius_min=0.6,
        obstacle_radius_max=0.9,
        obstacle_min_spacing=1.5,
        obstacle_start_x=10.0,
        center_ratio=0.45,
        num_points=512,
        sample_distance=20.0,
        success_rate_threshold=0.45,
        min_episodes=100,
        timesteps=3_000_000,
        min_timesteps_to_advance=500_000,
    ),
    
    CurriculumStage(
        name="Stage 7: 完整任务",
        description="最终基础难度，模拟真实场景",
        tunnel_radius=6.0,
        tunnel_length=50.0,
        num_obstacles=22,
        obstacle_radius_min=0.6,
        obstacle_radius_max=1.0,
        obstacle_min_spacing=1.3,
        obstacle_start_x=10.0,         # 【修改】从8.0提高到10.0，确保安全距离
        center_ratio=0.4,
        num_points=512,
        sample_distance=20.0,
        success_rate_threshold=0.40,
        min_episodes=100,
        timesteps=5_000_000,
        min_timesteps_to_advance=800_000,
    ),
    
    # ========== 【新增】高级阶段（泛化训练）==========
    CurriculumStage(
        name="Stage 8: 极密避障",
        description="25+障碍物，超高密度，极限挑战（启用随机化）",
        tunnel_radius=5.8,
        tunnel_length=55.0,
        num_obstacles=25,
        obstacle_radius_min=0.5,
        obstacle_radius_max=1.0,
        obstacle_min_spacing=1.2,
        obstacle_start_x=10.0,         # 【修改】从8.0提高到10.0
        center_ratio=0.5,
        num_points=512,
        sample_distance=20.0,
        success_rate_threshold=0.35,
        min_episodes=100,
        timesteps=5_000_000,
        min_timesteps_to_advance=800_000,
        # 【启用环境随机化】
        enable_randomization=True,
        radius_variation=0.3,
        obstacle_count_variation=2,
        obstacle_size_variation=0.15,
        spawn_offset_variation=0.8,
    ),
    
    CurriculumStage(
        name="Stage 9: 超长通道",
        description="80m超长通道，30+障碍物，耐久测试（强随机化）",
        tunnel_radius=5.5,
        tunnel_length=80.0,
        num_obstacles=30,
        obstacle_radius_min=0.5,
        obstacle_radius_max=1.1,
        obstacle_min_spacing=1.2,
        obstacle_start_x=10.0,         # 【修改】从6.0提高到10.0
        center_ratio=0.5,
        num_points=512,
        sample_distance=20.0,
        success_rate_threshold=0.30,
        min_episodes=100,
        timesteps=6_000_000,
        min_timesteps_to_advance=1_000_000,
        # 【强化环境随机化】
        enable_randomization=True,
        radius_variation=0.5,
        obstacle_count_variation=3,
        obstacle_size_variation=0.20,
        spawn_offset_variation=1.0,
    ),
    
    CurriculumStage(
        name="Stage 10: 终极挑战",
        description="100m通道，35+障碍物，最终泛化测试（极端随机化）",
        tunnel_radius=5.5,
        tunnel_length=100.0,
        num_obstacles=35,
        obstacle_radius_min=0.5,
        obstacle_radius_max=1.2,
        obstacle_min_spacing=1.0,
        obstacle_start_x=10.0,         # 【修改】从5.0提高到10.0
        center_ratio=0.5,
        num_points=512,
        sample_distance=20.0,
        success_rate_threshold=0.25,
        min_episodes=100,
        timesteps=8_000_000,
        min_timesteps_to_advance=1_500_000,
        # 【极端环境随机化】
        enable_randomization=True,
        radius_variation=0.8,
        obstacle_count_variation=5,
        obstacle_size_variation=0.25,
        spawn_offset_variation=1.5,
    ),
]


def get_stage_config(stage_index: int) -> CurriculumStage:
    """获取指定阶段的配置"""
    if stage_index < 0 or stage_index >= len(CURRICULUM_STAGES):
        raise ValueError(f"Invalid stage index: {stage_index}")
    return CURRICULUM_STAGES[stage_index]


def get_total_stages() -> int:
    """获取总阶段数"""
    return len(CURRICULUM_STAGES)


def print_curriculum_info():
    """打印课程学习信息"""
    print("\n" + "=" * 70)
    print("📚 课程学习配置 v2 (更平滑的难度梯度 + 高级阶段)")
    print("=" * 70)
    
    total_timesteps = 0
    for i, stage in enumerate(CURRICULUM_STAGES):
        print(f"\n🎯 Stage {i+1}: {stage.name}")
        print(f"   {stage.description}")
        print(f"   通道: 半径={stage.tunnel_radius}m, 长度={stage.tunnel_length}m")
        print(f"   障碍物: {stage.num_obstacles}个, 半径={stage.obstacle_radius_min}-{stage.obstacle_radius_max}m")
        print(f"   间距: {stage.obstacle_min_spacing}m, 中心比例: {stage.center_ratio*100:.0f}%")
        print(f"   晋级: 成功率≥{stage.success_rate_threshold*100:.0f}% + 最少{stage.min_timesteps_to_advance//1000}k步")
        print(f"   最大步数: {stage.timesteps:,}")
        total_timesteps += stage.min_timesteps_to_advance
    
    print(f"\n📊 预估最少总步数: {total_timesteps:,} (提前晋级时)")
    print(f"📊 最大总步数: {sum(s.timesteps for s in CURRICULUM_STAGES):,}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    print_curriculum_info()
