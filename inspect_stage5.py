#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inspect_stage5.py: 可视化课程5的不同progressive stage的障碍物布局

功能：
- 按数字键1-9切换不同progressive stage
- Stage 0: 固定障碍物位置（seed=42）
- Stage 1-10: 渐进调整（扰动量从±0.5m到±5.0m）
- Stage 11+: 完全随机障碍物位置
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pygame
from pygame.locals import *
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

from src.rl.tunnel import TunnelConfig, ProvingGround, Obstacle
from src.core.stage import LogicStage
from src.physics.submarine_actor import SubmarineActor
from src.gameplay.obstacle_actor import ObstacleActor
from src.gameplay.tunnel_actor import TunnelActor
from src.view.renderer import Renderer


class Stage5Inspector:
    """课程5 Progressive Stage 可视化工具"""

    def __init__(self):
        pygame.init()
        self.width, self.height = 1280, 900

        pygame.display.set_mode((self.width, self.height), DOUBLEBUF | OPENGL | RESIZABLE)
        pygame.display.set_caption("Stage 5 Progressive Inspector")

        # 课程5的TunnelConfig
        self.tunnel_config = TunnelConfig(
            radius=5.0,
            length=50.0,
            center_y=0.0,
            center_z=100.0,
            num_obstacles=15,
            obstacle_radius_min=0.7,
            obstacle_radius_max=1.0,
            obstacle_start_x=10.0,
            # Progressive mode配置
            progressive_mode=True,
            progressive_stage=0,  # 初始stage
            fixed_obstacle_seed=42,  # 固定障碍物布局的种子
            perturbation_amount=0.5,  # 每阶段递增0.5m
        )

        # 创建ProvingGround（不通过TunnelActor，直接操作）
        self.proving_ground = ProvingGround(config=self.tunnel_config)

        # 创建LogicStage用于渲染
        self.logic_stage = LogicStage()

        # TunnelActor用于渲染tunnel visual
        self.tunnel_actor = TunnelActor(num_obstacles=15)
        # 更新TunnelActor的config以匹配progressive配置
        self.tunnel_actor.config = self.tunnel_config
        self.logic_stage.add_actor(self.tunnel_actor)

        # 潜艇（用于参考）
        self.submarine = SubmarineActor("HeroSub")
        # 设置潜艇初始位置
        spawn_pos = self.proving_ground.get_submarine_spawn_position()
        self.submarine.eta[:3] = spawn_pos
        self.submarine._sync_state()
        self.logic_stage.add_actor(self.submarine)

        # 渲染器
        self.renderer = Renderer(self.width, self.height)
        self.renderer.tunnel_config = self.tunnel_config

        self.clock = pygame.time.Clock()
        self.running = True
        self.paused = False

        # 当前progressive stage
        self.current_stage = 0
        self.max_stage = 15  # 允许查看到stage 15（完全随机模式）

        # 初始化第一个stage
        self._load_stage(self.current_stage)

        # 显示信息
        self.show_info = True

    def _load_stage(self, stage: int):
        """加载指定的progressive stage"""
        print(f"\n{'='*60}")
        print(f"Loading Progressive Stage {stage}")
        print(f"{'='*60}")

        # 更新TunnelConfig的progressive_stage
        self.tunnel_config.progressive_stage = stage

        # 重置ProvingGround，生成新的障碍物布局
        self.proving_ground.reset(seed=None)

        # 清空并重新添加障碍物actor
        self.logic_stage.actors = [a for a in self.logic_stage.actors
                                    if not isinstance(a, ObstacleActor)]

        # 为每个障碍物创建ObstacleActor
        for i, obs in enumerate(self.proving_ground.obstacles):
            obs_actor = ObstacleActor(
                name=f"Obstacle_{i}",
                x=obs.position[0],
                y=obs.position[1],
                z=obs.position[2],
                radius=obs.radius
            )
            self.logic_stage.add_actor(obs_actor)

        # 更新渲染器的tunnel（用于障碍物位置查询）
        self.renderer.tunnel = self.proving_ground

        # 打印障碍物信息
        print(f"Stage {stage} Configuration:")
        if stage == 0:
            print("  Mode: FIXED (seed=42)")
            print("  Perturbation: 0.0m (no perturbation)")
        elif 1 <= stage <= 10:
            perturbation = stage * self.tunnel_config.perturbation_amount
            print(f"  Mode: PROGRESSIVE")
            print(f"  Perturbation: ±{perturbation:.1f}m")
        else:
            print("  Mode: RANDOM (completely random)")
            print("  Perturbation: N/A")

        print(f"  Number of obstacles: {len(self.proving_ground.obstacles)}")

        # 显示前几个障碍物的位置
        print("\n  Obstacle positions (first 5):")
        for i, obs in enumerate(self.proving_ground.obstacles[:5]):
            print(f"    Obs{i}: X={obs.position[0]:.2f}, Y={obs.position[1]:.2f}, Z={obs.position[2]:.2f}, R={obs.radius:.2f}")
        if len(self.proving_ground.obstacles) > 5:
            print(f"    ... and {len(self.proving_ground.obstacles) - 5} more")

        # 计算障碍物位置的统计信息
        positions = np.array([obs.position for obs in self.proving_ground.obstacles])
        mean_pos = np.mean(positions, axis=0)
        std_pos = np.std(positions, axis=0)
        print(f"\n  Position Statistics:")
        print(f"    Mean: X={mean_pos[0]:.2f}, Y={mean_pos[1]:.2f}, Z={mean_pos[2]:.2f}")
        print(f"    Std:  X={std_pos[0]:.2f}, Y={std_pos[1]:.2f}, Z={std_pos[2]:.2f}")

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.VIDEORESIZE:
                self.width, self.height = event.w, event.h
                pygame.display.set_mode((self.width, self.height), DOUBLEBUF | OPENGL | RESIZABLE)
                self.renderer.resize(self.width, self.height)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_r:
                    # 重新加载当前stage（重新生成随机）
                    self._load_stage(self.current_stage)
                elif pygame.K_1 <= event.key <= pygame.K_9:
                    # 数字键1-9切换到stage 0-8
                    new_stage = event.key - pygame.K_1
                    self.current_stage = new_stage
                    self._load_stage(self.current_stage)
                elif event.key == pygame.K_0:
                    # 0键切换到stage 9
                    self.current_stage = 9
                    self._load_stage(self.current_stage)
                elif event.key == pygame.K_q:
                    # Q键切换到stage 10
                    self.current_stage = 10
                    self._load_stage(self.current_stage)
                elif event.key == pygame.K_w:
                    # W键切换到stage 11
                    self.current_stage = 11
                    self._load_stage(self.current_stage)
                elif event.key == pygame.K_e:
                    # E键切换到stage 12
                    self.current_stage = 12
                    self._load_stage(self.current_stage)
                elif event.key == pygame.K_LEFT:
                    # 左键：上一个stage
                    self.current_stage = max(0, self.current_stage - 1)
                    self._load_stage(self.current_stage)
                elif event.key == pygame.K_RIGHT:
                    # 右键：下一个stage
                    self.current_stage = min(self.max_stage, self.current_stage + 1)
                    self._load_stage(self.current_stage)
                elif event.key == pygame.K_h:
                    # H键：切换信息显示
                    self.show_info = not self.show_info
                elif event.key == pygame.K_SPACE:
                    # 空格：暂停/继续
                    self.paused = not self.paused

    def get_stage_description(self) -> str:
        """获取当前stage的描述"""
        stage = self.current_stage
        if stage == 0:
            return "Stage 0: FIXED (no perturbation)"
        elif 1 <= stage <= 10:
            perturbation = stage * self.tunnel_config.perturbation_amount
            return f"Stage {stage}: PROGRESSIVE (±{perturbation:.1f}m perturbation)"
        else:
            return f"Stage {stage}: RANDOM (completely random)"

    def run(self):
        print("\n" + "="*60)
        print("Stage 5 Progressive Inspector")
        print("="*60)
        print("\nControls:")
        print("  1-9, 0: Switch to Stage 0-9")
        print("  Q, W, E: Switch to Stage 10, 11, 12")
        print("  LEFT/RIGHT: Previous/Next stage")
        print("  R: Reload current stage (re-randomize)")
        print("  H: Toggle info display")
        print("  Space: Pause/Continue")
        print("  ESC: Quit")
        print("\nCamera:")
        print("  WASD: Camera Move")
        print("  Mouse Right-Click + Drag: Camera Rotate")
        print("="*60)

        # 初始相机位置
        self.renderer.cam_pos = np.array([-5.0, 0.0, -10.0], dtype=np.float32)

        while self.running:
            dt_ms = self.clock.tick(30)
            dt = dt_ms / 1000.0
            if dt > 0.1:
                dt = 0.1

            self.handle_events()

            if not self.paused:
                # 更新物理
                self.logic_stage.update(dt)

            # 渲染
            # 为renderer提供必要的score_data（使用虚拟值）
            score_data = {
                'episode_reward': 0.0,
                'total_episodes': 0,
                'success_count': 0,
                'episode_steps': 0,
                'collision_predicted': False,
                # 自定义信息（用于显示在控制台）
                '_stage': self.current_stage,
                '_stage_description': self.get_stage_description(),
                '_num_obstacles': len(self.proving_ground.obstacles),
                '_show_info': self.show_info,
            }
            self.renderer.render(self.logic_stage, point_cloud=None, score_data=score_data)

        pygame.quit()


def main():
    app = Stage5Inspector()
    app.run()


if __name__ == "__main__":
    main()
