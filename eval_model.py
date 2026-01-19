#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_model.py: 评估训练好的模型 (与 train.py 配置匹配)

用法:
    python eval_model.py --model ./training_output/xxx/best_model/best_model.zip
    python eval_model.py --model ./training_output/xxx/final_model.zip --manual
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import argparse
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pygame
from pygame.locals import *

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.rl.submarine_env import SubmarineEnv
from src.rl.tunnel import TunnelConfig
from src.rl.point_cloud_sampler import PointCloudConfig
from src.core.stage import LogicStage
from src.physics.submarine_actor import SubmarineActor
from src.gameplay.obstacle_actor import ObstacleActor
from src.gameplay.tunnel_actor import TunnelActor
from src.view.renderer import Renderer


class EvalApp:
    """评估可视化应用 (支持命令行配置)"""
    
    def __init__(self, model_path: str, vecnorm_path: str = None,
                 manual_mode: bool = False, 
                 # 通道参数
                 tunnel_radius: float = 6.0,
                 tunnel_length: float = 50.0,
                 num_obstacles: int = 18,
                 obstacle_radius_min: float = 0.6,
                 obstacle_radius_max: float = 1.0,
                 obstacle_min_spacing: float = 1.5,
                 obstacle_start_x: float = 10.0,  # 【新增】障碍物起始X位置
                 # 点云参数
                 sample_distance: float = 15.0,
                 num_points: int = 512):
        pygame.init()
        self.width, self.height = 1280, 900
        
        pygame.display.set_mode((self.width, self.height), DOUBLEBUF | OPENGL | RESIZABLE)
        pygame.display.set_caption("Submarine RL Evaluation (v10)")
        
        self.manual_mode = manual_mode
        
        # ============ 使用传入的参数配置 ============
        self.tunnel_config = TunnelConfig(
            radius=tunnel_radius,
            length=tunnel_length,
            center_z=100.0,
            num_obstacles=num_obstacles,
            obstacle_radius_min=obstacle_radius_min,
            obstacle_radius_max=obstacle_radius_max,
            obstacle_min_spacing=obstacle_min_spacing,
            obstacle_start_x=obstacle_start_x,  # 【新增】传递障碍物起始位置
        )
        
        # 点云采样配置
        self.point_cloud_config = PointCloudConfig(
            num_points=num_points,  # 使用传入的点云数量
            obstacle_points_ratio=0.5,
            sample_distance=sample_distance,
            normalize_range=sample_distance,
            points_per_obstacle=32,
        )
        
        # 创建 Logic Stage 用于渲染
        self.logic_stage = LogicStage()
        self.submarine = SubmarineActor("HeroSub")
        self.logic_stage.add_actor(self.submarine)
        
        # Add Tunnel Actor
        self.tunnel_actor = TunnelActor(config=self.tunnel_config)
        self.logic_stage.add_actor(self.tunnel_actor)

        # 创建环境 (与 train.py 一致的参数)
        self.env = SubmarineEnv(
            tunnel_config=self.tunnel_config,
            point_cloud_config=self.point_cloud_config,
            point_cloud_history_len=8,  # 与 train.py 一致
            max_steps=1200,
        )
        
        # 加载模型
        self.model = None
        self.vec_normalize = None
        
        if model_path and os.path.exists(model_path):
            print(f"Loading model from: {model_path}")
            self.model = PPO.load(model_path)
            
            # 尝试加载 VecNormalize 统计信息
            if vecnorm_path and os.path.exists(vecnorm_path):
                print(f"Loading VecNormalize from: {vecnorm_path}")
                # 创建一个 dummy VecEnv 来加载 VecNormalize
                dummy_env = DummyVecEnv([lambda: SubmarineEnv(
                    tunnel_config=self.tunnel_config,
                    point_cloud_config=self.point_cloud_config,
                    point_cloud_history_len=8,
                    max_steps=1200,
                )])
                self.vec_normalize = VecNormalize.load(vecnorm_path, dummy_env)
                self.vec_normalize.training = False  # 评估模式
                self.vec_normalize.norm_reward = False
                print("VecNormalize loaded successfully!")
            else:
                print("Warning: VecNormalize file not found. Observations may not be normalized correctly.")
                print("  Expected path: vecnormalize.pkl in model directory")
        elif not manual_mode:
            print("No model specified. Running with random actions.")
        
        # 渲染器
        self.renderer = Renderer(self.width, self.height)
        self.renderer.tunnel_config = self.tunnel_config
        
        self.clock = pygame.time.Clock()
        self.running = True
        self.paused = False
        
        self.episode_reward = 0
        self.episode_steps = 0
        self.total_episodes = 0
        self.success_count = 0
        
        # 统计信息
        self.all_rewards = []
        self.all_steps = []
        
        # 初始化
        self._reset_episode()
    
    def _reset_episode(self):
        """重置回合"""
        obs, info = self.env.reset()
        self.obs = obs
        self.episode_reward = 0
        self.episode_steps = 0
        self._sync_submarine_state()
        self._sync_obstacle_actors()
    
    def _sync_obstacle_actors(self):
        """同步障碍物 Actor 到 logic_stage"""
        self.logic_stage.actors = [a for a in self.logic_stage.actors if not isinstance(a, ObstacleActor)]
        
        tunnel = self.env.get_tunnel()
        if tunnel:
            for i, obs in enumerate(tunnel.obstacles):
                obs_actor = ObstacleActor(
                    name=f"Obstacle_{i}",
                    x=obs.position[0],
                    y=obs.position[1],
                    z=obs.position[2],
                    radius=obs.radius
                )
                self.logic_stage.add_actor(obs_actor)
        
        self.renderer.tunnel = tunnel
    
    def _sync_submarine_state(self):
        """同步环境状态到潜艇 Actor"""
        state = self.env.get_state()
        self.submarine.eta = state['eta']
        self.submarine.nu = state['nu']
        self.submarine.u_actual = state['u_actual']
        self.submarine._sync_state()
    
    def _normalize_obs(self, obs):
        """如果有 VecNormalize，则归一化观测"""
        if self.vec_normalize is not None:
            # VecNormalize 期望 batch 输入
            obs_batch = {k: np.expand_dims(v, 0) for k, v in obs.items()}
            normalized = self.vec_normalize.normalize_obs(obs_batch)
            return {k: v[0] for k, v in normalized.items()}
        return obs
    
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
                    self._reset_episode()
                    print("Episode reset.")
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                    print("Paused." if self.paused else "Resumed.")
                elif event.key == pygame.K_s:
                    self._print_stats()
    
    def _print_stats(self):
        """打印统计信息"""
        print("\n" + "=" * 50)
        print("Evaluation Statistics:")
        print("=" * 50)
        print(f"  Total Episodes: {self.total_episodes}")
        print(f"  Success Count: {self.success_count}")
        if self.total_episodes > 0:
            print(f"  Success Rate: {self.success_count / self.total_episodes * 100:.1f}%")
            print(f"  Average Reward: {np.mean(self.all_rewards):.2f}")
            print(f"  Average Steps: {np.mean(self.all_steps):.0f}")
        print("=" * 50 + "\n")
    
    def get_manual_action(self, keys):
        """获取手动控制的动作"""
        pitch = 0.0
        yaw = 0.0
        thrust = 0.3
        
        if keys[pygame.K_i]:
            pitch = 0.5
        if keys[pygame.K_k]:
            pitch = -0.5
        if keys[pygame.K_j]:
            yaw = 0.5
        if keys[pygame.K_l]:
            yaw = -0.5
            
        if keys[pygame.K_UP]:
            thrust = 0.8
        if keys[pygame.K_DOWN]:
            thrust = 0.0
        
        return np.array([pitch, yaw, thrust], dtype=np.float32)
    
    def run(self):
        print("\n" + "=" * 50)
        print("Submarine RL Evaluation (v9)")
        print("=" * 50)
        print("Controls:")
        print("  WASD: Camera Move")
        print("  Mouse Right-Click + Drag: Camera Rotate")
        print("  IJKL: Manual control (pitch/yaw)")
        print("  Arrow Up/Down: Adjust thrust")
        print("  R: Reset episode")
        print("  Space: Pause/Resume")
        print("  S: Print statistics")
        print("  ESC: Quit")
        print("=" * 50 + "\n")
        
        self.renderer.cam_pos = np.array([-10.0, 0.0, -5.0], dtype=np.float32)
        
        while self.running:
            dt_ms = self.clock.tick(30)
            
            self.handle_events()
            
            if not self.paused:
                keys = pygame.key.get_pressed()
                
                if self.manual_mode:
                    action = self.get_manual_action(keys)
                elif self.model is not None:
                    # 归一化观测
                    normalized_obs = self._normalize_obs(self.obs)
                    action, _ = self.model.predict(normalized_obs, deterministic=True)
                else:
                    action = self.env.action_space.sample()
                    action[2] = 0.5
                
                # 执行动作
                self.obs, reward, terminated, truncated, info = self.env.step(action)
                self.episode_reward += reward
                self.episode_steps += 1
                
                # 同步状态
                self._sync_submarine_state()
                
                # 检查回合结束
                if terminated or truncated:
                    self.total_episodes += 1
                    success = info.get('goal_reached', False)
                    
                    self.all_rewards.append(self.episode_reward)
                    self.all_steps.append(self.episode_steps)
                    
                    if success:
                        self.success_count += 1
                        print(f"Episode {self.total_episodes}: ✓ SUCCESS! Reward: {self.episode_reward:.2f}, Steps: {self.episode_steps}")
                    else:
                        reason = info.get('collision_reason', 'timeout')
                        print(f"Episode {self.total_episodes}: ✗ FAILED ({reason}). Reward: {self.episode_reward:.2f}, Steps: {self.episode_steps}")
                    
                    success_rate = self.success_count / self.total_episodes * 100
                    print(f"  Success rate: {success_rate:.1f}% ({self.success_count}/{self.total_episodes})")
                    
                    self._reset_episode()
            
            # Extract Point Cloud for rendering
            point_cloud = None
            if isinstance(self.obs, dict) and 'point_cloud_seq' in self.obs:
                # Get latest frame from the sequence
                pc_seq = self.obs['point_cloud_seq']
                latest_pc = pc_seq[-1]  # (num_points * 3,)
                
                # Reshape to (N, 3)
                point_cloud = latest_pc.reshape(-1, 3)
                
                # Denormalize (multiply by normalize_range)
                point_cloud = point_cloud * self.point_cloud_config.normalize_range

            # 渲染
            self.renderer.render(self.logic_stage, point_cloud=point_cloud)
            
            # 定期打印调试信息
            if self.episode_steps % 60 == 0 and self.episode_steps > 0:
                pos = self.submarine.position
                progress = self.env.tunnel.get_progress(pos)
                speed = self.env.nu[0]
                print(f"  Step {self.episode_steps}: X={pos[0]:.1f}m, Progress={progress*100:.1f}%, Speed={speed:.2f}m/s")
        
        self._print_stats()
        pygame.quit()
        self.env.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained submarine model (v10 - 支持课程阶段配置)")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to trained model (.zip file)")
    parser.add_argument("--vecnorm", type=str, default=None,
                        help="Path to VecNormalize file (vecnormalize.pkl)")
    parser.add_argument("--manual", action="store_true",
                        help="Manual control mode")
    
    # 快捷选项：使用课程阶段配置
    # 动态获取总阶段数
    try:
        from src.rl.curriculum import get_total_stages
        total_stages = get_total_stages()
    except ImportError:
        total_stages = 10  # 默认值
    
    parser.add_argument("--stage", type=int, default=None, choices=list(range(1, total_stages + 1)),
                        help=f"Use curriculum stage config (1-{total_stages}). Overrides other tunnel params.")
    
    # 通道参数
    parser.add_argument("--tunnel-radius", type=float, default=6.0,
                        help="Tunnel radius in meters (default: 6.0)")
    parser.add_argument("--tunnel-length", type=float, default=50.0,
                        help="Tunnel length in meters (default: 50.0)")
    parser.add_argument("--num-obstacles", type=int, default=18,
                        help="Number of obstacles (default: 18)")
    parser.add_argument("--obstacle-radius-min", type=float, default=0.6,
                        help="Min obstacle radius (default: 0.6)")
    parser.add_argument("--obstacle-radius-max", type=float, default=1.0,
                        help="Max obstacle radius (default: 1.0)")
    parser.add_argument("--obstacle-spacing", type=float, default=1.5,
                        help="Min obstacle spacing (default: 1.5)")
    parser.add_argument("--obstacle-start-x", type=float, default=10.0,
                        help="Obstacle start X position (default: 10.0)")
    
    # 点云参数
    parser.add_argument("--sample-distance", type=float, default=15.0,
                        help="Point cloud sample distance (default: 15.0)")
    parser.add_argument("--num-points", type=int, default=512,
                        help="Number of point cloud points (default: 512)")
    
    args = parser.parse_args()
    
    # 如果指定了 --stage，使用课程阶段配置
    if args.stage is not None:
        try:
            from src.rl.curriculum import get_stage_config
            stage = get_stage_config(args.stage - 1)  # 0-indexed
            print(f"\n📚 使用课程阶段 {args.stage} 配置: {stage.name}")
            print(f"   通道: 半径={stage.tunnel_radius}m, 长度={stage.tunnel_length}m")
            print(f"   障碍物: {stage.num_obstacles}个\n")
            
            args.tunnel_radius = stage.tunnel_radius
            args.tunnel_length = stage.tunnel_length
            args.num_obstacles = stage.num_obstacles
            args.obstacle_radius_min = stage.obstacle_radius_min
            args.obstacle_radius_max = stage.obstacle_radius_max
            args.obstacle_spacing = stage.obstacle_min_spacing
            args.obstacle_start_x = stage.obstacle_start_x  # 【新增】读取障碍物起始位置
            args.sample_distance = stage.sample_distance
            args.num_points = stage.num_points
        except ImportError:
            print("Warning: Cannot import curriculum config, using default values")
    
    # 自动查找 vecnormalize.pkl
    vecnorm_path = args.vecnorm
    if vecnorm_path is None and args.model:
        model_dir = os.path.dirname(args.model)
        possible_paths = [
            os.path.join(model_dir, "vecnormalize.pkl"),
            os.path.join(os.path.dirname(model_dir), "vecnormalize.pkl"),
        ]
        for path in possible_paths:
            if os.path.exists(path):
                vecnorm_path = path
                break
    
    app = EvalApp(
        model_path=args.model,
        vecnorm_path=vecnorm_path,
        manual_mode=args.manual,
        tunnel_radius=args.tunnel_radius,
        tunnel_length=args.tunnel_length,
        num_obstacles=args.num_obstacles,
        obstacle_radius_min=args.obstacle_radius_min,
        obstacle_radius_max=args.obstacle_radius_max,
        obstacle_min_spacing=args.obstacle_spacing,
        obstacle_start_x=args.obstacle_start_x,  # 【新增】传递障碍物起始位置
        sample_distance=args.sample_distance,
        num_points=args.num_points,
    )
    app.run()


if __name__ == "__main__":
    main()
