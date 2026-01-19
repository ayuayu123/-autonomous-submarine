#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_sim.py: 评估训练好的模型并在 pygame 中可视化
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import argparse
import numpy as np

from src.gameplay.tunnel_actor import TunnelActor

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pygame
from pygame.locals import *

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from src.rl.submarine_env import SubmarineEnv
from src.rl.tunnel import TunnelConfig
from src.rl.point_cloud_sampler import PointCloudConfig
from src.core.stage import LogicStage
from src.physics.submarine_actor import SubmarineActor
from src.gameplay.obstacle_actor import ObstacleActor
from src.view.renderer import Renderer


class EvalApp:
    """评估可视化应用"""
    
    def __init__(self, model_path: str, use_lidar: bool = True,
                 manual_mode: bool = False,
                 save_data: bool = True, save_dir: str = "./eval_data",
                 vecnormalize_path: str = None,
                 save_csv: bool = True):
        pygame.init()
        self.width, self.height = 1280, 900
        
        pygame.display.set_mode((self.width, self.height), DOUBLEBUF | OPENGL | RESIZABLE)
        pygame.display.set_caption("Submarine RL Evaluation")
        
        self.manual_mode = manual_mode
        self.save_data = save_data
        self.save_dir = save_dir
        self.vecnormalize_path = vecnormalize_path
        self.save_csv = save_csv
        
        # 数据收集
        self.current_episode_data = {
            'point_clouds': [],
            'states': [],
            'actions': [],
            'rewards': [],
            'positions': []
        }
        self.episode_counter = 0
        
        # CSV数据收集
        self.csv_trajectory = []
        self.csv_obstacles = []
        self.csv_point_clouds = []
        
        # 创建 Logic Stage 用于渲染
        self.logic_stage = LogicStage()
        self.submarine = SubmarineActor("HeroSub")
        self.logic_stage.add_actor(self.submarine)
        
        # Add Tunnel Actor (使用 tunnel.py 中的默认配置)
        self.tunnel_actor = TunnelActor()
        self.logic_stage.add_actor(self.tunnel_actor)
        
        # Sync config for Env
        self.tunnel_config = self.tunnel_actor.config
        
        
        self.point_cloud_config = PointCloudConfig(
            num_points=256,
            normalize_range=50.0
        )

        # 创建环境
        raw_env = SubmarineEnv(
            tunnel_config=self.tunnel_config,
            point_cloud_config=self.point_cloud_config,
            point_cloud_history_len=60,
            max_steps=1200,
        )
        
        # 使用 VecNormalize 包装环境（如果提供了 vecnormalize_path）
        if self.vecnormalize_path and os.path.exists(self.vecnormalize_path):
            print(f"Loading VecNormalize statistics from: {self.vecnormalize_path}")
            vec_env = DummyVecEnv([lambda: raw_env])
            self.env = VecNormalize.load(self.vecnormalize_path, vec_env)
            self.env.training = False  # 评估模式，不更新统计信息
            self.env.norm_reward = False  # 评估时不归一化奖励
        else:
            self.env = raw_env
            if self.vecnormalize_path:
                print(f"Warning: VecNormalize file not found at {self.vecnormalize_path}")
        
        # 加载模型
        self.model = None
        if model_path and os.path.exists(model_path):
            print(f"Loading model from: {model_path}")
            self.model = PPO.load(model_path)
        elif not manual_mode:
            print("No model specified. Running with random actions.")
        
        # 渲染器
        self.renderer = Renderer(self.width, self.height)
        
        # 设置渲染器配置
        self.renderer.tunnel_config = self.tunnel_config
        
        self.clock = pygame.time.Clock()
        self.running = True
        self.paused = False
        
        self.episode_reward = 0
        self.episode_steps = 0
        self.total_episodes = 0
        self.success_count = 0
        
        # 初始化
        self._reset_episode()
    
    def _reset_episode(self):
        """重置回合"""
        obs, info = self.env.reset()
        self.obs = obs
        self.episode_reward = 0
        self.episode_steps = 0
        self._sync_submarine_state()
        
        # 同步更新可视化的障碍物 Actor
        self._sync_obstacle_actors()
        
        # 清空当前回合数据
        self.current_episode_data = {
            'point_clouds': [],
            'states': [],
            'actions': [],
            'rewards': [],
            'positions': []
        }
        
        # 清空CSV数据
        self.csv_trajectory = []
        self.csv_obstacles = []
        self.csv_point_clouds = []
    
    def _sync_obstacle_actors(self):
        """同步障碍物 Actor 到 logic_stage（每次 reset 后调用）"""
        # 移除旧的障碍物 Actor
        self.logic_stage.actors = [a for a in self.logic_stage.actors if not isinstance(a, ObstacleActor)]
        
        # 添加新的障碍物 Actor
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
        
        # 同时更新渲染器中的 tunnel 引用
        self.renderer.tunnel = tunnel
    
    def _collect_data(self, action, reward):
        """收集当前步的数据"""
        if not self.save_data:
            return
        
        # 收集点云数据
        if isinstance(self.obs, dict) and 'point_cloud_seq' in self.obs:
            pc_seq = self.obs['point_cloud_seq']
            latest_pc = pc_seq[-1].reshape(-1, 3)
            self.current_episode_data['point_clouds'].append(latest_pc)
        
        # 收集状态数据
        if isinstance(self.obs, dict) and 'state' in self.obs:
            self.current_episode_data['states'].append(self.obs['state'])
        
        # 收集动作和奖励
        self.current_episode_data['actions'].append(action)
        self.current_episode_data['rewards'].append(reward)
        
        # 收集位置数据
        state = self.env.get_state()
        self.current_episode_data['positions'].append(state['eta'][0:3])
    
    def _collect_csv_data(self, action, reward):
        """收集CSV格式数据"""
        if not self.save_csv:
            return
        
        state = self.env.get_state()
        eta = state['eta']
        
        # 收集轨迹数据
        trajectory_row = {
            'step': self.episode_steps,
            'x': eta[0],
            'y': eta[1],
            'z': eta[2],
            'roll': eta[3],
            'pitch': eta[4],
            'yaw': eta[5],
            'u': state['nu'][0],
            'v': state['nu'][1],
            'w': state['nu'][2],
            'p': state['nu'][3],
            'q': state['nu'][4],
            'r': state['nu'][5],
            'action_pitch': action[0],
            'action_yaw': action[1],
            'action_thrust': action[2],
            'reward': reward,
            'target_rpm': self.submarine.target_rpm,
            'rudder_angle': self.submarine.rudder_angle,
            'stern_angle': self.submarine.stern_angle
        }
        self.csv_trajectory.append(trajectory_row)
        
        # 收集障碍物数据（只收集一次）
        if len(self.csv_obstacles) == 0:
            tunnel = self.env.get_tunnel()
            if tunnel:
                for obs in tunnel.obstacles:
                    obstacle_row = {
                        'obstacle_id': len(self.csv_obstacles),
                        'x': obs.position[0],
                        'y': obs.position[1],
                        'z': obs.position[2],
                        'radius': obs.radius
                    }
                    self.csv_obstacles.append(obstacle_row)
        
        # 收集点云数据
        if isinstance(self.obs, dict) and 'point_cloud_seq' in self.obs:
            pc_seq = self.obs['point_cloud_seq']
            latest_pc = pc_seq[-1].reshape(-1, 3)
            latest_pc = latest_pc * 50.0  # Denormalize
            
            for i, point in enumerate(latest_pc):
                point_cloud_row = {
                    'step': self.episode_steps,
                    'point_id': i,
                    'x': point[0],
                    'y': point[1],
                    'z': point[2]
                }
                self.csv_point_clouds.append(point_cloud_row)
    
    def _save_episode_data(self, success: bool):
        """保存当前回合的数据"""
        if not self.save_data:
            return
        
        import os
        
        # 创建保存目录
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 转换为numpy数组
        point_clouds = np.array(self.current_episode_data['point_clouds'])
        states = np.array(self.current_episode_data['states'])
        actions = np.array(self.current_episode_data['actions'])
        rewards = np.array(self.current_episode_data['rewards'])
        positions = np.array(self.current_episode_data['positions'])
        
        # 保存文件
        episode_dir = os.path.join(self.save_dir, f"episode_{self.episode_counter:04d}")
        os.makedirs(episode_dir, exist_ok=True)
        
        np.save(os.path.join(episode_dir, 'point_clouds.npy'), point_clouds)
        np.save(os.path.join(episode_dir, 'states.npy'), states)
        np.save(os.path.join(episode_dir, 'actions.npy'), actions)
        np.save(os.path.join(episode_dir, 'rewards.npy'), rewards)
        np.save(os.path.join(episode_dir, 'positions.npy'), positions)
        
        # 保存元数据
        metadata = {
            'episode_id': int(self.episode_counter),
            'success': bool(success),
            'total_reward': float(np.sum(rewards)),
            'num_steps': int(len(rewards)),
            'point_clouds_shape': [int(x) for x in point_clouds.shape],
            'states_shape': [int(x) for x in states.shape],
            'actions_shape': [int(x) for x in actions.shape],
            'rewards_shape': [int(x) for x in rewards.shape],
            'positions_shape': [int(x) for x in positions.shape]
        }
        
        import json
        with open(os.path.join(episode_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Episode {self.episode_counter} data saved to {episode_dir}")
        print(f"  - point_clouds.npy: {point_clouds.shape}")
        print(f"  - states.npy: {states.shape}")
        print(f"  - actions.npy: {actions.shape}")
        print(f"  - rewards.npy: {rewards.shape}")
        print(f"  - positions.npy: {positions.shape}")
        print(f"  - metadata.json: {metadata}")
        
        self.episode_counter += 1
    
    def _save_csv_data(self, success: bool):
        """保存CSV格式数据（仅在成功时保存）"""
        if not self.save_csv or not success:
            return
        
        import os
        import pandas as pd
        
        # 创建保存目录
        csv_dir = os.path.join(self.save_dir, "csv_data")
        os.makedirs(csv_dir, exist_ok=True)
        
        # 保存轨迹CSV
        trajectory_df = pd.DataFrame(self.csv_trajectory)
        trajectory_path = os.path.join(csv_dir, f"trajectory_{self.episode_counter:04d}.csv")
        trajectory_df.to_csv(trajectory_path, index=False)
        print(f"CSV trajectory saved: {trajectory_path}")
        
        # 保存障碍物CSV
        obstacles_df = pd.DataFrame(self.csv_obstacles)
        obstacles_path = os.path.join(csv_dir, f"obstacles_{self.episode_counter:04d}.csv")
        obstacles_df.to_csv(obstacles_path, index=False)
        print(f"CSV obstacles saved: {obstacles_path}")
        
        # 保存点云CSV
        point_cloud_df = pd.DataFrame(self.csv_point_clouds)
        point_cloud_path = os.path.join(csv_dir, f"point_clouds_{self.episode_counter:04d}.csv")
        point_cloud_df.to_csv(point_cloud_path, index=False)
        print(f"CSV point clouds saved: {point_cloud_path}")
    
    def _sync_submarine_state(self):
        """同步环境状态到潜艇 Actor"""
        state = self.env.get_state()
        self.submarine.eta = state['eta']
        self.submarine.nu = state['nu']
        self.submarine.u_actual = state['u_actual']
        self.submarine._sync_state()
    
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
    
    def get_manual_action(self, keys):
        """获取手动控制的动作"""
        pitch = 0.0
        yaw = 0.0
        thrust = 0.3  # 默认推力
        
        # WASD used for Camera now. Use Arrow keys for steering?
        # Or IJKL?
        if keys[pygame.K_i]:
            pitch = 0.5
        if keys[pygame.K_k]:
            pitch = -0.5
        if keys[pygame.K_j]:
            yaw = 0.5
        if keys[pygame.K_l]:
            yaw = -0.5
            
        if keys[pygame.K_UP]:
            thrust = min(thrust + 0.1, 1.0)
        if keys[pygame.K_DOWN]:
            thrust = max(thrust - 0.1, 0.0)
        
        return np.array([pitch, yaw, thrust], dtype=np.float32)
    
    def run(self):
        print("\nControls:")
        print("  WASD: Camera Move")
        print("  Mouse Right-Click + Drag: Camera Rotate")
        print("  IJKL: Manual control (pitch/yaw) (if manual mode)")
        print("  Arrow Up/Down: Adjust thrust")
        print("  R: Reset episode")
        print("  Space: Pause/Resume")
        print("  ESC: Quit")
        print()
        
        # Set initial camera position
        self.renderer.cam_pos = np.array([-10.0, 0.0, -5.0], dtype=np.float32)
        
        while self.running:
            dt_ms = self.clock.tick(30)
            dt = dt_ms / 1000.0
            if dt > 0.1:
                dt = 0.1
            
            self.handle_events()
            
            if not self.paused:
                # 获取动作
                keys = pygame.key.get_pressed()
                
                if self.manual_mode:
                    action = self.get_manual_action(keys)
                elif self.model is not None:
                    action, _ = self.model.predict(self.obs, deterministic=True)
                else:
                    action = self.env.action_space.sample()
                    action[2] = 0.5  # 固定推力
                
                # 执行动作
                self.obs, reward, terminated, truncated, info = self.env.step(action)
                self.episode_reward += reward
                self.episode_steps += 1
                
                # 收集数据
                self._collect_data(action, reward)
                self._collect_csv_data(action, reward)
                
                # 同步状态
                self._sync_submarine_state()
                
                # 检查回合结束
                if terminated or truncated:
                    self.total_episodes += 1
                    success = info.get('goal_reached', False)
                    
                    if success:
                        self.success_count += 1
                        print(f"Episode {self.total_episodes}: SUCCESS! Reward: {self.episode_reward:.2f}, Steps: {self.episode_steps}")
                    else:
                        reason = info.get('collision_reason', 'timeout')
                        print(f"Episode {self.total_episodes}: FAILED ({reason}). Reward: {self.episode_reward:.2f}, Steps: {self.episode_steps}")
                    
                    success_rate = self.success_count / self.total_episodes * 100
                    print(f"  Success rate: {success_rate:.1f}% ({self.success_count}/{self.total_episodes})")
                    
                    # 保存数据
                    self._save_episode_data(success)
                    self._save_csv_data(success)
                    
                    self._reset_episode()
            
            # Extract Point Cloud for rendering
            point_cloud = None
            if isinstance(self.obs, dict) and 'point_cloud_seq' in self.obs:
                # Get latest frame: (N*3,) or (N, 3)?
                # Shape is (history_len, num_points * 3)
                # We want the last one
                pc_seq = self.obs['point_cloud_seq']
                latest_pc = pc_seq[-1] # (num_points * 3,)
                
                # Reshape to (N, 3)
                point_cloud = latest_pc.reshape(-1, 3)
                
                # Denormalize (multiply by 50.0)
                point_cloud = point_cloud * 50.0

            # 渲染
            score_data = {
                'episode_reward': self.episode_reward,
                'total_episodes': self.total_episodes,
                'success_count': self.success_count,
                'episode_steps': self.episode_steps
            }
            self.renderer.render(self.logic_stage, point_cloud=point_cloud, score_data=score_data)
            
            # Print debug info periodically
            if self.episode_steps % 30 == 0:
                pos = self.submarine.position
                print(f"Submarine Pos: X={pos[0]:.2f}, Y={pos[1]:.2f}, Z={pos[2]:.2f}")
                
                if point_cloud is not None and len(point_cloud) > 0:
                    # Sample 5 points
                    num_samples = min(5, len(point_cloud))
                    indices = np.random.choice(len(point_cloud), num_samples, replace=False)
                    samples = point_cloud[indices]
                    print(f"Point Cloud Samples (Rel to Sub, {num_samples} pts):")
                    for i, p in enumerate(samples):
                        print(f"  P{i}: [{p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}]")
                else:
                    print("Point Cloud: None")
                print("-" * 30)
        
        pygame.quit()
        self.env.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained submarine model")
    parser.add_argument("--model", type=str, default="./best_model2.zip",
                        help="Path to trained model (e.g., ./training_output/xxx/best_model/best_model.zip)")
    parser.add_argument("--vecnormalize", type=str, default=None,
                        help="Path to VecNormalize statistics (e.g., ./training_output/xxx/vecnormalize.pkl)")
    parser.add_argument("--no-lidar", action="store_true",
                        help="Disable LiDAR")
    parser.add_argument("--manual", action="store_true",
                        help="Manual control mode")
    parser.add_argument("--save-data", action="store_true", default=True,
                        help="Save episode data (point clouds, states, actions, rewards, positions)")
    parser.add_argument("--no-save-data", action="store_true",
                        help="Disable data saving")
    parser.add_argument("--save-dir", type=str, default="./eval_data",
                        help="Directory to save episode data")
    parser.add_argument("--no-save-csv", action="store_true",
                        help="Disable CSV saving")
    
    args = parser.parse_args()
    
    # 处理数据保存选项
    save_data = args.save_data and not args.no_save_data
    save_csv = not args.no_save_csv
    
    app = EvalApp(
        model_path=args.model,
        use_lidar=not args.no_lidar,
        manual_mode=args.manual,
        save_data=save_data,
        save_dir=args.save_dir,
        vecnormalize_path=args.vecnormalize,
        save_csv=save_csv,
    )
    app.run()


if __name__ == "__main__":
    main()
