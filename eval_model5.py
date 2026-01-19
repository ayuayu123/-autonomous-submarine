#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_model5.py: 评估 best_model (1).zip 模型，支持自定义网络架构
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import argparse
import numpy as np

from src.gameplay.tunnel_actor import TunnelActor

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
from src.rl.custom_extractor import PointCloudLSTMExtractor


class EvalApp:
    
    def __init__(self, model_path: str, use_lidar: bool = True,
                 manual_mode: bool = False,
                 save_data: bool = True, save_dir: str = "./eval_data_model5",
                 vecnormalize_path: str = None,
                 save_csv: bool = True,
                 lstm_hidden_size: int = 128,
                 lstm_num_layers: int = 1,
                 pc_encoder_dims: tuple = (256,),
                 state_feature_dim: int = 64,
                 dropout: float = 0.1,
                 use_layer_norm: bool = True,
                 bidirectional: bool = False,
                 pi_hidden: tuple = (256, 128),
                 vf_hidden: tuple = (256, 128)):
        pygame.init()
        self.width, self.height = 1280, 900
        
        pygame.display.set_mode((self.width, self.height), DOUBLEBUF | OPENGL | RESIZABLE)
        pygame.display.set_caption("Submarine RL Evaluation - Model 5")
        
        self.manual_mode = manual_mode
        self.save_data = save_data
        self.save_dir = save_dir
        self.vecnormalize_path = vecnormalize_path
        self.save_csv = save_csv
        
        self.current_episode_data = {
            'point_clouds': [],
            'states': [],
            'actions': [],
            'rewards': [],
            'positions': []
        }
        self.episode_counter = 0
        
        self.csv_trajectory = []
        self.csv_obstacles = []
        self.csv_point_clouds = []
        
        self.logic_stage = LogicStage()
        self.submarine = SubmarineActor("HeroSub")
        self.logic_stage.add_actor(self.submarine)
        
        self.tunnel_actor = TunnelActor(num_obstacles=15, radius=6)
        self.logic_stage.add_actor(self.tunnel_actor)
        
        self.tunnel_config = self.tunnel_actor.config
        
        self.point_cloud_config = PointCloudConfig(
            num_points=256,
            normalize_range=50.0
        )

        raw_env = SubmarineEnv(
            tunnel_config=self.tunnel_config,
            point_cloud_config=self.point_cloud_config,
            point_cloud_history_len=16,
            max_steps=1200,
        )
        
        self.vec_normalize = None
        if self.vecnormalize_path and os.path.exists(self.vecnormalize_path):
            print(f"Loading VecNormalize from: {self.vecnormalize_path}")
            dummy_env = DummyVecEnv([lambda: SubmarineEnv(
                tunnel_config=self.tunnel_config,
                point_cloud_config=self.point_cloud_config,
                point_cloud_history_len=16,
                max_steps=1200,
            )])
            self.vec_normalize = VecNormalize.load(self.vecnormalize_path, dummy_env)
            self.vec_normalize.training = False
            self.vec_normalize.norm_reward = False
            print("VecNormalize loaded successfully!")
        else:
            if self.vecnormalize_path:
                print(f"Warning: VecNormalize file not found at {self.vecnormalize_path}")
        
        self.env = raw_env
        
        self.model = None
        if model_path and os.path.exists(model_path):
            print(f"Loading model from: {model_path}")
            
            policy_kwargs = dict(
                features_extractor_class=PointCloudLSTMExtractor,
                features_extractor_kwargs=dict(
                    lstm_hidden_size=lstm_hidden_size,
                    lstm_num_layers=lstm_num_layers,
                    point_cloud_encoder_dims=pc_encoder_dims,
                    state_feature_dim=state_feature_dim,
                    dropout=dropout,
                    use_layer_norm=use_layer_norm,
                    bidirectional=bidirectional,
                ),
                net_arch=dict(
                    pi=list(pi_hidden),
                    vf=list(vf_hidden),
                ),
            )
            
            print(f"\nModel architecture:")
            print(f"  - LSTM hidden size: {lstm_hidden_size}")
            print(f"  - LSTM num layers: {lstm_num_layers}")
            print(f"  - Point cloud encoder dims: {pc_encoder_dims}")
            print(f"  - State feature dim: {state_feature_dim}")
            print(f"  - Dropout: {dropout}")
            print(f"  - LayerNorm: {use_layer_norm}")
            print(f"  - Bidirectional: {bidirectional}")
            print(f"  - Actor network: {list(pi_hidden)}")
            print(f"  - Critic network: {list(vf_hidden)}")
            
            self.model = PPO.load(model_path, policy_kwargs=policy_kwargs)
        elif not manual_mode:
            print("No model specified. Running with random actions.")
        
        self.renderer = Renderer(self.width, self.height)
        self.renderer.tunnel_config = self.tunnel_config
        
        self.clock = pygame.time.Clock()
        self.running = True
        self.paused = False
        
        self.episode_reward = 0
        self.episode_steps = 0
        self.total_episodes = 0
        self.success_count = 0
        
        self._reset_episode()
    
    def _reset_episode(self):
        obs, info = self.env.reset()
        self.obs = obs
        self.episode_reward = 0
        self.episode_steps = 0
        self._sync_submarine_state()
        self._sync_obstacle_actors()
        
        self.current_episode_data = {
            'point_clouds': [],
            'states': [],
            'actions': [],
            'rewards': [],
            'positions': []
        }
        
        self.csv_trajectory = []
        self.csv_obstacles = []
        self.csv_point_clouds = []
    
    def _normalize_obs(self, obs):
        if self.vec_normalize is not None:
            obs_batch = {k: np.expand_dims(v, 0) for k, v in obs.items()}
            normalized = self.vec_normalize.normalize_obs(obs_batch)
            return {k: v[0] for k, v in normalized.items()}
        return obs
    
    def _get_raw_env(self):
        if hasattr(self.env, 'env'):
            return self.env.env
        elif hasattr(self.env, 'envs'):
            return self.env.envs[0]
        else:
            return self.env
    
    def _sync_obstacle_actors(self):
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
    
    def _collect_data(self, action, reward):
        if not self.save_data:
            return
        
        if isinstance(self.obs, dict) and 'point_cloud_seq' in self.obs:
            pc_seq = self.obs['point_cloud_seq']
            latest_pc = pc_seq[-1].reshape(-1, 3)
            self.current_episode_data['point_clouds'].append(latest_pc)
        
        if isinstance(self.obs, dict) and 'state' in self.obs:
            self.current_episode_data['states'].append(self.obs['state'])
        
        self.current_episode_data['actions'].append(action[0] if hasattr(action, '__len__') else action)
        self.current_episode_data['rewards'].append(reward[0] if hasattr(reward, '__len__') else reward)
        
        raw_env = self._get_raw_env()
        state = raw_env.get_state()
        self.current_episode_data['positions'].append(state['eta'][0:3])
    
    def _collect_csv_data(self, action, reward):
        if not self.save_csv:
            return
        
        state = self.env.get_state()
        eta = state['eta']
        
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
        
        if isinstance(self.obs, dict) and 'point_cloud_seq' in self.obs:
            pc_seq = self.obs['point_cloud_seq']
            latest_pc = pc_seq[-1].reshape(-1, 3)
            latest_pc = latest_pc * 50.0
            
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
        if not self.save_data:
            return
        
        import os
        
        os.makedirs(self.save_dir, exist_ok=True)
        
        point_clouds = np.array(self.current_episode_data['point_clouds'])
        states = np.array(self.current_episode_data['states'])
        actions = np.array(self.current_episode_data['actions'])
        rewards = np.array(self.current_episode_data['rewards'])
        positions = np.array(self.current_episode_data['positions'])
        
        episode_dir = os.path.join(self.save_dir, f"episode_{self.episode_counter:04d}")
        os.makedirs(episode_dir, exist_ok=True)
        
        np.save(os.path.join(episode_dir, 'point_clouds.npy'), point_clouds)
        np.save(os.path.join(episode_dir, 'states.npy'), states)
        np.save(os.path.join(episode_dir, 'actions.npy'), actions)
        np.save(os.path.join(episode_dir, 'rewards.npy'), rewards)
        np.save(os.path.join(episode_dir, 'positions.npy'), positions)
        
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
        if not self.save_csv or not success:
            return
        
        import os
        import pandas as pd
        
        csv_dir = os.path.join(self.save_dir, "csv_data")
        os.makedirs(csv_dir, exist_ok=True)
        
        trajectory_df = pd.DataFrame(self.csv_trajectory)
        trajectory_path = os.path.join(csv_dir, f"trajectory_{self.episode_counter:04d}.csv")
        trajectory_df.to_csv(trajectory_path, index=False)
        print(f"CSV trajectory saved: {trajectory_path}")
        
        obstacles_df = pd.DataFrame(self.csv_obstacles)
        obstacles_path = os.path.join(csv_dir, f"obstacles_{self.episode_counter:04d}.csv")
        obstacles_df.to_csv(obstacles_path, index=False)
        print(f"CSV obstacles saved: {obstacles_path}")
        
        point_cloud_df = pd.DataFrame(self.csv_point_clouds)
        point_cloud_path = os.path.join(csv_dir, f"point_clouds_{self.episode_counter:04d}.csv")
        point_cloud_df.to_csv(point_cloud_path, index=False)
        print(f"CSV point clouds saved: {point_cloud_path}")
    
    def _sync_submarine_state(self):
        raw_env = self._get_raw_env()
        state = raw_env.get_state()
        self.submarine.eta = state['eta']
        self.submarine.nu = state['nu']
        self.submarine.u_actual = state['u_actual']
        
        u_control = state['u_control']
        self.submarine.target_rpm = u_control[4]
        self.submarine.rudder_angle = u_control[1]
        self.submarine.stern_angle = u_control[3]
        
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
        
        self.renderer.cam_pos = np.array([-10.0, 0.0, -5.0], dtype=np.float32)
        
        while self.running:
            dt_ms = self.clock.tick(30)
            dt = dt_ms / 1000.0
            if dt > 0.1:
                dt = 0.1
            
            self.handle_events()
            
            if not self.paused:
                keys = pygame.key.get_pressed()
                
                if self.manual_mode:
                    action = self.get_manual_action(keys)
                elif self.model is not None:
                    normalized_obs = self._normalize_obs(self.obs)
                    action, _ = self.model.predict(normalized_obs, deterministic=True)
                else:
                    action = self.env.action_space.sample()
                    action[2] = 0.5
                
                self.obs, reward, terminated, truncated, info = self.env.step(action)
                self.episode_reward += reward
                self.episode_steps += 1
                
                self._sync_submarine_state()
                self._collect_data(action, reward)
                self._collect_csv_data(action, reward)
                
                if terminated or truncated:
                    self.total_episodes += 1
                    success = info.get('goal_reached', False)
                    
                    if success:
                        self.success_count += 1
                        print("=" * 60)
                        print(f"✓ Episode {self.total_episodes}: SUCCESS!")
                        print(f"  Reward: {self.episode_reward:.2f}, Steps: {self.episode_steps}")
                        print("=" * 60)
                    else:
                        reason = info.get('collision_reason', 'timeout')
                        print(f"Episode {self.total_episodes}: FAILED ({reason}). Reward: {self.episode_reward:.2f}, Steps: {self.episode_steps}")
                    
                    success_rate = self.success_count / self.total_episodes * 100
                    print(f"  Success rate: {success_rate:.1f}% ({self.success_count}/{self.total_episodes})")
                    
                    self._save_episode_data(success)
                    self._save_csv_data(success)
                    self._reset_episode()
            
            point_cloud = None
            if isinstance(self.obs, dict) and 'point_cloud_seq' in self.obs:
                pc_seq = self.obs['point_cloud_seq']
                latest_pc = pc_seq[-1]
                point_cloud = latest_pc.reshape(-1, 3)
                point_cloud = point_cloud * 50.0

            # Predict collision
            collision_predicted = False
            raw_env = self._get_raw_env()
            
            if hasattr(raw_env, '_predict_collision'):
                collision_predicted = raw_env._predict_collision()

            score_data = {
                'episode_reward': self.episode_reward,
                'total_episodes': self.total_episodes,
                'success_count': self.success_count,
                'episode_steps': self.episode_steps,
                'collision_predicted': collision_predicted
            }
            self.renderer.render(self.logic_stage, point_cloud=point_cloud, score_data=score_data)
            
            if self.episode_steps % 30 == 0:
                pos = self.submarine.position
                print(f"Submarine Pos: X={pos[0]:.2f}, Y={pos[1]:.2f}, Z={pos[2]:.2f}")
                
                if point_cloud is not None and len(point_cloud) > 0:
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
    parser = argparse.ArgumentParser(description="Evaluate best_model (1).zip with custom network architecture")
    parser.add_argument("--model", type=str, default="best_model (10).zip",
                        help="Path to trained model")
    parser.add_argument("--vecnormalize", type=str, default="submarine_ppo_2400480_steps_normalize.pkl",
                        help="Path to VecNormalize statistics")
    parser.add_argument("--no-lidar", action="store_true",
                        help="Disable LiDAR")
    parser.add_argument("--manual", action="store_true",
                        help="Manual control mode")
    parser.add_argument("--save-data", action="store_true", default=True,
                        help="Save episode data")
    parser.add_argument("--no-save-data", action="store_true",
                        help="Disable data saving")
    parser.add_argument("--save-dir", type=str, default="./eval_data_model5",
                        help="Directory to save episode data")
    parser.add_argument("--no-save-csv", action="store_true",
                        help="Disable CSV saving")
    
    parser.add_argument("--lstm-hidden-size", type=int, default=128,
                        help="LSTM hidden layer size")
    parser.add_argument("--lstm-num-layers", type=int, default=1,
                        help="LSTM number of layers")
    parser.add_argument("--pc-encoder-dims", type=str, default="256",
                        help="Point cloud encoder dimensions (comma-separated)")
    parser.add_argument("--state-feature-dim", type=int, default=64,
                        help="State feature dimension")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout probability")
    parser.add_argument("--use-layer-norm", action="store_true", default=True,
                        help="Use Layer Normalization")
    parser.add_argument("--no-layer-norm", action="store_true",
                        help="Disable Layer Normalization")
    parser.add_argument("--bidirectional", action="store_true",
                        help="Use bidirectional LSTM")
    
    parser.add_argument("--pi-hidden", type=str, default="256,128",
                        help="Actor network hidden dimensions (comma-separated)")
    parser.add_argument("--vf-hidden", type=str, default="256,128",
                        help="Critic network hidden dimensions (comma-separated)")
    
    args = parser.parse_args()
    
    save_data = args.save_data and not args.no_save_data
    save_csv = not args.no_save_csv
    
    pc_encoder_dims = tuple(int(x) for x in args.pc_encoder_dims.split(','))
    pi_hidden = tuple(int(x) for x in args.pi_hidden.split(','))
    vf_hidden = tuple(int(x) for x in args.vf_hidden.split(','))
    
    use_layer_norm = args.use_layer_norm and not args.no_layer_norm
    
    app = EvalApp(
        model_path=args.model,
        use_lidar=not args.no_lidar,
        manual_mode=args.manual,
        save_data=save_data,
        save_dir=args.save_dir,
        vecnormalize_path=args.vecnormalize,
        save_csv=save_csv,
        lstm_hidden_size=args.lstm_hidden_size,
        lstm_num_layers=args.lstm_num_layers,
        pc_encoder_dims=pc_encoder_dims,
        state_feature_dim=args.state_feature_dim,
        dropout=args.dropout,
        use_layer_norm=use_layer_norm,
        bidirectional=args.bidirectional,
        pi_hidden=pi_hidden,
        vf_hidden=vf_hidden,
    )
    app.run()


if __name__ == "__main__":
    main()
