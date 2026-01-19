#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_visual.py: 可视化训练脚本 - 使用3个并行环境进行训练，并通过pygame实时渲染

特性:
- 3个并行环境进行高效训练
- Pygame实时渲染其中一个环境
- 支持课程学习和渐进式训练
- 实时显示训练统计信息
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import threading
import time
from datetime import datetime
import numpy as np
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pygame
from pygame.locals import *

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
    BaseCallback
)
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from src.rl.submarine_env import SubmarineEnv
from src.rl.tunnel import TunnelConfig
from src.rl.point_cloud_sampler import PointCloudConfig
from src.core.stage import LogicStage
from src.physics.submarine_actor import SubmarineActor
from src.gameplay.obstacle_actor import ObstacleActor
from src.gameplay.tunnel_actor import TunnelActor
from src.view.renderer import Renderer
from src.rl.custom_extractor import PointCloudLSTMExtractor


class TrainingConfig:
    """可视化训练配置"""

    def __init__(self):
        self.total_timesteps = 9_000_000
        self.n_envs = 3  # 3个并行环境
        self.seed = 42

        self.learning_rate = 3e-5
        self.n_steps = 2048
        self.batch_size = 512
        self.n_epochs = 10
        self.gamma = 0.99

        self.num_obstacles = 20

        self.point_cloud_history_len = 16
        self.lstm_hidden_size = 128
        self.lstm_num_layers = 1
        self.bidirectional = False

        self.pc_encoder_dims = "256"
        self.state_feature_dim = 64

        self.dropout = 0.1
        self.use_layer_norm = True

        self.gae_lambda = 0.99
        self.clip_range = 0.2
        self.ent_coef = 0.01

        self.pi_hidden_1 = 256
        self.pi_hidden_2 = 128
        self.vf_hidden_1 = 256
        self.vf_hidden_2 = 128

        self.output_dir = "./training_output"
        self.save_freq = 50000
        self.eval_freq = 20000

        self.device = "cuda"

        self.resume_model_path = None
        self.resume_normalize_path = None
        self.resume_stage = 1

        # 可视化配置
        self.visualize = True
        self.vis_env_index = 0  # 渲染第0个环境


class CurriculumConfig:
    """课程配置"""

    def __init__(self, stage: int, num_obstacles: int,
                 w_velocity: float = 1.0):
        self.stage = stage
        self.num_obstacles = num_obstacles
        self.w_velocity = w_velocity


class Stage5ProgressiveManager:
    """课程5的渐进式训练管理器"""

    def __init__(self, consecutive_success_threshold: int = 3):
        self.progressive_stage = 0
        self.consecutive_success_threshold = consecutive_success_threshold
        self.recent_results = []
        self.max_perturbation_stage = 10

    def add_result(self, success: bool):
        self.recent_results.append(success)
        if len(self.recent_results) > self.consecutive_success_threshold:
            self.recent_results.pop(0)

    def should_advance(self) -> bool:
        if self.progressive_stage >= self.max_perturbation_stage + 1:
            return False
        if len(self.recent_results) >= self.consecutive_success_threshold:
            if all(self.recent_results[-self.consecutive_success_threshold:]):
                return True
        return False

    def advance(self):
        if self.progressive_stage <= self.max_perturbation_stage:
            self.progressive_stage += 1
            self.recent_results = []
            return True
        return False


class CurriculumManager:
    """课程管理器"""

    def __init__(self, eval_episodes: int = 10, success_threshold: float = 0.8, start_stage: int = 1):
        self.current_stage = start_stage
        self.eval_episodes = eval_episodes
        self.success_threshold = success_threshold
        self.episode_results = []
        self.stage_thresholds = {1: 0.8, 2: 0.8, 3: 0.8, 4: 0.8, 5: 0.8}
        self.stage5_progressive = Stage5ProgressiveManager(consecutive_success_threshold=3)

    def start_new_evaluation(self):
        self.episode_results = []

    def add_episode_result(self, success: bool, progress: float):
        self.episode_results.append({'success': success, 'progress': progress})
        if self.current_stage == 5:
            self.stage5_progressive.add_result(success)

    def should_advance(self) -> bool:
        if self.current_stage >= len(CURRICULUM_STAGES):
            return False
        if len(self.episode_results) < self.eval_episodes:
            return False
        success_rate = sum(r['success'] for r in self.episode_results) / len(self.episode_results)
        threshold = self.stage_thresholds.get(self.current_stage, self.success_threshold)
        return success_rate >= threshold

    def advance(self):
        if self.current_stage < len(CURRICULUM_STAGES):
            self.current_stage += 1
            self.episode_results = []
            self.stage5_progressive = Stage5ProgressiveManager(consecutive_success_threshold=3)
            return True
        return False

    def get_current_config(self) -> CurriculumConfig:
        return CURRICULUM_STAGES[self.current_stage]

    def get_recent_success_rate(self, window_size: int = 15) -> float:
        if len(self.episode_results) == 0:
            return 0.0
        recent_results = self.episode_results[-min(window_size, len(self.episode_results)):]
        return sum(r['success'] for r in recent_results) / len(recent_results)


# 课程阶段配置
CURRICULUM_STAGES = {
    1: CurriculumConfig(stage=1, num_obstacles=3, w_velocity=0.2),
    2: CurriculumConfig(stage=2, num_obstacles=5, w_velocity=0.2),
    3: CurriculumConfig(stage=3, num_obstacles=7, w_velocity=0.2),
    4: CurriculumConfig(stage=4, num_obstacles=9, w_velocity=0.2),
    5: CurriculumConfig(stage=5, num_obstacles=15, w_velocity=0.2),
}


class VisualizationThread(threading.Thread):
    """
    可视化线程 - 在独立线程中运行pygame渲染

    与训练并行运行，定期同步模型和环境状态

    支持多环境网格显示模式
    """

    def __init__(self, model_path: str, vecnormalize_path: str,
                 tunnel_config: TunnelConfig,
                 point_cloud_config: PointCloudConfig,
                 point_cloud_history_len: int = 16,
                 lstm_hidden_size: int = 128,
                 lstm_num_layers: int = 1,
                 pc_encoder_dims: tuple = (256,),
                 state_feature_dim: int = 64,
                 dropout: float = 0.1,
                 use_layer_norm: bool = True,
                 bidirectional: bool = False,
                 pi_hidden: tuple = (256, 128),
                 vf_hidden: tuple = (256, 128),
                 n_envs: int = 3,
                 grid_mode: bool = True):
        super().__init__(daemon=True)  # 设为守护线程，主线程退出时自动退出
        self.model_path = model_path
        self.vecnormalize_path = vecnormalize_path
        self.tunnel_config = tunnel_config
        self.point_cloud_config = point_cloud_config
        self.point_cloud_history_len = point_cloud_history_len

        # 模型架构参数
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.pc_encoder_dims = pc_encoder_dims
        self.state_feature_dim = state_feature_dim
        self.dropout = dropout
        self.use_layer_norm = use_layer_norm
        self.bidirectional = bidirectional
        self.pi_hidden = pi_hidden
        self.vf_hidden = vf_hidden

        # 多环境配置
        self.n_envs = n_envs
        self.grid_mode = grid_mode
        # 计算网格布局 (尽量接近正方形)
        if grid_mode:
            self.grid_cols = int(np.ceil(np.sqrt(n_envs)))
            self.grid_rows = int(np.ceil(n_envs / self.grid_cols))
        else:
            self.grid_rows = 1
            self.grid_cols = 1

        self.running = False
        self.paused = False
        self.model = None
        self.vec_normalize = None
        self.envs = []  # 多个环境实例
        self.renderer = None
        self.logic_stages = []  # 多个 logic_stage
        self.submarine_actors = []  # 多个潜艇 actor
        self.tunnel_actors = []  # 多个隧道 actor

        # 多环境观测和状态
        self.obs_list = []
        self.episode_rewards = [0] * n_envs
        self.episode_steps_list = [0] * n_envs

        # 训练统计信息 (从主线程更新)
        self.training_stats = {
            'timesteps': 0,
            'total_episodes': 0,
            'success_count': 0,
            'current_stage': 1,
            'num_obstacles': 3,
            'progressive_stage': 0,
        }

        self.lock = threading.Lock()

    def init_pygame(self):
        """初始化pygame和渲染器"""
        pygame.init()
        # 增大窗口尺寸以支持多环境显示
        self.width, self.height = 1920, 1080
        pygame.display.set_mode((self.width, self.height), DOUBLEBUF | OPENGL | RESIZABLE)
        pygame.display.set_caption(f"Submarine RL Training - {self.n_envs} Environments Grid Visualization")

        # 创建渲染器 (支持网格模式)
        self.renderer = Renderer(
            self.width,
            self.height,
            grid_mode=self.grid_mode,
            grid_rows=self.grid_rows,
            grid_cols=self.grid_cols
        )
        self.renderer.tunnel_config = self.tunnel_config

        self.clock = pygame.time.Clock()

    def init_envs(self):
        """初始化多个评估环境"""
        from src.rl.submarine_env import SubmarineEnv

        for i in range(self.n_envs):
            raw_env = SubmarineEnv(
                tunnel_config=self.tunnel_config,
                point_cloud_config=self.point_cloud_config,
                point_cloud_history_len=self.point_cloud_history_len,
                max_steps=1200,
            )
            self.envs.append(raw_env)

            # 创建逻辑舞台
            logic_stage = LogicStage()
            self.logic_stages.append(logic_stage)

            # 创建潜艇Actor
            submarine_actor = SubmarineActor(f"Submarine_{i}")
            logic_stage.add_actor(submarine_actor)
            self.submarine_actors.append(submarine_actor)

            # 创建隧道Actor
            tunnel_actor = TunnelActor(config=self.tunnel_config)
            logic_stage.add_actor(tunnel_actor)
            self.tunnel_actors.append(tunnel_actor)

        # 加载VecNormalize
        if self.vecnormalize_path and os.path.exists(self.vecnormalize_path):
            print(f"[Vis] Loading VecNormalize from: {self.vecnormalize_path}")
            dummy_env = DummyVecEnv([lambda: self.envs[0]])
            self.vec_normalize = VecNormalize.load(self.vecnormalize_path, dummy_env)
            self.vec_normalize.training = False
            self.vec_normalize.norm_reward = False
            print(f"[Vis] Loaded VecNormalize for {self.n_envs} environments")

    def load_model(self):
        """加载模型"""
        if self.model_path and os.path.exists(self.model_path):
            print(f"[Vis] Loading model from: {self.model_path}")

            policy_kwargs = dict(
                features_extractor_class=PointCloudLSTMExtractor,
                features_extractor_kwargs=dict(
                    lstm_hidden_size=self.lstm_hidden_size,
                    lstm_num_layers=self.lstm_num_layers,
                    point_cloud_encoder_dims=self.pc_encoder_dims,
                    state_feature_dim=self.state_feature_dim,
                    dropout=self.dropout,
                    use_layer_norm=self.use_layer_norm,
                    bidirectional=self.bidirectional,
                ),
                net_arch=dict(
                    pi=list(self.pi_hidden),
                    vf=list(self.vf_hidden),
                ),
            )

            self.model = PPO.load(self.model_path, policy_kwargs=policy_kwargs)
            print("[Vis] Model loaded successfully!")

    def reset_episodes(self):
        """重置所有环境episode"""
        self.obs_list = []
        for i, env in enumerate(self.envs):
            obs, info = env.reset()
            self.obs_list.append(obs)
            self.episode_rewards[i] = 0
            self.episode_steps_list[i] = 0

            # 同步潜艇状态
            state = env.get_state()
            self.submarine_actors[i].eta = state['eta']
            self.submarine_actors[i].nu = state['nu']
            self.submarine_actors[i].u_actual = state['u_actual']

            u_control = state['u_control']
            self.submarine_actors[i].target_rpm = u_control[4]
            self.submarine_actors[i].rudder_angle = u_control[1]
            self.submarine_actors[i].stern_angle = u_control[3]
            self.submarine_actors[i]._sync_state()

            # 同步障碍物
            self._sync_obstacles(i)

    def _sync_obstacles(self, env_idx: int):
        """同步指定环境的障碍物Actor"""
        env = self.envs[env_idx]
        logic_stage = self.logic_stages[env_idx]

        # 清除旧障碍物
        logic_stage.actors = [a for a in logic_stage.actors
                              if not isinstance(a, ObstacleActor)]

        tunnel = env.get_tunnel()
        if tunnel:
            for i, obs in enumerate(tunnel.obstacles):
                obs_actor = ObstacleActor(
                    name=f"Obstacle_{i}",
                    x=obs.position[0],
                    y=obs.position[1],
                    z=obs.position[2],
                    radius=obs.radius
                )
                logic_stage.add_actor(obs_actor)

        # 设置渲染器的隧道引用 (使用第一个环境的隧道)
        if env_idx == 0:
            self.renderer.tunnel = tunnel

    def normalize_obs(self, obs):
        """归一化观测"""
        if self.vec_normalize is not None:
            obs_batch = {k: np.expand_dims(v, 0) for k, v in obs.items()}
            normalized = self.vec_normalize.normalize_obs(obs_batch)
            return {k: v[0] for k, v in normalized.items()}
        return obs

    def step_all(self):
        """对所有环境执行一步"""
        actions = []
        rewards = []

        for i, obs in enumerate(self.obs_list):
            if self.model is None:
                action = self.envs[i].action_space.sample()
                action[2] = 0.5
            else:
                normalized_obs = self.normalize_obs(obs)
                action, _ = self.model.predict(normalized_obs, deterministic=True)

            actions.append(action)

            # 执行步骤
            new_obs, reward, terminated, truncated, info = self.envs[i].step(action)
            rewards.append(reward)
            self.episode_rewards[i] += reward
            self.episode_steps_list[i] += 1

            # 同步潜艇状态
            state = self.envs[i].get_state()
            self.submarine_actors[i].eta = state['eta']
            self.submarine_actors[i].nu = state['nu']
            self.submarine_actors[i].u_actual = state['u_actual']

            u_control = state['u_control']
            self.submarine_actors[i].target_rpm = u_control[4]
            self.submarine_actors[i].rudder_angle = u_control[1]
            self.submarine_actors[i].stern_angle = u_control[3]
            self.submarine_actors[i]._sync_state()

            # 检查episode结束
            if terminated or truncated:
                # 更新统计
                with self.lock:
                    self.training_stats['total_episodes'] += 1
                    if info.get('goal_reached', False):
                        self.training_stats['success_count'] += 1

                # 重置该环境
                obs, info = self.envs[i].reset()
                self.obs_list[i] = obs
                self.episode_rewards[i] = 0
                self.episode_steps_list[i] = 0

                # 同步潜艇状态
                state = self.envs[i].get_state()
                self.submarine_actors[i].eta = state['eta']
                self.submarine_actors[i].nu = state['nu']
                self.submarine_actors[i].u_actual = state['u_actual']

                u_control = state['u_control']
                self.submarine_actors[i].target_rpm = u_control[4]
                self.submarine_actors[i].rudder_angle = u_control[1]
                self.submarine_actors[i].stern_angle = u_control[3]
                self.submarine_actors[i]._sync_state()

                # 同步障碍物
                self._sync_obstacles(i)
            else:
                self.obs_list[i] = new_obs

        return actions, rewards

    def handle_events(self):
        """处理pygame事件"""
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
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    self.reset_episodes()

    def run(self):
        """可视化主循环"""
        self.running = True
        self.init_pygame()
        self.init_envs()
        self.load_model()
        self.reset_episodes()

        print(f"[Vis] Visualization thread started ({self.n_envs} environments)")
        print("[Vis] Controls: Space=Pause, R=Reset all, ESC=Quit")
        if self.grid_mode:
            print(f"[Vis] Grid Layout: {self.grid_rows}x{self.grid_cols}")
            print("[Vis] Mouse over viewport and right-drag to rotate camera")

        last_model_check = time.time()
        model_check_interval = 5.0  # 每5秒检查一次模型更新

        while self.running:
            dt_ms = self.clock.tick(30)
            dt = dt_ms / 1000.0
            if dt > 0.1:
                dt = 0.1

            self.handle_events()

            # 更新相机 (在网格模式下为当前鼠标所在视口)
            if self.grid_mode:
                cell_width = self.width // self.grid_cols
                cell_height = self.height // self.grid_rows
                mx, my = pygame.mouse.get_pos()
                col = mx // cell_width
                row = (self.height - 1 - my) // cell_height
                viewport_idx = row * self.grid_cols + col
                if 0 <= viewport_idx < len(self.renderer.camera_states):
                    self.renderer.update_grid_camera_input(viewport_idx)

            # 定期重新加载模型 (如果训练期间保存了新模型)
            current_time = time.time()
            if current_time - last_model_check >= model_check_interval:
                if self.model_path and os.path.exists(self.model_path):
                    try:
                        # 重新加载模型以获取最新参数
                        policy_kwargs = dict(
                            features_extractor_class=PointCloudLSTMExtractor,
                            features_extractor_kwargs=dict(
                                lstm_hidden_size=self.lstm_hidden_size,
                                lstm_num_layers=self.lstm_num_layers,
                                point_cloud_encoder_dims=self.pc_encoder_dims,
                                state_feature_dim=self.state_feature_dim,
                                dropout=self.dropout,
                                use_layer_norm=self.use_layer_norm,
                                bidirectional=self.bidirectional,
                            ),
                            net_arch=dict(
                                pi=list(self.pi_hidden),
                                vf=list(self.vf_hidden),
                            ),
                        )
                        self.model = PPO.load(self.model_path, policy_kwargs=policy_kwargs)
                    except Exception as e:
                        pass  # 忽略加载错误，可能在保存过程中
                last_model_check = current_time

            if not self.paused:
                # 执行所有环境的一步
                self.step_all()

            # 准备渲染数据
            point_clouds_list = []
            score_data_list = []

            for i, obs in enumerate(self.obs_list):
                # 提取点云
                point_cloud = None
                if isinstance(obs, dict) and 'point_cloud_seq' in obs:
                    pc_seq = obs['point_cloud_seq']
                    latest_pc = pc_seq[-1]
                    point_cloud = latest_pc.reshape(-1, 3)
                    point_cloud = point_cloud * 50.0
                point_clouds_list.append(point_cloud)

                # 预测碰撞
                collision_predicted = False
                if hasattr(self.envs[i], '_predict_collision'):
                    collision_predicted = self.envs[i]._predict_collision()

                # 准备统计数据
                score_data_list.append({
                    'episode_reward': self.episode_rewards[i],
                    'total_episodes': self.training_stats['total_episodes'],
                    'success_count': self.training_stats['success_count'],
                    'episode_steps': self.episode_steps_list[i],
                    'collision_predicted': collision_predicted,
                    'timesteps': self.training_stats['timesteps'],
                    'current_stage': self.training_stats['current_stage'],
                    'num_obstacles': self.training_stats['num_obstacles'],
                    'progressive_stage': self.training_stats['progressive_stage'],
                })

            # 网格渲染
            if self.grid_mode:
                self.renderer.render_grid(
                    self.logic_stages,
                    point_clouds_list,
                    score_data_list
                )
            else:
                # 单环境渲染 (兼容旧代码)
                if self.obs_list:
                    point_cloud = point_clouds_list[0]
                    score_data = score_data_list[0]
                    self.renderer.render(self.logic_stages[0], point_cloud=point_cloud, score_data=score_data)

        pygame.quit()
        print("[Vis] Visualization thread stopped")


class SaveNormalizeOnCheckpointCallback(BaseCallback):
    """在每次保存 checkpoint 时同时保存归一化参数"""

    def __init__(self, vec_env, save_freq: int, save_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.vec_env = vec_env
        self.save_freq = save_freq
        self.save_path = save_path
        self.last_save_step = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_save_step >= self.save_freq:
            self.last_save_step = self.num_timesteps
            checkpoint_name = f"submarine_ppo_{self.num_timesteps}_steps_normalize.pkl"
            normalize_path = os.path.join(self.save_path, checkpoint_name)
            self.vec_env.save(normalize_path)
            if self.verbose > 0:
                print(f"[Normalize] Saved VecNormalize statistics to: {normalize_path}")
        return True


class SaveNormalizeOnBestModelCallback(BaseCallback):
    """在保存 best model 时同时保存归一化参数"""

    def __init__(self, vec_env, save_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.vec_env = vec_env
        self.save_path = save_path
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> bool:
        if self.training_env is not None and hasattr(self.training_env, 'best_mean_reward'):
            current_best = self.training_env.best_mean_reward
            if current_best > self.best_mean_reward:
                self.best_mean_reward = current_best
                normalize_path = os.path.join(self.save_path, "vecnormalize.pkl")
                self.vec_env.save(normalize_path)
                if self.verbose > 0:
                    print(f"[Normalize] Saved best model VecNormalize to: {normalize_path}")
        return True


class CurriculumCallback(BaseCallback):
    """课程切换回调函数"""

    def __init__(self, curriculum_manager: CurriculumManager,
                 eval_env, curriculum_state, eval_freq: int = 5000,
                 verbose: int = 0):
        super().__init__(verbose)
        self.curriculum_manager = curriculum_manager
        self.eval_env = eval_env
        self.curriculum_state = curriculum_state
        self.eval_freq = eval_freq
        self.last_eval_time = 0
        self.eval_episodes_per_check = 10
        self.min_timesteps_before_eval = 50000
        self.vis_thread = None  # 可视化线程引用

    def set_visualization_thread(self, vis_thread):
        """设置可视化线程引用，用于更新统计信息"""
        self.vis_thread = vis_thread

    def _on_step(self) -> bool:
        # 更新可视化线程的统计信息
        if self.vis_thread is not None:
            with self.vis_thread.lock:
                self.vis_thread.training_stats['timesteps'] = self.num_timesteps
                self.vis_thread.training_stats['current_stage'] = self.curriculum_state.get('current_stage', 1)
                self.vis_thread.training_stats['num_obstacles'] = self.curriculum_state.get('num_obstacles', 3)
                self.vis_thread.training_stats['progressive_stage'] = self.curriculum_state.get('progressive_stage', 0)

        if self.num_timesteps - self.last_eval_time >= self.eval_freq:
            self.last_eval_time = self.num_timesteps
            if self.num_timesteps >= self.min_timesteps_before_eval:
                self._evaluate_and_check()
        return True

    def _evaluate_and_check(self):
        """评估并检查是否需要切换课程或渐进式阶段"""
        if self.verbose > 0:
            print(f"\n[Evaluation] Evaluating curriculum progress at timestep {self.num_timesteps}")

        self.curriculum_manager.start_new_evaluation()

        for episode_idx in range(self.eval_episodes_per_check):
            obs = self.eval_env.reset()
            done = np.array([False])
            success = False
            progress = 0.0

            while not done[0]:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.eval_env.step(action)

                if done[0]:
                    if info[0].get('goal_reached', False):
                        success = True
                    progress = info[0].get('progress', 0.0)

            self.curriculum_manager.add_episode_result(success, progress)

        success_rate = sum(r['success'] for r in self.curriculum_manager.episode_results) / len(
            self.curriculum_manager.episode_results)
        avg_progress = sum(r['progress'] for r in self.curriculum_manager.episode_results) / len(
            self.curriculum_manager.episode_results)

        if self.verbose > 0:
            print(f"[Evaluation] Overall success rate: {success_rate:.2%}, Avg progress: {avg_progress:.2%}")

        # 检查课程5的渐进式训练阶段
        if self.curriculum_manager.current_stage == 5:
            progressive_manager = self.curriculum_manager.stage5_progressive
            if progressive_manager.should_advance():
                old_stage = progressive_manager.progressive_stage
                if progressive_manager.advance():
                    new_stage = progressive_manager.progressive_stage
                    self.curriculum_state['progressive_stage'] = new_stage

                    mode_desc = "Fixed" if new_stage == 0 else (f"Perturbed (±{new_stage * 0.5:.1f}m)" if new_stage <= 10 else "Random")
                    print(f"\n{'=' * 60}")
                    print(f"PROGRESSIVE STAGE ADVANCED: Stage {old_stage} → Stage {new_stage}")
                    print(f"  - Mode: {mode_desc}")
                    print(f"{'=' * 60}\n")

        # 检查是否需要切换主课程阶段
        if self.curriculum_manager.should_advance():
            old_stage = self.curriculum_manager.current_stage
            if self.curriculum_manager.advance():
                new_stage = self.curriculum_manager.current_stage
                new_config = self.curriculum_manager.get_current_config()

                self.curriculum_state['current_stage'] = new_config.stage
                self.curriculum_state['num_obstacles'] = new_config.num_obstacles
                self.curriculum_state['w_velocity'] = new_config.w_velocity

                if new_stage == 5:
                    self.curriculum_state['progressive_mode'] = True
                    self.curriculum_state['progressive_stage'] = 0
                else:
                    self.curriculum_state['progressive_mode'] = False

                print(f"\n{'=' * 60}")
                print(f"CURRICULUM ADVANCED: Stage {old_stage} → Stage {new_stage}")
                print(f"  - Obstacles: {new_config.num_obstacles}")
                print(f"  - Velocity weight: {new_config.w_velocity}")
                if new_stage == 5:
                    print(f"  - Progressive Training: ENABLED")
                print(f"{'=' * 60}\n")


def make_env(rank: int, seed: int = 0,
             point_cloud_history_len: int = 8,
             tunnel_config: TunnelConfig = None,
             point_cloud_config: PointCloudConfig = None,
             curriculum_state=None):
    """创建单个环境实例的工厂函数"""

    def _init():
        if curriculum_state is not None:
            current_stage = curriculum_state.get('current_stage', 1)
            num_obstacles = curriculum_state.get('num_obstacles', 3)
            w_velocity = curriculum_state.get('w_velocity', 0.5)
            progressive_mode = curriculum_state.get('progressive_mode', False)
            progressive_stage = curriculum_state.get('progressive_stage', 0)
        else:
            current_stage = 1
            num_obstacles = 3
            w_velocity = 0.5
            progressive_mode = False
            progressive_stage = 0

        if tunnel_config is not None:
            tunnel_config.num_obstacles = num_obstacles
            tunnel_config.progressive_mode = progressive_mode
            tunnel_config.progressive_stage = progressive_stage
            if progressive_mode and progressive_stage == 0:
                tunnel_config.fixed_obstacle_seed = 42

        env = SubmarineEnv(
            tunnel_config=tunnel_config,
            point_cloud_config=point_cloud_config,
            point_cloud_history_len=point_cloud_history_len,
            max_steps=1200
        )

        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env

    return _init


def train(config):
    """训练主函数"""

    from multiprocessing import Manager

    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_name = "visual_hybrid_lstm"

    if config.resume_model_path:
        # 获取模型文件的目录路径
        model_dir = os.path.dirname(config.resume_model_path)
        # 如果模型在当前目录（model_dir为空），使用父目录或当前目录
        if not model_dir:
            # 检查是否有上级目录的路径结构
            resume_dir = os.path.dirname(os.path.abspath(config.resume_model_path))
            # 如果在根目录，使用默认输出目录
            if not resume_dir or resume_dir == os.path.dirname(os.getcwd()):
                log_dir = os.path.join(config.output_dir, f"resume_{timestamp}")
            else:
                log_dir = resume_dir
        else:
            resume_dir = os.path.dirname(model_dir)
            log_dir = resume_dir if resume_dir else config.output_dir
    else:
        log_dir = os.path.join(config.output_dir, f"submarine_ppo_{mode_name}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)

    print(f"Training output directory: {log_dir}")

    # 检查是否继续训练
    if config.resume_model_path:
        if not os.path.exists(config.resume_model_path):
            raise FileNotFoundError(f"Resume model path not found: {config.resume_model_path}")
        if config.resume_normalize_path and not os.path.exists(config.resume_normalize_path):
            raise FileNotFoundError(f"Resume normalize path not found: {config.resume_normalize_path}")
        print(f"\nResuming training from:")
        print(f"  - Model: {config.resume_model_path}")
        print(f"  - VecNormalize: {config.resume_normalize_path}")

    # 创建课程管理器
    curriculum_manager = CurriculumManager(eval_episodes=10, success_threshold=0.8, start_stage=config.resume_stage)

    # 创建共享课程状态
    manager = Manager()

    if config.resume_stage in CURRICULUM_STAGES:
        resume_config = CURRICULUM_STAGES[config.resume_stage]

        # 对于阶段5，需要初始化渐进式训练模式
        progressive_mode = (resume_config.stage == 5)
        progressive_stage = 0 if progressive_mode else 0

        curriculum_state = manager.dict({
            'current_stage': resume_config.stage,
            'num_obstacles': resume_config.num_obstacles,
            'w_velocity': resume_config.w_velocity,
            'progressive_mode': progressive_mode,
            'progressive_stage': progressive_stage,
        })
        print(f"\nResuming from Curriculum Stage {config.resume_stage}:")
        print(f"  - Obstacles: {resume_config.num_obstacles}")
        print(f"  - Velocity weight: {resume_config.w_velocity}")
        if progressive_mode:
            print(f"  - Progressive Training: ENABLED (Stage {progressive_stage})")
    else:
        curriculum_state = manager.dict({
            'current_stage': 1,
            'num_obstacles': 3,
            'w_velocity': 0.5,
        })

    # 环境配置
    tunnel_config = TunnelConfig(
        radius=5.0,
        length=50.0,
        center_z=100.0,
        num_obstacles=config.num_obstacles,
        obstacle_radius_min=0.7,
        obstacle_radius_max=1.0,
    )

    # 点云采样配置
    point_cloud_config = PointCloudConfig(
        num_points=256,
        obstacle_points_ratio=0.7,
        normalize_range=50.0,
    )

    # 创建并行环境 (3个)
    print(f"\nCreating {config.n_envs} parallel environments for training...")
    print("Mode: VISUAL HYBRID LSTM (点云序列 → LSTM → 特征 + 状态 → PPO)")
    print(f"  - Point cloud history length: {config.point_cloud_history_len}")
    print(f"  - Current curriculum: Stage {curriculum_state['current_stage']} ({curriculum_state['num_obstacles']} obstacles)")

    env = SubprocVecEnv([
        make_env(i, config.seed,
                 point_cloud_history_len=config.point_cloud_history_len,
                 tunnel_config=tunnel_config,
                 point_cloud_config=point_cloud_config,
                 curriculum_state=curriculum_state)
        for i in range(config.n_envs)
    ])

    # 创建评估环境
    eval_env = DummyVecEnv([
        make_env(0, config.seed + 1000,
                 point_cloud_history_len=config.point_cloud_history_len,
                 tunnel_config=tunnel_config,
                 point_cloud_config=point_cloud_config,
                 curriculum_state=curriculum_state)
    ])

    # 添加归一化包装器
    print("\nAdding VecNormalize for observation and reward normalization...")

    if config.resume_normalize_path:
        env = VecNormalize.load(config.resume_normalize_path, env)
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
        eval_env.obs_rms = env.obs_rms
        print(f"Loaded VecNormalize statistics from: {config.resume_normalize_path}")
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=2, clip_reward=2)
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
        eval_env.obs_rms = env.obs_rms

    # 解析点云编码器维度
    encoder_dims = tuple(int(x) for x in config.pc_encoder_dims.split(','))

    policy_kwargs = dict(
        features_extractor_class=PointCloudLSTMExtractor,
        features_extractor_kwargs=dict(
            lstm_hidden_size=config.lstm_hidden_size,
            lstm_num_layers=config.lstm_num_layers,
            point_cloud_encoder_dims=encoder_dims,
            state_feature_dim=config.state_feature_dim,
            dropout=config.dropout,
            use_layer_norm=config.use_layer_norm,
            bidirectional=config.bidirectional,
        ),
        net_arch=dict(
            pi=[config.pi_hidden_1, config.pi_hidden_2],
            vf=[config.vf_hidden_1, config.vf_hidden_2],
        ),
    )

    print(f"\nUsing PPO with PointCloudLSTMExtractor:")
    print(f"  - LSTM hidden size: {config.lstm_hidden_size}")
    print(f"  - LSTM num layers: {config.lstm_num_layers}")
    print(f"  - Bidirectional: {config.bidirectional}")
    print(f"  - Point cloud encoder: {encoder_dims}")
    print(f"  - State feature dim: {config.state_feature_dim}")
    print(f"  - Dropout: {config.dropout}")
    print(f"  - Layer Norm: {config.use_layer_norm}")
    print(f"  - Actor network: [{config.pi_hidden_1}, {config.pi_hidden_2}]")
    print(f"  - Critic network: [{config.vf_hidden_1}, {config.vf_hidden_2}]")

    # 加载或创建模型
    if config.resume_model_path:
        print(f"\nLoading existing model from: {config.resume_model_path}")
        model = PPO.load(config.resume_model_path, env=env, tensorboard_log=log_dir, device=config.device)
        if config.learning_rate is not None:
            print(f"--> Overriding learning rate to: {config.learning_rate}")
            model.learning_rate = config.learning_rate
            for param_group in model.policy.optimizer.param_groups:
                param_group["lr"] = config.learning_rate

        if config.batch_size is not None:
            print(f"  -> Overriding Batch Size: {model.batch_size} -> {config.batch_size}")
            model.batch_size = config.batch_size

        if config.n_epochs is not None:
            print(f"  -> Overriding N Epochs: {model.n_epochs} -> {config.n_epochs}")
            model.n_epochs = config.n_epochs

        print("Model loaded successfully, continuing training...")
    else:
        model = PPO(
            "MultiInputPolicy",
            env,
            learning_rate=3e-5,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            n_epochs=10,
            gamma=config.gamma,
            gae_lambda=0.99,
            clip_range=config.clip_range,
            ent_coef=config.ent_coef,
            target_kl=0.03,
            policy_kwargs=policy_kwargs,
            verbose=1,  # 降低详细程度，避免与可视化冲突
            tensorboard_log=log_dir,
            device=config.device,
        )

    # 回调函数
    checkpoint_callback = CheckpointCallback(
        save_freq=config.save_freq,
        save_path=os.path.join(log_dir, "checkpoints"),
        name_prefix="submarine_ppo",
    )

    save_normalize_callback = SaveNormalizeOnCheckpointCallback(
        vec_env=env,
        save_freq=config.save_freq,
        save_path=os.path.join(log_dir, "checkpoints"),
        verbose=0
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(log_dir, "best_model"),
        log_path=os.path.join(log_dir, "eval_logs"),
        eval_freq=config.eval_freq,
        n_eval_episodes=5,
        deterministic=True,
    )

    save_best_normalize_callback = SaveNormalizeOnBestModelCallback(
        vec_env=env,
        save_path=os.path.join(log_dir, "best_model"),
        verbose=0
    )

    # 课程学习回调
    curriculum_callback = CurriculumCallback(
        curriculum_manager=curriculum_manager,
        eval_env=eval_env,
        curriculum_state=curriculum_state,
        eval_freq=config.eval_freq,
        verbose=1
    )

    # 启动可视化线程
    vis_thread = None
    if config.visualize:
        print("\n" + "=" * 60)
        print("Starting Visualization Thread...")
        print("=" * 60)

        # 使用best_model的路径作为可视化模型路径
        vis_model_path = os.path.join(log_dir, "best_model", "best_model.zip")
        vis_vecnormalize_path = os.path.join(log_dir, "best_model", "vecnormalize.pkl")

        vis_thread = VisualizationThread(
            model_path=vis_model_path,
            vecnormalize_path=vis_vecnormalize_path,
            tunnel_config=tunnel_config,
            point_cloud_config=point_cloud_config,
            point_cloud_history_len=config.point_cloud_history_len,
            lstm_hidden_size=config.lstm_hidden_size,
            lstm_num_layers=config.lstm_num_layers,
            pc_encoder_dims=encoder_dims,
            state_feature_dim=config.state_feature_dim,
            dropout=config.dropout,
            use_layer_norm=config.use_layer_norm,
            bidirectional=config.bidirectional,
            pi_hidden=(config.pi_hidden_1, config.pi_hidden_2),
            vf_hidden=(config.vf_hidden_1, config.vf_hidden_2),
            n_envs=config.n_envs,  # 传递并行环境数量
            grid_mode=True,  # 启用网格模式
        )
        vis_thread.start()
        curriculum_callback.set_visualization_thread(vis_thread)

        print("[Vis] Visualization controls:")
        print("  - Space: Pause/Resume visualization")
        print("  - R: Reset all visualization episodes")
        print("  - Mouse over viewport + Right-drag: Rotate camera")
        print("  - ESC: Quit (training will continue)")
        print(f"  - Grid Layout: {vis_thread.grid_rows}x{vis_thread.grid_cols}")
        print("=" * 60 + "\n")

    callbacks = CallbackList([checkpoint_callback, save_normalize_callback,
                              eval_callback, save_best_normalize_callback,
                              curriculum_callback])

    # 训练信息
    print(f"\n{'=' * 60}")
    print("Training Configuration:")
    print(f"{'=' * 60}")
    print(f"  - Total timesteps: {config.total_timesteps:,}")
    print(f"  - Parallel environments: {config.n_envs}")
    print(f"  - Learning rate: {config.learning_rate}")
    print(f"  - Batch size: {config.batch_size}")
    print(f"  - N steps: {config.n_steps}")
    print(f"  - N epochs: {config.n_epochs}")
    print(f"  - Gamma: {config.gamma}")
    print(f"  - Mode: Visual Hybrid LSTM")
    print(f"  - Obstacles: {config.num_obstacles}")
    print(f"  - Device: {config.device}")
    print(f"{'=' * 60}\n")

    try:
        model.learn(
            total_timesteps=config.total_timesteps,
            callback=callbacks,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")

    # 保存最终模型
    final_model_path = os.path.join(log_dir, "final_model")
    model.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")

    # 保存归一化参数
    normalize_path = os.path.join(log_dir, "vecnormalize.pkl")
    env.save(normalize_path)
    print(f"VecNormalize statistics saved to: {normalize_path}")

    # 停止可视化线程
    if vis_thread is not None:
        print("\nStopping visualization thread...")
        vis_thread.running = False
        vis_thread.join(timeout=5)

    # 清理
    env.close()
    eval_env.close()

    return log_dir


def main():
    parser = argparse.ArgumentParser(description='Submarine Obstacle Avoidance - Visual Training')

    parser.add_argument('--resume_model', type=str, default=None,
                        help='Path to existing model to resume training from')
    parser.add_argument('--resume_normalize', type=str, default=None,
                        help='Path to VecNormalize file to resume training from')
    parser.add_argument('--resume_stage', type=int, default=1,
                        help='Curriculum stage to resume from (1-5)')
    parser.add_argument('--total_timesteps', type=int, default=None,
                        help='Total training timesteps')
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='Learning rate')
    parser.add_argument('--n_envs', type=int, default=None,
                        help='Number of parallel environments')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for training results')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use for training (auto, cpu, cuda)')

    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override batch size (e.g. 128, 256, 512)')
    parser.add_argument('--n_epochs', type=int, default=None,
                        help='Override number of epochs (e.g. 10, 20)')
    parser.add_argument('--no_visual', action='store_true',
                        help='Disable visualization')

    args = parser.parse_args()

    config = TrainingConfig()

    if args.resume_model is not None:
        config.resume_model_path = args.resume_model
    if args.resume_normalize is not None:
        config.resume_normalize_path = args.resume_normalize
    if args.resume_stage is not None:
        config.resume_stage = args.resume_stage
    if args.total_timesteps is not None:
        config.total_timesteps = args.total_timesteps
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.n_envs is not None:
        config.n_envs = args.n_envs
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.device is not None:
        config.device = args.device
    if args.no_visual:
        config.visualize = False

    print("=" * 60)
    print("Submarine Obstacle Avoidance - PPO Training (Visual Hybrid LSTM)")
    print("=" * 60)
    if config.resume_model_path:
        print("Mode: Resume Training")
    else:
        print("Mode: Fresh Training")
    print(f"Parallel Environments: {config.n_envs}")
    print(f"Visualization: {'Enabled' if config.visualize else 'Disabled'}")
    print("=" * 60)

    train(config)


if __name__ == "__main__":
    main()
