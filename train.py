#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py: 使用 Stable-Baselines3 PPO 训练潜艇避障策略

模式:
- 混合 LSTM 模式: 点云序列经 LSTM 处理后与状态融合 (Hybrid LSTM)
"""
import os
import sys
from datetime import datetime
import numpy as np
import argparse
from copy import deepcopy

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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


class TrainingConfig:
    """训练配置 - 硬编码超参"""

    def __init__(self):
        self.total_timesteps = 9_000_000
        self.n_envs = 15
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


class CurriculumConfig:
    """课程配置"""

    def __init__(self, stage: int, num_obstacles: int,
                 w_velocity: float = 1.0):
        self.stage = stage
        self.num_obstacles = num_obstacles
        self.w_velocity = w_velocity


class Stage5ProgressiveManager:
    """
    课程5的渐进式训练管理器

    训练策略:
    - Stage 0: 固定障碍物位置，成功率≥70%代表学会
    - Stage 1-10: 每次调整障碍物位置，调整量逐渐增大
    - Stage 11+: 完全随机化训练
    """

    def __init__(self, min_eval_episodes: int = 10, success_threshold: float = 0.7):
        self.progressive_stage = 0  # 0=固定, 1-10=渐进, 11+=随机
        self.min_eval_episodes = min_eval_episodes  # 最少评估episode数
        self.success_threshold = success_threshold  # 成功率阈值
        self.episode_results = []  # 存储最近的成功/失败结果
        self.max_perturbation_stage = 10  # 最大渐进调整次数

    def add_result(self, success: bool):
        """添加一次评估结果"""
        self.episode_results.append(success)
        # 只保留最近的结果
        if len(self.episode_results) > self.min_eval_episodes:
            self.episode_results.pop(0)

    def should_advance(self) -> bool:
        """检查是否应该进入下一渐进阶段"""
        if self.progressive_stage >= self.max_perturbation_stage + 1:
            # 已经进入完全随机模式
            return False

        # 检查是否有足够的评估数据
        if len(self.episode_results) < self.min_eval_episodes:
            return False

        # 检查成功率是否达到阈值
        success_rate = sum(self.episode_results) / len(self.episode_results)
        return success_rate >= self.success_threshold

    def advance(self):
        """进入下一渐进阶段"""
        if self.progressive_stage <= self.max_perturbation_stage:
            self.progressive_stage += 1
            self.episode_results = []
            return True
        return False

    def reset_for_new_stage(self):
        """进入新阶段时重置结果"""
        self.episode_results = []

    def is_in_random_mode(self) -> bool:
        """检查是否已进入完全随机模式"""
        return self.progressive_stage > self.max_perturbation_stage


class CurriculumManager:
    """课程管理器"""

    def __init__(self, eval_episodes: int = 10, success_threshold: float = 0.8, start_stage: int = 1):
        self.current_stage = start_stage
        self.base_eval_episodes = eval_episodes  # 基础评估次数
        self.success_threshold = success_threshold
        self.episode_results = []

        # 定义各阶段的通过阈值
        self.stage_thresholds = {
            1: 0.8, 2: 0.8, 3: 0.8, 4: 0.8,
            5: 0.8,  # Stage 5 基础阈值
            6: 0.9  # Stage 6 严格阈值 (90%)
        }

        # 课程5的渐进式训练管理器 (70%成功率阈值)
        self.stage5_progressive = Stage5ProgressiveManager(
            min_eval_episodes=eval_episodes,
            success_threshold=0.7
        )

    def start_new_evaluation(self):
        """开始新的评估，清空之前的结果"""
        self.episode_results = []

    def get_required_eval_episodes(self) -> int:
        """获取当前阶段所需的评估轮数"""
        # Stage 6 需要 30 次评估以确保稳定性
        if self.current_stage == 6:
            return 30
        return self.base_eval_episodes

    def add_episode_result(self, success: bool, progress: float):
        """添加一个episode的结果"""
        self.episode_results.append({
            'success': success,
            'progress': progress,
        })

        # 如果当前是课程5，同时添加到渐进式管理器
        if self.current_stage == 5:
            self.stage5_progressive.add_result(success)

    def should_advance(self) -> bool:
        """判断是否应该进入下一课程"""
        if self.current_stage >= len(CURRICULUM_STAGES):
            return False

        # 检查评估次数是否达标
        required_eps = self.get_required_eval_episodes()
        if len(self.episode_results) < required_eps:
            return False

        # 特殊保护：如果是 Stage 5，必须先完成渐进式训练（进入随机模式）才能晋升到 Stage 6
        if self.current_stage == 5:
            if not self.stage5_progressive.is_in_random_mode():
                return False

        success_rate = sum(r['success'] for r in self.episode_results) / len(self.episode_results)
        threshold = self.stage_thresholds.get(self.current_stage, self.success_threshold)
        return success_rate >= threshold

    def advance(self):
        """进入下一课程"""
        if self.current_stage < len(CURRICULUM_STAGES):
            self.current_stage += 1
            self.episode_results = []
            # 重置课程5的渐进式管理器
            self.stage5_progressive = Stage5ProgressiveManager(
                min_eval_episodes=self.base_eval_episodes,
                success_threshold=0.7
            )
            return True
        return False

    def get_current_config(self) -> CurriculumConfig:
        """获取当前课程配置"""
        return CURRICULUM_STAGES[self.current_stage]

    def get_recent_success_rate(self, window_size: int = 15) -> float:
        """获取最近 N 次的成功率"""
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
    # Stage 6: 15个障碍物，随机位置，要求 90% 成功率 (30次评估)
    6: CurriculumConfig(stage=6, num_obstacles=15, w_velocity=0.2),
}


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
        """在每个 rollout 结束后检查是否需要保存归一化"""
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
                 log_dir: str = "", verbose: int = 0):
        super().__init__(verbose)
        self.curriculum_manager = curriculum_manager
        self.eval_env = eval_env
        self.curriculum_state = curriculum_state
        self.eval_freq = eval_freq
        self.log_dir = log_dir
        self.last_eval_time = 0
        self.min_timesteps_before_eval = 10000
        self.solved_stage6 = False  # 标记是否已经通关Stage 6

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_eval_time >= self.eval_freq:
            self.last_eval_time = self.num_timesteps

            if self.num_timesteps >= self.min_timesteps_before_eval:
                self._evaluate_and_check()
            elif self.verbose > 0:
                print(
                    f"[Curriculum] Skipping evaluation: only {self.num_timesteps} timesteps trained (need {self.min_timesteps_before_eval})")

        return True

    def _evaluate_and_check(self):
        """评估并检查是否需要切换课程或渐进式阶段"""
        if self.verbose > 0:
            print(f"\n[Evaluation] Evaluating curriculum progress at timestep {self.num_timesteps}")

        self.curriculum_manager.start_new_evaluation()

        # 获取当前阶段需要评估的次数 (Stage 6 为 30 次，其他为 10 次)
        num_evals = self.curriculum_manager.get_required_eval_episodes()
        print(f"[Evaluation] Running {num_evals} episodes for Stage {self.curriculum_manager.current_stage}...")

        for episode_idx in range(num_evals):
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

        # 统计结果
        results = self.curriculum_manager.episode_results
        success_rate = sum(r['success'] for r in results) / len(results)
        avg_progress = sum(r['progress'] for r in results) / len(results)

        if self.verbose > 0:
            print(
                f"[Evaluation] Overall success rate: {success_rate:.2%} ({len(results)} eps), Avg progress: {avg_progress:.2%}")

            # 如果是课程5，显示渐进式训练的成功率
            if self.curriculum_manager.current_stage == 5:
                progressive_manager = self.curriculum_manager.stage5_progressive
                if len(progressive_manager.episode_results) > 0:
                    prog_success_rate = sum(progressive_manager.episode_results) / len(
                        progressive_manager.episode_results)
                    print(
                        f"[Progressive] Current stage: {progressive_manager.progressive_stage}, Success rate: {prog_success_rate:.2%} (threshold: 70%)")

        # ============ Stage 6 通关判断逻辑 ============
        if self.curriculum_manager.current_stage == 6:
            # Stage 6 要求 90% 成功率 (基于 30 次评估)
            if success_rate >= 0.90:
                print(f"\n{'#' * 60}")
                print(f"STAGE 6 SOLVED! Success Rate: {success_rate:.2%} (>= 90% over {num_evals} episodes)")
                print(f"{'#' * 60}")

                # 保存通关模型
                solved_save_path = os.path.join(self.log_dir, "stage6_solved_model")
                self.model.save(solved_save_path)

                # 保存归一化参数
                norm_save_path = os.path.join(self.log_dir, "stage6_solved_vecnormalize.pkl")
                self.eval_env.save(norm_save_path)

                print(f"Saved SOLVED model to: {solved_save_path}")
                print(f"Saved SOLVED normalization to: {norm_save_path}")
                self.solved_stage6 = True
        # ============================================

        # 检查课程5的渐进式训练阶段
        if self.curriculum_manager.current_stage == 5:
            progressive_manager = self.curriculum_manager.stage5_progressive
            # 只有当没有达到随机模式时，才允许在这里 advance
            if not progressive_manager.is_in_random_mode() and progressive_manager.should_advance():
                old_stage = progressive_manager.progressive_stage
                if progressive_manager.advance():
                    new_stage = progressive_manager.progressive_stage
                    # 更新课程状态中的渐进式阶段
                    self.curriculum_state['progressive_stage'] = new_stage

                    mode_desc = "Fixed" if new_stage == 0 else (
                        f"Perturbed (±{new_stage * 0.5:.1f}m)" if new_stage <= 10 else "Random")
                    print(f"\n{'=' * 60}")
                    print(f"PROGRESSIVE STAGE ADVANCED: Stage {old_stage} -> Stage {new_stage}")
                    print(f"  - Mode: {mode_desc}")
                    print(f"{'=' * 60}\n")

        # 检查是否需要切换主课程阶段 (例如 5 -> 6)
        if self.curriculum_manager.should_advance():
            old_stage = self.curriculum_manager.current_stage
            if self.curriculum_manager.advance():
                new_stage = self.curriculum_manager.current_stage
                new_config = self.curriculum_manager.get_current_config()

                # 更新全局共享状态
                self.curriculum_state['current_stage'] = new_config.stage
                self.curriculum_state['num_obstacles'] = new_config.num_obstacles
                self.curriculum_state['w_velocity'] = new_config.w_velocity

                # 处理 Stage 5 和 Stage 6 的渐进式模式逻辑
                if new_stage == 5:
                    self.curriculum_state['progressive_mode'] = True
                    self.curriculum_state['progressive_stage'] = 0
                elif new_stage == 6:
                    # 进入 Stage 6：开启渐进式，但直接设置为 11 (Random)
                    self.curriculum_state['progressive_mode'] = True
                    self.curriculum_state['progressive_stage'] = 11
                else:
                    self.curriculum_state['progressive_mode'] = False
                    self.curriculum_state['progressive_stage'] = 0

                print(f"\n{'=' * 60}")
                print(f"CURRICULUM ADVANCED: Stage {old_stage} -> Stage {new_stage}")
                print(f"  - Obstacles: {new_config.num_obstacles}")
                print(f"  - Velocity weight: {new_config.w_velocity}")
                if new_stage == 5:
                    print(f"  - Progressive Training: ENABLED (Start from Fixed)")
                elif new_stage == 6:
                    print(f"  - Progressive Training: ENABLED (Mode: FULL RANDOM)")
                    print(f"  - Target: >90% Success Rate over 30 episodes")
                print(f"{'=' * 60}\n")


def make_env(rank: int, seed: int = 0,
             point_cloud_history_len: int = 8,
             tunnel_config: TunnelConfig = None,
             point_cloud_config: PointCloudConfig = None,
             curriculum_state=None):
    """
    创建单个环境实例的工厂函数
    """

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

        # 调试日志
        if progressive_mode and rank == 0:
            # 只在rank 0打印一次，避免刷屏
            pass

            # 为每个环境创建独立的 tunnel_config 副本
        env_tunnel_config = deepcopy(tunnel_config) if tunnel_config else None
        if env_tunnel_config is not None:
            env_tunnel_config.num_obstacles = num_obstacles
            # 设置渐进式训练模式
            env_tunnel_config.progressive_mode = progressive_mode
            env_tunnel_config.progressive_stage = progressive_stage
            # 对于课程5，设置固定的障碍物种子
            if progressive_mode and progressive_stage == 0:
                env_tunnel_config.fixed_obstacle_seed = 42  # 使用固定种子生成基础布局

        env = SubmarineEnv(
            tunnel_config=env_tunnel_config,
            point_cloud_config=point_cloud_config,
            point_cloud_history_len=point_cloud_history_len,
            max_steps=1200,
            curriculum_state=curriculum_state  # 传递课程状态，使环境能够动态同步
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
    mode_name = "hybrid_lstm"

    if config.resume_model_path:
        resume_dir = os.path.dirname(os.path.dirname(config.resume_model_path))
        log_dir = resume_dir
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

    # 创建课程管理器，支持从指定阶段开始
    curriculum_manager = CurriculumManager(eval_episodes=10, success_threshold=0.8, start_stage=config.resume_stage)

    # 创建共享课程状态，支持从指定阶段恢复
    manager = Manager()

    if config.resume_stage in CURRICULUM_STAGES:
        resume_config = CURRICULUM_STAGES[config.resume_stage]

        # 判断是否为渐进式模式 (Stage 5 或 6)
        progressive_mode = (resume_config.stage >= 5)

        # 如果是 Stage 6，直接设置为随机模式 (11)，否则从 0 开始
        progressive_stage = 11 if resume_config.stage == 6 else 0

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

    # 创建并行环境
    print(f"\nCreating {config.n_envs} parallel environments...")
    print("Mode: HYBRID LSTM (点云序列 -> LSTM -> 特征 + 状态 -> PPO)")
    print(f"  - Point cloud history length: {config.point_cloud_history_len}")
    print(
        f"  - Current curriculum: Stage {curriculum_state['current_stage']} ({curriculum_state['num_obstacles']} obstacles, velocity reward weight={curriculum_state['w_velocity']})")

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
        eval_env.obs_rms = env.obs_rms  # 使用训练环境的统计信息

    # ============ 混合 LSTM 模式 ============

    # 混合 LSTM 模式: 使用自定义特征提取器 + 标准 PPO
    from src.rl.custom_extractor import PointCloudLSTMExtractor

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
            pi=[config.pi_hidden_1, config.pi_hidden_2],  # Actor 网络
            vf=[config.vf_hidden_1, config.vf_hidden_2],  # Critic 网络
        ),
    )

    print(f"\nUsing PPO with PointCloudLSTMExtractor (优化版):")
    print(f"  - LSTM hidden size: {config.lstm_hidden_size}")
    print(f"  - LSTM num layers: {config.lstm_num_layers}")
    print(f"  - Bidirectional: {config.bidirectional}")
    print(f"  - Point cloud encoder: {encoder_dims}")
    print(f"  - State feature dim: {config.state_feature_dim}")
    print(f"  - Dropout: {config.dropout}")
    print(f"  - Layer Norm: {config.use_layer_norm}")
    print(f"  - Actor network: [{config.pi_hidden_1}, {config.pi_hidden_2}]")
    print(f"  - Critic network: [{config.vf_hidden_1}, {config.vf_hidden_2}]")
    print(f"  - Policy: MultiInputPolicy")

    if config.resume_model_path:
        print(f"\nLoading existing model from: {config.resume_model_path}")
        model = PPO.load(config.resume_model_path, env=env, tensorboard_log=log_dir, device=config.device)
        if config.learning_rate is not None:
            print(f"--> Overriding learning rate to: {config.learning_rate}")
            model.learning_rate = config.learning_rate
            for param_group in model.policy.optimizer.param_groups:
                param_group["lr"] = config.learning_rate

        # 3. 强制更新其他训练超参数 (如果需要)
        # 注意：n_steps 不能修改，因为这涉及 buffer 的形状，修改会导致报错
        # 但是可以修改 n_epochs (每次更新循环次数) 或 batch_size (如果显存允许)
        if config.batch_size is not None:
            print(f"  -> Overriding Batch Size: {model.batch_size} -> {config.batch_size}")
            model.batch_size = config.batch_size

        # 4. 强制更新 N Epochs
        # 默认 argparse 传入的是 TrainingConfig 里的值 (10)
        if config.n_epochs is not None:
            print(f"  -> Overriding N Epochs: {model.n_epochs} -> {config.n_epochs}")
            model.n_epochs = config.n_epochs

        # ==================== 修改结束 ====================

        print("Model loaded successfully, continuing training...")
    else:
        model = PPO(
            "MultiInputPolicy",  # 使用 Dict 观测空间
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
            verbose=2,
            tensorboard_log=log_dir,
            device=config.device,
        )

    # 回调函数
    checkpoint_callback = CheckpointCallback(
        save_freq=config.save_freq,
        save_path=os.path.join(log_dir, "checkpoints"),
        name_prefix="submarine_ppo",
    )

    # 每次保存 checkpoint 时同时保存归一化参数
    save_normalize_callback = SaveNormalizeOnCheckpointCallback(
        vec_env=env,
        save_freq=config.save_freq,
        save_path=os.path.join(log_dir, "checkpoints"),
        verbose=1
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(log_dir, "best_model"),
        log_path=os.path.join(log_dir, "eval_logs"),
        eval_freq=config.eval_freq,
        n_eval_episodes=5,
        deterministic=True,
    )

    # 保存 best model 时同时保存归一化参数
    save_best_normalize_callback = SaveNormalizeOnBestModelCallback(
        vec_env=env,
        save_path=os.path.join(log_dir, "best_model"),
        verbose=1
    )

    # 课程学习回调
    curriculum_callback = CurriculumCallback(
        curriculum_manager=curriculum_manager,
        eval_env=eval_env,
        curriculum_state=curriculum_state,
        eval_freq=config.eval_freq,
        log_dir=log_dir,  # 传入 log_dir 用于保存通关模型
        verbose=1
    )

    callbacks = CallbackList([checkpoint_callback, save_normalize_callback, eval_callback, save_best_normalize_callback,
                              curriculum_callback])

    # 训练信息
    print(f"\n{'=' * 60}")
    print("Training Configuration:")
    print(f"{'=' * 60}")
    print(f"  - Total timesteps: {config.total_timesteps:,}")
    print(f"  - Learning rate: {config.learning_rate}")
    print(f"  - Batch size: {config.batch_size}")
    print(f"  - N steps: {config.n_steps}")
    print(f"  - N epochs: {config.n_epochs}")
    print(f"  - Gamma: {config.gamma}")
    print(f"  - Mode: Hybrid LSTM")
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

    # 清理
    env.close()
    eval_env.close()

    return log_dir


def main():
    parser = argparse.ArgumentParser(description='Submarine Obstacle Avoidance - PPO Training')

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
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.n_epochs is not None:
        config.n_epochs = args.n_epochs

    print("=" * 60)
    print("Submarine Obstacle Avoidance - PPO Training (Hybrid LSTM)")
    print("=" * 60)
    if config.resume_model_path:
        print("Mode: Resume Training")
    else:
        print("Mode: Fresh Training")
    print("=" * 60)

    train(config)


if __name__ == "__main__":
    main()