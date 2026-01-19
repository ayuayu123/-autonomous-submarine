#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py: 使用 Stable-Baselines3 PPO 训练潜艇避障策略

模式:
- 混合 LSTM 模式: 点云序列经 LSTM 处理后与状态融合 (Hybrid LSTM)
"""
import os
import sys
import argparse
from datetime import datetime
import numpy as np

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
from src.rl.reward_logger_callback import RewardLoggerCallback


class SuccessRateCallback(BaseCallback):
    """
    自定义回调：追踪训练过程中的成功率
    
    每隔一定步数输出:
    - 成功率 (到达终点的比例)
    - 碰撞类型分布 (边界/障碍物)
    - 平均进度
    """
    
    def __init__(self, check_freq: int = 10000, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        
        # 统计变量
        self.episode_count = 0
        self.success_count = 0
        self.boundary_collision_count = 0
        self.obstacle_collision_count = 0
        self.timeout_count = 0
        self.total_progress = 0.0
        
        # 用于周期性重置的计数
        self.period_episode_count = 0
        self.period_success_count = 0
        self.period_boundary_collision_count = 0
        self.period_obstacle_collision_count = 0
        self.period_timeout_count = 0
        self.period_total_progress = 0.0
        
    def _on_step(self) -> bool:
        # 检查是否有 episode 结束
        # VecEnv 的 infos 是一个列表，dones 是 numpy array
        infos = self.locals.get('infos', [])
        dones = self.locals.get('dones', [])
        
        # 确保 dones 是可迭代的
        if hasattr(dones, '__iter__'):
            for i, done in enumerate(dones):
                if done:
                    self.episode_count += 1
                    self.period_episode_count += 1
                    
                    # 获取对应的 info
                    if i < len(infos):
                        info = infos[i]
                        # 尝试从不同位置获取终止信息
                        terminal_info = info.get('terminal_info', info)
                        
                        # 获取详细信息
                        goal_reached = terminal_info.get('goal_reached', False)
                        collision = terminal_info.get('collision', False)
                        collision_reason = terminal_info.get('collision_reason', '')
                        progress = terminal_info.get('progress', 0.0)
                        
                        self.total_progress += progress
                        self.period_total_progress += progress
                        
                        if goal_reached:
                            self.success_count += 1
                            self.period_success_count += 1
                        elif collision:
                            if collision_reason == 'boundary':
                                self.boundary_collision_count += 1
                                self.period_boundary_collision_count += 1
                            else:  # obstacle
                                self.obstacle_collision_count += 1
                                self.period_obstacle_collision_count += 1
                        else:  # timeout
                            self.timeout_count += 1
                            self.period_timeout_count += 1
        
        # 周期性输出统计（无论有没有episode完成都输出）
        if self.n_calls % self.check_freq == 0:
            self._print_stats()
            self._reset_period_stats()
        
        return True
    
    def _print_stats(self):
        """打印统计信息"""
        print(f"\n{'='*60}")
        print(f"[SuccessRateCallback] Step {self.num_timesteps:,}")
        print(f"{'='*60}")
        
        if self.period_episode_count > 0:
            success_rate = self.period_success_count / self.period_episode_count * 100
            avg_progress = self.period_total_progress / self.period_episode_count * 100
            print(f"  Period Episodes: {self.period_episode_count}")
            print(f"  Success Rate: {success_rate:.1f}% ({self.period_success_count}/{self.period_episode_count})")
            print(f"  Avg Progress: {avg_progress:.1f}%")
            print(f"  Failures: boundary={self.period_boundary_collision_count}, obstacle={self.period_obstacle_collision_count}, timeout={self.period_timeout_count}")
        else:
            print(f"  Period Episodes: 0 (no episodes completed in this period)")
        
        # 总体统计
        if self.episode_count > 0:
            total_success_rate = self.success_count / self.episode_count * 100
            print(f"  --- Total: {total_success_rate:.1f}% success ({self.success_count}/{self.episode_count}) ---")
        else:
            print(f"  --- Total: No episodes completed yet ---")
        print(f"{'='*60}\n")
    
    def _reset_period_stats(self):
        """重置周期统计"""
        self.period_episode_count = 0
        self.period_success_count = 0
        self.period_boundary_collision_count = 0
        self.period_obstacle_collision_count = 0
        self.period_timeout_count = 0
        self.period_total_progress = 0.0


class SaveVecNormalizeCallback(BaseCallback):
    """
    自定义回调：在每次保存 best_model 时也保存 VecNormalize 统计信息
    
    这样 eval_model.py 就能自动找到对应的 vecnormalize.pkl
    """
    
    def __init__(self, save_path: str, train_env, verbose: int = 1):
        super().__init__(verbose)
        self.save_path = save_path
        self.train_env = train_env
        self.last_best_model_path = None
        
    def _on_step(self) -> bool:
        # 检查 best_model 目录下是否有新的模型被保存
        best_model_path = os.path.join(self.save_path, "best_model.zip")
        
        if os.path.exists(best_model_path):
            # 检查文件修改时间，判断是否是新保存的
            current_mtime = os.path.getmtime(best_model_path)
            
            if self.last_best_model_path != current_mtime:
                # 新的 best_model 被保存了，同时保存 VecNormalize
                vecnorm_path = os.path.join(self.save_path, "vecnormalize.pkl")
                self.train_env.save(vecnorm_path)
                
                if self.verbose > 0:
                    print(f"[SaveVecNormalizeCallback] VecNormalize saved to: {vecnorm_path}")
                
                self.last_best_model_path = current_mtime
        
        return True


def make_env(rank: int, seed: int = 0, 
             point_cloud_history_len: int = 8,
             tunnel_config: TunnelConfig = None, 
             point_cloud_config: PointCloudConfig = None):
    """
    创建单个环境实例的工厂函数
    """
    def _init():
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


def train(args):
    """训练主函数"""
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_name = "hybrid_lstm"
    
    log_dir = os.path.join(args.output_dir, f"submarine_ppo_{mode_name}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"Training output directory: {log_dir}")
    
    # 环境配置 (v13 - 增大管道半径，给更多机动空间)
    tunnel_config = TunnelConfig(
        radius=8.0,              # 从5.0增大到8.0，更多机动空间
        length=50.0,
        center_z=100.0,
        num_obstacles=args.num_obstacles,
        obstacle_radius_min=0.7,
        obstacle_radius_max=1.0,
    )
    
    # 点云采样配置 (固定采样版本)
    point_cloud_config = PointCloudConfig(
        num_points=384,               # 增加到384点
        obstacle_points_ratio=0.6,    # 60%给障碍物
        normalize_range=25.0,
        points_per_obstacle=32,       # 每个障碍物32个固定采样点
    )
    
    # 创建并行环境
    print(f"\nCreating {args.n_envs} parallel environments...")
    print("Mode: HYBRID LSTM (点云序列 → LSTM → 特征 + 状态 → PPO)")
    print(f"  - Point cloud history length: {args.point_cloud_history_len}")
    
    env = SubprocVecEnv([
        make_env(i, args.seed, 
                 point_cloud_history_len=args.point_cloud_history_len,
                 tunnel_config=tunnel_config, 
                 point_cloud_config=point_cloud_config) 
        for i in range(args.n_envs)
    ])
    
    # 创建评估环境
    eval_env = DummyVecEnv([
        make_env(0, args.seed + 1000, 
                 point_cloud_history_len=args.point_cloud_history_len,
                 tunnel_config=tunnel_config, 
                 point_cloud_config=point_cloud_config)
    ])
    
    # 添加归一化包装器
    # 【关键修复】只归一化 state，不归一化 point_cloud_seq (已经是 [-1,1])
    print("\nAdding VecNormalize for observation and reward normalization...")
    print("  - Only normalizing 'state' key (point_cloud_seq already normalized)")
    env = VecNormalize(
        env, 
        norm_obs=True, 
        norm_reward=True, 
        clip_obs=10.0,      # 增大clip范围避免截断
        clip_reward=10.0,
        norm_obs_keys=["state"]  # 只归一化state，不处理点云
    )
    eval_env = VecNormalize(
        eval_env, 
        norm_obs=True, 
        norm_reward=False, 
        training=False,
        norm_obs_keys=["state"]
    )
    eval_env.obs_rms = env.obs_rms  # 使用训练环境的统计信息
    
    # ============ 混合 LSTM 模式 ============
    
    # 混合 LSTM 模式: 使用自定义特征提取器 + 标准 PPO
    from src.rl.custom_extractor import PointCloudLSTMExtractor
    
    # 解析点云编码器维度
    encoder_dims = tuple(int(x) for x in args.pc_encoder_dims.split(','))
    
    policy_kwargs = dict(
        features_extractor_class=PointCloudLSTMExtractor,
        features_extractor_kwargs=dict(
            lstm_hidden_size=args.lstm_hidden_size,
            lstm_num_layers=args.lstm_num_layers,
            point_cloud_encoder_dims=encoder_dims,
            state_feature_dim=args.state_feature_dim,
            dropout=args.dropout,
            use_layer_norm=args.use_layer_norm,
            bidirectional=args.bidirectional,
        ),
        net_arch=dict(
            pi=[args.pi_hidden_1, args.pi_hidden_2],  # Actor 网络
            vf=[args.vf_hidden_1, args.vf_hidden_2],  # Critic 网络
        ),
    )
    
    print(f"\nUsing PPO with PointCloudLSTMExtractor (优化版):")
    print(f"  - LSTM hidden size: {args.lstm_hidden_size}")
    print(f"  - LSTM num layers: {args.lstm_num_layers}")
    print(f"  - Bidirectional: {args.bidirectional}")
    print(f"  - Point cloud encoder: {encoder_dims}")
    print(f"  - State feature dim: {args.state_feature_dim}")
    print(f"  - Dropout: {args.dropout}")
    print(f"  - Layer Norm: {args.use_layer_norm}")
    print(f"  - Actor network: [{args.pi_hidden_1}, {args.pi_hidden_2}]")
    print(f"  - Critic network: [{args.vf_hidden_1}, {args.vf_hidden_2}]")
    print(f"  - Policy: MultiInputPolicy")
    
    model = PPO(
        "MultiInputPolicy",  # 使用 Dict 观测空间
        env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=log_dir,
        device=args.device,
    )
    
    # 回调函数
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=os.path.join(log_dir, "checkpoints"),
        name_prefix="submarine_ppo",
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(log_dir, "best_model"),
        log_path=os.path.join(log_dir, "eval_logs"),
        eval_freq=args.eval_freq,
        n_eval_episodes=5,
        deterministic=True,
    )
        
    # 成功率追踪回调
    success_rate_callback = SuccessRateCallback(
        check_freq=2000,  # 每2000步输出一次成功率
        verbose=1
    )
    
    # 【新增】VecNormalize 自动保存回调
    # 每次 best_model 更新时，同时保存 vecnormalize.pkl
    save_vecnorm_callback = SaveVecNormalizeCallback(
        save_path=os.path.join(log_dir, "best_model"),
        train_env=env,
        verbose=1
    )
    
    # 【新增】奖励组件日志回调
    # 每个 episode 结束时将奖励组件统计写入 CSV 文件
    reward_logger_callback = RewardLoggerCallback(
        log_dir=log_dir,
        log_freq=1,  # 每个 episode 都记录
        verbose=1,
        csv_filename="reward_components.csv"
    )
    
    callbacks = CallbackList([
        checkpoint_callback, 
        eval_callback, 
        success_rate_callback, 
        save_vecnorm_callback,
        reward_logger_callback  # 奖励日志
    ])
    
    # 训练信息
    print(f"\n{'='*60}")
    print("Training Configuration:")
    print(f"{'='*60}")
    print(f"  - Total timesteps: {args.total_timesteps:,}")
    print(f"  - Learning rate: {args.learning_rate}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - N steps: {args.n_steps}")
    print(f"  - N epochs: {args.n_epochs}")
    print(f"  - Gamma: {args.gamma}")
    print(f"  - Mode: Hybrid LSTM")
    print(f"  - Obstacles: {args.num_obstacles}")
    print(f"  - Device: {args.device}")
    print(f"{'='*60}\n")
    
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
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
    parser = argparse.ArgumentParser(description="Train submarine obstacle avoidance with PPO (Hybrid LSTM Mode)")
    
    # 训练参数
    parser.add_argument("--total-timesteps", type=int, default=2_000_000,
                        help="Total training timesteps")
    parser.add_argument("--n-envs", type=int, default=4,
                        help="Number of parallel environments")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    # PPO 超参数
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--n-steps", type=int, default=2048,
                        help="Number of steps per environment per update")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Minibatch size")
    parser.add_argument("--n-epochs", type=int, default=10,
                        help="Number of epochs per update")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    
    # 环境参数
    parser.add_argument("--num-obstacles", type=int, default=14,
                        help="Number of obstacles in tunnel")
    
    # 混合 LSTM 模式参数
    parser.add_argument("--point-cloud-history-len", type=int, default=8,
                        help="Number of point cloud frames to keep in history (序列长度, 从40减少到8)")
    parser.add_argument("--lstm-hidden-size", type=int, default=128,
                        help="LSTM hidden layer size (隐藏层大小)")
    parser.add_argument("--lstm-num-layers", type=int, default=1,
                        help="Number of LSTM layers (LSTM层数, 1-3)")
    parser.add_argument("--bidirectional", action="store_true",
                        help="Use bidirectional LSTM (双向LSTM, 输出维度翻倍)")
    
    # 特征提取器参数
    parser.add_argument("--pc-encoder-dims", type=str, default="128,64",
                        help="Point cloud encoder hidden dimensions (点云编码器维度, 两层768->128->64)")
    parser.add_argument("--state-feature-dim", type=int, default=64,
                        help="State encoder output dimension (状态特征维度)")
    
    # 正则化参数
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout probability (0-0.5, 推荐0.1-0.2)")
    parser.add_argument("--use-layer-norm", action="store_true", default=True,
                        help="Use Layer Normalization (推荐开启)")
    parser.add_argument("--no-layer-norm", action="store_true",
                        help="Disable Layer Normalization")
    
    # PPO 额外超参数
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="GAE lambda parameter (0.9-0.99)")
    parser.add_argument("--clip-range", type=float, default=0.2,
                        help="PPO clip range (0.1-0.3)")
    parser.add_argument("--ent-coef", type=float, default=0.01,
                        help="Entropy coefficient for exploration (0.001-0.05)")
    
    # Actor/Critic 网络架构
    parser.add_argument("--pi-hidden-1", type=int, default=128,
                        help="Actor network first hidden layer size")
    parser.add_argument("--pi-hidden-2", type=int, default=64,
                        help="Actor network second hidden layer size")
    parser.add_argument("--vf-hidden-1", type=int, default=128,
                        help="Critic network first hidden layer size")
    parser.add_argument("--vf-hidden-2", type=int, default=64,
                        help="Critic network second hidden layer size")
    
    # 保存参数
    parser.add_argument("--output-dir", type=str, default="./training_output",
                        help="Output directory for logs and models")
    parser.add_argument("--save-freq", type=int, default=10000,
                        help="Checkpoint save frequency")
    parser.add_argument("--eval-freq", type=int, default=5000,
                        help="Evaluation frequency")
    
    # 设备
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (auto, cuda, cpu)")
    
    args = parser.parse_args()
    
    # 处理 --no-layer-norm 标志
    if args.no_layer_norm:
        args.use_layer_norm = False
    
    print("=" * 60)
    print("Submarine Obstacle Avoidance - PPO Training (Hybrid LSTM)")
    print("=" * 60)
    
    train(args)


if __name__ == "__main__":
    main()
