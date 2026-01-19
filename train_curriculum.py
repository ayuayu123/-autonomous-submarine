#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_curriculum.py: 课程学习训练脚本

使用渐进式难度训练，从简单任务开始逐步增加挑战。
每个阶段训练到达标后自动晋级到下一阶段。

用法:
    python train_curriculum.py --device cuda
    python train_curriculum.py --start-stage 2  # 从第2阶段开始
    python train_curriculum.py --resume ./training_output/xxx/stage_2_model.zip
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
from src.rl.curriculum import (
    CurriculumStage, 
    CURRICULUM_STAGES, 
    get_stage_config, 
    get_total_stages,
    print_curriculum_info
)
from src.rl.reward_logger_callback import RewardLoggerCallback


class CurriculumCallback(BaseCallback):
    """课程学习回调 - 检测阶段达标并允许提前晋级（滑动窗口统计）"""
    
    def __init__(self, stage: 'CurriculumStage', min_timesteps: int, verbose: int = 1,
                 window_size: int = 200):  # 滑动窗口大小
        super().__init__(verbose)
        self.stage = stage
        self.min_timesteps = min_timesteps  # 达标后至少训练这么多步
        self.window_size = window_size  # 统计窗口大小
        
        # 累积统计（用于总体信息）
        self.total_episodes = 0
        self.total_successes = 0
        self.total_progress = 0.0
        
        # 【滑动窗口】最近 N 个 episode 的结果（用于晋级判断）
        from collections import deque
        self.recent_successes = deque(maxlen=window_size)  # 1=成功, 0=失败
        self.recent_progress = deque(maxlen=window_size)
        
        # 【新增】速度和碰撞统计（滑动窗口）
        self.total_speed = 0.0
        self.speed_samples = 0
        self.recent_boundary = deque(maxlen=window_size)  # 1=边界碰撞, 0=其他
        self.recent_obstacle = deque(maxlen=window_size)  # 1=障碍物碰撞, 0=其他
        
        # 晋级状态
        self.can_advance = False
        self.current_success_rate = 0.0  # 滑动窗口成功率
        self.advanced = False  # 是否已经提前晋级
        
    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [])
        dones = self.locals.get('dones', [])
        
        # 每步记录速度
        for info in infos:
            if 'forward_speed' in info:
                self.total_speed += abs(info['forward_speed'])
                self.speed_samples += 1
        
        if hasattr(dones, '__iter__'):
            for i, done in enumerate(dones):
                if done and i < len(infos):
                    self.total_episodes += 1
                    info = infos[i]
                    terminal_info = info.get('terminal_info', info)
                    
                    goal_reached = terminal_info.get('goal_reached', False)
                    progress = terminal_info.get('progress', 0.0)
                    collision_reason = terminal_info.get('collision_reason', '')
                    
                    # 累积统计
                    self.total_progress += progress
                    if goal_reached:
                        self.total_successes += 1
                    
                    # 【滑动窗口】记录最近 N 个 episode
                    self.recent_successes.append(1 if goal_reached else 0)
                    self.recent_progress.append(progress)
                    
                    # 碰撞类型（滑动窗口）
                    self.recent_boundary.append(1 if collision_reason == 'boundary' else 0)
                    self.recent_obstacle.append(1 if collision_reason == 'obstacle' else 0)
        
        # 【滑动窗口】计算最近 N 个 episode 的成功率
        if len(self.recent_successes) >= self.stage.min_episodes:
            self.current_success_rate = sum(self.recent_successes) / len(self.recent_successes)
            if self.current_success_rate >= self.stage.success_rate_threshold:
                self.can_advance = True
        
        # 定期打印统计
        if self.n_calls % 5000 == 0 and self.total_episodes > 0:
            # 累积统计
            total_rate = self.total_successes / self.total_episodes * 100
            avg_progress_total = self.total_progress / self.total_episodes * 100
            
            # 滑动窗口统计
            window_len = len(self.recent_successes)
            window_rate = sum(self.recent_successes) / max(window_len, 1) * 100
            window_progress = sum(self.recent_progress) / max(window_len, 1) * 100
            window_boundary = sum(self.recent_boundary)
            window_obstacle = sum(self.recent_obstacle)
            
            avg_speed = self.total_speed / max(self.speed_samples, 1)
            
            print(f"\n[Curriculum] {self.stage.name}")
            print(f"  Total Episodes: {self.total_episodes}")
            print(f"  【近{window_len}局】 成功率: {window_rate:.1f}%, 进度: {window_progress:.1f}%")
            print(f"  【近{window_len}局】 碰撞: 边界={window_boundary}, 障碍物={window_obstacle}")
            print(f"  Avg Speed: {avg_speed:.2f} m/s")
            print(f"  Target: {self.stage.success_rate_threshold*100:.0f}% (滑动窗口 {self.window_size} eps)")
            if self.can_advance:
                print(f"  ✓ Ready to advance!")
        
        # 【关键】达标且超过最小步数后提前结束
        if self.can_advance and self.num_timesteps >= self.min_timesteps:
            if not self.advanced:
                self.advanced = True
                avg_speed = self.total_speed / max(self.speed_samples, 1)
                window_len = len(self.recent_successes)
                print(f"\n{'='*60}")
                print(f"🎉 [{self.stage.name}] 达标! 提前晋级!")
                print(f"   滑动窗口成功率: {self.current_success_rate*100:.1f}% (目标: {self.stage.success_rate_threshold*100:.0f}%)")
                print(f"   窗口大小: {window_len} episodes")
                print(f"   平均速度: {avg_speed:.2f} m/s")
                print(f"   训练步数: {self.num_timesteps:,} (最小: {self.min_timesteps:,})")
                print(f"{'='*60}\n")
                return False  # 停止训练，进入下一阶段
        
        return True


class SaveVecNormalizeCallback(BaseCallback):
    """在保存 best_model 时同时保存 VecNormalize"""
    
    def __init__(self, save_path: str, train_env, verbose: int = 1):
        super().__init__(verbose)
        self.save_path = save_path
        self.train_env = train_env
        self.last_best_model_path = None
        
    def _on_step(self) -> bool:
        best_model_path = os.path.join(self.save_path, "best_model.zip")
        
        if os.path.exists(best_model_path):
            current_mtime = os.path.getmtime(best_model_path)
            
            if self.last_best_model_path != current_mtime:
                vecnorm_path = os.path.join(self.save_path, "vecnormalize.pkl")
                self.train_env.save(vecnorm_path)
                
                if self.verbose > 0:
                    print(f"[SaveVecNormalize] Saved to: {vecnorm_path}")
                
                self.last_best_model_path = current_mtime
        
        return True


def make_env(rank: int, seed: int, stage: CurriculumStage):
    """创建环境工厂函数"""
    def _init():
        tunnel_config = TunnelConfig(
            radius=stage.tunnel_radius,
            length=stage.tunnel_length,
            center_z=100.0,
            num_obstacles=stage.num_obstacles,
            obstacle_radius_min=stage.obstacle_radius_min,
            obstacle_radius_max=stage.obstacle_radius_max,
            obstacle_start_x=stage.obstacle_start_x,
            obstacle_min_spacing=stage.obstacle_min_spacing,
            center_obstacle_ratio=stage.center_ratio,
        )
        
        point_cloud_config = PointCloudConfig(
            num_points=stage.num_points,
            obstacle_points_ratio=0.6,
            sample_distance=stage.sample_distance,
            normalize_range=stage.sample_distance,
            points_per_obstacle=32,
        )
        
        env = SubmarineEnv(
            tunnel_config=tunnel_config,
            point_cloud_config=point_cloud_config,
            point_cloud_history_len=8,
            max_steps=1200,
        )
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def train_stage(stage_index: int, args, log_dir: str, 
                previous_model_path: str = None,
                previous_vecnorm_path: str = None):
    """训练单个阶段"""
    stage = get_stage_config(stage_index)
    
    print(f"\n{'='*70}")
    print(f"🎯 开始训练: {stage.name}")
    print(f"{'='*70}")
    print(f"  {stage.description}")
    print(f"  通道: 半径={stage.tunnel_radius}m, 长度={stage.tunnel_length}m")
    print(f"  障碍物: {stage.num_obstacles}个")
    print(f"  目标成功率: {stage.success_rate_threshold*100:.0f}%")
    print(f"  训练步数: {stage.timesteps:,}")
    print(f"{'='*70}\n")
    
    stage_log_dir = os.path.join(log_dir, f"stage_{stage_index+1}")
    os.makedirs(stage_log_dir, exist_ok=True)
    
    # 创建环境
    env = SubprocVecEnv([
        make_env(i, args.seed, stage) 
        for i in range(args.n_envs)
    ])
    
    eval_env = DummyVecEnv([
        make_env(0, args.seed + 1000, stage)
    ])
    
    # VecNormalize
    env = VecNormalize(
        env, 
        norm_obs=True, 
        norm_reward=True, 
        clip_obs=10.0,
        clip_reward=10.0,
        norm_obs_keys=["state"]
    )
    
    # 如果有之前的 vecnormalize，加载统计信息
    if previous_vecnorm_path and os.path.exists(previous_vecnorm_path):
        print(f"Loading VecNormalize stats from previous stage: {previous_vecnorm_path}")
        prev_vec = VecNormalize.load(previous_vecnorm_path, env.venv)
        env.obs_rms = prev_vec.obs_rms
        env.ret_rms = prev_vec.ret_rms
    
    eval_env = VecNormalize(
        eval_env, 
        norm_obs=True, 
        norm_reward=False, 
        training=False,
        norm_obs_keys=["state"]
    )
    eval_env.obs_rms = env.obs_rms
    
    # 创建或加载模型
    from src.rl.custom_extractor import PointCloudLSTMExtractor
    
    policy_kwargs = dict(
        features_extractor_class=PointCloudLSTMExtractor,
        features_extractor_kwargs=dict(
            # 【v26 扩容】更大的网络适合12K维输入
            lstm_hidden_size=256,              # 从128增加到256
            lstm_num_layers=1,                 # 保持1层（多层训练不稳定）
            point_cloud_encoder_dims=(256, 128),  # 从(128,64)扩展
            state_feature_dim=128,             # 从64增加到128
            dropout=0.1,
            use_layer_norm=True,
            bidirectional=False,
        ),
        # 【v26 扩容】更大的Policy网络
        net_arch=dict(pi=[256, 128], vf=[256, 128]),  # 从[128,64]扩展
    )
    
    if previous_model_path and os.path.exists(previous_model_path):
        print(f"Loading model from previous stage: {previous_model_path}")
        model = PPO.load(previous_model_path, env=env, device=args.device)
        # 重置学习率（可选：课程学习中可能需要调整）
        model.learning_rate = args.learning_rate
    else:
        model = PPO(
            "MultiInputPolicy",
            env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=stage_log_dir,
            device=args.device,
        )
    
    # 回调
    best_model_path = os.path.join(stage_log_dir, "best_model")
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_model_path,
        log_path=os.path.join(stage_log_dir, "eval_logs"),
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True,
    )
    
    curriculum_callback = CurriculumCallback(
        stage, 
        min_timesteps=stage.min_timesteps_to_advance,  # 使用配置的最小步数
        verbose=1
    )
    
    save_vecnorm_callback = SaveVecNormalizeCallback(
        save_path=best_model_path,
        train_env=env,
        verbose=1
    )
    
    # 【新增】奖励组件日志回调
    reward_logger_callback = RewardLoggerCallback(
        log_dir=stage_log_dir,
        log_freq=1,
        verbose=1,
        csv_filename=f"reward_components_stage{stage_index+1}.csv"
    )
    
    callbacks = CallbackList([
        eval_callback, 
        curriculum_callback, 
        save_vecnorm_callback,
        reward_logger_callback  # 奖励日志
    ])
    
    # 训练
    try:
        model.learn(
            total_timesteps=stage.timesteps,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=(previous_model_path is None),
        )
    except KeyboardInterrupt:
        print("\n训练中断")
    
    # 保存最终模型
    final_model_path = os.path.join(stage_log_dir, f"stage_{stage_index+1}_final.zip")
    model.save(final_model_path)
    
    final_vecnorm_path = os.path.join(stage_log_dir, "vecnormalize.pkl")
    env.save(final_vecnorm_path)
    
    print(f"\n[Stage {stage_index+1}] 训练完成!")
    print(f"  最终成功率: {curriculum_callback.current_success_rate*100:.1f}%")
    print(f"  模型保存: {final_model_path}")
    
    # 清理
    env.close()
    eval_env.close()
    
    return {
        'model_path': final_model_path,
        'vecnorm_path': final_vecnorm_path,
        'best_model_path': os.path.join(best_model_path, "best_model.zip"),
        'best_vecnorm_path': os.path.join(best_model_path, "vecnormalize.pkl"),
        'success_rate': curriculum_callback.current_success_rate,
        'can_advance': curriculum_callback.can_advance,
    }


def main():
    parser = argparse.ArgumentParser(description="Curriculum Learning for Submarine RL")
    
    parser.add_argument("--start-stage", type=int, default=0,
                        help="Starting stage index (0-based)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to model to resume from")
    parser.add_argument("--resume-vecnorm", type=str, default=None,
                        help="Path to vecnormalize.pkl to resume from")
    
    # 训练参数
    parser.add_argument("--n-envs", type=int, default=8,
                        help="Number of parallel environments")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--n-steps", type=int, default=4096,  # 从2048增加到4096
                        help="Steps per environment per update")
    parser.add_argument("--batch-size", type=int, default=512,  # 从256增加到512
                        help="Minibatch size")
    parser.add_argument("--n-epochs", type=int, default=10,
                        help="Epochs per update")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (auto, cuda, cpu)")
    
    parser.add_argument("--output-dir", type=str, default="./training_output",
                        help="Output directory")
    
    args = parser.parse_args()
    
    # 打印课程信息
    print_curriculum_info()
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.output_dir, f"curriculum_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"\n📁 训练输出目录: {log_dir}\n")
    
    # 逐阶段训练
    previous_model = args.resume
    previous_vecnorm = args.resume_vecnorm
    
    for stage_idx in range(args.start_stage, get_total_stages()):
        result = train_stage(
            stage_idx, 
            args, 
            log_dir,
            previous_model_path=previous_model,
            previous_vecnorm_path=previous_vecnorm,
        )
        
        # 始终使用 final_model 作为下一阶段的起点（保证训练连续性）
        previous_model = result['model_path']  # stage_X_final.zip
        previous_vecnorm = result['vecnorm_path']
        
        print(f"\n{'='*70}")
        if result['can_advance']:
            print(f"✓ Stage {stage_idx+1} 达标! 成功率: {result['success_rate']*100:.1f}%")
            print(f"  进入下一阶段...")
        else:
            print(f"⚠ Stage {stage_idx+1} 未达标 (成功率: {result['success_rate']*100:.1f}%)")
            print(f"  继续进入下一阶段 (可能需要更多训练)")
        print(f"{'='*70}\n")
    
    print("\n🎉 课程学习训练完成!")
    print(f"最终模型: {previous_model}")


if __name__ == "__main__":
    main()
