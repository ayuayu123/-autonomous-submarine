#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reward_logger_callback.py: 奖励组件日志记录回调

在训练时记录每个 episode 的奖励组件分解，写入 CSV 文件。
训练完成后可用于分析奖励函数的平衡性。

使用方法:
    在 train.py 中添加:
    
    from src.rl.reward_logger_callback import RewardLoggerCallback
    
    reward_callback = RewardLoggerCallback(
        log_dir="./logs",
        log_freq=1,  # 每个 episode 都记录
        verbose=1
    )
    
    model.learn(
        total_timesteps=1000000,
        callback=[eval_callback, reward_callback]
    )
"""

import os
import csv
import numpy as np
from collections import defaultdict
from typing import Dict, List, Any, Optional
from stable_baselines3.common.callbacks import BaseCallback


class RewardLoggerCallback(BaseCallback):
    """
    训练时奖励组件日志记录回调
    
    功能:
    - 收集每步的奖励组件分解
    - 在 episode 结束时计算统计并写入 CSV
    - 支持多环境 (VecEnv)
    """
    
    def __init__(
        self,
        log_dir: str = "./logs",
        log_freq: int = 1,
        verbose: int = 0,
        csv_filename: str = "reward_components.csv"
    ):
        """
        Args:
            log_dir: 日志目录
            log_freq: 记录频率 (每 N 个 episode 记录一次)
            verbose: 日志级别 (0=静默, 1=摘要, 2=详细)
            csv_filename: CSV 文件名
        """
        super().__init__(verbose)
        self.log_dir = log_dir
        self.log_freq = log_freq
        self.csv_filename = csv_filename
        
        # 每个环境的数据收集器
        self.episode_rewards: Dict[int, List[Dict[str, float]]] = defaultdict(list)
        self.episode_count = 0
        self.csv_path = None
        self.csv_initialized = False
        
    def _init_callback(self) -> bool:
        """初始化回调"""
        # 创建日志目录
        os.makedirs(self.log_dir, exist_ok=True)
        self.csv_path = os.path.join(self.log_dir, self.csv_filename)
        
        if self.verbose >= 1:
            print(f"[RewardLoggerCallback] 日志将保存到: {self.csv_path}")
        
        return True
    
    def _on_step(self) -> bool:
        """每步调用"""
        # 从 infos 中获取奖励组件（兼容 SubprocVecEnv）
        infos = self.locals.get('infos', [])
        dones = self.locals.get('dones', self.locals.get('done', [False]))
        
        if not isinstance(dones, (list, np.ndarray)):
            dones = [dones]
        
        # 收集每步的奖励组件
        for env_idx, info in enumerate(infos):
            if isinstance(info, dict) and 'reward_components' in info:
                components = info['reward_components']
                if components:
                    self.episode_rewards[env_idx].append(components.copy())
        
        # 检查 episode 是否结束
        for env_idx, done in enumerate(dones):
            if done and len(self.episode_rewards[env_idx]) > 0:
                self._log_episode(env_idx)
                self.episode_rewards[env_idx] = []
                self.episode_count += 1
        
        return True
    
    def _get_base_env(self, env) -> Any:
        """递归获取底层 SubmarineEnv"""
        # 尝试多种方式获取底层环境
        if hasattr(env, 'get_reward_components'):
            return env
        
        # VecNormalize 包装
        if hasattr(env, 'venv'):
            return self._get_base_env(env.venv)
        
        # DummyVecEnv / SubprocVecEnv
        if hasattr(env, 'envs'):
            if len(env.envs) > 0:
                return self._get_base_env(env.envs[0])
        
        # Monitor 包装
        if hasattr(env, 'env'):
            return self._get_base_env(env.env)
        
        # Gymnasium 包装
        if hasattr(env, 'unwrapped'):
            if env.unwrapped is not env:
                return self._get_base_env(env.unwrapped)
        
        return None
    
    def _log_episode(self, env_idx: int):
        """记录一个 episode 的奖励组件"""
        rewards = self.episode_rewards[env_idx]
        if not rewards:
            return
        
        # 是否需要记录 (基于 log_freq)
        if self.episode_count % self.log_freq != 0:
            return
        
        # 计算统计
        stats = self._compute_stats(rewards)
        stats['episode'] = self.episode_count
        stats['env_idx'] = env_idx
        stats['timestep'] = self.num_timesteps
        stats['num_steps'] = len(rewards)
        
        # 写入 CSV
        self._write_to_csv(stats)
        
        # 打印摘要
        if self.verbose >= 1 and self.episode_count % 100 == 0:
            self._print_summary(stats)
    
    def _compute_stats(self, rewards: List[Dict[str, float]]) -> Dict[str, float]:
        """计算奖励组件统计"""
        stats = {}
        
        # 获取所有组件名称
        all_keys = set()
        for r in rewards:
            all_keys.update(r.keys())
        
        for key in all_keys:
            values = [r.get(key, 0.0) for r in rewards]
            # 累计
            stats[f'{key}_sum'] = sum(values)
            # 均值
            stats[f'{key}_mean'] = np.mean(values)
            # 最大最小
            stats[f'{key}_max'] = max(values)
            stats[f'{key}_min'] = min(values)
        
        return stats
    
    def _write_to_csv(self, stats: Dict[str, Any]):
        """写入 CSV 文件"""
        # 首次写入时创建文件头
        if not self.csv_initialized:
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=sorted(stats.keys()))
                writer.writeheader()
                writer.writerow(stats)
            self.csv_initialized = True
        else:
            with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=sorted(stats.keys()))
                writer.writerow(stats)
    
    def _print_summary(self, stats: Dict[str, Any]):
        """打印摘要信息"""
        print(f"\n[Episode {stats['episode']}] 奖励组件摘要 (步数: {stats['num_steps']})")
        print("-" * 60)
        
        # 主要组件 (v26更新 - 添加持续偏移惩罚和居中奖励)
        main_components = ['progress', 'speed_penalty', 'path_clearance', 
                          'proximity_penalty', 'boundary_penalty', 
                          'persistent_offset_penalty', 'centering_reward',
                          'prediction_penalty', 'total']
        
        for comp in main_components:
            sum_key = f'{comp}_sum'
            mean_key = f'{comp}_mean'
            if sum_key in stats:
                print(f"  {comp:25s}: 累计={stats[sum_key]:+8.2f}, 均值={stats[mean_key]:+.4f}")
    
    def _on_training_end(self):
        """训练结束时调用"""
        if self.verbose >= 1:
            print(f"\n[RewardLoggerCallback] 训练完成!")
            print(f"  总 episode 数: {self.episode_count}")
            print(f"  日志文件: {self.csv_path}")


class RewardComponentMonitor:
    """
    环境包装器，用于收集奖励组件
    
    使用方法:
        env = SubmarineEnv(...)
        env = RewardComponentMonitor(env)
    """
    
    def __init__(self, env):
        self.env = env
        self._episode_components = []
        
    def __getattr__(self, name):
        return getattr(self.env, name)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 获取奖励组件
        if hasattr(self.env, 'get_reward_components'):
            components = self.env.get_reward_components()
            self._episode_components.append(components)
            
            # 添加到 info 中
            info['reward_components'] = components
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        # 重置时清空组件记录
        self._episode_components = []
        return self.env.reset(**kwargs)
    
    def get_episode_stats(self) -> Dict[str, float]:
        """获取当前 episode 的奖励组件统计"""
        if not self._episode_components:
            return {}
        
        stats = {}
        all_keys = set()
        for c in self._episode_components:
            all_keys.update(c.keys())
        
        for key in all_keys:
            values = [c.get(key, 0.0) for c in self._episode_components]
            stats[f'{key}_sum'] = sum(values)
            stats[f'{key}_mean'] = np.mean(values)
        
        return stats
