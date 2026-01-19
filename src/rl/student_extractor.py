#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
student_extractor.py: Student 模型特征提取器 (扩大版)

用于策略蒸馏，比 Teacher 模型具有更大的网络容量。

架构设计对比:
┌──────────────────────┬──────────────────┬──────────────────┐
│        组件          │   Teacher (小)    │   Student (大)   │
├──────────────────────┼──────────────────┼──────────────────┤
│ 点云编码器           │   768 → 256      │  768 → 512 → 256 │
│ LSTM 隐藏层         │   128            │  256             │
│ LSTM 层数           │   1              │  2               │
│ 双向 LSTM           │   否             │  是              │
│ 状态特征维度         │   64             │  128             │
│ 特征输出维度         │   192            │  640             │
└──────────────────────┴──────────────────┴──────────────────┘
"""
import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Dict, Tuple, Optional


class StudentPointCloudExtractor(BaseFeaturesExtractor):
    """
    Student 模型特征提取器 (扩大版)
    
    相比 Teacher 模型 (PointCloudLSTMExtractor):
    - 更深的点云编码器
    - 更大的 LSTM 隐藏层和双向处理
    - 多头注意力机制 (可选)
    - 更强的正则化
    
    输入观测空间 (Dict):
        - "state": (state_dim,) 状态向量
        - "point_cloud_seq": (seq_len, point_cloud_dim) 点云序列
        
    输出:
        - (features_dim,) 特征向量
    """
    
    def __init__(
        self, 
        observation_space: gym.spaces.Dict,
        # LSTM 配置 (扩大)
        lstm_hidden_size: int = 256,        # Teacher: 128
        lstm_num_layers: int = 2,           # Teacher: 1
        bidirectional: bool = True,         # Teacher: False
        # 点云编码器配置 (扩大)
        point_cloud_encoder_dims: tuple = (512, 256),  # Teacher: (256,)
        # 状态编码器配置 (扩大)
        state_feature_dim: int = 128,       # Teacher: 64
        # 注意力配置 (新增)
        use_attention: bool = True,
        attention_heads: int = 4,
        # 正则化配置
        dropout: float = 0.15,              # Teacher: 0.1
        use_layer_norm: bool = True,
    ):
        """
        Args:
            observation_space: Dict 类型的观测空间
            lstm_hidden_size: LSTM 隐藏层大小 (比 Teacher 大)
            lstm_num_layers: LSTM 层数 (比 Teacher 多)
            bidirectional: 是否使用双向 LSTM
            point_cloud_encoder_dims: 点云编码器隐藏层维度 (比 Teacher 深)
            state_feature_dim: 状态特征输出维度 (比 Teacher 大)
            use_attention: 是否使用多头注意力
            attention_heads: 注意力头数
            dropout: Dropout 概率
            use_layer_norm: 是否使用 Layer Normalization
        """
        # 计算输出特征维度
        lstm_output_size = lstm_hidden_size * (2 if bidirectional else 1)
        features_dim = lstm_output_size + state_feature_dim
        super().__init__(observation_space, features_dim=features_dim)
        
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.bidirectional = bidirectional
        self.use_layer_norm = use_layer_norm
        self.use_attention = use_attention
        self.dropout_rate = dropout
        
        # 获取观测空间维度
        point_cloud_shape = observation_space["point_cloud_seq"].shape
        self.seq_len = point_cloud_shape[0]
        self.point_cloud_dim = point_cloud_shape[1]
        self.state_dim = observation_space["state"].shape[0]
        
        # ============ 点云编码器 (更深) ============
        pc_encoder_layers = []
        input_dim = self.point_cloud_dim
        num_encoder_layers = len(point_cloud_encoder_dims)
        
        for i, hidden_dim in enumerate(point_cloud_encoder_dims):
            # 线性层
            pc_encoder_layers.append(nn.Linear(input_dim, hidden_dim))
            
            # Layer Normalization
            if use_layer_norm:
                pc_encoder_layers.append(nn.LayerNorm(hidden_dim))
            
            # 激活函数 (使用 GELU 替代 ReLU，更平滑)
            pc_encoder_layers.append(nn.GELU())
            
            # Dropout (在激活之后)
            if dropout > 0 and i < num_encoder_layers - 1:
                pc_encoder_layers.append(nn.Dropout(dropout))
                
            input_dim = hidden_dim
            
        self.point_cloud_encoder = nn.Sequential(*pc_encoder_layers)
        self.encoder_output_dim = input_dim
        
        # ============ LSTM 层 (更大) ============
        self.lstm = nn.LSTM(
            input_size=self.encoder_output_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        
        # ============ 多头注意力 (新增) ============
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=lstm_output_size,
                num_heads=attention_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.attention_norm = nn.LayerNorm(lstm_output_size)
        else:
            self.attention = None
            self.attention_norm = None
        
        # LSTM 输出后的 Layer Normalization
        if use_layer_norm:
            self.lstm_layer_norm = nn.LayerNorm(lstm_output_size)
        else:
            self.lstm_layer_norm = nn.Identity()
        
        # ============ 状态编码器 (更深) ============
        # 动态计算中间维度
        state_hidden_dim = max(128, state_feature_dim * 2, self.state_dim * 4)
        
        self.state_encoder = nn.Sequential(
            nn.Linear(self.state_dim, state_hidden_dim),
            nn.LayerNorm(state_hidden_dim) if use_layer_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(state_hidden_dim, state_hidden_dim // 2),
            nn.LayerNorm(state_hidden_dim // 2) if use_layer_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(state_hidden_dim // 2, state_feature_dim),
            nn.LayerNorm(state_feature_dim) if use_layer_norm else nn.Identity(),
            nn.GELU(),
        )
        
        # ============ 权重初始化 ============
        self._initialize_weights()
        
        # 打印配置信息
        self._print_config(features_dim, lstm_output_size)
    
    def _print_config(self, features_dim: int, lstm_output_size: int):
        """打印配置信息"""
        print(f"\n{'='*60}")
        print(f"[StudentPointCloudExtractor] 初始化完成 (扩大版)")
        print(f"{'='*60}")
        print(f"  点云编码器:")
        print(f"    - 输入维度: {self.point_cloud_dim}")
        print(f"    - 编码器输出: {self.encoder_output_dim}")
        print(f"  LSTM 配置:")
        print(f"    - 序列长度: {self.seq_len}")
        print(f"    - 隐藏层大小: {self.lstm_hidden_size}")
        print(f"    - 层数: {self.lstm_num_layers}")
        print(f"    - 双向: {self.bidirectional}")
        print(f"    - 输出维度: {lstm_output_size}")
        print(f"  注意力机制:")
        print(f"    - 启用: {self.use_attention}")
        print(f"  状态编码器:")
        print(f"    - 输入维度: {self.state_dim}")
        print(f"    - 输出维度: {self._features_dim - lstm_output_size}")
        print(f"  正则化:")
        print(f"    - Dropout: {self.dropout_rate}")
        print(f"    - LayerNorm: {self.use_layer_norm}")
        print(f"  总输出特征维度: {features_dim}")
        print(f"{'='*60}\n")
        
    def _initialize_weights(self):
        """
        初始化权重
        
        - LSTM: 正交初始化
        - 线性层: Xavier 均匀初始化
        - 遗忘门偏置: 设为 1
        """
        # LSTM 权重初始化
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                n = param.size(0)
                param.data[n//4:n//2].fill_(1.0)
        
        # 线性层权重初始化
        for module in [self.point_cloud_encoder, self.state_encoder]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
        
        # 注意力层初始化
        if self.attention is not None:
            nn.init.xavier_uniform_(self.attention.in_proj_weight)
            nn.init.xavier_uniform_(self.attention.out_proj.weight)
        
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        前向传播

        Args:
            observations: Dict 类型的观测
                - "state": (batch, state_dim) 或 (batch, n_envs, state_dim)
                - "point_cloud_seq": (batch, seq_len, point_cloud_dim) 或 (batch, n_envs, seq_len, point_cloud_dim)

        Returns:
            features: (batch * n_envs, features_dim) 特征向量
        """
        point_cloud_seq = observations["point_cloud_seq"]
        state = observations["state"]

        # 处理多环境并行训练的情况
        # 当使用 SubprocVecEnv 时，数据形状可能是 (batch, n_envs, seq_len, dim)
        # 需要展平为 (batch * n_envs, seq_len, dim)
        if point_cloud_seq.dim() == 4:
            # 形状: (batch, n_envs, seq_len, dim) -> (batch * n_envs, seq_len, dim)
            n_envs = point_cloud_seq.shape[1]
            batch_size = point_cloud_seq.shape[0]
            point_cloud_seq = point_cloud_seq.permute(0, 1, 3, 2).contiguous()
            point_cloud_seq = point_cloud_seq.view(batch_size * n_envs, self.point_cloud_dim, self.seq_len)
            point_cloud_seq = point_cloud_seq.permute(0, 2, 1).contiguous()

            # 同样处理 state
            if state.dim() == 3:
                state = state.view(batch_size * n_envs, self.state_dim)
            elif state.dim() == 2 and state.shape[0] != batch_size * n_envs:
                # 如果 state 已经被正确展平，则不需要处理
                pass

        batch_size = point_cloud_seq.shape[0]

        # ============ 处理点云序列 ============
        # point_cloud_seq: (batch, seq_len, point_cloud_dim)

        # 通过编码器降维 (逐帧处理)
        pc_flat = point_cloud_seq.view(batch_size * self.seq_len, self.point_cloud_dim)
        pc_encoded = self.point_cloud_encoder(pc_flat)  # (batch * seq_len, encoder_output_dim)

        # 恢复序列形状
        pc_encoded = pc_encoded.view(batch_size, self.seq_len, -1)  # (batch, seq_len, encoder_output_dim)
        
        # 通过 LSTM
        lstm_out, (h_n, c_n) = self.lstm(pc_encoded)
        # lstm_out: (batch, seq_len, hidden_size * num_directions)
        
        # 应用注意力机制 (可选)
        if self.attention is not None:
            # 自注意力
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
            # 残差连接 + LayerNorm
            lstm_out = self.attention_norm(lstm_out + attn_out)
        
        # 提取最终特征
        if self.bidirectional:
            # 双向 LSTM: 拼接正向和反向的最后隐藏状态
            h_forward = h_n[-2]  # (batch, hidden_size)
            h_backward = h_n[-1]  # (batch, hidden_size)
            point_cloud_features = torch.cat([h_forward, h_backward], dim=1)
        else:
            point_cloud_features = h_n[-1]  # (batch, hidden_size)
        
        # 应用 Layer Normalization
        point_cloud_features = self.lstm_layer_norm(point_cloud_features)
        
        # ============ 处理状态 ============
        state_features = self.state_encoder(state)  # (batch, state_feature_dim)
        
        # ============ 拼接输出 ============
        combined = torch.cat([point_cloud_features, state_features], dim=1)
        
        return combined


# 预设配置
STUDENT_CONFIGS = {
    # 中等规模 Student (推荐首次尝试)
    "medium": {
        "lstm_hidden_size": 192,
        "lstm_num_layers": 2,
        "bidirectional": True,
        "point_cloud_encoder_dims": (384, 256),
        "state_feature_dim": 96,
        "use_attention": False,
        "dropout": 0.12,
    },
    # 大规模 Student
    "large": {
        "lstm_hidden_size": 256,
        "lstm_num_layers": 2,
        "bidirectional": True,
        "point_cloud_encoder_dims": (512, 256),
        "state_feature_dim": 128,
        "use_attention": True,
        "attention_heads": 4,
        "dropout": 0.15,
    },
    # 超大规模 Student
    "xlarge": {
        "lstm_hidden_size": 384,
        "lstm_num_layers": 3,
        "bidirectional": True,
        "point_cloud_encoder_dims": (768, 512, 256),
        "state_feature_dim": 192,
        "use_attention": True,
        "attention_heads": 8,
        "dropout": 0.2,
    },
}


def get_student_config(size: str = "medium") -> dict:
    """
    获取 Student 模型配置
    
    Args:
        size: 模型规模 ("medium", "large", "xlarge")
        
    Returns:
        配置字典
    """
    if size not in STUDENT_CONFIGS:
        raise ValueError(f"Unknown size: {size}. Choose from {list(STUDENT_CONFIGS.keys())}")
    return STUDENT_CONFIGS[size].copy()
