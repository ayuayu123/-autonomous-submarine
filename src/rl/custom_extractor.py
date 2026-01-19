#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
custom_extractor.py: 自定义特征提取器，用于混合架构 (优化版)

架构设计：
- 点云序列 → MLP编码器 → LSTM → LayerNorm → 时序特征
- 状态数据 → MLP → 状态特征
- 拼接后输出给 PPO 的 Actor/Critic 网络

优化点：
1. 添加 Dropout 正则化 (防止过拟合)
2. 添加 Layer Normalization (稳定训练)
3. 正交初始化 LSTM 权重 (改善梯度流)
4. 遗忘门偏置初始化为 1 (帮助学习长期依赖)
5. 支持双向 LSTM (可选，捕捉更丰富的时序特征)
6. 动态计算中间维度
"""
import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Dict, Tuple, Optional


class PointCloudLSTMExtractor(BaseFeaturesExtractor):
    """
    混合特征提取器 (优化版)
    
    - 对点云序列使用 LSTM 提取时序特征
    - 对状态数据使用 MLP 提取状态特征
    - 拼接两者作为最终输出
    
    输入观测空间 (Dict):
        - "state": (state_dim,) 状态向量
        - "point_cloud_seq": (seq_len, point_cloud_dim) 点云序列
        
    输出:
        - (lstm_output_size + state_feature_dim,) 特征向量
    """
    
    def __init__(
        self, 
        observation_space: gym.spaces.Dict,
        lstm_hidden_size: int = 128,
        lstm_num_layers: int = 1,
        point_cloud_encoder_dims: tuple = (256,),
        state_feature_dim: int = 64,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
        bidirectional: bool = False,
    ):
        """
        Args:
            observation_space: Dict 类型的观测空间
            lstm_hidden_size: LSTM 隐藏层大小
            lstm_num_layers: LSTM 层数
            point_cloud_encoder_dims: 点云编码器隐藏层维度 (tuple)
            state_feature_dim: 状态特征输出维度
            dropout: Dropout 概率 (0 表示不使用, 推荐 0.1-0.3)
            use_layer_norm: 是否使用 Layer Normalization (推荐 True)
            bidirectional: 是否使用双向 LSTM (输出维度翻倍)
        """
        # 计算输出特征维度
        lstm_output_size = lstm_hidden_size * (2 if bidirectional else 1)
        features_dim = lstm_output_size + state_feature_dim
        super().__init__(observation_space, features_dim=features_dim)
        
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.bidirectional = bidirectional
        self.use_layer_norm = use_layer_norm
        self.dropout_rate = dropout
        
        # 获取观测空间维度
        point_cloud_shape = observation_space["point_cloud_seq"].shape
        self.seq_len = point_cloud_shape[0]
        self.point_cloud_dim = point_cloud_shape[1]
        self.state_dim = observation_space["state"].shape[0]
        
        # ============ 点云编码器 (带正则化) ============
        pc_encoder_layers = []
        input_dim = self.point_cloud_dim
        num_encoder_layers = len(point_cloud_encoder_dims)
        
        for i, hidden_dim in enumerate(point_cloud_encoder_dims):
            # 线性层
            pc_encoder_layers.append(nn.Linear(input_dim, hidden_dim))
            
            # Layer Normalization (在激活之前，稳定分布)
            if use_layer_norm:
                pc_encoder_layers.append(nn.LayerNorm(hidden_dim))
            
            # 激活函数
            pc_encoder_layers.append(nn.ReLU())
            
            # Dropout (在激活之后，最后一层除外以保留更多信息给 LSTM)
            if dropout > 0 and i < num_encoder_layers - 1:
                pc_encoder_layers.append(nn.Dropout(dropout))
                
            input_dim = hidden_dim
            
        self.point_cloud_encoder = nn.Sequential(*pc_encoder_layers)
        self.encoder_output_dim = input_dim
        
        # ============ LSTM 层 ============
        self.lstm = nn.LSTM(
            input_size=self.encoder_output_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0,  # 多层时在层间使用 dropout
            bidirectional=bidirectional,
        )
        
        # LSTM 输出后的 Layer Normalization (稳定输出尺度)
        if use_layer_norm:
            self.lstm_layer_norm = nn.LayerNorm(lstm_output_size)
        else:
            self.lstm_layer_norm = nn.Identity()
        
        # ============ 状态编码器 (带正则化) ============
        # 动态计算中间维度，确保足够的表达能力
        state_hidden_dim = max(64, state_feature_dim, self.state_dim * 2)
        
        state_encoder_layers = [
            nn.Linear(self.state_dim, state_hidden_dim),
        ]
        if use_layer_norm:
            state_encoder_layers.append(nn.LayerNorm(state_hidden_dim))
        state_encoder_layers.append(nn.ReLU())
        
        if dropout > 0:
            state_encoder_layers.append(nn.Dropout(dropout))
            
        state_encoder_layers.append(nn.Linear(state_hidden_dim, state_feature_dim))
        if use_layer_norm:
            state_encoder_layers.append(nn.LayerNorm(state_feature_dim))
        state_encoder_layers.append(nn.ReLU())
        
        self.state_encoder = nn.Sequential(*state_encoder_layers)
        
        # ============ 权重初始化 ============
        self._initialize_weights()
        
        # 打印配置信息
        self._print_config(features_dim, lstm_output_size)
    
    def _print_config(self, features_dim: int, lstm_output_size: int):
        """打印配置信息"""
        print(f"\n{'='*60}")
        print(f"[PointCloudLSTMExtractor] 初始化完成 (优化版)")
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
        
        - LSTM: 正交初始化 (改善梯度流和长期依赖学习)
        - 线性层: Xavier 均匀初始化
        - 遗忘门偏置: 设为 1 (帮助 LSTM 记住长期信息)
        """
        # LSTM 权重初始化
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                # 输入到隐藏的权重: Xavier 初始化
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                # 隐藏到隐藏的权重: 正交初始化 (防止梯度消失/爆炸)
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                # 偏置初始化
                param.data.fill_(0)
                # 设置遗忘门偏置为 1 (LSTM 的第二个门)
                # bias 布局: [input_gate, forget_gate, cell_gate, output_gate]
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
        
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        前向传播
        
        Args:
            observations: Dict 类型的观测
                - "state": (batch, state_dim)
                - "point_cloud_seq": (batch, seq_len, point_cloud_dim)
                
        Returns:
            features: (batch, features_dim) 特征向量
        """
        point_cloud_seq = observations["point_cloud_seq"]
        state = observations["state"]
        
        batch_size = point_cloud_seq.shape[0]
        
        # ============ 处理点云序列 ============
        # point_cloud_seq: (batch, seq_len, point_cloud_dim)
        
        # 通过编码器降维 (逐帧处理)
        # 先展开为 (batch * seq_len, point_cloud_dim)
        pc_flat = point_cloud_seq.view(batch_size * self.seq_len, self.point_cloud_dim)
        pc_encoded = self.point_cloud_encoder(pc_flat)  # (batch * seq_len, encoder_output_dim)
        
        # 恢复序列形状
        pc_encoded = pc_encoded.view(batch_size, self.seq_len, -1)  # (batch, seq_len, encoder_output_dim)
        
        # 通过 LSTM
        # lstm_out: (batch, seq_len, hidden_size * num_directions)
        # h_n: (num_layers * num_directions, batch, hidden_size)
        lstm_out, (h_n, c_n) = self.lstm(pc_encoded)
        
        if self.bidirectional:
            # 双向 LSTM: 拼接正向和反向的最后隐藏状态
            # h_n 形状: (num_layers * 2, batch, hidden_size)
            # 最后一层的正向: h_n[-2], 反向: h_n[-1]
            h_forward = h_n[-2]  # (batch, hidden_size)
            h_backward = h_n[-1]  # (batch, hidden_size)
            point_cloud_features = torch.cat([h_forward, h_backward], dim=1)
        else:
            # 单向 LSTM: 取最后一层的隐藏状态
            point_cloud_features = h_n[-1]  # (batch, hidden_size)
        
        # 应用 Layer Normalization
        point_cloud_features = self.lstm_layer_norm(point_cloud_features)
        
        # ============ 处理状态 ============
        state_features = self.state_encoder(state)  # (batch, state_feature_dim)
        
        # ============ 拼接输出 ============
        combined = torch.cat([point_cloud_features, state_features], dim=1)
        
        return combined


# 为了向后兼容，保留旧版本的别名
PointCloudLSTMExtractorV1 = PointCloudLSTMExtractor
