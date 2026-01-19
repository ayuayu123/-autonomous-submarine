# 奖励函数更改日志

---

## 版本索引

| 版本 | 日期 | 核心主题 |
|------|------|----------|
| v30 | 2026-01-17 | 短距离避障和连续避障增强 |
| v29 | 2026-01-16 | 彻底解决固定轨迹问题 |
| v27 | 2026-01-15 | 降低proximity惩罚累积 |
| v26 | 2026-01-15 | 防止固定轨迹避障 |
| v25 | 2025-01-15 | 修复惩罚叠加过重 |
| v24 | 2025-01-15 | 速度误罚修复、预测增强（失败） |
| v23 | - | 恢复proximity_penalty |
| v22 | - | 舵角一致性判断 |
| v21h | - | 引入预测碰撞惩罚 |

---

## v30 - 2026-01-17

**问题**：小规模避障效果好，但短距离避障和连续避障仍有问题
- 短距离避障问题：
  1. 预测碰撞时间步过大（0.5s起步），无法检测<0.5s的碰撞
  2. 预测只用线性轨迹，可能误判
  3. `proximity_penalty` 触发距离太短（2.0m），反应时间不足
- 连续避障问题：
  1. 居中奖励与避障冲突：刚躲完一个障碍后立刻被"拉回中心"
  2. 没有"轨迹规划"信号：只看最近障碍，没考虑接下来2-3个障碍的布局
  3. 避障完成后没有"巩固"时间：动作不连贯

**改进1：更细粒度的预测时间步**

```python
# 原时间步（无法检测<0.5s的碰撞）
time_steps = [0.5, 1.0, 1.5, 2.0]

# 新时间步（可检测0.2s的碰撞）
prediction_time_steps = [0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 2.5]
```

**改进2：连续障碍物序列观测**

将前方通道分成3段（0-4m, 4-8m, 8-12m），为每段提供：

| 信息 | 说明 |
|------|------|
| `density` | 该段障碍物密度 (0-1) |
| `avg_y` | 障碍物平均Y位置 (-1到1) |
| `avg_z` | 障碍物平均Z位置 (-1到1) |
| `urgency` | 紧迫度 (第1段=1.0, 第3段=0.33) |

观测空间维度变化：22维 → 35维（+12维序列信息 +1维惯性状态）

**改进3：避障惯性机制**

```python
# 当Agent正在避障时，设置惯性=1.0
# 避障结束后，惯性按0.9指数衰减
# 惯性期内，居中奖励减弱（只保留30%）

if self._avoidance_inertia > 0:
    inertia_factor = 0.3 + 0.7 * (1.0 - self._avoidance_inertia)
    centering_reward *= inertia_factor
```

目的：让连续避障动作更连贯，不要刚躲完就被"拉回中心"

**改进4：紧急碰撞惩罚**

当预测碰撞时间 < 0.5s 时，额外给予强惩罚：

```python
emergency_time_threshold = 0.5   # 碰撞时间<0.5s视为紧急
p_emergency = -3.0               # 紧急惩罚（比普通prediction更重）

if time_to_collision < emergency_time_threshold:
    emergency_penalty = p_emergency * (1.0 - time_to_collision / emergency_time_threshold)
```

目的：让Agent学会"不要走到这一步"，强化早期避障意识

**参数汇总 (v30)**：

| 参数 | 值 | 说明 |
|------|------|------|
| `prediction_time_steps` | [0.2...2.5] | 更细的预测时间步 |
| `emergency_time_threshold` | 0.5s | 紧急碰撞阈值 |
| `p_emergency` | -3.0 | 紧急碰撞惩罚 |
| `avoidance_inertia_steps` | 15 | 避障惯性持续步数 |
| `avoidance_inertia_decay` | 0.9 | 惯性衰减率 |
| `num_sequence_segments` | 3 | 前方分段数 |
| `base_obs_dim` | 35 | 观测维度（原22） |

**预期效果**：
- 更早检测到短距离碰撞（0.2s vs 0.5s）
- Agent能看到前方多个障碍物的布局，做更远的规划
- 连续避障动作更流畅，不会被居中奖励干扰
- 紧急情况下的强惩罚信号，促进更早避障

---

## v29 - 2026-01-16

**问题**：模型仍然学到"固定轨迹"避障策略
- 现象：Agent学会向下/某个固定方向移动来躲避大多数障碍物
- 原因分析：
  1. 障碍物统计分布可被利用（40%中心 + 60%外围，角度随机）
  2. `offset_threshold = 0.25` 太高，Agent可以持续偏移24%而不受惩罚
  3. 居中奖励只在"前方畅通"时触发，Agent偏移后收不到回中心的信号
  4. 历史窗口太长（40步），短期偏移不会被惩罚

**修改 (障碍物生成 v5)**：

| 改进 | 说明 |
|------|------|
| **角度区域均匀分布** | 将圆周分成8个扇区（每个45°），强制每个扇区都有障碍物 |
| **中心比例提高** | 从40%提高到50%，更多障碍物在中心区域 |
| **强制外围障碍物** | 每隔3个X区段，强制在外围生成1个障碍物 |
| **中心区域更集中** | 中心区域半径从 `max_r*0.4` 缩小到 `max_r*0.35` |

**修改 (奖励函数 v29)**：

| 参数 | v28 | v29 | 说明 |
|------|-----|-----|------|
| `offset_threshold` | 0.25 | **0.15** | 更早触发惩罚 |
| `p_persistent_offset` | -0.5 | **-1.0** | 惩罚翻倍 |
| `offset_history_len` | 40 | **25** | 更快检测偏移 |
| `offset_history_check_start` | 15 | **10** | 更早开始检查 |
| `w_centering` | 0.12 | **0.15** | 增强居中激励 |
| `w_centering_reduced` | - | **0.05** | 新增：前方有障碍时仍给居中奖励 |
| 前方检测距离 | 4.0m | **3.0m** | 进一步缩短 |
| 碰撞通道宽度 | 1.5m | **1.2m** | 进一步缩窄 |

**核心逻辑改进**：

1. **居中奖励始终提供**
   ```python
   # 旧：只有前方畅通时给居中奖励
   if front_clear:
       centering_reward = w_centering * base_centering
   
   # 新：始终给予，只是强度不同
   if front_clear:
       centering_reward = w_centering * base_centering  # 满额
   else:
       centering_reward = w_centering_reduced * base_centering  # 30%
   ```

2. **障碍物角度均匀分布**
   ```python
   # 追踪8个扇区的障碍物数量
   angle_sector_counts = [0] * 8
   
   # 优先在最稀疏的扇区生成障碍物
   min_sector_count = min(angle_sector_counts)
   sparse_sectors = [i for i, c in enumerate(angle_sector_counts) if c == min_sector_count]
   target_sector = random.choice(sparse_sectors)
   ```

**预期效果**：
- 障碍物在各个方向均匀分布，无法通过固定方向策略避开
- 持续偏移惩罚更早触发、更强烈
- Agent始终有回到中心的激励信号
- 迫使Agent学习真正的动态避障能力

---

## v27 - 2026-01-15

**问题**：Stage 3 成功率持续下降 (42% → 30.5%)
- 现象：`proximity_penalty` 在失败episode中累积过重 (-76 ~ -165)
- 原因：agent靠近障碍物后无法快速逃脱，惩罚持续累积
- 障碍物碰撞率高：101→131/200 episodes

**诊断**：
1. `proximity_penalty` 权重 (-8.0) 过高
2. Stage 2 → Stage 3 难度跳跃过大 (6→12 障碍物)
3. 预警距离不足 (2.0m)，反应时间太短

**修改 (奖励函数)**：

| 参数 | v26 | v27 | 说明 |
|------|-----|-----|------|
| d_danger_obstacle | 2.0m | **2.5m** | 增加预警距离 |
| p_danger_obstacle | -8.0 | **-5.0** | 降低惩罚权重 |
| prediction_look_ahead | 2.0s | **2.5s** | 更早预测 |
| p_prediction | -1.5 | **-1.0** | 降低预测惩罚 |

**修改 (课程学习)**：

| Stage | 旧配置 | 新配置 |
|-------|--------|--------|
| Stage 2 障碍物 | 6个 | **8个** |
| Stage 2 隧道半径 | 7.0m | **6.5m** |
| Stage 2 达标率 | 70% | **75%** |
| Stage 3 障碍物 | 12个 | **10个** |
| Stage 3 隧道 | 40m | **45m** |

**预期**：
- 更平滑的难度过渡
- 降低失败episode的惩罚累积
- 提高Stage 3成功率稳定性

---

## v26 - 2026-01-15

**问题**：v25训练的二阶段模型学会了"固定轨迹避障"
- 现象：agent学会向下/某个固定方向移动来躲避大多数障碍物
- 问题：这不是通用避障策略，而是局部最优解
- Training Stage 3 成功率: 27-33%（不稳定）
- 碰撞: 障碍物=126-135/200

**修改**：

| 参数 | v25 | v26 |
|------|-----|-----|
| p_danger_boundary | -0.8 | **-1.2** |
| w_centering | 0.02 | **0.08** |
| p_persistent_offset | - | **-0.15** (新增) |
| offset_history_len | - | **50** (新增) |
| offset_threshold | - | **0.3** (新增) |

**新增奖励组件**：

1. **persistent_offset_penalty** - 持续偏移惩罚
   - 追踪最近50步的横向位置历史
   - 如果平均偏移 > 30%，开始惩罚
   - 惩罚公式: `-0.15 * (offset_magnitude - 0.3)`
   - 目的: 阻止agent学习固定方向的避障路线

2. **centering_reward** - 条件居中奖励（增强版）
   - 只有前方6m内无障碍物时才触发
   - 奖励公式: `0.08 * (1 - dist_from_center / radius)`
   - 目的: 鼓励agent在安全时回到隧道中心

**预期**：
- agent学会动态多向避障，而非固定轨迹
- 提高对不同障碍物分布的泛化能力
- 成功率稳定在较高水平

---

## v25 - 2025-01-15

**问题**：v24的prediction和proximity惩罚叠加太重
- mean_ep_length: 1100+（agent在拖延）
- 成功率: 37-40%（比v23的47-53%还低）
- 障碍物碰撞: 101-111/200（更多了）
- prediction_penalty: -47~-53（太重）
- 成功通关的episode total还是-679（负分）

**修改**：
| 参数 | v24 | v25 |
|------|-----|-----|
| w_progress | 0.8 | **1.5** |
| d_danger_obstacle | 3.0m | **2.0m** |
| p_danger_obstacle | -20 | **-8** |
| prediction_look_ahead | 2.5s | **2.0s** |
| p_prediction | -3.0 | **-1.5** |

**预期**：成功通关有正收益，agent愿意前进

---

## 参数汇总 (v26)

| 参数 | 值 | 说明 |
|------|------|------|
| `w_goal` | 500.0 | 到达终点奖励 |
| `w_progress` | 1.5 | 进度奖励权重 |
| `p_collision` | -100.0 | 碰撞惩罚 |
| `p_danger_boundary` | **-1.2** | 边界惩罚强度 (增强) |
| `w_centering` | **0.08** | 居中奖励 (增强) |
| `p_persistent_offset` | **-0.15** | 持续偏移惩罚 (新增) |
| `offset_threshold` | **0.3** | 偏移惩罚阈值 (新增) |
| `p_danger_obstacle` | -8.0 | 障碍物惩罚强度 |
| `d_danger_obstacle` | 2.0m | 障碍物预警距离 |
| `prediction_look_ahead` | 2.0s | 碰撞预测时间 |
| `p_prediction` | -1.5 | 预测碰撞惩罚权重 |

## v24 - 2025-01-15

### 📋 修改原因

基于 Stage 2 训练后期日志分析（约 6000+ episodes）发现以下问题：

1. **速度惩罚误伤问题**
   - 现象：即使成功的 episode 也有 -9 ~ -15 的速度惩罚
   - 原因：速度惩罚区间 (0.1-0.5 m/s) 过宽，正常避障速度也被惩罚
   - 影响：与进度奖励冲突，可能导致策略振荡

2. **预测碰撞信号滞后**
   - 现象：`prediction_penalty` 只在失败 episode 才出现（-18 ~ -21）
   - 原因：预测时间 2.0 秒太短，惩罚权重 -2.0 太弱
   - 影响：agent 没有足够的早期预警信号

3. **障碍物碰撞率偏高**
   - 现象：200 局中有 73-83 次障碍物碰撞（约 40%）
   - 原因：障碍物预警范围 2.0m 可能不足
   - 影响：成功率停滞在 47-53%，距离目标 70% 还有差距

### 🔧 具体变更

#### 1. 速度惩罚优化

```python
# 修改前 (v21f)
self.v_safe = 1.5
# 低速惩罚区间: 0.1-0.5 m/s

# 修改后 (v24)
self.v_safe = 1.8             # 提高超速阈值
self.v_min = 0.3              # 新增：最低速度阈值
self.p_overspeed = -0.15      # 从 -0.2 降低

# 新惩罚逻辑：
# - 0.0-0.1 m/s: -0.8 (完全静止)
# - 0.1-0.3 m/s: 渐进惩罚 (0 到 -0.4)
# - 0.3-1.8 m/s: 无惩罚 ← 关键改进！
# - >1.8 m/s: 超速惩罚
```

#### 2. 预测碰撞惩罚增强

```python
# 修改前 (v23)
look_ahead_time = 2.0
prediction_penalty = -2.0 * (2.0 - time_to_collision) / 2.0

# 修改后 (v24)
self.prediction_look_ahead = 2.5  # 预测时间延长
self.p_prediction = -3.0          # 惩罚权重增强

prediction_penalty = self.p_prediction * (
    self.prediction_look_ahead - time_to_collision
) / self.prediction_look_ahead
```

#### 3. 障碍物接近惩罚增强

```python
# 修改前
self.d_danger_obstacle = 2.0   # 预警范围
self.p_danger_obstacle = -15.0 # 惩罚强度

# 修改后 (v24)
self.d_danger_obstacle = 3.0   # 扩大预警范围
self.p_danger_obstacle = -20.0 # 增强惩罚
```

### ⚡ 预期效果

| 指标 | 修改前 | 预期修改后 |
|------|--------|----------|
| 成功 episode 的 speed_penalty | -9 ~ -15 | ~0 |
| prediction_penalty 触发时机 | 碰撞前 2 秒 | **碰撞前 2.5 秒** |
| prediction_penalty 峰值 | -2.0 | **-3.0** |
| 障碍物预警触发距离 | 2.0m | **3.0m** |
| 预估障碍物碰撞率 | ~40% | 下降 10-15% |

---

## v23

### 📋 修改原因

- `proximity_penalty` 在 v22 中被移除（与 prediction_penalty 重复），但导致障碍物距离信号缺失
- 需要恢复基于真实距离的梯度惩罚

### 🔧 具体变更

1. **恢复 proximity_penalty**
   - 重新引入基于 `get_nearby_obstacles()` 的距离惩罚
   - 二次曲线：越近惩罚越大

2. **调整 path_clearance 权重**
   - 从 2.0 降低到 1.2
   - 避免与其他惩罚叠加过重

3. **增强 prediction_penalty 权重**
   - 从 -1.5 增加到 -2.0

### ⚡ 预期效果

- 提供更平滑的距离梯度信号
- 降低过度惩罚导致的策略保守

---

## v22

### 📋 修改原因

- 之前只根据横向速度判断躲避意图，但速度响应慢
- 舵角（动作）是更直接的意图体现

### 🔧 具体变更

1. **基于舵角判断躲避意图**
   ```python
   # 舵角一致性
   yaw_alignment = self.last_yaw_cmd * best_direction[0]
   pitch_alignment = self.last_pitch_cmd * best_direction[1]
   action_alignment = yaw_alignment + pitch_alignment
   
   # 综合评分：速度(0.4) + 动作(0.6)
   combined_alignment = 0.4 * velocity_alignment + 0.6 * action_alignment
   ```

2. **移除 proximity_penalty**（后在 v23 恢复）
   - 原因：与 prediction_penalty 功能重复

3. **调整 path_clearance 奖惩逻辑**
   - 正确躲避：+0.8 * urgency
   - 明显错误：-0.4 * urgency（从 -0.6 降低）
   - 中性：0

### ⚡ 预期效果

- 更灵敏响应躲避动作
- 减少对潜在合理策略的干扰

---

## v21h - 预测碰撞版

### 📋 修改原因

- 之前的惩罚都是"事后惩罚"（碰撞了才知道错了）
- 需要"预警惩罚"让 agent 提前意识到危险轨迹

### 🔧 具体变更

1. **新增 `_predict_collision()` 方法**
   ```python
   def _predict_collision(self, look_ahead_time: float = 2.0):
       """基于当前速度线性预测未来轨迹是否碰撞"""
       time_steps = [0.5, 1.0, 1.5, 2.0]
       for t in time_steps:
           future_pos = pos + vel * t
           if check_collision(future_pos):
               return True, t
       return False, inf
   ```

2. **添加 prediction_penalty**
   - 预测到碰撞时，根据剩余时间给予惩罚
   - 时间越短（越危险）惩罚越大

3. **降低 p_danger_obstacle**
   - 从更高值降至 -15.0
   - 避免与 prediction_penalty 叠加过重

### ⚡ 预期效果

- agent 会主动调整轨迹避免预测碰撞
- 提前 2 秒就开始收到避障信号

---

## v21f

### 📋 修改原因

- 观察到 agent 可能通过"静止"来避免负奖励（reward hacking）
- 需要强制前进

### 🔧 具体变更

1. **大幅增强低速惩罚**
   ```python
   if forward_speed < 0.1:
       speed_penalty = -1.0  # 完全静止：强惩罚
   elif forward_speed < 0.5:
       low_speed_ratio = (0.5 - forward_speed) / 0.4
       speed_penalty = -0.6 * low_speed_ratio  # 渐进惩罚
   ```

2. **提高 v_safe**
   - 从 1.2 提高到 1.5
   - 允许更快的巡航速度

### ⚡ 预期效果

- 消除静止/极低速策略
- 鼓励积极前进

---

## v21d

### 📋 修改原因

- 边界碰撞偶有发生
- 需要在接近边界时给予渐进警告

### 🔧 具体变更

1. **新增边界距离惩罚**
   ```python
   dist_to_boundary = radius - dist_from_center
   boundary_warning_dist = max(radius * 0.4, submarine_radius + 1.0)
   
   if dist_to_boundary < boundary_warning_dist:
       boundary_ratio = 1.0 - (dist_to_boundary / boundary_warning_dist)
       boundary_penalty = p_danger_boundary * (boundary_ratio ** 2)
   ```

2. **新增边界速度惩罚**
   - 如果正在向边界移动，额外惩罚

### ⚡ 预期效果

- 减少边界碰撞
- agent 会主动远离边界

---

## v15b

### 📋 修改原因

- 障碍物碰撞是主要失败原因
- 需要专门的障碍物接近惩罚和成功避障奖励

### 🔧 具体变更

1. **障碍物危险惩罚**
   ```python
   d_danger_obstacle = 2.0   # 预警距离
   p_danger_obstacle = -15.0 # 惩罚强度
   num_danger_obstacles = 3  # 检测最近 3 个
   ```

2. **主动避障奖励**
   ```python
   w_avoidance = 5.0         # 成功避障奖励
   d_avoidance_trigger = 2.5 # 进入范围开始追踪
   d_avoidance_safe = 3.0    # 远离视为成功
   ```

### ⚡ 预期效果

- 提供清晰的障碍物距离信号
- 奖励成功的避障行为

---

## v15

### 📋 修改原因

- 边界惩罚需要参数化，便于调优

### 🔧 具体变更

```python
d_danger_boundary = 2.0   # 预警距离
p_danger_boundary = -0.8  # 惩罚强度（原硬编码 -1.5）
```

### ⚡ 预期效果

- 边界惩罚可调
- 避免过激的边界躲避

---

## v14/v14b

### 📋 修改原因

- 点云信息虽然丰富，但 agent 难以直接从中提取危险方向
- 需要显式的危险方向信号

### 🔧 具体变更

1. **扩展观测空间**
   - 新增 `num_obstacle_info * 3 = 9` 维
   - 每组：`[y_direction, z_direction, urgency]`

2. **新增 `_get_danger_direction_from_pointcloud()`**
   - 从点云计算最近点的加权平均方向
   - 可部署（不依赖障碍物精确位置）

### ⚡ 预期效果

- 提供显式的危险方向信号
- 帮助 agent 理解"往哪边躲"

---

## 早期版本 (v1-v13)

早期版本主要建立了基础奖励结构：

- **w_goal**: 到达终点大奖励
- **w_progress**: 进度奖励（每前进 1% 得分）
- **p_collision**: 碰撞惩罚
- **c_step**: 每步小惩罚（防止拖延）
- **w_centering**: 居中奖励（防止走边缘）

这些基础参数在后续版本中持续调优。

---

## 参数汇总表 (v24)

| 参数 | 值 | 说明 |
|------|------|------|
| `w_goal` | 500.0 | 到达终点奖励 |
| `w_progress` | 0.80 | 进度奖励权重 |
| `p_collision` | -100.0 | 碰撞惩罚 |
| `c_step` | -0.01 | 每步惩罚 |
| `v_safe` | 1.8 | 安全速度上限 |
| `v_min` | 0.3 | 最低速度阈值 |
| `p_overspeed` | -0.15 | 超速惩罚 |
| `p_drift` | -0.1 | 横向漂移惩罚 |
| `d_danger_boundary` | 2.0 | 边界预警距离 |
| `p_danger_boundary` | -0.8 | 边界惩罚强度 |
| `d_danger_obstacle` | 3.0 | 障碍物预警距离 |
| `p_danger_obstacle` | -20.0 | 障碍物惩罚强度 |
| `prediction_look_ahead` | 2.5 | 碰撞预测时间 (秒) |
| `p_prediction` | -3.0 | 预测碰撞惩罚权重 |
| `w_avoidance` | 5.0 | 成功避障奖励 |
| `w_centering` | 0.02 | 居中奖励 |

---

## 调优建议

基于历史经验，以下是调优的一般原则：

1. **进度奖励 vs 安全惩罚**
   - 如果 agent 太保守（走得慢），增加 `w_progress`
   - 如果 agent 太激进（碰撞多），增加惩罚权重

2. **预警距离 vs 惩罚强度**
   - 预警距离决定"何时开始警告"
   - 惩罚强度决定"警告多响亮"
   - 两者需要平衡，距离太大+强度太高会导致过度保守

3. **速度限制**
   - 不惩罚区间要足够宽，避免干扰正常导航
   - 只惩罚极端情况（静止、超速）

4. **预测碰撞**
   - 预测时间不能太长（2-3 秒合适），否则误报过多
   - 惩罚权重要足够强，但不能超过实际碰撞
