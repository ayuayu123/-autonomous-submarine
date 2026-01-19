# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**SubmarineHunting** is a high-fidelity AUV (Autonomous Underwater Vehicle) simulation platform that combines 6-DOF physics modeling with deep reinforcement learning. The project trains PPO agents with hybrid LSTM architectures to navigate through underwater tunnels while avoiding obstacles.

**Technology Stack**: Python 3.10+, NumPy, Pygame/PyOpenGL, Stable-Baselines3, PyTorch, Gymnasium

---

## Essential Commands

### Running the Project

```bash
# Interactive manual control (pygame-based)
python pygame_sim.py

# Interactive RL environment with human control
python run_human.py

# Start training (uses default TrainingConfig settings)
python train.py

# Resume training with custom hyperparameters
python train.py \
    --resume_model ./training_output/xxx/checkpoints/submarine_ppo_500000_steps.zip \
    --resume_normalize ./training_output/xxx/checkpoints/submarine_ppo_500000_steps_normalize.pkl \
    --resume_stage 5 \
    --learning_rate 1e-5 \
    --n_epochs 20 \
    --total_timesteps 5000000

# Evaluate trained model
python eval_sim.py \
    --model ./training_output/xxx/best_model/best_model.zip \
    --vecnormalize ./training_output/xxx/vecnormalize.pkl

# Manual evaluation mode (human control)
python eval_sim.py --manual
```

### Monitoring Training

```bash
tensorboard --logdir ./training_output
```

### Environment Setup

```bash
conda create -n sub_hunt python=3.10
conda activate sub_hunt
pip install -r requirements.txt
```

---

## Architecture Overview

### Core Design Pattern: Actor-Stage

The simulation uses an **Actor-Stage pattern** to decouple physics from scene management:

- **Actor** (`src/core/actor.py`): Base class for all entities. Each has an `update(dt)` method called every frame.
  - `SubmarineActor`: Handles submarine physics and state synchronization
  - `ObstacleActor`: Represents obstacles
  - `TunnelActor`: Manages tunnel visualization

- **LogicStage** (`src/core/stage.py`): Container managing all actors. Calls `update()` on all entities each simulation step.

### Layered Architecture

```
lib/                           # Mathematical foundations (DO NOT modify without physics expertise)
├── gnc.py                    # Guidance, Navigation, Control math
├── models.py                 # 6-DOF dynamics equations (Fossen 2021)
├── control.py                # PID and SMC controllers
├── actuator.py               # Actuator dynamics
└── guidance.py               # Line-of-sight guidance

src/                          # Simulation engine
├── core/                     # Actor-Stage framework
│   ├── actor.py             # Base Actor class
│   └── stage.py             # LogicStage container
├── physics/
│   └── submarine_actor.py   # Submarine entity wrapper
├── gameplay/
│   ├── obstacle_actor.py    # Obstacle entity
│   └── tunnel_actor.py      # Tunnel visualization
├── rl/                       # Reinforcement Learning layer
│   ├── submarine_env.py     # Gymnasium environment (MAIN ENTRY POINT for RL)
│   ├── custom_extractor.py  # Hybrid LSTM feature extractor
│   ├── tunnel.py            # TunnelConfig, ProvingGround (obstacle generation)
│   └── point_cloud_sampler.py # Surface point cloud perception
└── view/
    └── renderer.py          # OpenGL 3D renderer

torpedo.py                    # Physics engine (REMUS 100 AUV model)

train.py                      # Training script with curriculum learning
eval_sim.py                   # Evaluation and visualization
pygame_sim.py                 # Standalone 3D simulator
run_human.py                  # Interactive RL environment
```

---

## Hybrid LSTM Neural Architecture

The RL system uses a sophisticated multi-modal architecture defined in `src/rl/custom_extractor.py`:

1. **Point Cloud Stream**: Sequences of (history_len=16, num_points=256, 3) → PointNet-style encoder → LSTM → temporal features
2. **State Stream**: 13D state vector → MLP → static features
3. **Fusion**: Concatenated features → PPO policy networks (Actor-Critic)

**Key Configuration** (in `TrainingConfig`):
- `point_cloud_history_len`: LSTM sequence length (default: 16)
- `lstm_hidden_size`: LSTM hidden dimension (default: 128)
- `pc_encoder_dims`: Point cloud encoder dimensions (default: "256")
- `state_feature_dim`: State feature dimension after MLP (default: 64)

---

## Curriculum Learning System

The training uses a 5-stage curriculum defined in `train.py`:

```python
CURRICULUM_STAGES = {
    1: CurriculumConfig(stage=1, num_obstacles=3, w_velocity=0.2),
    2: CurriculumConfig(stage=2, num_obstacles=5, w_velocity=0.2),
    3: CurriculumConfig(stage=3, num_obstacles=7, w_velocity=0.2),
    4: CurriculumConfig(stage=4, num_obstacles=9, w_velocity=0.2),
    5: CurriculumConfig(stage=5, num_obstacles=15, w_velocity=0.2),
}
```

**Stage 5 Progressive Training** (new feature):
- Stage 0: Fixed obstacle positions (seed=42), advance after 3 consecutive successes
- Stage 1-10: Incremental perturbation (±0.5m to ±5.0m)
- Stage 11+: Fully random obstacle positions

**Key Classes**:
- `CurriculumManager`: Manages main curriculum progression
- `Stage5ProgressiveManager`: Manages Stage 5 progressive sub-stages
- `CurriculumCallback`: Evaluates and advances curriculum during training

---

## Environment Configuration

### TunnelConfig (`src/rl/tunnel.py`)

```python
TunnelConfig(
    radius=5.0,                      # Tunnel radius (meters)
    length=50.0,                     # Tunnel length (meters)
    center_z=100.0,                  # Tunnel depth (meters)
    num_obstacles=15,                # Number of obstacles
    obstacle_radius_min=0.7,         # Min obstacle radius
    obstacle_radius_max=1.0,         # Max obstacle radius
    # Progressive training mode (Stage 5)
    progressive_mode=False,
    progressive_stage=0,
    fixed_obstacle_seed=None,
    perturbation_amount=0.5,
)
```

### Observation Space (Hybrid LSTM Mode)

```python
{
    "state": Box(shape=(13,)),           # [rel_y, rel_z, nu(6), orientation(3), dist_boundary, progress]
    "point_cloud_seq": Box(shape=(16, 768))  # (history_len, num_points * 3)
}
```

### Action Space

```python
Box(shape=(3,))  # [pitch_cmd, yaw_cmd, thrust_cmd] in ranges [-1,1], [-1,1], [-1,1]
```

### Reward Function (Progress-Dominant)

- `+7.0 × progress_delta × 100`: Progress reward (only when exceeding historical max)
- `+1000`: Goal reached reward
- `-550`: Collision or timeout penalty
- `-4.5`: Predicted collision penalty (per step, 3-second trajectory prediction)
- `-0.5`: Step penalty (encourages efficiency)

---

## Modifying Training Hyperparameters

### For Fresh Training

Edit the `TrainingConfig` class in `train.py` (lines 33-77):

```python
class TrainingConfig:
    def __init__(self):
        self.learning_rate = 3e-5
        self.n_epochs = 10
        self.batch_size = 512
        # ... other parameters
```

### For Resuming Training

Use command-line arguments to override saved settings:

```bash
python train.py \
    --resume_model path/to/model.zip \
    --resume_normalize path/to/vecnormalize.pkl \
    --learning_rate 1e-5 \
    --n_epochs 20 \
    --batch_size 256
```

The code automatically updates both the model's learning rate and the optimizer's param groups when resuming.

---

## Important Implementation Notes

### Shared Curriculum State

When using `SubprocVecEnv` for parallel training, curriculum state is shared via `multiprocessing.Manager().dict()`. This ensures all workers synchronize when the curriculum advances.

### VecNormalize

The environment uses `VecNormalize` for automatic observation/reward normalization. When resuming training, **always load both the model AND the VecNormalize statistics**:

```bash
python train.py \
    --resume_model model.zip \
    --resume_normalize vecnormalize.pkl
```

### Physics Engine

The 6-DOF dynamics are implemented in `torpedo.py` based on Fossen (2021). The mathematical foundations in `lib/` should not be modified without deep understanding of underwater vehicle hydrodynamics.

### Point Cloud Sampling

The system uses **surface point cloud sampling** (not raycasting). Points are sampled directly from obstacle and tunnel wall surfaces. See `src/rl/point_cloud_sampler.py` for details.

---

## Common Development Tasks

### Adding a New Curriculum Stage

1. Add stage config to `CURRICULUM_STAGES` in `train.py`
2. Update `stage_thresholds` in `CurriculumManager.__init__()` if needed
3. No other changes required - automatic progression

### Modifying Neural Network Architecture

Edit `policy_kwargs` in the `train()` function (line 431-446):

```python
policy_kwargs = dict(
    features_extractor_class=PointCloudLSTMExtractor,
    features_extractor_kwargs=dict(
        lstm_hidden_size=config.lstm_hidden_size,
        # ... other kwargs
    ),
    net_arch=dict(
        pi=[config.pi_hidden_1, config.pi_hidden_2],  # Actor
        vf=[config.vf_hidden_1, config.vf_hidden_2],  # Critic
    ),
)
```

### Changing Observation/Action Spaces

Modify `src/rl/submarine_env.py`:
- `_setup_observation_space()`: Define new observation structure
- `_get_observation()`: Return observations matching new space
- `action_space`: Define new action dimensions

---

## File Locations Reference

| Purpose | File |
|---------|------|
| Main training entry | `train.py` |
| RL environment | `src/rl/submarine_env.py` |
| Neural network | `src/rl/custom_extractor.py` |
| Tunnel/obstacles | `src/rl/tunnel.py` |
| Point cloud | `src/rl/point_cloud_sampler.py` |
| Physics engine | `torpedo.py` |
| 3D renderer | `src/view/renderer.py` |
| Evaluation | `eval_sim.py` |
| Interactive mode | `run_human.py`, `pygame_sim.py` |

---

## Documentation

- **README.md**: Comprehensive project documentation (727 lines, Chinese)
- **RL环境配置文档.md**: Detailed RL environment specs (831 lines, Chinese)

Both files are actively maintained and contain extensive architecture details, troubleshooting guides, and examples.
