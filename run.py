#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SubmarineHunting - 潜艇自主避障强化学习系统

统一入口脚本，支持两种模式：
1. 演示模式 (demo): 加载训练好的模型进行可视化演示
2. 训练模式 (train): 使用课程学习进行模型训练

用法:
    # 演示模式 (默认)
    python run.py demo --model ./best_model/best_model.zip --stage 5
    
    # 训练模式
    python run.py train --output ./training_output
    
    # 查看帮助
    python run.py --help
    python run.py demo --help
    python run.py train --help
"""

import os
import sys
import argparse

# 设置环境变量，解决可能的库冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


def run_demo(args):
    """运行演示模式"""
    print("\n" + "=" * 60)
    print("🚀 SubmarineHunting - 演示模式")
    print("=" * 60)
    
    # 导入演示模块
    from eval_model import EvalApp
    from src.rl.curriculum import get_stage_config, get_total_stages
    
    # 处理阶段配置
    if args.stage is not None:
        stage = get_stage_config(args.stage - 1)
        print(f"\n📚 使用课程阶段 {args.stage} 配置: {stage.name}")
        print(f"   通道: 半径={stage.tunnel_radius}m, 长度={stage.tunnel_length}m")
        print(f"   障碍物: {stage.num_obstacles}个")
        print(f"   障碍物起始位置: X={stage.obstacle_start_x}m\n")
        
        tunnel_radius = stage.tunnel_radius
        tunnel_length = stage.tunnel_length
        num_obstacles = stage.num_obstacles
        obstacle_radius_min = stage.obstacle_radius_min
        obstacle_radius_max = stage.obstacle_radius_max
        obstacle_min_spacing = stage.obstacle_min_spacing
        obstacle_start_x = stage.obstacle_start_x
        sample_distance = stage.sample_distance
        num_points = stage.num_points
    else:
        tunnel_radius = args.tunnel_radius
        tunnel_length = args.tunnel_length
        num_obstacles = args.num_obstacles
        obstacle_radius_min = args.obstacle_radius_min
        obstacle_radius_max = args.obstacle_radius_max
        obstacle_min_spacing = args.obstacle_spacing
        obstacle_start_x = args.obstacle_start_x
        sample_distance = args.sample_distance
        num_points = args.num_points
    
    # 自动查找 vecnormalize.pkl
    vecnorm_path = args.vecnorm
    if vecnorm_path is None and args.model:
        model_dir = os.path.dirname(args.model)
        possible_paths = [
            os.path.join(model_dir, "vecnormalize.pkl"),
            os.path.join(os.path.dirname(model_dir), "vecnormalize.pkl"),
        ]
        for path in possible_paths:
            if os.path.exists(path):
                vecnorm_path = path
                break
    
    # 创建并运行演示应用
    app = EvalApp(
        model_path=args.model,
        vecnorm_path=vecnorm_path,
        manual_mode=args.manual,
        tunnel_radius=tunnel_radius,
        tunnel_length=tunnel_length,
        num_obstacles=num_obstacles,
        obstacle_radius_min=obstacle_radius_min,
        obstacle_radius_max=obstacle_radius_max,
        obstacle_min_spacing=obstacle_min_spacing,
        obstacle_start_x=obstacle_start_x,
        sample_distance=sample_distance,
        num_points=num_points,
    )
    app.run()


def run_train(args):
    """运行训练模式"""
    print("\n" + "=" * 60)
    print("🎓 SubmarineHunting - 训练模式")
    print("=" * 60)
    
    # 导入训练模块
    import train_curriculum
    
    # 设置训练参数
    sys.argv = ['train_curriculum.py']
    
    if args.output:
        sys.argv.extend(['--output', args.output])
    if args.start_stage:
        sys.argv.extend(['--start-stage', str(args.start_stage)])
    if args.resume:
        sys.argv.extend(['--resume', args.resume])
    if args.num_envs:
        sys.argv.extend(['--num-envs', str(args.num_envs)])
    
    print(f"\n开始课程学习训练...")
    print(f"输出目录: {args.output or './training_output'}")
    
    # 运行训练
    train_curriculum.main()


def main():
    """主入口"""
    parser = argparse.ArgumentParser(
        description="SubmarineHunting - 潜艇自主避障强化学习系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  演示模式:
    python run.py demo --model ./best_model/best_model.zip --stage 5
    python run.py demo --manual  # 手动控制模式
    
  训练模式:
    python run.py train --output ./my_training
    python run.py train --start-stage 3  # 从第3阶段开始
        """
    )
    
    subparsers = parser.add_subparsers(dest='mode', help='运行模式')
    
    # ==================== 演示模式 ====================
    demo_parser = subparsers.add_parser('demo', help='演示模式：可视化展示训练好的模型')
    
    demo_parser.add_argument("--model", type=str, default="./best_model/best_model.zip",
                             help="模型文件路径 (默认: ./best_model/best_model.zip)")
    demo_parser.add_argument("--vecnorm", type=str, default=None,
                             help="VecNormalize 文件路径 (自动查找)")
    demo_parser.add_argument("--manual", action="store_true",
                             help="手动控制模式")
    
    # 阶段配置
    demo_parser.add_argument("--stage", type=int, default=None,
                             help="使用课程阶段配置 (1-10)")
    
    # 通道参数
    demo_parser.add_argument("--tunnel-radius", type=float, default=6.0,
                             help="通道半径 (默认: 6.0m)")
    demo_parser.add_argument("--tunnel-length", type=float, default=50.0,
                             help="通道长度 (默认: 50.0m)")
    demo_parser.add_argument("--num-obstacles", type=int, default=18,
                             help="障碍物数量 (默认: 18)")
    demo_parser.add_argument("--obstacle-radius-min", type=float, default=0.6,
                             help="最小障碍物半径 (默认: 0.6)")
    demo_parser.add_argument("--obstacle-radius-max", type=float, default=1.0,
                             help="最大障碍物半径 (默认: 1.0)")
    demo_parser.add_argument("--obstacle-spacing", type=float, default=1.5,
                             help="障碍物最小间距 (默认: 1.5)")
    demo_parser.add_argument("--obstacle-start-x", type=float, default=10.0,
                             help="障碍物起始X位置 (默认: 10.0)")
    demo_parser.add_argument("--sample-distance", type=float, default=15.0,
                             help="点云采样距离 (默认: 15.0)")
    demo_parser.add_argument("--num-points", type=int, default=512,
                             help="点云采样点数 (默认: 512)")
    
    # ==================== 训练模式 ====================
    train_parser = subparsers.add_parser('train', help='训练模式：使用课程学习训练模型')
    
    train_parser.add_argument("--output", type=str, default="./training_output",
                              help="训练输出目录 (默认: ./training_output)")
    train_parser.add_argument("--start-stage", type=int, default=None,
                              help="起始阶段 (1-10, 默认从头开始)")
    train_parser.add_argument("--resume", type=str, default=None,
                              help="从检查点恢复训练")
    train_parser.add_argument("--num-envs", type=int, default=8,
                              help="并行环境数量 (默认: 8)")
    
    args = parser.parse_args()
    
    # 如果没有指定模式，默认进入演示模式
    if args.mode is None:
        print("\n未指定模式，使用 --help 查看帮助")
        print("\n快速开始:")
        print("  演示模式: python run.py demo --stage 5")
        print("  训练模式: python run.py train")
        parser.print_help()
        return
    
    if args.mode == 'demo':
        run_demo(args)
    elif args.mode == 'train':
        run_train(args)


if __name__ == "__main__":
    main()
