#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run.py: 项目入口脚本
通过交互式输入选择进入训练或评估模式
使用 subprocess 调用原始脚本，避免 PyInstaller 打包后的导入问题
"""
import os
import sys
import subprocess


def print_banner():
    """打印欢迎横幅"""
    print("=" * 60)
    print("  SubmarineHunting - AUV 仿真与强化学习平台")
    print("=" * 60)
    print()


def print_menu():
    """打印主菜单"""
    print("请选择运行模式:")
    print("  [1] 训练模式 (train.py)")
    print("  [2] 评估模式 (eval_model5.py)")
    print("  [0] 退出")
    print()


def run_train_mode():
    """运行训练模式"""
    print("\n" + "=" * 40)
    print("启动训练模式...")
    print("=" * 40)
    print()

    # 获取脚本路径
    if getattr(sys, 'frozen', False):
        # 打包后的可执行文件，找到源码目录
        base_dir = os.path.dirname(sys.executable)
        # 尝试向上查找项目根目录
        while base_dir and not os.path.exists(os.path.join(base_dir, 'train.py')):
            parent = os.path.dirname(base_dir)
            if parent == base_dir:
                # 无法找到，使用当前目录
                base_dir = os.path.dirname(os.path.abspath(__file__))
                break
            base_dir = parent
        script_path = os.path.join(base_dir, 'train.py')
    else:
        # 正常 Python 运行
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train.py')

    if not os.path.exists(script_path):
        print(f"错误: 找不到 train.py，路径: {script_path}")
        return

    # 使用 subprocess 调用训练脚本
    try:
        result = subprocess.run([sys.executable, script_path], check=True)
        print(f"\n训练完成，退出码: {result.returncode}")
    except subprocess.CalledProcessError as e:
        print(f"训练异常退出，退出码: {e.returncode}")
    except FileNotFoundError:
        print(f"错误: 无法找到 Python 解释器: {sys.executable}")
    except Exception as e:
        print(f"训练启动失败: {e}")
        import traceback
        traceback.print_exc()


def run_eval_mode():
    """运行评估模式"""
    print("\n" + "=" * 40)
    print("评估模式配置")
    print("=" * 40)

    # 获取模型路径
    default_model = "best_model (10).zip"
    model_path = input(f"模型路径 [默认: {default_model}]: ").strip()
    if not model_path:
        model_path = default_model

    # 获取 VecNormalize 路径
    default_vecnorm = "submarine_ppo_2400480_steps_normalize.pkl"
    vecnormalize_path = input(f"VecNormalize 路径 [默认: {default_vecnorm}]: ").strip()
    if not vecnormalize_path:
        vecnormalize_path = default_vecnorm

    # 询问是否手动控制
    manual_input = input("手动控制模式? (y/N): ").strip().lower()
    manual_mode = manual_input == 'y' or manual_input == 'yes'

    # 询问是否保存数据
    save_input = input("保存评估数据? (Y/n): ").strip().lower()
    save_data = save_input != 'n' and save_input != 'no'

    # 询问保存目录
    default_save_dir = "./eval_data_model5"
    if save_data:
        save_dir = input(f"保存目录 [默认: {default_save_dir}]: ").strip()
        if not save_dir:
            save_dir = default_save_dir
    else:
        save_dir = default_save_dir

    print("\n" + "=" * 40)
    print("启动评估模式...")
    print(f"  模型: {model_path}")
    print(f"  VecNormalize: {vecnormalize_path}")
    print(f"  手动控制: {manual_mode}")
    print(f"  保存数据: {save_data}")
    if save_data:
        print(f"  保存目录: {save_dir}")
    print("=" * 40)
    print()

    # 获取脚本路径
    if getattr(sys, 'frozen', False):
        # 打包后的可执行文件，找到源码目录
        base_dir = os.path.dirname(sys.executable)
        # 尝试向上查找项目根目录
        while base_dir and not os.path.exists(os.path.join(base_dir, 'eval_model5.py')):
            parent = os.path.dirname(base_dir)
            if parent == base_dir:
                base_dir = os.path.dirname(os.path.abspath(__file__))
                break
            base_dir = parent
        script_path = os.path.join(base_dir, 'eval_model5.py')
    else:
        # 正常 Python 运行
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'eval_model5.py')

    if not os.path.exists(script_path):
        print(f"错误: 找不到 eval_model5.py，路径: {script_path}")
        return

    # 构建命令行参数
    cmd = [sys.executable, script_path]
    cmd.extend([
        '--model', model_path,
        '--vecnormalize', vecnormalize_path,
        '--save-dir', save_dir,
    ])

    if manual_mode:
        cmd.append('--manual')

    if not save_data:
        cmd.append('--no-save-data')

    # 使用 subprocess 调用评估脚本
    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n评估完成，退出码: {result.returncode}")
    except subprocess.CalledProcessError as e:
        print(f"评估异常退出，退出码: {e.returncode}")
    except FileNotFoundError:
        print(f"错误: 无法找到 Python 解释器: {sys.executable}")
    except Exception as e:
        print(f"评估启动失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""
    print_banner()

    while True:
        print_menu()
        choice = input("请输入选项 (0-2): ").strip()

        if choice == '1':
            run_train_mode()
            # 训练结束后询问是否继续
            print("\n训练已完成。")
            break
        elif choice == '2':
            run_eval_mode()
            # 评估结束后询问是否继续
            print("\n评估已完成。")
            break
        elif choice == '0':
            print("退出程序。")
            break
        else:
            print("无效选项，请重新输入。")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序被用户中断。")
    except Exception as e:
        print(f"\n程序异常退出: {e}")
        import traceback
        traceback.print_exc()
