#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_human.py: 手动控制模式启动脚本
"""

import sys
import os
import numpy as np
import pygame
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.rl.submarine_env import SubmarineEnv

def get_manual_action(keys, current_thrust):
    """
    根据键盘输入获取动作
    
    Controls:
    - W/S: Pitch (上下俯仰)
    - A/D: Yaw (左右转向)
    - UP/DOWN: Thrust (加减速)
    """
    pitch = 0.0
    yaw = 0.0
    
    # Pitch control
    if keys[pygame.K_w]:
        pitch = 0.8  # 下潜/抬头? 根据坐标系定义，通常 pitch > 0 是抬头
    if keys[pygame.K_s]:
        pitch = -0.8
        
    # Yaw control
    if keys[pygame.K_a]:
        yaw = 0.8   # 左转
    if keys[pygame.K_d]:
        yaw = -0.8  # 右转
        
    # Thrust control (accumulation handled outside or simple step here)
    # 这里我们只返回推力的增量或目标值，或者由外部维护状态
    thrust_delta = 0.0
    if keys[pygame.K_UP]:
        thrust_delta = 0.02
    if keys[pygame.K_DOWN]:
        thrust_delta = -0.02
        
    return pitch, yaw, thrust_delta

def main():
    print("Initializing SubmarineEnv in Human Mode...")
    print("Controls:")
    print("  W/S: Pitch (Nose Up/Down)")
    print("  A/D: Yaw (Turn Left/Right)")
    print("  UP/DOWN: Adjust Speed")
    print("  ESC: Quit")
    
    # 初始化环境
    env = SubmarineEnv(render_mode="human")
    obs, info = env.reset()
    
    running = True
    current_thrust = 0.5 # 初始推力
    
    try:
        while running:
            # 1. 渲染环境 (这会处理窗口事件，包括关闭窗口)
            # 注意：SubmarineEnv.render() 内部调用了 pygame.event.get()，
            # 如果它检测到 QUIT，会调用 close()。
            env.render()
            
            # 检查环境是否已经关闭
            if env._renderer is None:
                running = False
                break
            
            # 2. 获取键盘状态 (get_pressed 不需要事件队列)
            keys = pygame.key.get_pressed()
            
            if keys[pygame.K_ESCAPE]:
                running = False
                break
                
            # 3. 计算动作
            pitch, yaw, thrust_delta = get_manual_action(keys, current_thrust)
            
            # 更新推力状态
            current_thrust = np.clip(current_thrust + thrust_delta, 0.0, 1.0)
            
            # 组合动作 [pitch, yaw, thrust]
            action = np.array([pitch, yaw, current_thrust], dtype=np.float32)
            
            # 4. 环境步进
            obs, reward, terminated, truncated, info = env.step(action)
            
            # 打印一些调试信息 (可选)
            # print(f"Thrust: {current_thrust:.2f}, Reward: {reward:.2f}")
            
            if terminated or truncated:
                reason = info.get('collision_reason', 'unknown')
                result = "Success" if info.get('goal_reached', False) else f"Failed ({reason})"
                print(f"Episode Finished: {result}. Resetting...")
                obs, info = env.reset()
                current_thrust = 0.5 # 重置推力
            
            # 控制帧率
            # env.render() 通常没有内置帧率控制，这里手动加一点延迟或者依靠 vsync
            # pygame.time.Clock().tick(30) # 如果我们在外部持有 clock
            # 由于 env 内部每次 render 都可能比较快，我们加个小 sleep 避免 CPU 占用过高
            # 实际上 pygame_sim 也是 30fps。
            time.sleep(0.03) 
            
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        env.close()
        print("Environment closed.")

if __name__ == "__main__":
    main()
