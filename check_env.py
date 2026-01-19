#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查当前环境是否包含所有必需的依赖
"""
import sys
import importlib
import io

# 修复Windows控制台编码问题
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

REQUIRED_PACKAGES = {
    'numpy': 'NumPy',
    'scipy': 'SciPy',
    'matplotlib': 'Matplotlib',
    'pygame': 'Pygame',
    'OpenGL': 'PyOpenGL',
    'gymnasium': 'Gymnasium',
    'stable_baselines3': 'Stable-Baselines3',
    'torch': 'PyTorch',
    'pandas': 'Pandas',
    'tqdm': 'TQDM',
}

MISSING_OK = {
    'tensorboard': 'TensorBoard (可选)',
}

def check_package(name):
    """检查包是否可导入"""
    try:
        module = importlib.import_module(name)
        version = getattr(module, '__version__', 'unknown')
        return True, version
    except ImportError:
        return False, None

def main():
    print("=" * 60)
    print("SubmarineHunting 环境依赖检查")
    print("=" * 60)
    print(f"Python 版本: {sys.version}")
    print(f"Python 路径: {sys.executable}")
    print("-" * 60)

    all_ok = True

    # 检查必需包
    print("\n【必需依赖】")
    for module_name, display_name in REQUIRED_PACKAGES.items():
        ok, version = check_package(module_name)
        status = "✓" if ok else "✗"
        version_str = f" ({version})" if version else ""
        print(f"  {status} {display_name:25s}{version_str}")
        if not ok:
            all_ok = False

    # 检查可选包
    print("\n【可选依赖】")
    for module_name, display_name in MISSING_OK.items():
        ok, version = check_package(module_name)
        status = "✓" if ok else "○"
        version_str = f" ({version})" if version else ""
        print(f"  {status} {display_name:25s}{version_str}")

    print("\n" + "=" * 60)
    if all_ok:
        print("✓ 所有必需依赖已安装，环境就绪！")
        return 0
    else:
        print("✗ 缺少必需依赖，请先安装:")
        print("  conda activate py310_env")
        print("  pip install -r requirements.txt")
        return 1

if __name__ == "__main__":
    sys.exit(main())
