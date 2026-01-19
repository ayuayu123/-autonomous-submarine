# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for train.py
打包训练脚本，支持离线训练环境
"""
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# 项目根目录
project_root = os.path.getcwd()

# 收集所有需要的数据文件和隐式导入
datas = []
binaries = []
hiddenimports = []

# 收集 torch 相关的隐式导入
hiddenimports += collect_submodules('torch')
hiddenimports += collect_submodules('torch.nn')
hiddenimports += collect_submodules('torch.optim')
hiddenimports += collect_submodules('torch.distributed')

# 收集 stable_baselines3 相关
hiddenimports += collect_submodules('stable_baselines3')
hiddenimports += collect_submodules('stable_baselines3.common')
hiddenimports += collect_submodules('stable_baselines3.common.vec_env')
hiddenimports += collect_submodules('stable_baselines3.common.callbacks')
hiddenimports += collect_submodules('stable_baselines3.common.monitor')

# 收集 gymnasium 相关
hiddenimports += collect_submodules('gymnasium')
hiddenimports += collect_submodules('gymnasium.spaces')
hiddenimports += ['gymnasium.wrappers', 'gymnasium.envs']

# 收集项目自定义模块
hiddenimports += [
    'src',
    'src.core',
    'src.core.actor',
    'src.core.stage',
    'src.physics',
    'src.physics.submarine_actor',
    'src.gameplay',
    'src.gameplay.obstacle_actor',
    'src.gameplay.tunnel_actor',
    'src.rl',
    'src.rl.submarine_env',
    'src.rl.tunnel',
    'src.rl.point_cloud_sampler',
    'src.rl.custom_extractor',
    'src.view',
    'src.view.renderer',
    'torpedo',
    'lib',
    'lib.gnc',
    'lib.models',
    'lib.control',
    'lib.actuator',
    'lib.guidance',
]

# 隐式导入的模块
hiddenimports += [
    'numpy',
    'numpy.core._multiarray_umath',
    'scipy',
    'scipy.spatial',
    'pandas',
    'tqdm',
    'shimmy',
    'cloudpickle',
    'zmq',
    'multiprocessing',
    'multiprocessing.shared_memory',
    'torch._dynamo',
    'torch._inductor',
    'torch.backends.cudnn',
    'tensorboard',
    'tensorboardX',
]

block_cipher = None

a = Analysis(
    ['train.py'],
    pathex=[project_root],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='SubmarineTrain',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # 显示控制台输出
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # 可选: 添加图标文件路径
)
