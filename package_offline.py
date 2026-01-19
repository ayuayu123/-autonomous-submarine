#!/usr/bin/env python3
"""
SubmarineHunting 离线部署包打包脚本

这个脚本会创建一个完整的离线部署包，包含：
1. 便携式 Python 环境
2. 所有依赖包
3. 项目源代码
4. 预训练模型（可选）
5. 部署文档和安装脚本

使用方法:
    python package_offline.py [--with-models] [--output-dir DIR]
"""

import os
import sys
import shutil
import argparse
import zipfile
import subprocess
from pathlib import Path
from datetime import datetime


class OfflinePackageBuilder:
    """离线部署包构建器"""

    def __init__(self, output_dir="dist", with_models=False, portable_python_dir=None):
        self.project_root = Path(__file__).parent.resolve()
        self.output_dir = Path(output_dir)
        self.with_models = with_models
        self.portable_python_dir = portable_python_dir
        self.package_name = f"SubmarineHunting_Offline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 临时打包目录
        self.temp_dir = self.output_dir / self.package_name

    def log(self, message):
        """打印日志"""
        print(f"[INFO] {message}")

    def copy_directory(self, src, dst, exclude_patterns=None):
        """复制目录，支持排除模式"""
        src = Path(src)
        dst = Path(dst)
        dst.mkdir(parents=True, exist_ok=True)

        exclude_patterns = exclude_patterns or []

        for item in src.rglob("*"):
            if item.is_file():
                # 检查是否在排除列表中
                if any(pattern in str(item) for pattern in exclude_patterns):
                    continue

                # 计算相对路径
                rel_path = item.relative_to(src)
                dest_file = dst / rel_path
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, dest_file)

        self.log(f"Copied: {src} -> {dst}")

    def copy_project_files(self):
        """复制项目文件"""
        self.log("=" * 60)
        self.log("复制项目源代码...")
        self.log("=" * 60)

        # 需要复制的目录
        dirs_to_copy = [
            ("lib", "lib"),
            ("src", "src"),
        ]

        # 需要复制的文件
        files_to_copy = [
            "torpedo.py",
            "train.py",
            "eval_sim.py",
            "eval_model5.py",
            "pygame_sim.py",
            "run_human.py",
            "requirements.txt",
            "python-3.13.0-amd64.exe",
            "get-pip.py",
            "CLAUDE.md",
            "DEPLOYMENT.md",
            "README.md",
            "RL环境配置文档.md",
        ]

        # 复制目录
        for src_dir, dst_dir in dirs_to_copy:
            src_path = self.project_root / src_dir
            if src_path.exists():
                self.copy_directory(
                    src_path,
                    self.temp_dir / dst_dir,
                    exclude_patterns=["__pycache__", ".pyc", ".git"]
                )

        # 复制文件
        for file_name in files_to_copy:
            src_file = self.project_root / file_name
            if src_file.exists():
                shutil.copy2(src_file, self.temp_dir / file_name)
                self.log(f"Copied: {file_name}")

    def copy_packages(self):
        """复制依赖包"""
        self.log("=" * 60)
        self.log("复制依赖包...")
        self.log("=" * 60)

        packages_dir = self.project_root / "packages"
        target_dir = self.temp_dir / "packages"

        if packages_dir.exists():
            shutil.copytree(packages_dir, target_dir)
            self.log(f"Copied packages directory ({len(list(packages_dir.glob('*.whl')))} wheels)")
        else:
            self.log("Warning: packages directory not found!")

    def copy_portable_python(self):
        """复制或下载便携式 Python"""
        self.log("=" * 60)
        self.log("准备便携式 Python 环境...")
        self.log("=" * 60)

        python_dir = self.temp_dir / "python"

        if self.portable_python_dir:
            # 使用用户指定的 Python 目录
            src_python = Path(self.portable_python_dir)
            if src_python.exists():
                self.copy_directory(
                    src_python,
                    python_dir,
                    exclude_patterns=["__pycache__", "*.pyc", "test", "tests"]
                )
            else:
                raise FileNotFoundError(f"指定的 Python 目录不存在: {src_python}")
        else:
            # 检查是否有便携式 Python 的压缩包
            python_archive = self.project_root / "python_embed.7z"
            if python_archive.exists():
                self.log(f"找到 Python 压缩包: {python_archive}")
                self.log("请手动解压到部署包中的 python 目录")
                shutil.copy2(python_archive, self.temp_dir / "python_embed.7z")
            else:
                self.log("未找到便携式 Python，需要手动准备")
                self.log("请从以下地址下载 Python 3.13 嵌入式版本:")
                self.log("https://www.python.org/downloads/windows/")
                self.log("下载 'Windows embeddable package (64-bit)'")
                self.log(f"并解压到: {python_dir}")

        return python_dir

    def setup_pip_in_embedded_python(self, python_dir):
        """在嵌入式 Python 中设置 pip"""
        self.log("设置 pip...")

        # 检查是否已有 pip
        if (python_dir / "Scripts" / "pip.exe").exists():
            self.log("pip 已存在")
            return

        # 下载 get-pip.py
        get_pip = python_dir / "get-pip.py"
        if not get_pip.exists():
            self.log("需要下载 get-pip.py 来安装 pip")
            self.log("请访问: https://bootstrap.pypa.io/get-pip.py")

    def copy_models(self):
        """复制预训练模型（可选）"""
        if not self.with_models:
            return

        self.log("=" * 60)
        self.log("复制预训练模型...")
        self.log("=" * 60)

        models_dir = self.temp_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        # 查找所有模型文件
        model_files = list(self.project_root.glob("*.zip")) + list(self.project_root.glob("*.pkl"))

        for model_file in model_files:
            if any(x in model_file.name for x in ["best_model", "vecnormalize", "submarine_ppo"]):
                shutil.copy2(model_file, models_dir / model_file.name)
                self.log(f"Copied model: {model_file.name}")

    def create_install_scripts(self):
        """创建安装脚本"""
        self.log("=" * 60)
        self.log("创建安装脚本...")
        self.log("=" * 60)

        # Windows 批处理脚本
        install_bat = self.temp_dir / "install.bat"
        install_bat.write_text("""@echo off
REM SubmarineHunting 离线部署安装脚本
echo ====================================
echo SubmarineHunting 离线部署安装
echo ====================================
echo.

set PROJECT_DIR=%~dp0

REM 检查 Python 是否已安装
python --version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Python 已安装，跳过安装步骤
    goto INSTALL_PACKAGES
)

echo.
echo [1/4] 安装 Python 3.13...
echo 这可能需要几分钟...
python-3.13.0-amd64.exe /quiet InstallAllUsers=0 PrependPath=0 Include_test=0
if errorlevel 1 (
    echo 错误: Python 安装失败
    pause
    exit /b 1
)

REM 刷新环境变量
set "PATH=%LOCALAPPDATA%\\Programs\\Python\\Python313;%LOCALAPPDATA%\\Programs\\Python\\Python313\\Scripts;%PATH%"

:INSTALL_PACKAGES
echo.
echo [2/4] 验证 Python 版本...
python --version
if errorlevel 1 (
    echo 错误: Python 未正确安装
    echo 请关闭此窗口，重新以管理员身份运行
    pause
    exit /b 1
)

echo.
echo [3/4] 安装依赖包...
echo 这可能需要 5-10 分钟...
python -m pip install --no-index --find-links="%PROJECT_DIR%packages" -r "%PROJECT_DIR%requirements.txt"
if errorlevel 1 (
    echo 警告: 部分依赖安装失败，尝试继续...
)

echo.
echo [4/4] 验证安装...
python -c "import torch; import gymnasium; import stable_baselines3; print('所有依赖导入成功!')"
if errorlevel 1 (
    echo 错误: 依赖验证失败
    pause
    exit /b 1
)

echo.
echo 创建启动脚本...

REM 创建训练启动脚本
echo @echo off > train.bat
echo setlocal EnableDelayedExpansion >> train.bat
echo set "PYTHON_EXE=python" >> train.bat
echo where python ^>nul 2^>^&1 >> train.bat
echo if errorlevel 1 ( >> train.bat
echo     set "PYTHON_EXE=%%LOCALAPPDATA%%\Programs\Python\Python313\python.exe" >> train.bat
echo ) >> train.bat
echo "!PYTHON_EXE!" %%~dp0train.py %%* >> train.bat

REM 创建评估启动脚本
echo @echo off > eval.bat
echo setlocal EnableDelayedExpansion >> eval.bat
echo set "PYTHON_EXE=python" >> eval.bat
echo where python ^>nul 2^>^&1 >> eval.bat
echo if errorlevel 1 ( >> eval.bat
echo     set "PYTHON_EXE=%%LOCALAPPDATA%%\Programs\Python\Python313\python.exe" >> eval.bat
echo ) >> eval.bat
echo "!PYTHON_EXE!" %%~dp0eval_model5.py %%* >> eval.bat

REM 创建交互式仿真启动脚本
echo @echo off > pygame_sim.bat
echo setlocal EnableDelayedExpansion >> pygame_sim.bat
echo set "PYTHON_EXE=python" >> pygame_sim.bat
echo where python ^>nul 2^>^&1 >> pygame_sim.bat
echo if errorlevel 1 ( >> pygame_sim.bat
echo     set "PYTHON_EXE=%%LOCALAPPDATA%%\Programs\Python\Python313\python.exe" >> pygame_sim.bat
echo ) >> pygame_sim.bat
echo "!PYTHON_EXE!" %%~dp0pygame_sim.py >> pygame_sim.bat

echo.
echo ====================================
echo 安装完成!
echo ====================================
echo.
echo 可用命令:
echo   - 训练: train.bat
echo   - 评估: eval.bat
echo   - 交互仿真: pygame_sim.bat
echo.
echo 完整文档请参阅: OFFLINE_DEPLOYMENT.md
echo.
pause
""")

        # Linux/Mac shell 脚本
        install_sh = self.temp_dir / "install.sh"
        install_sh.write_text("""#!/bin/bash
# SubmarineHunting 离线部署安装脚本

echo "===================================="
echo "SubmarineHunting 离线部署安装"
echo "===================================="
echo ""

# 设置路径
PYTHON_DIR="$(cd "$(dirname "$0")"/python && pwd)"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 添加 Python 到 PATH
export PATH="$PYTHON_DIR:$PYTHON_DIR/Scripts:$PATH"

echo ""
echo "[1/4] 检查 Python 环境..."
python3 --version || python --version
if [ $? -ne 0 ]; then
    echo "错误: Python 未正确配置"
    echo "请确保已将便携式 Python 解压到 python 目录"
    exit 1
fi

echo ""
echo "[2/4] 安装依赖包..."
python3 -m pip install --no-index --find-links="$PROJECT_DIR/packages" -r "$PROJECT_DIR/requirements.txt" || \
python -m pip install --no-index --find-links="$PROJECT_DIR/packages" -r "$PROJECT_DIR/requirements.txt"
if [ $? -ne 0 ]; then
    echo "警告: 部分依赖安装失败，尝试继续..."
fi

echo ""
echo "[3/4] 验证安装..."
python3 -c "import torch; import gymnasium; import stable_baselines3; print('所有依赖导入成功!')" || \
python -c "import torch; import gymnasium; import stable_baselines3; print('所有依赖导入成功!')"

echo ""
echo "[4/4] 创建启动脚本..."

# 创建训练启动脚本
cat > train.sh << 'EOF'
#!/bin/bash
PYTHON_DIR="$(cd "$(dirname "$0")"/python && pwd)"
export PATH="$PYTHON_DIR:$PYTHON_DIR/Scripts:$PATH"
python3 "$(dirname "$0")/train.py" "$@"
EOF
chmod +x train.sh

# 创建评估启动脚本
cat > eval.sh << 'EOF'
#!/bin/bash
PYTHON_DIR="$(cd "$(dirname "$0")"/python && pwd)"
export PATH="$PYTHON_DIR:$PYTHON_DIR/Scripts:$PATH"
python3 "$(dirname "$0")/eval_model5.py" "$@"
EOF
chmod +x eval.sh

# 创建交互式仿真启动脚本
cat > pygame_sim.sh << 'EOF'
#!/bin/bash
PYTHON_DIR="$(cd "$(dirname "$0")"/python && pwd)"
export PATH="$PYTHON_DIR:$PYTHON_DIR/Scripts:$PATH"
python3 "$(dirname "$0")/pygame_sim.py"
EOF
chmod +x pygame_sim.sh

echo ""
echo "===================================="
echo "安装完成!"
echo "===================================="
echo ""
echo "可用命令:"
echo "  - 训练: ./train.sh"
echo "  - 评估: ./eval.sh"
echo "  - 交互仿真: ./pygame_sim.sh"
echo ""
echo "完整文档请参阅: OFFLINE_DEPLOYMENT.md"
echo ""
""")

        # 创建简单的运行脚本（不安装）
        run_bat = self.temp_dir / "run.bat"
        run_bat.write_text("""@echo off
REM SubmarineHunting 快速启动脚本（用于已安装环境）

set PYTHON_DIR=%~dp0python
set PATH=%PYTHON_DIR%;%PYTHON_DIR%Scripts;%PATH%

python %~dp0%* || python %~dp0%*
""")

        self.log("Created install scripts: install.bat, install.sh, run.bat")

    def create_deployment_doc(self):
        """创建离线部署文档"""
        self.log("=" * 60)
        self.log("创建离线部署文档...")
        self.log("=" * 60)

        doc_content = """# SubmarineHunting 离线部署指南

## 目录

- [部署包内容](#部署包内容)
- [系统要求](#系统要求)
- [安装步骤](#安装步骤)
- [使用说明](#使用说明)
- [故障排除](#故障排除)

---

## 部署包内容

```
SubmarineHunting_Offline/
├── python/                 # 便携式 Python 环境（需手动准备）
├── packages/               # Python 依赖包（wheel 文件）
├── lib/                    # 数学库
├── src/                    # 仿真引擎
├── models/                 # 预训练模型（可选）
├── torpedo.py              # 物理引擎
├── train.py                # 训练脚本
├── eval_sim.py             # 评估脚本
├── pygame_sim.py           # 交互式仿真
├── run_human.py            # 人工控制环境
├── requirements.txt        # 依赖清单
├── install.bat             # Windows 安装脚本
├── install.sh              # Linux/Mac 安装脚本
└── OFFLINE_DEPLOYMENT.md   # 本文档
```

---

## 系统要求

### 硬件要求

| 组件 | 最低配置 | 推荐配置 |
|------|---------|---------|
| CPU | 4核心 | 8核心+ |
| 内存 | 8 GB | 16 GB+ |
| 显卡 | 集成显卡 | 独立显卡（支持 OpenGL 3.3+）|
| 存储 | 5 GB 可用空间 | 10 GB+ SSD |
| GPU（训练）| 无 | NVIDIA GPU（CUDA 支持）|

### 软件要求

- **操作系统**: Windows 10/11, Linux, 或 macOS
- **显卡驱动**: 支持 OpenGL 3.3+ 的最新驱动

---

## 安装步骤

### 准备便携式 Python（首次使用）

#### 方法 1: 使用 Python 嵌入式版本

1. **下载 Python 3.13 嵌入式版本**

   访问: https://www.python.org/downloads/windows/

   下载 "Windows embeddable package (64-bit)"

2. **解压到部署包**

   将下载的 zip 文件解压到 `python` 目录：

   ```
   SubmarineHunting_Offline/
   └── python/              # 解压到这里
       ├── python.exe
       ├── python3.dll
       └── ...
   ```

3. **修改配置文件**

   编辑 `python/python313._pth` 文件，取消注释以下行：

   ```text
   import site
   ```

   这样可以启用 site-packages 目录。

#### 方法 2: 使用 Conda 打包的环境

如果你有一台有 Conda 的机器，可以打包现有环境：

```bash
# 在有 Conda 的机器上
conda create -n sub_hunt python=3.13
conda activate sub_hunt
pip install -r requirements.txt

# 安装 conda-pack
conda install -c conda-forge conda-pack

# 打包环境
conda pack -n sub_hunt -o python_env.tar.gz

# 在目标机器上解压到 python 目录
tar -xzf python_env.tar.gz -C /path/to/SubmarineHunting_Offline/python
```

### 安装步骤

#### Windows 系统

1. **确保便携式 Python 已准备好**

   检查 `python` 目录是否存在且包含 `python.exe`

2. **运行安装脚本**

   双击 `install.bat` 或在命令行运行：

   ```cmd
   install.bat
   ```

3. **等待安装完成**

   安装脚本会：
   - 检查 Python 环境
   - 安装所有依赖包
   - 验证安装
   - 创建启动脚本

#### Linux/macOS 系统

1. **确保便携式 Python 已准备好**

   检查 `python` 目录是否存在且包含 `python` 可执行文件

2. **运行安装脚本**

   ```bash
   chmod +x install.sh
   ./install.sh
   ```

---

## 使用说明

### 快速开始

安装完成后，使用以下命令：

| 功能 | Windows | Linux/Mac |
|------|---------|-----------|
| 训练模型 | `train.bat` | `./train.sh` |
| 评估模型 | `eval.bat` | `./eval.sh` |
| 交互仿真 | `pygame_sim.bat` | `./pygame_sim.sh` |

### 训练模型

#### 从头开始训练

```cmd
REM Windows
train.bat

REM Linux/Mac
./train.sh
```

#### 恢复训练

```cmd
REM Windows
train.bat --resume_model ./models/best_model.zip --resume_normalize ./models/vecnormalize.pkl --learning_rate 1e-5

REM Linux/Mac
./train.sh --resume_model ./models/best_model.zip --resume_normalize ./models/vecnormalize.pkl --learning_rate 1e-5
```

### 评估模型

#### 使用预训练模型评估

```cmd
REM Windows
eval.bat --model ./models/best_model.zip --vecnormalize ./models/vecnormalize.pkl

REM Linux/Mac
./eval.sh --model ./models/best_model.zip --vecnormalize ./models/vecnormalize.pkl
```

#### 人工控制模式

```cmd
eval.bat --manual
```

### 交互式仿真

```cmd
REM Windows
pygame_sim.bat

REM Linux/Mac
./pygame_sim.sh
```

**控制方式：**
- **WASD**: 移动相机
- **鼠标右键拖拽**: 旋转视角
- **R**: 重置仿真
- **空格**: 暂停/继续
- **ESC**: 退出

### 监控训练进度

```cmd
REM 启动 TensorBoard
python -m tensorboard --logdir ./training_output

REM 然后访问 http://localhost:6006
```

---

## 故障排除

### 常见问题

#### 1. Python 未找到

**问题**: 运行脚本时提示 "python 不是内部或外部命令"

**解决方案**:
- 检查 `python` 目录是否存在
- 确认便携式 Python 已正确解压
- 确保 `python.exe` 在 `python` 目录中

#### 2. 导入模块失败

**问题**: `ModuleNotFoundError: No module named 'xxx'`

**解决方案**:
```cmd
cd /path/to/SubmarineHunting_Offline
python -m pip install --no-index --find-links=./packages -r requirements.txt
```

#### 3. OpenGL 错误

**问题**: 渲染时出现 OpenGL 相关错误

**解决方案**:
- 更新显卡驱动到最新版本
- 确认显卡支持 OpenGL 3.3+
- 如果是虚拟机，启用 3D 加速

#### 4. CUDA 相关错误

**问题**: 使用 GPU 训练时出错

**解决方案**:
- 检查是否安装了 CUDA 版本的 PyTorch
- 运行 `python -c "import torch; print(torch.cuda.is_available())"` 检查 CUDA 可用性
- 如不可用，修改配置使用 CPU 训练

#### 5. 内存不足

**问题**: 训练时内存溢出

**解决方案**:
- 减少 `n_envs` 参数（并行环境数）
- 减少 `batch_size` 参数
- 使用更小的神经网络模型

### 调试模式

启用详细日志：

```cmd
REM Windows
set PYTHONVERBOSE=1
train.bat

REM Linux/Mac
export PYTHONVERBOSE=1
./train.sh
```

---

## 高级配置

### 修改训练参数

编辑 `train.py` 中的 `TrainingConfig` 类：

```python
class TrainingConfig:
    def __init__(self):
        self.learning_rate = 3e-5      # 学习率
        self.n_envs = 15               # 并行环境数
        self.batch_size = 512          # 批次大小
        self.total_timesteps = 9000000 # 总训练步数
        # ... 更多参数
```

### 修改神经网络架构

编辑 `train.py` 中的 `policy_kwargs`：

```python
policy_kwargs = dict(
    features_extractor_class=PointCloudLSTMExtractor,
    features_extractor_kwargs=dict(
        lstm_hidden_size=128,          # LSTM 隐藏层大小
        point_cloud_history_len=16,    # 点云序列长度
        # ...
    ),
)
```

### 环境配置

修改 `src/rl/tunnel.py` 中的 `TunnelConfig`：

```python
TunnelConfig(
    radius=5.0,           # 隧道半径
    length=50.0,          # 隧道长度
    num_obstacles=15,     # 障碍物数量
    # ...
)
```

---

## 技术支持

- **完整文档**: 参阅 `README.md` 和 `RL环境配置文档.md`
- **问题反馈**: https://github.com/anthropics/claude-code/issues

---

## 附录

### 依赖包列表

```
numpy>=1.20.0
matplotlib>=3.5.0
scipy>=1.7.0
pygame>=2.1.0
PyOpenGL>=3.1.0
PyOpenGL_accelerate>=3.1.0
gymnasium>=0.29.0
stable-baselines3[extra]>=2.0.0
shimmy>=0.2.1
torch>=2.0.0
pandas>=1.3.0
tqdm>=4.62.0
```

### 文件结构说明

| 目录/文件 | 说明 |
|----------|------|
| `lib/` | 数学库（GNC、动力学模型、控制器）|
| `src/core/` | Actor-Stage 框架 |
| `src/physics/` | 物理实体 |
| `src/gameplay/` | 游戏实体（障碍物、隧道）|
| `src/rl/` | 强化学习层（环境、网络架构）|
| `src/view/` | OpenGL 3D 渲染器 |
| `torpedo.py` | REMUS 100 AUV 物理引擎 |

---

**最后更新**: 2026-01-19
"""

        doc_file = self.temp_dir / "OFFLINE_DEPLOYMENT.md"
        doc_file.write_text(doc_content, encoding="utf-8")
        self.log(f"Created deployment documentation: {doc_file}")

    def create_readme(self):
        """创建简单的 README 文件"""
        readme_content = """# SubmarineHunting 离线部署包

## 快速开始

### Windows 用户

1. **准备 Python 环境**
   - 下载 [Python 3.13 Embeddable Package](https://www.python.org/downloads/windows/)
   - 解压到 `python` 目录
   - 编辑 `python/python313._pth`，取消注释 `import site`

2. **运行安装**
   ```
   双击 install.bat
   ```

3. **开始使用**
   ```
   train.bat      # 训练
   eval.bat       # 评估
   pygame_sim.bat # 交互仿真
   ```

### Linux/macOS 用户

1. **准备 Python 环境**
   ```bash
   # 使用系统 Python 或准备便携式 Python
   ```

2. **运行安装**
   ```bash
   chmod +x install.sh
   ./install.sh
   ```

3. **开始使用**
   ```bash
   ./train.sh      # 训练
   ./eval.sh       # 评估
   ./pygame_sim.sh # 交互仿真
   ```

## 完整文档

- **OFFLINE_DEPLOYMENT.md** - 详细部署指南
- **README.md** - 项目文档
- **DEPLOYMENT.md** - 原始部署文档

## 系统要求

- Python 3.10+
- 8 GB+ 内存
- 支持 OpenGL 3.3+ 的显卡
- (可选) NVIDIA GPU 用于加速训练

## 问题排查

如果遇到问题，请参阅 `OFFLINE_DEPLOYMENT.md` 中的故障排除章节。
"""

        readme_file = self.temp_dir / "README.txt"
        readme_file.write_text(readme_content, encoding="utf-8")
        self.log("Created README.txt")

    def create_package_archive(self):
        """创建部署包压缩文件"""
        self.log("=" * 60)
        self.log("创建部署包压缩文件...")
        self.log("=" * 60)

        # 创建 ZIP 文件
        zip_file = self.output_dir / f"{self.package_name}.zip"

        with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zf:
            for file in self.temp_dir.rglob("*"):
                if file.is_file():
                    arcname = file.relative_to(self.temp_dir)
                    zf.write(file, arcname)

        # 计算大小
        size_mb = sum(f.stat().st_size for f in self.temp_dir.rglob("*") if f.is_file()) / (1024 * 1024)

        self.log(f"创建压缩包: {zip_file}")
        self.log(f"压缩包大小: {size_mb:.1f} MB")

        return zip_file

    def build(self):
        """构建完整的离线部署包"""
        self.log("=" * 60)
        self.log("开始构建离线部署包...")
        self.log("=" * 60)
        self.log(f"输出目录: {self.output_dir}")
        self.log(f"包名称: {self.package_name}")
        print()

        # 清理临时目录
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        self.temp_dir.mkdir(parents=True)

        # 构建步骤
        self.copy_project_files()
        self.copy_packages()
        self.copy_portable_python()
        self.copy_models()
        self.create_install_scripts()
        self.create_deployment_doc()
        self.create_readme()

        # 创建压缩包
        zip_file = self.create_package_archive()

        print()
        self.log("=" * 60)
        self.log("构建完成!")
        self.log("=" * 60)
        self.log(f"部署包位置: {zip_file}")
        print()
        self.log("部署步骤:")
        self.log("1. 将压缩包复制到目标机器")
        self.log("2. 解压缩包")
        self.log("3. 准备便携式 Python（参考 OFFLINE_DEPLOYMENT.md）")
        self.log("4. 运行 install.bat (Windows) 或 install.sh (Linux/Mac)")
        self.log("5. 开始使用!")

        return zip_file


def main():
    parser = argparse.ArgumentParser(
        description="SubmarineHunting 离线部署包打包工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基础打包（不含模型）
  python package_offline.py

  # 包含预训练模型
  python package_offline.py --with-models

  # 指定输出目录
  python package_offline.py --output-dir dist

  # 使用已准备的便携式 Python
  python package_offline.py --portable-python /path/to/python

  # 完整示例
  python package_offline.py --with-models --output-dir dist --portable-python C:/Python313
        """
    )

    parser.add_argument(
        "--with-models",
        action="store_true",
        help="包含预训练模型（会增加包大小）"
    )

    parser.add_argument(
        "--output-dir",
        default="dist",
        help="输出目录（默认: dist）"
    )

    parser.add_argument(
        "--portable-python",
        default=None,
        help="便携式 Python 目录路径（如果提供，将打包到部署包中）"
    )

    args = parser.parse_args()

    try:
        builder = OfflinePackageBuilder(
            output_dir=args.output_dir,
            with_models=args.with_models,
            portable_python_dir=args.portable_python
        )
        builder.build()
        return 0
    except Exception as e:
        print(f"[ERROR] 构建失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
