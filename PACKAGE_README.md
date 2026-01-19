# SubmarineHunting - PyInstaller 打包指南

本指南说明如何使用 PyInstaller 将 SubmarineHunting 项目打包为独立的 Windows 可执行文件，以便在内网环境离线运行。

---

## 文件说明

| 文件 | 说明 |
|------|------|
| `eval_model5.spec` | eval_model5.py 的 PyInstaller 配置文件 |
| `train.spec` | train.py 的 PyInstaller 配置文件 |
| `build_eval.bat` | 打包 eval_model5.py 的批处理脚本 |
| `build_train.bat` | 打包 train.py 的批处理脚本 |
| `build_all.bat` | 同时打包两个脚本的批处理脚本 |

---

## 环境要求

### Conda 环境配置

打包脚本默认使用名为 `normal` 的 conda 环境。如需修改环境名称，编辑 `.bat` 文件中的：

```batch
set "CONDA_ENV=normal"
```

### 创建 Conda 环境

如果尚未创建环境：

```bash
# 创建环境
conda create -n normal python=3.10

# 激活环境
conda activate normal

# 安装依赖
pip install -r requirements.txt
```

---

## 快速开始

> **注意**：打包脚本会自动激活 `normal` conda 环境，请确保该环境已创建并安装了所有依赖。

### 方式一：打包单个脚本

#### 打包评估脚本 (eval_model5.py)

```bash
build_eval.bat
```

生成文件：`dist\SubmarineEval.exe`

#### 打包训练脚本 (train.py)

```bash
build_train.bat
```

生成文件：`dist\SubmarineTrain.exe`

### 方式二：同时打包两个脚本

```bash
build_all.bat
```

生成文件夹：`dist_package\` (包含两个可执行文件)

---

## 手动打包步骤

如果需要手动执行打包过程：

### 1. 激活 Conda 环境

```bash
conda activate normal
```

### 2. 安装 PyInstaller

```bash
pip install pyinstaller
```

### 3. 安装项目依赖

```bash
pip install -r requirements.txt
```

### 4. 执行打包

```bash
# 打包评估脚本
pyinstaller --clean eval_model5.spec

# 打包训练脚本
pyinstaller --clean train.spec
```

---

## 使用打包后的程序

### eval_model5 (SubmarineEval.exe)

评估/可视化训练好的模型：

```bash
# 使用训练好的模型评估
SubmarineEval.exe --model "path\to\model.zip" --vecnormalize "path\to\vecnormalize.pkl"

# 手动控制模式
SubmarineEval.exe --manual

# 禁用数据保存
SubmarineEval.exe --model "model.zip" --no-save-data

# 自定义模型架构参数
SubmarineEval.exe --model "model.zip" --lstm-hidden-size 256 --pc-encoder-dims "512,256"
```

### train (SubmarineTrain.exe)

从头开始训练：

```bash
# 默认配置训练
SubmarineTrain.exe

# 自定义训练参数
SubmarineTrain.exe --total_timesteps 5000000 --learning_rate 1e-5
```

恢复训练：

```bash
# 从已有模型恢复训练
SubmarineTrain.exe --resume_model "path\to\model.zip" --resume_normalize "path\to\vecnormalize.pkl" --resume_stage 3

# 调整学习率恢复训练
SubmarineTrain.exe --resume_model "model.zip" --resume_normalize "vecnormalize.pkl" --learning_rate 5e-6 --n_epochs 20
```

---

## 离线部署

### 部署步骤

1. **复制文件到目标机器**

   将 `dist_package` 文件夹复制到目标机器

2. **准备模型文件（用于评估）**

   如果使用评估功能，需要将训练好的模型文件一并复制：

   ```
   dist_package\
   ├── eval\
   │   ├── SubmarineEval.exe
   │   └── models\              # 模型文件目录
   │       ├── best_model.zip
   │       └── vecnormalize.pkl
   └── train\
       └── SubmarineTrain.exe
   ```

3. **运行程序**

   双击 `SubmarineEval.exe` 或 `SubmarineTrain.exe`，或通过命令行运行

---

## 注意事项

### 1. 文件体积

打包后的可执行文件体积较大（约 500MB - 1GB），这是因为包含了：
- Python 解释器
- PyTorch 深度学习框架
- NumPy、SciPy 等科学计算库
- Stable-Baselines3 强化学习框架
- Pygame/OpenGL 渲染库

### 2. 首次运行

首次运行时，可执行文件会自动解压到临时目录，可能需要几秒钟启动时间。

### 3. 防病毒软件

某些防病毒软件可能会误报 PyInstaller 打包的程序。如果遇到此问题：
- 将程序添加到白名单
- 或使用数字签名对程序进行签名

### 4. CUDA 支持

如果目标机器有 NVIDIA GPU 并希望使用 GPU 加速：

1. 打包机器上安装 CUDA 版本的 PyTorch
2. 打包时会自动包含 CUDA 库
3. 确保目标机器安装了兼容的 NVIDIA 驱动程序

如果不需要 GPU 支持，可以使用 CPU 版本的 PyTorch 打包以减小体积。

### 5. 数据目录

程序运行时会生成以下目录：
- `./eval_data_model5/` - 评估数据（由 eval_model5 生成）
- `./training_output/` - 训练输出（由 train 生成）

---

## 常见问题

### Q: 打包失败，提示找不到模块

**A:** 在对应的 `.spec` 文件中的 `hiddenimports` 列表中添加缺失的模块。

### Q: 运行时提示 "DLL load failed"

**A:** 确保目标机器安装了 Visual C++ Redistributable：
- [Visual C++ 2015-2022 Redistributable (x64)](https://aka.ms/vs/17/release/vc_redist.x64.exe)

### Q: 程序体积过大

**A:** 可以通过以下方式减小体积：
1. 使用 UPX 压缩（已默认启用）
2. 排除不需要的模块（在 `.spec` 文件的 `excludes` 中添加）
3. 使用 CPU 版本的 PyTorch 而非 CUDA 版本

### Q: 如何更新打包的程序

**A:** 源代码修改后，重新运行对应的 `.bat` 脚本即可重新打包。

---

## 技术支持

如有问题，请检查：
1. **Conda 环境**：`normal` 环境是否已创建并激活
   ```bash
   conda activate normal
   ```
2. **Python 版本**：确保为 3.10+
   ```bash
   python --version
   ```
3. **依赖安装**：所有依赖是否正确安装
   ```bash
   pip list
   ```
4. **项目结构**：项目目录结构是否完整

---

## 许可证

本项目仅用于教育和研究目的。
