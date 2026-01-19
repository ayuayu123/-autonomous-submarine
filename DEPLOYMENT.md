# SubmarineHunting 部署指南

## 📦 打包说明

本项目支持使用 PyInstaller 打包为独立可执行文件，可在无 Python 环境的机器上运行。

### 打包前准备

1. **确保所有依赖已安装**:
   ```bash
   pip install -r requirements.txt
   pip install pyinstaller
   ```

2. **确保模型文件存在**:
   - `best_model/best_model.zip` - 训练好的模型
   - `best_model/vecnormalize.pkl` - 观测归一化参数

### 打包步骤

**方法1: 使用批处理脚本 (推荐)**
```bash
# Windows
build_package.bat
```

**方法2: 手动打包**
```bash
pyinstaller submarine_hunting.spec --noconfirm
```

### 打包输出

打包完成后，文件位于 `dist/SubmarineHunting/` 目录：

```
dist/SubmarineHunting/
├── SubmarineHunting.exe    # 主程序
├── best_model/             # 模型文件
├── lib/                    # 依赖库
├── src/                    # 源代码模块
└── ... (其他依赖)
```

---

## 🚀 运行说明

### 演示模式 (Demo)

用于客户演示，加载训练好的模型进行可视化展示。

```bash
# 使用默认配置
SubmarineHunting.exe demo

# 指定课程阶段 (1-10)
SubmarineHunting.exe demo --stage 5

# 完整参数
SubmarineHunting.exe demo --model ./best_model/best_model.zip --stage 7

# 手动控制模式
SubmarineHunting.exe demo --manual
```

**演示模式控制键**:
| 按键 | 功能 |
|------|------|
| WASD | 移动摄像机 |
| 鼠标右键拖动 | 旋转视角 |
| IJKL | 手动控制潜艇 (俯仰/偏航) |
| 上/下箭头 | 调节推力 |
| R | 重置回合 |
| Space | 暂停/继续 |
| S | 打印统计 |
| ESC | 退出 |

### 训练模式 (Train)

用于训练新模型（需要 GPU 支持以获得更好性能）。

```bash
# 默认训练
SubmarineHunting.exe train

# 指定输出目录
SubmarineHunting.exe train --output ./my_training

# 从指定阶段开始
SubmarineHunting.exe train --start-stage 3

# 调整并行环境数
SubmarineHunting.exe train --num-envs 4
```

---

## 📊 课程阶段说明

| 阶段 | 名称 | 障碍物 | 通道半径 | 难度 |
|------|------|--------|----------|------|
| 1 | 基础导航 | 0 | 6.0m | ⭐ |
| 2 | 简单避障 | 6 | 7.0m | ⭐⭐ |
| 3 | 初级密集 | 8 | 6.5m | ⭐⭐ |
| 4 | 中等避障 | 10 | 6.5m | ⭐⭐⭐ |
| 5 | 过渡密集 | 13 | 6.2m | ⭐⭐⭐ |
| 6 | 密集避障 | 18 | 6.0m | ⭐⭐⭐⭐ |
| 7 | 完整任务 | 22 | 6.0m | ⭐⭐⭐⭐ |
| 8 | 极密避障 | 25 | 5.8m | ⭐⭐⭐⭐⭐ |
| 9 | 超长通道 | 30 | 5.5m | ⭐⭐⭐⭐⭐ |
| 10 | 终极挑战 | 35 | 5.5m | ⭐⭐⭐⭐⭐ |

---

## 🔧 内网部署

### 离线安装依赖

如果目标机器无法访问外网，可以提前下载依赖包：

1. **在联网机器上下载依赖**:
   ```bash
   pip download -r requirements.txt -d ./packages
   ```

2. **复制到目标机器并安装**:
   ```bash
   pip install --no-index --find-links=./packages -r requirements.txt
   ```

### 最小化部署

如果只需要演示功能，可以只复制以下文件：

```
SubmarineHunting/
├── SubmarineHunting.exe
├── best_model/
│   ├── best_model.zip
│   └── vecnormalize.pkl
└── (PyInstaller 依赖文件)
```

---

## ❓ 常见问题

### Q: 打包后运行报错 "找不到模块"
A: 检查 `submarine_hunting.spec` 中的 `hiddenimports` 是否包含所有必要模块。

### Q: 演示模式黑屏
A: 确保显卡驱动已更新，支持 OpenGL 3.3+。

### Q: 训练模式很慢
A: 建议使用 NVIDIA GPU 并安装 CUDA 版本的 PyTorch。

### Q: 模型文件丢失
A: 重新从源目录复制 `best_model/` 文件夹。

---

## 📞 技术支持

如有问题，请联系开发团队。
