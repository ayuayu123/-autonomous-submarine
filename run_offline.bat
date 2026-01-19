@echo off
REM SubmarineHunting 离线运行脚本
REM 使用本地 conda 环境，无需激活

setlocal

REM 设置项目根目录
set PROJECT_DIR=%~dp0
set CONDA_ENV=%PROJECT_DIR%conda
set PYTHON_EXE=%CONDA_ENV%\python.exe

REM 检查 Python 是否存在
if not exist "%PYTHON_EXE%" (
    echo 错误: 找不到 Python 解释器
    echo 路径: %PYTHON_EXE%
    echo 请确认 conda 环境已解压到: %CONDA_ENV%
    pause
    exit /b 1
)

REM 运行主程序
echo ========================================
echo   SubmarineHunting - AUV 仿真平台
echo ========================================
echo Python: %PYTHON_EXE%
echo 项目目录: %PROJECT_DIR%
echo ========================================
echo.

"%PYTHON_EXE%" "%PROJECT_DIR%run.py"

REM 如果程序异常退出，暂停以便查看错误信息
if errorlevel 1 (
    echo.
    echo 程序异常退出，错误代码: %errorlevel%
    pause
)
