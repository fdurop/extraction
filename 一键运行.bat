@echo off
chcp 65001 >nul
title 多模态PDF处理项目 - 一键运行

echo ================================================
echo 多模态PDF处理项目 - 一键运行脚本
echo ================================================
echo.

echo 步骤1: 检查Python环境...
python --version
if %errorlevel% neq 0 (
    echo ❌ 错误: 未找到Python，请先安装Python 3.8+
    echo 下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
)
echo ✓ Python环境正常
echo.

echo 步骤2: 检查依赖包...
python check_environment.py
if %errorlevel% neq 0 (
    echo.
    echo ❌ 环境检查失败，正在自动安装依赖...
    echo.
    python install_environment.py
    echo.
    echo 重新检查环境...
    python check_environment.py
)
echo.

echo 步骤3: 检查输入文件...
if not exist "input\*.pdf" (
    echo ❌ 警告: input目录中没有找到PDF文件
    echo 请将PDF文件放入input目录中
    echo.
    pause
    exit /b 1
)
echo ✓ 找到输入文件
echo.

echo 步骤4: 开始处理PDF文件...
echo 正在启动多模态预处理器...
python src/multimodal_preprocessor.py

echo.
echo ================================================
echo 处理完成！
echo 请查看output目录中的结果
echo ================================================
pause

