#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目环境配置脚本
自动安装所有必要的依赖包
"""

import subprocess
import sys
import os

def install_package(package_name, version=None):
    """安装指定的包"""
    if version:
        package_spec = f"{package_name}=={version}"
    else:
        package_spec = package_name
    
    print(f"正在安装 {package_spec}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_spec])
        print(f"✓ {package_name} 安装成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {package_name} 安装失败: {e}")
        return False

def install_spacy_model():
    """安装spaCy中文模型"""
    print("正在安装spaCy中文模型...")
    try:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "zh_core_web_sm"])
        print("✓ spaCy中文模型安装成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ spaCy中文模型安装失败: {e}")
        return False

def main():
    print("=" * 50)
    print("开始配置项目环境...")
    print("=" * 50)
    
    # 基础依赖包列表
    packages = [
        "torch",           # PyTorch深度学习框架
        "torchvision",     # PyTorch计算机视觉工具
        "transformers",    # Hugging Face转换器库
        "PyMuPDF",         # PDF处理库
        "pillow",          # 图像处理库
        "spacy",           # 自然语言处理库
        "numpy",           # 数值计算库
        "scikit-learn",    # 机器学习工具
        "matplotlib",      # 绘图库
        "seaborn",         # 统计图表库
        "pandas",          # 数据处理库
        "tqdm",            # 进度条库
    ]
    
    # 安装基础包
    failed_packages = []
    for package in packages:
        if not install_package(package):
            failed_packages.append(package)
    
    # 安装spaCy中文模型
    if not install_spacy_model():
        print("警告: spaCy中文模型安装失败，可能影响实体识别功能")
    
    # 检查CUDA可用性
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA可用，检测到GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("ℹ 使用CPU模式运行")
    except ImportError:
        print("✗ PyTorch导入失败")
    
    # 安装结果总结
    print("\n" + "=" * 50)
    print("环境配置完成！")
    print("=" * 50)
    
    if failed_packages:
        print(f"以下包安装失败: {', '.join(failed_packages)}")
        print("请手动安装这些包或检查网络连接")
    else:
        print("所有依赖包安装成功！")
    
    print("\n下一步操作:")
    print("1. 运行 'python src/multimodal_preprocessor.py' 开始处理PDF文件")
    print("2. 确保input/目录中有PDF文件")
    print("3. 检查output/目录中的处理结果")

if __name__ == "__main__":
    main()

