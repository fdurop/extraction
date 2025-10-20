# 多模态数据提取系统

基于 DeepSeek-VL 的智能化教育助手，支持从 PDF/PPTX 文件中提取文本、图像、公式、表格和代码，并生成专业术语库。

## 功能特性

- 📄 支持 PDF/PPTX 文件解析
- 🖼️ 图像智能描述（基于 DeepSeek-VL）
- 🔢 公式识别与提取
- 💻 代码识别与提取
- 📊 表格提取
- 📚 专业术语库自动生成
- 🤝 交互式人工补充机制
- 📝 完整的日志记录系统

## 环境要求

- Python 3.12
- CUDA 12.1+（GPU 版本）
- 8GB+ GPU 显存（推荐）

## 安装步骤

### 方法一：使用国内镜像源（推荐，速度快）

```bash
# 1. 安装 PyTorch（CUDA 12.1）
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121

# 2. 安装 PaddlePaddle GPU 版本
pip install paddlepaddle-gpu==2.6.0 -i https://mirrors.aliyun.com/pypi/simple/

# 3. 安装其他依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 方法二：使用官方源

```bash
# 1. 安装 PyTorch（CUDA 12.1）
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121

# 2. 安装 PaddlePaddle GPU 版本
pip install paddlepaddle-gpu==2.6.0

# 3. 安装其他依赖
pip install -r requirements.txt
```

### CPU 版本安装

如果没有 GPU，可以安装 CPU 版本：

```bash
# 1. 安装 PyTorch CPU 版本
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu

# 2. 安装 PaddlePaddle CPU 版本
pip install paddlepaddle==2.6.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

# 3. 安装其他依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 验证安装

```bash
python3 -c "import torch, transformers, paddleocr; print('✅ 所有依赖安装成功')"
```

## 运行

### 基本运行

```bash
python3 src/multimodal_preprocessor.py
```

### 处理流程

1. 将待处理的 PDF/PPTX 文件放入 `input/` 目录
2. 运行程序
3. 查看 `output/` 目录中的结果
4. 根据提示进行交互式补充（如需要）
5. 查看生成的专业术语库：`output/professional_terms_library.json`
6. 查看完整日志：`logs/processing_YYYYMMDD_HHMMSS.log`

## 目录结构

```
extraction-main/
├─ input/                    # 输入文件（PDF/PPTX）
├─ output/                   # 输出结果
│  ├─ text/                  # 文本内容（JSON格式）
│  ├─ images/                # 图片及描述（PNG + JSON元数据）
│  ├─ formulas/              # 公式（JSON格式）
│  ├─ tables/                # 表格（JSON格式）
│  ├─ code/                  # 代码（.py + JSON元数据）
│  └─ professional_terms_library.json  # 专业术语库
├─ logs/                     # 日志文件
│  └─ processing_YYYYMMDD_HHMMSS.log
├─ src/                      # 源代码
│  ├─ multimodal_preprocessor.py      # 主程序
│  ├─ advanced_pptx_processor.py      # PPTX处理器
│  ├─ deepseek_vl_wrapper.py          # DeepSeek-VL封装
│  └─ logger_config.py                # 日志配置
├─ models/                   # DeepSeek-VL模型
│  └─ deepseek-vl-7b-chat/
├─ requirements.txt          # 依赖列表
└─ README.md                 # 本文件
```

## 输出说明

### 文本提取
- 位置：`output/text/`
- 格式：`{文件名}_slide_{编号}.json`
- 内容：每页的文本内容、标题、关键信息

### 图像描述
- 位置：`output/images/`
- 格式：
  - 图片：`{文件名}_slide_{编号}_img_{序号}.png`
  - 元数据：`{文件名}_slide_{编号}_img_{序号}_metadata.json`
- 内容：图像描述、核心概念、应用场景、教学建议

### 公式识别
- 位置：`output/formulas/`
- 格式：`{文件名}_slide_{编号}_formulas.json`
- 内容：LaTeX格式公式、变量说明、物理意义

### 代码识别
- 位置：`output/code/`
- 格式：
  - 代码：`{文件名}_slide_{编号}_code.py`
  - 元数据：`{文件名}_slide_{编号}_code_metadata.json`
- 内容：识别出的代码、语言类型、功能说明

### 专业术语库
- 位置：`output/professional_terms_library.json`
- 内容：从所有文档中提取的专业术语列表
- 用途：用于构建知识图谱、术语检索

### 日志文件
- 位置：`logs/processing_YYYYMMDD_HHMMSS.log`
- 级别：DEBUG, INFO, WARNING, ERROR
- 内容：完整的处理过程记录、错误追踪

## 常见问题

### 1. 显存不足
如果遇到 CUDA out of memory 错误：
- 减少批处理大小
- 使用 CPU 模式运行
- 增加系统虚拟内存

### 2. 依赖冲突
如果遇到包冲突：
```bash
# 清理缓存
pip cache purge

# 重新安装
pip uninstall torch torchvision torchaudio -y
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 3. 模型加载失败
确保模型文件完整：
- 检查 `models/deepseek-vl-7b-chat/` 目录
- 确保包含所有 `.safetensors` 文件
- 检查网络连接（首次运行会下载 tokenizer）

## 技术栈

- **多模态模型**：DeepSeek-VL 7B
- **OCR引擎**：PaddleOCR
- **PDF处理**：PyMuPDF, pdfplumber, Camelot
- **图像处理**：Pillow, OpenCV, Wand
- **深度学习框架**：PyTorch 2.2.2
- **文档解析**：python-pptx, python-docx

## 许可证

本项目使用的 DeepSeek-VL 模型遵循其原始许可证。详见 `DeepSeek-VL-main/LICENSE-MODEL`。
