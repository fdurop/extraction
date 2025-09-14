# 多模态PDF数据预处理器

基于CLIP模型的PDF文档多模态数据提取与预处理工具。

## 📋 功能特性

- **PDF文档解析**: 自动提取PDF中的文本和图像内容
- **图像增强处理**: 对提取的图像进行对比度和锐度增强
- **CLIP语义理解**: 使用CLIP模型生成文本和图像的语义向量表示
- **智能图像描述**: 自动为图像生成描述性标签
- **🆕 公式提取**: 识别并提取LaTeX格式的数学公式，支持OCR识别图像中的公式
- **🆕 表格提取**: 自动识别并提取表格数据，保存为CSV和JSON格式
- **🆕 代码提取**: 识别各种编程语言的代码块，支持多种代码格式
- **结构化输出**: 所有数据以JSON和CSV格式规范存储

## 🛠️ 技术栈

- **深度学习**: PyTorch + Transformers (CLIP模型)
- **PDF处理**: PyMuPDF 
- **图像处理**: Pillow
- **数据格式**: JSON

## 📦 安装要求

### 系统要求
- Python 3.8+
- Windows 10/11 或 Linux/macOS
- 至少 4GB RAM
- 可选: NVIDIA GPU (支持CUDA加速)

### 依赖安装

```bash
# 克隆或下载项目
cd multimodal-pdf-processor

# 安装依赖
pip install -r src/requirements.txt

# 或使用自动安装脚本 (Windows)
install_environment.bat
```

## 🚀 快速开始

### 1. 准备输入文件
将PDF文件放入 `input/` 目录中

### 2. 运行处理器

**方法一：直接运行**
```bash
python src/multimodal_preprocessor.py
```

**方法二：使用批处理 (Windows)**
```bash
一键运行.bat
```

### 3. 查看结果
处理完成后，结果保存在 `output/` 目录：

```
output/
├── images/           # 提取的图像和增强图像
│   ├── filename_p1_img1.png
│   ├── filename_p1_img1_enhanced.png
│   └── filename_p1_img1.json
├── text/            # 文本数据
│   └── filename_p1.json
├── formulas/        # 🆕 数学公式数据
│   ├── filename_p1_formulas.json
│   └── filename_p1_formulas.csv
├── tables/          # 🆕 表格数据
│   ├── filename_p1_tables.json
│   └── filename_p1_table1.csv
├── code/            # 🆕 代码块数据
│   ├── filename_p1_code.json
│   ├── filename_p1_code.csv
│   └── filename_p1_code1.py
└── filename_metadata.json  # 处理结果汇总
```

## 📄 输出格式

### 文本数据 (text/*.json)
```json
{
  "type": "text",
  "page": 1,
  "raw_text": "页面文本内容...",
  "word_count": 150,
  "associated_images": ["path/to/image1.png"],
  "text_vector": [0.1, 0.2, ...]  // 512维CLIP向量
}
```

### 图像数据 (images/*.json)
```json
{
  "type": "image",
  "image_path": "output/images/file_p1_img1.png",
  "enhanced_path": "output/images/file_p1_img1_enhanced.png",
  "width": 800,
  "height": 600,
  "format": "PNG",
  "mode": "RGB",
  "page_text_context": "相关页面文本...",
  "image_vector": [0.3, 0.4, ...],  // 512维CLIP向量
  "clip_descriptions": [
    {"description": "科学图表", "confidence": 0.85},
    {"description": "数据图表", "confidence": 0.72}
  ]
}
```

### 🆕 公式数据 (formulas/*.json)
```json
{
  "type": "formula",
  "page": 1,
  "formula_id": "filename_p1_formula1",
  "content": "x = \\frac{-b \\pm \\sqrt{b^2-4ac}}{2a}",
  "format": "latex",
  "extraction_method": "pattern_1",
  "context": "求解二次方程的公式..."
}
```

### 🆕 表格数据 (tables/*.json)
```json
{
  "type": "table",
  "page": 1,
  "table_id": "filename_p1_table1",
  "rows": 5,
  "columns": 3,
  "data": [
    {"Column1": "Value1", "Column2": "Value2", "Column3": "Value3"},
    ...
  ],
  "extraction_method": "camelot",
  "accuracy": 0.95
}
```

### 🆕 代码数据 (code/*.json)
```json
{
  "type": "code",
  "page": 1,
  "code_id": "filename_p1_code1",
  "content": "def hello_world():\n    print('Hello, World!')",
  "language": "python",
  "extraction_method": "pattern_1",
  "line_count": 2,
  "context": "示例代码..."
}
```

### 元数据汇总 (metadata.json)
```json
{
  "processing_date": "2025-08-20 23:15:42",
  "source_file": "filename",
  "statistics": {
    "total_pages": 77,
    "total_text_blocks": 77,
    "total_images": 180,
    "total_formulas": 25,
    "total_tables": 12,
    "total_table_rows": 156,
    "total_code_blocks": 8,
    "total_code_lines": 245,
    "code_languages": ["python", "javascript", "sql"]
  },
  "files": {
    "text_files": [...],
    "image_files": [...],
    "formula_files": [...],
    "table_files": [...],
    "code_files": [...]
  },
  "processing_info": {
    "clip_model": "openai/clip-vit-base-patch32",
    "device": "cpu/cuda",
    "output_format": "JSON/CSV",
    "ocr_enabled": true,
    "extraction_features": ["text", "images", "formulas", "tables", "code"]
  }
}
```

## 🔧 配置选项

### CLIP模型
项目使用本地CLIP模型 (`./clip-model/`)，首次运行会自动加载。

### 图像描述类别
支持的描述类别包括：
- 科学图表、数学公式、数据图表
- 流程图、实验装置、分子结构
- 技术示意图、概念图、网络图
- 照片、插图、表格、代码截图

## 📁 项目结构

```
multimodal-pdf-processor/
├── src/
│   ├── multimodal_preprocessor.py  # 主处理器
│   └── requirements.txt            # 项目依赖
├── clip-model/                     # CLIP模型文件
├── input/                          # 输入PDF文件目录
├── output/                         # 输出结果目录
│   ├── images/                     # 图像文件和元数据
│   ├── text/                       # 文本数据
│   ├── formulas/                   # 🆕 数学公式数据
│   ├── tables/                     # 🆕 表格数据（JSON/CSV）
│   └── code/                       # 🆕 代码块数据
├── install_environment.bat         # 环境安装脚本
├── 一键运行.bat                    # 快速运行脚本
└── README.md                       # 项目文档
```

## 🚨 常见问题

### Q: 程序运行没有输出？
A: 确保使用 `py` 命令而不是 `python`，或检查Python环境变量配置。

### Q: CUDA相关错误？
A: 程序会自动检测GPU，如果没有CUDA设备会自动使用CPU模式，不影响功能。

### Q: 内存不足？
A: 确保有足够内存，大型PDF文件需要更多资源。可以先处理较小的文件测试。

### Q: 依赖安装失败？
A: 尝试使用国内镜像源：
```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r src/requirements.txt
```

## 📊 性能说明

- **处理速度**: 约1-2页/秒 (取决于页面复杂度和硬件)
- **内存占用**: 2-4GB (加载CLIP模型)
- **输出大小**: 每页约1-10MB (包含向量数据)

## 🔄 更新日志

### v2.0 (2025-08-20)
- ✅ 简化为纯数据预处理功能
- ✅ 移除复杂的实体识别模块
- ✅ 优化CLIP图像描述生成
- ✅ 统一JSON输出格式
- ✅ 改进错误处理和用户体验

### v1.0
- 初始版本，包含实体识别功能

## 📜 许可证

本项目仅供学习和研究使用。

## 🤝 贡献

欢迎提交Issue和Pull Request。

---

**技术支持**: 如遇问题请查看常见问题部分或提交Issue。
