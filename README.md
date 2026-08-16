# 多模态课件内容抽取

本项目负责教学助手的前置知识抽取阶段：将用户上传的 PDF、PPTX 课件解析为文本、图片、表格、公式和代码等结构化内容，并生成供下游知识点抽取、知识图谱构建和混合检索使用的统一数据与语义向量。

本项目不直接完成知识点实体抽取和完整知识图谱构建。`kg_data` 中的 `Page`、`TextChunk`、`Figure`、`Table`、`Formula`、`CodeBlock` 是带来源信息的内容证据节点，下游还需要从这些内容中抽取 `Concept/Entity`，完成实体归一、关系抽取和证据关联。

## 核心技术路线

```text
PDF/PPTX
  -> 文档解析与页面拆分
  -> 文本、图片、原生表格等内容提取
  -> 无效图片、重复图片、校徽和水印过滤
  -> 多模态 API 理解图片、公式、表格和代码
  -> 生成统一内容节点及结构关系
  -> 文本 Embedding API 生成语义向量
  -> 输出 debug 人工检查数据和 kg_data 下游交付数据
```

当前采用两套相互独立的 API 配置：

| API | 当前模型 | 主要作用 | 选择原因 |
| --- | --- | --- | --- |
| 多模态理解 API | `qwen3.7-plus` | 理解图片中的教学内容，并辅助识别公式、表格和代码 | 中文教学材料理解较好；支持图文输入；使用远程 API 可以避免本地部署大型视觉语言模型带来的显存、内存和维护成本 |
| 文本向量 API | `qwen3.7-text-embedding` | 将各类内容节点的 `embedding_text` 转换为 1024 维语义向量 | 专门面向检索；支持区分 `document` 与 `query`；相比本地 Hash 向量能够表达语义相关性，1024 维兼顾检索质量、存储量和计算开销 |

两套 API 可以使用不同的 API Key。当前如果 `embedding.api_key` 留空，会复用 `vlm.api_key`；图片理解与向量生成仍然是两个独立接口、两个独立模型，并分别记录模型来源和调用状态。

## 环境安装

在 PowerShell 中进入项目目录：

```powershell
cd D:\extraction-main\extraction
```

首次使用时创建项目隔离环境并安装依赖：

```powershell
python -m venv .venv_api
.\.venv_api\Scripts\python.exe -m pip install -r requirements.txt
```

当前运行应始终使用 `.venv_api` 中的 Python，避免使用系统 Python 导致依赖版本混乱。

PaddleOCR 是可选增强依赖。未安装时程序会记录警告并继续运行，不影响远程多模态 API 主流程。`models/` 中保留的本地模型不会被当前 API 流程加载。

## API 配置

复制配置模板：

```powershell
Copy-Item config\vlm_api.example.yaml config\vlm_api.yaml
```

编辑 `config/vlm_api.yaml`：

```yaml
vlm:
  enabled: true
  provider: qwen
  api_key: "填写百炼 API Key"
  model: qwen3.7-plus
  base_url: https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions
  timeout_seconds: 120
  retry_count: 2
  max_tokens: 700
  temperature: 0.1
  image_detail: auto

embedding:
  enabled: true
  provider: qwen
  api_key: ""  # 留空时复用 vlm.api_key，也可以填写独立 Key
  model: qwen3.7-text-embedding
  base_url: https://dashscope.aliyuncs.com/api/v1
  dimension: 1024
  batch_size: 10
  timeout_seconds: 120
  retry_count: 2
  output_type: dense
  query_instruction: "Retrieve relevant evidence from Chinese educational course materials."
```

真实配置文件 `config/vlm_api.yaml` 已被 Git 忽略，不要把 API Key 写入示例配置、代码、日志或提交记录。也可以通过 `VLM_API_KEY`、`EMBEDDING_API_KEY` 或 `DASHSCOPE_API_KEY` 环境变量提供密钥。

## 使用方法

1. 将待处理的 `.pdf` 和 `.pptx` 文件放入：

   ```text
   D:\extraction-main\extraction\input\
   ```

2. 从项目根目录启动：

   ```powershell
   cd D:\extraction-main\extraction
   .\.venv_api\Scripts\python.exe run_src2.py
   ```

   `run_src2.py` 是当前唯一正式入口，不建议直接执行 `run_extraction.py` 或使用系统 Python。

3. 只测试一个输入文件时：

   ```powershell
   $env:EXTRACTION_MAX_FILES = "1"
   .\.venv_api\Scripts\python.exe run_src2.py
   Remove-Item Env:\EXTRACTION_MAX_FILES
   ```

每次启动都会创建 `output/{provider}_{model}_{递增序号}/`。新序号等于已有最大序号加一，因此不会覆盖、删除或混入旧结果。

## 完整处理流程

### 1. 文档解析

- PDF 使用 PyMuPDF 按页解析。
- PPTX 使用 `python-pptx` 按幻灯片解析。
- 提取页面文本、内嵌图片、原生表格、页码、文件类型和文档元数据。
- 保存页面、文档及原始内容之间的来源映射。

### 2. 图片预处理

- 过滤尺寸过小、纯色、近似空白和明显无意义的图片。
- 根据重复位置、重复内容和版式特征过滤校徽、水印及装饰元素。
- 对保留图片进行必要的增强，原图和增强图保留在 `debug/images/` 供人工复核。

### 3. 多模态内容理解

`qwen3.7-plus` 接收图片及页面上下文，生成适合教学材料的中文描述。根据内容类型进一步识别：

- 图片：主题、关键对象、流程、结构、图中文字和教学含义。
- 公式：LaTeX、公式名称、变量含义和适用条件。
- 表格：表头、行列内容、单位、主题和关键结论。
- 代码：语言、源码、功能说明和关键逻辑。

API 生成结果会保存 `provider`、`model`、`backend`、`status` 和错误信息。接口失败时不会伪装成成功的本地模型结果。

### 4. 统一结构化输出

程序把抽取结果统一为 `Page`、`TextChunk`、`Figure`、`Table`、`Formula` 和 `CodeBlock` 等内容节点，同时生成课程包含文档、文档包含页面、页面前后顺序和页面包含内容节点等关系。

这些关系主要描述文档结构和证据来源，不等同于最终知识点之间的语义关系。

### 5. 语义向量生成

不同节点先生成 `embedding_text`：

| 节点 | 当前向量原料 |
| --- | --- |
| `Page` | 页面标题、页面摘要和正文前 500 字 |
| `TextChunk` | 清洗文本、关键点和专业术语 |
| `Figure` | 多模态模型生成的图片内容描述 |
| `Formula` | LaTeX 与公式含义 |
| `Table` | 表头、说明和表格正文前 1500 字 |
| `CodeBlock` | 编程语言、功能说明和代码前 1500 字 |

所有输入文档处理完成后，程序才会将节点批量提交给 `qwen3.7-text-embedding`，避免每处理一个文档就重复生成前面节点的向量并重复计费。

入库材料使用 `text_type=document`；查询阶段使用 `text_type=query` 和教学材料检索指令。返回的 1024 维向量会进行 L2 归一化，检索时向量点积等价于余弦相似度。API 失败时索引状态会写为 `failed`，不会静默回退到本地 Hash 向量。

## 输出目录

```text
output/
  qwen_qwen3.7-plus_1/              # 某次独立运行，不覆盖旧结果
    kg_data/                        # 下游知识图谱与检索正式输入
      course_manifest.json          # 课程、用户、文档清单和 schema 版本
      documents.json                # Document 数据
      pages.json                    # Page 数据及页面顺序
      content_list.json             # 页面内细粒度内容单元
      multimodal_nodes.json         # 统一多模态内容节点
      relationships.json            # 文档结构与内容归属关系
      schema.json                   # 节点、关系和文件结构定义
      knowledge_export_summary.json # 节点数、向量状态和调用摘要
      vectors/
        vector_index.json           # embedding_id、node_id、模型、维度和原料文本
        vector_matrix.npy           # 与 vector_index.json 顺序对应的向量矩阵
    debug/                          # 人工复核和问题定位材料
      text/                         # 每页文本抽取结果
      images/                       # 图片、增强图和图片理解 metadata
      formulas/                     # 公式识别结果
      tables/                       # 表格识别结果
      code/                         # 代码与 metadata
      logs/                         # 模型调用、错误及运行日志
      document_metadata/            # 每个输入文件的解析元数据
      professional_terms_library.json # 专业术语汇总
```

下游构建知识图谱时应优先读取 `kg_data/`；`debug/` 用于人工抽查识别精度、定位错误和调整提示词，不应作为主要图谱输入。

## 检查运行结果

运行完成后优先检查：

1. `debug/logs/`：确认视觉模型和向量模型、API 调用状态、重试及错误。
2. `debug/images/`：确认图片过滤是否误删教学内容，是否仍残留水印或校徽。
3. `debug/formulas/`、`debug/tables/`、`debug/code/`：抽查结构和原文一致性。
4. `kg_data/knowledge_export_summary.json`：确认节点和向量数量。
5. `kg_data/vectors/vector_index.json`：确认 `status=success`、模型为 `qwen3.7-text-embedding`、维度为 1024，且条目数与 `vector_matrix.npy` 行数一致。

## 项目目录

```text
config/       API 配置模板和本地私密配置
input/        待处理的 PDF/PPTX
logs/         运行日志
models/       保留的本地模型文件；当前 API 主流程不加载
output/       按模型与递增序号保存的历次结果
src2/         当前抽取程序
run_src2.py   唯一正式运行入口
```
