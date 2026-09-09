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

在 PowerShell 中进入外层工作区根目录：

```powershell
cd extraction-main
```

首次使用时创建项目隔离环境并安装依赖：

```powershell
python -m venv extraction\.venv_api
.\extraction\.venv_api\Scripts\python.exe -m pip install -r extraction\requirements.txt
```

当前运行应始终使用 `.venv_api` 中的 Python，避免使用系统 Python 导致依赖版本混乱。

PaddleOCR 是可选增强依赖。未安装时程序会记录警告并继续运行，不影响远程多模态 API 主流程。`models/` 中保留的本地模型不会被当前 API 流程加载。

## API 配置

复制配置模板：

```powershell
Copy-Item extraction\config\vlm_api.example.yaml extraction\config\vlm_api.yaml
```

编辑 `extraction/config/vlm_api.yaml`：

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

真实配置文件 `extraction/config/vlm_api.yaml` 已被 Git 忽略，不要把 API Key 写入示例配置、代码、日志或提交记录。也可以通过 `VLM_API_KEY`、`EMBEDDING_API_KEY` 或 `DASHSCOPE_API_KEY` 环境变量提供密钥。

## 使用方法

1. 将待处理的 `.pdf` 和 `.pptx` 文件放入：

   ```text
   extraction/input/
   ```

2. 从 `extraction-main` 根目录启动：

   ```powershell
   cd extraction-main
   .\extraction\.venv_api\Scripts\python.exe .\extraction\run_src2.py
   ```

   `run_src2.py` 是当前唯一正式入口，不建议直接执行 `run_extraction.py` 或使用系统 Python。

3. 只测试一个输入文件时：

   ```powershell
   .\extraction\.venv_api\Scripts\python.exe .\extraction\run_src2.py `
     --input-file ".\extraction\input\example.pdf" `
     --user-id "user-001" `
     --username "teacher-zhang" `
     --course-id "course-001" `
     --course-name "机器人基础" `
     --job-id "job-001"
   ```

4. 与前端或任务调度层集成时，约定由调用方在任务输入目录生成 `job_manifest.json`。服务器任务进程可直接启动：

   ```powershell
   .\extraction\.venv_api\Scripts\python.exe .\extraction\run_src2.py `
     --job-manifest ".\frontend-backend\input\user-001\course-001\job-001\job_manifest.json"
   ```

   命令行显式参数的优先级高于 manifest。任务身份包含 `user_id`、`username`、`course_id`、`course_name` 和 `job_id`，并写入课程清单、文档、页面、内容节点、图谱节点、导出摘要及向量 metadata。当前改动只在 extraction 中提供该接口，尚未修改前端上传代码；旧的无参数启动方式仍然可用于本地批量测试。

每次启动都会创建 `extraction/output/{provider}_{model}_{递增序号}/`。新序号等于已有最大序号加一，因此不会覆盖、删除或混入旧结果。交付 JSON 中的文件路径也以 `extraction-main` 为基准，例如 `extraction/input/...` 和 `extraction/output/...`。

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
- PPTX 中的 EMF、WMF、SVG 素材依次尝试 ImageMagick、Inkscape 和 Wand 转换；单个素材仍失败时保留原矢量文件，并将对应幻灯片整页高分辨率渲染为兜底图片，避免流程图、控制框图和代码截图丢失。
- 疑似表格图片先放大短边至 1600 像素并做轻量对比度、锐度增强；首次结构校验未通过时才尝试旋转 90/270 度，已经通过的表格不会产生额外 API 调用。

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
| `Page` | 页面标题和清洗后的正文前 800 字 |
| `TextChunk` | 清洗文本、关键点和专业术语 |
| `Figure` | 图片标题和清洗后的模型描述；原始描述仍完整保留，但校徽、水印、邮箱、网址和相关性套话不进入向量 |
| `Formula` | 清洗后的公式名、LaTeX、公式含义、符号和适用条件 |
| `Table` | 清洗后的标题、表头、说明、单位和表格正文前 1500 字 |
| `CodeBlock` | 清洗后的编程语言、功能说明和代码前 1500 字 |

所有输入文档处理完成后，程序才会将节点批量提交给 `qwen3.7-text-embedding`，避免每处理一个文档就重复生成前面节点的向量并重复计费。

入库材料使用 `text_type=document`；查询阶段使用 `text_type=query` 和教学材料检索指令。返回的 1024 维向量会进行 L2 归一化，检索时向量点积等价于余弦相似度。API 失败时索引状态会写为 `failed`，不会静默回退到本地 Hash 向量。

## 输出目录

```text
extraction/output/
  images/                           # 早期版本遗留的公共图片目录，新流程不再写入
  qwen_qwen3.7-plus/                # 该模型的首次/旧版运行结果，保留用于历史对照
  qwen_qwen3.7-plus_N/              # 第 N 次独立运行；N 按已有最大序号加 1
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
    ex_items/                       # 图片理解失败或课程无关的人工审核队列
      *_metadata.json               # 审核原因、模型状态、原描述和来源信息
      *_enhanced.png                # 供前端展示及人工修正的图片预览
    formula_review_items/           # LaTeX残缺、不确定字符或低置信度公式
      *.json                        # 公式、验证问题、置信度和来源信息
      *.png                         # 可用时保存对应公式来源图
    table_review_items/             # 结构不完整或低置信度表格
      *.json                        # 表头、单元格、验证问题和来源信息
      *.png                         # 可用时保存对应表格来源图
    debug/                          # 人工复核和问题定位材料
      text/                         # 每页文本抽取结果
      images/                       # 图片、增强图、表格识别候选、整页渲染兜底图和理解 metadata
      formulas/                     # 公式识别结果
      tables/                       # 表格识别结果
      code/                         # 代码与 metadata
      logs/                         # 模型调用、错误及运行日志
      document_metadata/            # 每个输入文件的解析元数据
      professional_terms_library.json # 专业术语汇总
```

### 输出目录职责

| 目录 | 作用与生成方式 | 是否作为下游正式输入 |
| --- | --- | --- |
| `extraction/output/` | 所有模型、所有历史运行结果的总目录。程序只在这里新建本次运行目录，不清理旧结果。 | 否，下游应选择其中一次完整运行。 |
| `extraction/output/images/` | 早期版本曾使用的公共图片目录。当前流程已将图片放入各次运行的 `debug/images/`，该目录仅为历史兼容保留。 | 否。 |
| `extraction/output/{provider}_{model}/` | 某模型早期或首次运行的结果，例如 `qwen_qwen3.7-plus/`。目录名由 API 提供方和视觉模型名称组成。 | 可以，但应先确认是否为需要使用的历史版本。 |
| `extraction/output/{provider}_{model}_{N}/` | 当前标准运行目录，例如 `qwen_qwen3.7-plus_6/`。启动时扫描同模型现有目录，使用最大序号加 1 创建新目录，因此不会覆盖或混入上一次结果。 | 是，选定一次运行后读取其 `kg_data/`。 |
| `kg_data/` | 本轮通过校验、允许进入后续流程的正式结构化数据。包含文档、页面、多模态内容节点、结构关系、schema、统计摘要和向量索引。 | 是，知识图谱构建与检索程序的主要输入。 |
| `kg_data/vectors/` | 正式节点的向量数据。`vector_index.json` 保存节点、模型、维度、原料文本和向量行号；`vector_matrix.npy` 保存按相同顺序排列的 1024 维向量。 | 是，向量检索直接读取。 |
| `debug/` | 本轮完整的可读中间结果，用于人工抽查、错误定位、提示词调整和与原课件对照。内容可能包含尚未进入正式数据的过程信息。 | 否，不应整体导入知识图谱。 |
| `debug/text/` | PDF/PPTX 每一页的文本抽取与清洗结果，包括正文、关键点、专业术语和文本质量信息。由解析器和文本处理器生成。 | 默认否；正式文本节点已经汇总到 `kg_data/`。 |
| `debug/images/` | 从文档中拆出的原始图片、增强图片、表格放大/旋转候选、PPTX 矢量素材和整页渲染兜底图，以及每张图片的理解 metadata。由解析器、图片过滤器、表格预处理器和视觉 API 生成。 | 默认否；正式图片描述已汇总到 `kg_data/`，原图路径可供追溯。 |
| `debug/formulas/` | 按页保存全部公式候选及校验结果，通常同时提供 JSON 和 CSV，便于检查 LaTeX、置信度和来源。 | 默认否；只有通过校验的公式进入 `kg_data/`。 |
| `debug/tables/` | 按表保存原生解析或视觉识别得到的 JSON、CSV、表头、单元格和校验信息。 | 默认否；只有通过校验的表格进入 `kg_data/`。 |
| `debug/code/` | 保存识别出的源代码文件及对应 metadata，包括语言、来源图片、模型、描述和原始响应。 | 默认否；正式代码节点已经汇总到 `kg_data/`。 |
| `debug/logs/` | 本轮运行日志副本，记录文件进度、各 API 使用的模型、token、重试、超时和错误。 | 否，仅用于运行审计。 |
| `debug/document_metadata/` | 每个输入文件的解析元数据，包括类型、页数、文件路径、页面与图片映射等。 | 否，主要用于追溯和排错。 |
| `ex_items/` | 图片人工审核队列。API 调用失败、描述为空或模型判断为课程无关的图片会保存 metadata 和可用的预览图，供前端展示和人工修正。 | 否，人工确认前不会进入正式节点或向量。 |
| `formula_review_items/` | 公式人工审核队列。LaTeX 为空或残缺、括号不平衡、字符不确定、JSON 部分恢复、置信度不足等候选保存在这里。 | 否，审核通过并重新写回前不进入正式数据。 |
| `table_review_items/` | 表格人工审核队列。API 失败、行列不一致、合并表头无法可靠恢复、存在不确定单元格或置信度不足的表格保存在这里。 | 否，审核通过并重新写回前不进入正式数据。 |

下游构建知识图谱时，应先选定一个完整的 `extraction/output/{provider}_{model}_{N}/`，然后只把其中的 `kg_data/` 作为正式输入。`debug/` 和三个审核目录用于质量控制与人工修正；它们保留了更多过程信息，但不能未经筛选直接入图。部分目录在本轮没有对应内容时可能为空，这表示没有生成该模态或没有发现需要审核的项目，并不一定是运行失败。

## 检查运行结果

运行完成后优先检查：

1. `debug/logs/`：确认视觉模型和向量模型、API 调用状态、重试及错误。
2. `debug/images/`：确认图片过滤是否误删教学内容，是否仍残留水印或校徽。
3. `debug/formulas/`、`debug/tables/`、`debug/code/`：抽查结构和原文一致性。
4. `kg_data/knowledge_export_summary.json`：确认节点和向量数量。
5. `kg_data/vectors/vector_index.json`：确认 `status=success`、模型为 `qwen3.7-text-embedding`、维度为 1024，且条目数与 `vector_matrix.npy` 行数一致。

## 项目目录

```text
extraction/config/       API 配置模板和本地私密配置
extraction/input/        待处理的 PDF/PPTX
extraction/logs/         运行日志
extraction/models/       保留的本地模型文件；当前 API 主流程不加载
extraction/output/       按模型与递增序号保存的历次结果
extraction/src2/         当前抽取程序
extraction/run_src2.py   唯一正式运行入口
```
## API 失败补漏

完整抽取结束后，可以只扫描因超时、断网、代理异常、限流或服务端 5xx
造成的最终失败项。程序不会重试课程无关、低置信度或结构校验失败的数据，也
不会覆盖原运行结果：

```powershell
.\extraction\.venv_api\Scripts\python.exe .\extraction\retry_failed_api.py --dry-run
.\extraction\.venv_api\Scripts\python.exe .\extraction\retry_failed_api.py
```

默认选择最新的完整输出，也可以显式指定：

```powershell
.\extraction\.venv_api\Scripts\python.exe .\extraction\retry_failed_api.py --run-dir extraction/output/qwen_qwen3.7-plus_7
```

结果写入原运行目录下递增编号的
`api_recovery/retry_N/manifest.json` 和 `api_recovery/retry_N/items/`。这是补漏覆盖层，
保留原始失败证据；下游合并时应优先采用其中 `retry_status=success` 的记录。

## 断点续跑

程序在每个文档完整处理后更新一次 `kg_data/` 检查点。运行意外停止时，下面的
入口会选择最近修改且向量状态不是 `success` 的运行目录，恢复已经完成的文档，
跳过它们并从下一个文档继续；中断时尚未完成的文档会从头重新处理。旧结果目录
不会被删除，也不会创建一个混入历史结果的新编号目录。

```powershell
cd D:\extraction-main
.\extraction\.venv_api\Scripts\python.exe .\extraction\resume_extraction.py
```

显式指定目录更稳妥，尤其是同时存在多个未完成任务时：

```powershell
.\extraction\.venv_api\Scripts\python.exe .\extraction\resume_extraction.py --run-dir extraction/output/qwen_qwen3.7-plus_8
```

续跑依据当前输入文件的“文件名 + 扩展名”与 `documents.json` 对照。请勿在中断后
用同名文件替换原输入；面向网页的并发部署应改用任务 ID 和文件哈希进行恢复，
不能依赖“最新目录”。
