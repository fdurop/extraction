# 教学助手技术路线与检索生成方案

更新日期：2026-08-14

## 1. 项目目标与边界

本项目面向多用户、多课程的教学资料管理。用户上传 PDF/PPTX 后，系统先抽取文本、图片、表格、公式和代码等多模态证据，再从证据中构建以知识点为核心的课程知识图谱，最后通过文本、视觉、关键词和图结构的混合检索，为大模型提供可追溯证据并生成回答。

项目不应把“文档—页面—元素”的目录树直接当成最终知识图谱。Document、Page、TextChunk、Figure、Table、Formula、CodeBlock 是来源与证据层；Concept/Entity 才是知识层的主要节点。页码、文件名和原始资源路径用于溯源，不应取代知识点之间的语义关系。

## 2. 当前已完成状态

当前抽取入口为 `run_src2.py`，主要流程位于 `src2/`：

1. 使用 PyMuPDF 和 python-pptx 解析 PDF/PPTX。
2. 过滤尺寸过小、重复、水印和装饰性图片。
3. 调用配置的远程多模态 API 理解图片、公式、代码和表格。
4. 保留 `debug/` 人工检查数据，并在 `kg_data/` 输出稳定的下游交付格式。
5. 每次运行创建 `output/{provider}_{model}_{递增序号}/`，不覆盖以前的结果。

最新样例 `output/qwen_qwen3.7-plus_1/kg_data/` 包含 4 个文档、60 页、112 个内容单元和 172 个可索引节点。其中有 60 个 TextChunk、41 个 Figure、8 个 Formula、3 个 CodeBlock，当前样例没有 Table。已有 232 条关系全部属于课程、文档、页面和内容的结构关系，还没有实际生成 Entity/Concept 节点及知识关系。

当前 `vectors/` 使用的是 `hash_text_v1`：将英文词、中文字符和中文二元组哈希到 384 维向量。它是为了固定输出格式而提供的轻量占位索引，不是语义 embedding，不能作为最终检索质量基线。当前 60 个 Page 的 `page_image_path` 均为空，因此现在只能对抽出的 Figure 图片建立视觉索引，尚不能直接实现 ColPali/VisRAG 式整页视觉检索。

## 3. 最终总体技术路线

```text
PDF/PPTX
  -> 多模态内容抽取（当前 extraction）
  -> 质量校验与证据规范化
  -> 知识点/实体/关系抽取与消歧
  -> 概念知识图谱 + 证据图
  -> BM25 文本索引 + 语义向量索引 + 视觉向量索引
  -> 查询分类与多路并行召回
  -> 排名融合 + 图扩展 + 重排
  -> 多模态证据组装
  -> Qwen 文本模型或视觉模型生成带来源回答
```

建议把系统分成三个可独立重跑的阶段：

- `extraction`：只负责忠实抽取和描述多模态证据，不判断最终知识关系。
- `graph_builder`：从证据中抽取知识点、统一同义词、建立知识关系并连接证据。
- `retrieval_generation`：建立索引、执行混合检索、组织证据和生成回答。

阶段之间通过版本化 JSON 契约连接。每个阶段记录输入版本、模型、提示词版本、时间和错误状态，避免某次模型升级后必须重新处理所有原始文件。

## 4. 抽取层还需补齐的交付

### 4.1 整页图像

为每个 Page 生成稳定的整页 PNG/JPEG，并填写 `pages.json.page_image_path`：

- PDF 直接按固定 DPI（建议 144–200 DPI）渲染整页。
- PPTX 优先通过 LibreOffice 转 PDF 后渲染，保证文字、公式和图形的组合布局不丢失。
- Figure 继续保留裁剪图；整页图和局部图同时建立视觉索引，但不能互相替代。

### 4.2 表格与证据质量

当前样例 Table 数为 0，需要先确认课件确实没有表格，还是表格检测漏检。表格至少保留原图、单元格结构、Markdown/HTML 表示和自然语言摘要。公式保留 LaTeX 与原图，代码保留源码、语言和原图，任何 API 失败项均应有 `status/api_error/indexable`，不能静默生成空节点。

### 4.3 稳定身份与增量处理

Document、Page、Evidence ID 应由 `course_id + 文件内容哈希 + 页码 + 元素位置/序号` 稳定生成。新增或修改文件时只重建受影响文档、知识点和索引，未变化的课程内容直接复用缓存。

## 5. 知识图谱构建

### 5.1 输入

`graph_builder` 只读取每次运行的 `kg_data/`：

- `course_manifest.json`：用户和课程边界。
- `documents.json`、`pages.json`：来源、顺序和页码。
- `content_list.json`、`multimodal_nodes.json`：多模态证据及其描述。
- `relationships.json`：已有目录结构关系。
- 原图、公式图、表格图和代码文件：用于证据回看与多模态重排。

### 5.2 两层图结构

**知识层**以 Concept/Entity 为核心，建议至少包含：

- `IS_A`：上下位或类型关系。
- `PART_OF`：组成关系。
- `PREREQUISITE_OF`：先修关系。
- `CAUSES`、`AFFECTS`：因果或影响关系。
- `IMPLEMENTS`：算法、代码或部件实现某概念。
- `DERIVED_FROM`：公式或结论的推导关系。
- `COMPARES_WITH`、`RELATED_TO`：对比和一般关联。

**证据层**保留 Course、Document、Page 和各类内容节点，主要关系为：

- `HAS_EVIDENCE`：Concept -> TextChunk/Figure/Table/Formula/CodeBlock。
- `LOCATED_IN`：Evidence -> Page。
- `PAGE_OF`、`DOCUMENT_OF`、`NEXT_PAGE`：来源和顺序。
- `EXPLAINS`、`ILLUSTRATES`、`DEMONSTRATES`：可在质量足够时替代笼统的 HAS_EVIDENCE。

页码应该同时保留在 Evidence 和关系属性中，Concept 本身不要绑定单一页，因为同一知识点可能跨页、跨文件出现。

### 5.3 构建过程

1. 按页及相邻页组合证据，使用 LLM 输出严格 JSON 的候选 Concept、Entity、关系和证据 ID。
2. 对候选名称做字符串规范化、别名归并和语义相似度聚类；相似但不确定的概念不自动合并，进入人工检查列表。
3. 使用课程内已有概念词表约束第二轮抽取，减少同一知识点被重复命名。
4. 只保留能回指至少一个 Evidence 的知识节点和关系；每条关系记录 `confidence`、`source_evidence_ids` 和 `extractor_version`。
5. 同一课程跨文件的相同概念合并为一个 Concept，通过多个 HAS_EVIDENCE 自然形成跨文件连接。
6. 对 Concept 图运行 Leiden/Louvain 社区发现，离线生成章节/主题摘要，供全局问题使用；不必每次问答都重新总结。

## 6. 索引设计

不要把大向量直接存入知识图节点。图数据库保存 `embedding_id/visual_embedding_id`，向量库保存向量和同一个 `node_id/evidence_id`，以此对齐。

### 6.1 关键词索引

对 Concept 名称、别名、TextChunk、公式变量、代码标识符和表格表头建立 BM25 索引。它对准确术语、缩写、函数名、变量名和公式符号非常重要，不能被语义向量完全替代。

### 6.2 文本语义索引

第一选择为开源 `Qwen3-Embedding-0.6B`，对中文、代码和长文本较友好，参数规模可控；利用 MRL 将输出统一为 512 或 1024 维。备选为 `BAAI/bge-m3`，它可同时提供 dense、sparse 和 multi-vector 表示。

以下内容分别建立向量，不要把整门课合成一个长向量：

- Concept：名称 + 别名 + 定义 + 关系摘要。
- TextChunk：标题 + 正文 + 专业术语。
- Figure/Table/Formula/CodeBlock：类型标签 + 精确识别结果 + 描述 + 页面上下文。
- Page：标题 + 页面摘要 + 页面正文，用作父级召回和结果聚合。

### 6.3 视觉索引

建议第一版使用开源 `Qwen3-VL-Embedding-2B` 对整页图和 Figure 裁剪图生成单向量。该模型支持文本、图片及混合输入，模型约 2B 参数、权重约 4.27 GB，工程上比多向量 late interaction 更容易接入普通向量库。

同时把 `ColQwen2-v1.0` 作为实验对照。它继承 ColPali 的多向量 late-interaction 思路，对复杂版式检索有很强的论文与 ViDoRe 支持，但索引体积、计算和向量库实现复杂度更高。若时间不足，不应把 ColPali/ColQwen 设为第一版唯一方案。

视觉向量的 metadata 至少包含 `user_id/course_id/document_id/page_id/evidence_id/modality/image_path/model/version`。检索必须先按用户和课程过滤，防止不同用户之间的数据越权召回。

## 7. 检索流程

### 7.1 查询分析

先识别以下信息，但不要让查询改写成为硬依赖：

- 用户和课程范围。
- 问题类型：知识点事实、关系/多跳、代码、公式、表格、视觉图示或全课程总结。
- 关键词、候选概念、代码符号和公式变量。
- 是否必须返回图片或跨文件证据。

### 7.2 多路召回

第一版建议并行执行：

1. BM25：Top 40，保证术语和符号命中。
2. 文本 dense retrieval：Top 40，召回语义相近的 Concept 与 Evidence。
3. 视觉 retrieval：视觉型问题或文本召回置信度低时取 Top 20；对关键教学问题也可常开 Top 10。
4. 图检索：用命中的 Concept 作为种子，限制在 1–2 跳内扩展关系、邻接 Concept 和 HAS_EVIDENCE，设置最大节点数，避免图爆炸。

对于“这门课主要讲什么”一类全局问题，读取预生成的社区摘要并使用 map-reduce；对于普通知识点问题，采用局部图搜索，不需要扫描全图。

### 7.3 融合与重排

不同检索器分数不可直接相加，建议先使用加权 Reciprocal Rank Fusion（RRF）：

```text
score(d) = Σ_m weight_m / (60 + rank_m(d))
```

默认权重可从 `BM25 1.0 / text dense 1.2 / visual 1.0 / graph 1.1` 开始，再通过验证集调整。视觉问题提高 visual 权重，代码/公式问题提高 BM25 和对应模态节点权重。

融合后取 Top 20–30 重排：

- 纯文本候选使用 `Qwen3-Reranker-0.6B` 或同级 API reranker。
- 含关键图片且预算允许时，只对前 10–20 个候选调用 `Qwen3-VL-Reranker-2B` 或 VLM 相关性判断。
- 最终保留约 6–12 条互补证据，并限制每页、每文档的最大数量，避免同一页重复内容占满上下文。

### 7.4 图扩展顺序

推荐“先召回种子，后扩图”，而不是先遍历整张图：

1. 文本/视觉/BM25 找到高置信 Concept 或 Evidence。
2. Evidence 通过 HAS_EVIDENCE 反查 Concept；Concept 在白名单关系上扩展 1–2 跳。
3. 为扩展到的 Concept 取最高质量证据。
4. 将扩展结果与原始召回结果一起重排。

这样图负责补足上下文、跨页和跨文件关系，向量负责找到入口，二者不会互相替代。

## 8. 证据组装与回答生成

最终上下文不是简单拼接节点描述，而是按 Concept 和 Page 分组的证据包：

```json
{
  "concepts": ["时间中断", "Timer0"],
  "relations": [{"source": "Timer0", "type": "IMPLEMENTS", "target": "时间中断"}],
  "evidence": [
    {
      "evidence_id": "code_xxx",
      "type": "CodeBlock",
      "document": "w10-1-直线运动平台（下）.pptx",
      "page_no": 7,
      "text": "...",
      "asset_path": "..."
    }
  ]
}
```

生成器选择：

- 最终证据只有文本、代码和 LaTeX 时，调用成本较低的文本模型。
- 最终证据包含必须查看的图、图表、复杂表格或页面布局时，调用 Qwen 视觉模型，并把原图而不只是图片描述传入。

系统提示词应要求：只根据证据回答；每个关键结论标注 `[文件名 p.页码]`；证据不足时明确说明；代码、公式和表格不得凭空补值。回答接口同时返回 `answer`、`citations`、`retrieved_node_ids`、`retrieval_scores` 和 `trace_id`，方便前端展示来源和后续评测。

## 9. 推荐的最小实现顺序

### P0：形成可评测基线

1. 补齐整页图片和 `page_image_path`。
2. 将 `hash_text_v1` 替换为真实文本 embedding。
3. 实现 Concept/Entity 抽取、同义词归并和 HAS_EVIDENCE。
4. 实现 BM25 + dense + 一跳图扩展 + RRF。
5. 生成带文件名、页码和 Evidence ID 的回答。

### P1：加入真正多模态检索

1. 使用 Qwen3-VL-Embedding-2B 建立 Page/Figure 视觉索引。
2. 加入查询类型路由和模态权重。
3. 实现同页证据聚合、跨文件 Concept 合并和多样性去重。
4. 建立人工标注测试集并完成消融实验。

### P2：提升复杂问题能力

1. 图社区检测与课程级摘要，用于全局问题。
2. 视觉 reranker 或 ColQwen2 late-interaction 对照实验。
3. 增量索引、缓存、失败恢复和运行成本统计。
4. 个性化学习路径可在核心问答稳定后再接入，不应阻塞检索主线。

## 10. 评测方案

从实际课程资料人工构造至少 80–150 个问题，并标注答案、证据页和问题模态：文本、图片、表格、公式、代码、跨页、跨文件、关系推理和全局总结。

检索指标：`Recall@5/10`、`MRR@10`、`nDCG@5/10`、证据页命中率、跨文件命中率。生成指标：答案正确率、关键事实覆盖率、引用准确率、无证据拒答率，并抽样人工评价忠实性。

至少比较以下消融版本：

1. 当前 hash 索引。
2. BM25。
3. 文本 dense。
4. BM25 + dense。
5. BM25 + dense + Graph。
6. BM25 + dense + Graph + visual。
7. 完整方案 + reranker。

ViDoRe/ViDoRe V2 可用于检查视觉检索模型的通用能力，但最终结论必须以你们自己的中文课件测试集为主，因为教学资料、代码、公式和中文术语分布与公开基准不同。

## 11. 成本与可持续运行

- 抽取、知识点抽取和社区摘要是离线索引成本；同一内容按文件哈希缓存，不重复调用 API。
- 文本 embedding 优先使用 0.6B 开源模型本地批处理；若部署资源不足再切换 API，接口保持一致。
- 视觉 embedding 只在上传或文件变化时计算；问答阶段只编码查询，不重复编码文档。
- 视觉 reranker 按查询路由触发，不对所有问题调用。
- Graph 全局摘要只在课程图变化后重建；普通问答使用局部检索。
- 每次运行记录模型、请求数、输入/输出 token、耗时、缓存命中率和失败次数，形成可向老师展示的预算报告。

## 12. 研究与开源依据

以下来源均为论文、作者团队或官方开源项目：

1. Edge et al., [From Local to Global: A Graph RAG Approach to Query-Focused Summarization](https://arxiv.org/abs/2404.16130), 2024。支持实体图、社区摘要以及局部/全局 GraphRAG 的设计；[Microsoft GraphRAG 官方仓库](https://github.com/microsoft/graphrag)提供工程参考。
2. Guo et al., [LightRAG: Simple and Fast Retrieval-Augmented Generation](https://arxiv.org/abs/2410.05779), 2024。支持图结构与向量表示结合、低层与高层检索以及增量更新。
3. Guo et al., [RAG-Anything: All-in-One RAG Framework](https://arxiv.org/abs/2510.12323), 2025；[官方仓库](https://github.com/HKUDS/RAG-Anything)。支持把多模态内容视为互联知识实体、双图构建和跨模态混合检索。本项目借鉴思想与数据契约，不要求完整安装该框架。
4. Faysse et al., [ColPali: Efficient Document Retrieval with Vision Language Models](https://arxiv.org/abs/2407.01449), 2024；[官方 ColPali/ColQwen 仓库](https://github.com/illuin-tech/colpali)。支持直接对文档页面图像做多向量 embedding 与 late interaction 检索。
5. Yu et al., [VisRAG: Vision-based Retrieval-augmented Generation on Multi-modality Documents](https://arxiv.org/abs/2410.10594), 2024；[官方仓库](https://github.com/OpenBMB/VisRAG)。支持整页视觉检索和把原始页面图交给 VLM 生成，减少解析造成的信息损失。
6. Cho et al., [M3DocRAG: Multi-modal Retrieval is What You Need for Multi-page Multi-document Understanding](https://arxiv.org/abs/2411.04952), 2024。支持面向多页、多文档及视觉证据的检索生成，也是跨文件课程问答的重要依据。
7. Macé et al., [ViDoRe Benchmark V2: Raising the Bar for Visual Retrieval](https://arxiv.org/abs/2505.17166), 2025；[官方评测仓库](https://github.com/illuin-tech/vidore-benchmark)。为视觉文档检索提供 nDCG 等公开评测方法和更复杂的跨文档场景。
8. Zhang et al., [Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models](https://arxiv.org/abs/2506.05176), 2025；[官方仓库](https://github.com/QwenLM/Qwen3-Embedding)。支持 0.6B–8B 多语言文本 embedding/reranker、长文本和可变向量维度。
9. Qwen Team, [Qwen3-VL-Embedding and Qwen3-VL-Reranker](https://arxiv.org/abs/2601.04720), 2026；[官方仓库](https://github.com/QwenLM/Qwen3-VL-Embedding)。支持文本、图片、截图、视频和混合输入的多模态向量与重排，适合作为当前优先视觉索引方案。
10. Chen et al., [BGE M3-Embedding](https://arxiv.org/abs/2402.03216), 2024；[FlagEmbedding 官方仓库](https://github.com/FlagOpen/FlagEmbedding)。支持多语言、dense/sparse/multi-vector 和长文本，是文本检索的备选方案。
11. Bruch et al., [An Analysis of Fusion Functions for Hybrid Retrieval](https://arxiv.org/abs/2210.11934), 2022。系统分析了 lexical 与 semantic 混合检索中的加权融合和 RRF，为多路排名融合提供依据。
12. Abdelmagied et al., [Leveraging Graph Retrieval-Augmented Generation to Support Learners' Understanding of Knowledge Concepts in MOOCs](https://arxiv.org/abs/2505.10074), 2025。提供 GraphRAG 与教育知识图谱结合用于课程概念问答和学习引导的直接研究参考。

## 13. 最终决策摘要

本项目应采用“概念知识图谱 + 多模态证据层 + 多索引混合检索”，而不是在纯目录图和纯向量 RAG 之间二选一。近期可落地的默认组合是：

```text
图数据库：现有图数据库（Concept 为中心，Evidence 可追溯）
关键词索引：BM25
文本向量：Qwen3-Embedding-0.6B（512/1024 维）
视觉向量：Qwen3-VL-Embedding-2B（Page + Figure）
融合：加权 RRF
图扩展：种子节点后 1–2 跳受限扩展
重排：Qwen3-Reranker-0.6B；视觉重排按需启用
生成：有视觉证据用 Qwen VLM，否则用成本较低的文本模型
输出：答案 + 文件/页码引用 + evidence/node ID + 检索轨迹
```

ColQwen2/ColPali 保留为对照实验和后续增强方案；RAG-Anything 作为架构参考，而不是必须整体安装的运行依赖。
