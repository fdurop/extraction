# 多模态课件内容抽取

本项目通过远程多模态大模型 API，将 PDF/PPTX 课件拆分并抽取为文本、图片、表格、公式和代码等结构化内容，同时生成供下游知识图谱与检索使用的数据文件。

## 运行方式

1. 创建虚拟环境并安装依赖：

   ```powershell
   python -m venv .venv_api
   .\.venv_api\Scripts\python.exe -m pip install -r requirements.txt
   ```

2. 参考 `config/vlm_api.example.yaml` 创建 `config/vlm_api.yaml`，填写 API Key 和模型名称。真实配置已被 Git 忽略。

3. 将 PDF/PPTX 放入 `input/`，从项目根目录运行：

   ```powershell
   .\.venv_api\Scripts\python.exe run_src2.py
   ```

可通过环境变量 `EXTRACTION_MAX_FILES=1` 限制单次处理文件数。

## 目录

```text
config/       API 配置模板；本地 vlm_api.yaml 不提交
input/        待处理课件；仅提交目录占位
logs/         运行日志；仅提交目录占位
models/       本地模型文件保留，但当前 API 流程不会加载
output/       历次抽取结果，每次使用模型名和递增序号创建新目录
src2/         当前抽取程序
run_src2.py   唯一运行入口
```

每次运行的正式输出位于 `output/{provider}_{model}_{序号}/kg_data/`，人工检查材料位于同次运行目录的 `debug/`。程序不会覆盖或删除此前的运行结果。

## 说明

- 当前主流程只调用配置的远程多模态 API，不加载本地 DeepSeek-VL 或 CLIP。
- PDF 使用 PyMuPDF 解析，PPTX 使用 python-pptx 解析，图片增强使用 Pillow。
- PaddleOCR 是可选增强项；未安装时程序会记录警告并继续运行。
