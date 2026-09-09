"""Resume the latest incomplete extraction run from a document checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


EXTRACTION_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = EXTRACTION_ROOT.parent
os.chdir(WORKSPACE_ROOT)
sys.path.insert(0, str(EXTRACTION_ROOT / "src2"))
sys.path.insert(0, str(EXTRACTION_ROOT))

from logger_config import LoggerSetup, get_logger  # noqa: E402
from multimodal_preprocessor import MultimodalPreprocessor  # noqa: E402
from utils import FileUtils  # noqa: E402


def _read_json(path: Path, default):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _is_incomplete(run_dir: Path) -> bool:
    summary = _read_json(run_dir / "kg_data" / "knowledge_export_summary.json", {})
    vector_status = summary.get("vectors", {}).get("status")
    return vector_status != "success"


def find_latest_incomplete_run(output_root: Path) -> Path:
    candidates = [
        path
        for path in output_root.iterdir()
        if path.is_dir() and (path / "kg_data" / "documents.json").exists()
    ]
    incomplete = [path for path in candidates if _is_incomplete(path)]
    if not incomplete:
        raise FileNotFoundError("没有找到可续跑的未完成结果目录")
    return max(incomplete, key=lambda path: path.stat().st_mtime)


def _document_key(name: str, file_type: str) -> tuple[str, str]:
    return (str(name).strip().casefold(), str(file_type).lstrip(".").strip().casefold())


def completed_document_keys(run_dir: Path) -> set[tuple[str, str]]:
    documents = _read_json(run_dir / "kg_data" / "documents.json", [])
    return {
        _document_key(item.get("file_name", ""), item.get("file_type", ""))
        for item in documents
        if item.get("file_name") and item.get("file_type")
    }


def input_key(path: str) -> tuple[str, str]:
    file_path = Path(path)
    return _document_key(file_path.stem, file_path.suffix)


def parse_args():
    parser = argparse.ArgumentParser(description="继续最近一次未完成的多模态抽取任务")
    parser.add_argument(
        "--run-dir",
        help="显式指定结果目录，例如 extraction/output/qwen_qwen3.7-plus_8",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_root = EXTRACTION_ROOT / "output"
    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.is_absolute():
            run_dir = WORKSPACE_ROOT / run_dir
        run_dir = run_dir.resolve()
    else:
        run_dir = find_latest_incomplete_run(output_root)

    if not (run_dir / "kg_data" / "documents.json").exists():
        raise FileNotFoundError(f"结果目录缺少检查点: {run_dir}")

    completed = completed_document_keys(run_dir)
    input_dir = EXTRACTION_ROOT / "input"
    input_files = sorted(
        FileUtils.get_files(str(input_dir), extensions=[".pdf", ".pptx"]),
        key=lambda value: Path(value).name.casefold(),
    )
    pending = [path for path in input_files if input_key(path) not in completed]

    print("\n" + "=" * 60)
    print("多模态抽取断点续跑")
    print("=" * 60)
    print(f"恢复目录: {run_dir.relative_to(WORKSPACE_ROOT)}")
    print(f"输入文件: {len(input_files)} 个")
    print(f"已完成:   {len(completed)} 个")
    print(f"待处理:   {len(pending)} 个")

    processor = MultimodalPreprocessor(
        output_base_dir=str(run_dir.relative_to(WORKSPACE_ROOT)),
        resume=True,
    )
    logger = get_logger("Resume")
    LoggerSetup.log_session_start(logger)

    for index, file_path in enumerate(pending, 1):
        print(f"\n[{index}/{len(pending)}] 续跑处理: {Path(file_path).name}")
        logger.info("续跑处理文件 [%s/%s]: %s", index, len(pending), file_path)
        processor.process_file(file_path)

    if processor.professional_terms:
        processor.output_manager.save_professional_terms(processor.professional_terms)

    # Also rebuild when every document was done but the original run stopped
    # immediately before vector generation.
    processor.output_manager.finalize()

    log_file = LoggerSetup.get_log_file_path()
    if log_file and os.path.exists(log_file):
        log_dir = run_dir / "debug" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(log_file, log_dir / Path(log_file).name)

    print("\n续跑完成，正式数据和向量索引已刷新。")
    print(f"结果目录: {run_dir.relative_to(WORKSPACE_ROOT)}")


if __name__ == "__main__":
    main()
