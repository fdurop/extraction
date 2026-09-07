"""Retry transient VLM failures from a completed extraction run.

The original run is never modified. Recovered results are written as an
overlay under ``api_recovery/retry_N`` so they can be inspected or merged by a
later export step without losing the original evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parent
SRC2_DIR = PROJECT_ROOT / "src2"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(SRC2_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from core import FormulaValidator, ImageProcessor, TableValidator  # noqa: E402
from utils import create_vlm_client  # noqa: E402
from logger_config import get_logger  # noqa: E402


TRANSIENT_ERROR_MARKERS = (
    "timed out",
    "timeout",
    "connection",
    "proxyerror",
    "remote end closed",
    "max retries exceeded",
    "temporarily unavailable",
    "connection reset",
    "connection aborted",
    "name resolution",
    "dns",
    "http 429",
    "http 500",
    "http 502",
    "http 503",
    "http 504",
    "rate limit",
)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def project_path(value: Any) -> Optional[Path]:
    if not value:
        return None
    path = Path(str(value).replace("/", os.sep))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def is_transient_error(value: Any) -> bool:
    text = str(value or "").lower()
    return bool(text) and any(marker in text for marker in TRANSIENT_ERROR_MARKERS)


def next_retry_dir(run_dir: Path) -> Path:
    root = run_dir / "api_recovery"
    root.mkdir(parents=True, exist_ok=True)
    indices = []
    for item in root.iterdir():
        match = re.fullmatch(r"retry_(\d+)", item.name) if item.is_dir() else None
        if match:
            indices.append(int(match.group(1)))
    result = root / f"retry_{max(indices, default=0) + 1}"
    (result / "items").mkdir(parents=True, exist_ok=False)
    return result


def latest_completed_run() -> Path:
    output_root = PROJECT_ROOT / "output"
    candidates = [
        path for path in output_root.iterdir()
        if path.is_dir() and (path / "kg_data" / "knowledge_export_summary.json").exists()
    ]
    if not candidates:
        raise RuntimeError("output 中没有找到已完成且带 knowledge_export_summary.json 的运行目录")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def page_context(run_dir: Path, document: str, page_num: int) -> str:
    path = run_dir / "debug" / "text" / f"{document}_page_{page_num}.json"
    if not path.exists():
        return ""
    data = load_json(path)
    return str(data.get("cleaned_text") or data.get("processed_text") or data.get("raw_text") or "")[:1200]


def source_identity(path: Path, data: Dict[str, Any]) -> tuple[str, int, Optional[int]]:
    document = str(data.get("source_document") or "")
    page_num = int(data.get("page_num") or data.get("page") or 0)
    image_index = data.get("image_index")

    match = re.match(r"(.+)_slide_(\d+)_img_(\d+)_metadata$", path.stem)
    if match:
        document = document or match.group(1)
        page_num = page_num or int(match.group(2))
        image_index = int(image_index or match.group(3))
    return document, page_num, int(image_index) if image_index is not None else None


def candidate(
    operation: str,
    metadata_path: Path,
    data: Dict[str, Any],
    source_image: Any,
    error: Any,
    document: str,
    page_num: int,
    image_index: Optional[int],
) -> Optional[Dict[str, Any]]:
    image_path = project_path(source_image)
    if not image_path or not image_path.exists() or not is_transient_error(error):
        return None
    return {
        "operation": operation,
        "metadata_path": str(metadata_path.relative_to(PROJECT_ROOT)).replace(os.sep, "/"),
        "source_image": str(image_path.relative_to(PROJECT_ROOT)).replace(os.sep, "/"),
        "api_error": str(error),
        "source_document": document,
        "page_num": page_num,
        "image_index": image_index,
    }


def collect_candidates(run_dir: Path) -> List[Dict[str, Any]]:
    found: List[Dict[str, Any]] = []

    for path in (run_dir / "debug" / "images").glob("*_metadata.json"):
        data = load_json(path)
        if data.get("review_reason") != "api_failed" and data.get("status") not in {
            "api_failed", "no_model_succeeded", "failed"
        }:
            continue
        document, page_num, image_index = source_identity(path, data)
        item = candidate(
            "image_description", path, data,
            data.get("enhanced_path") or data.get("original_path"),
            data.get("api_error") or data.get("error"),
            document, page_num, image_index,
        )
        if item:
            found.append(item)

    for path in (run_dir / "debug" / "formulas").glob("*_formulas.json"):
        data = load_json(path)
        for formula in data.get("formulas") or []:
            if formula.get("review_reason") != "api_failed":
                continue
            document = path.name.rsplit("_page_", 1)[0]
            page_num = int(data.get("page") or formula.get("page_num") or 0)
            item = candidate(
                "formula", path, formula, formula.get("source_image"), formula.get("api_error"),
                document, page_num, formula.get("image_index"),
            )
            if item:
                found.append(item)

    for path in (run_dir / "debug" / "tables").glob("*.json"):
        data = load_json(path)
        if data.get("review_reason") != "api_failed":
            continue
        document = path.name.rsplit("_page_", 1)[0]
        page_match = re.search(r"_page_(\d+)_table_", path.name)
        page_num = int(data.get("page_num") or (page_match.group(1) if page_match else 0))
        item = candidate(
            "table", path, data, data.get("source_image"), data.get("api_error"),
            document, page_num, data.get("image_index"),
        )
        if item:
            found.append(item)

    for path in (run_dir / "debug" / "code").glob("*_metadata.json"):
        data = load_json(path)
        if data.get("review_reason") != "api_failed" and data.get("status") != "api_failed":
            continue
        document = path.name.rsplit("_page_", 1)[0]
        page_num = int(data.get("page") or 0)
        item = candidate(
            "code", path, data, data.get("source_image"), data.get("api_error"),
            document, page_num, data.get("image_index"),
        )
        if item:
            found.append(item)

    # Code recognition failures were historically only logged. New logs carry
    # the image path, so a last-attempt failure can still be retried even when
    # no modality metadata was written.
    log_pattern = re.compile(
        r"VLM API attempt failed: operation=(\S+) model=\S+ image=(.*?) "
        r"attempt=(\d+)/(\d+) error=(.*)$"
    )
    supported = {"image_description", "formula", "table", "code"}
    for log_path in (run_dir / "debug" / "logs").glob("*.log"):
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = log_pattern.search(line)
            if not match or match.group(1) not in supported:
                continue
            if int(match.group(3)) != int(match.group(4)):
                continue
            image_path = project_path(match.group(2).strip())
            error = match.group(5).strip()
            if not image_path or not image_path.exists() or not is_transient_error(error):
                continue
            document, page_num, image_index = image_identity_from_metadata(run_dir, image_path)
            item = candidate(
                match.group(1), log_path, {}, image_path, error,
                document, page_num, image_index,
            )
            if item:
                found.append(item)

    unique: Dict[tuple[str, str], Dict[str, Any]] = {}
    for item in found:
        unique[(item["operation"], item["source_image"])] = item
    return list(unique.values())


def image_identity_from_metadata(
    run_dir: Path, image_path: Path
) -> tuple[str, int, Optional[int]]:
    target = image_path.resolve()
    for metadata_path in (run_dir / "debug" / "images").glob("*_metadata.json"):
        data = load_json(metadata_path)
        paths = (data.get("original_path"), data.get("enhanced_path"))
        if any(project_path(value) == target for value in paths if value):
            return source_identity(metadata_path, data)
    return "", 0, None


def description_routes(description: Any) -> Dict[str, bool]:
    text = str(description or "")
    lowered = text.lower()
    code_markers = (
        "类型：代码", "类型:代码", "代码片段", "源代码", "程序代码",
        "code snippet", "source code", "void setup", "void loop", "int main",
        "def ", "class ", "#include", "pinmode", "digitalwrite", "attachinterrupt",
    )
    table_markers = ("表格", "数据表", "统计表", "对比表", "参数表", "真值表", "table", "tabular")
    formula_markers = (
        "公式", "方程", "数学表达式", "计算式", "积分", "微分", "导数",
        "矩阵", "分式", "根号", "formula", "equation", "latex",
    )
    has_code = any(marker.lower() in lowered for marker in code_markers)
    has_table = any(marker.lower() in lowered for marker in table_markers) and not has_code
    has_formula = (
        any(marker.lower() in lowered for marker in formula_markers)
        or bool(re.search(r"[A-Za-zα-ωΑ-Ω]\s*[=≈<>]\s*[^，。；\n]{2,80}", text))
    ) and not has_code and not has_table
    return {"code": has_code, "table": has_table, "formula": has_formula}


def retry_one(
    item: Dict[str, Any],
    client: Any,
    image_processor: ImageProcessor,
    formula_validator: FormulaValidator,
    table_validator: TableValidator,
    run_dir: Path,
) -> Dict[str, Any]:
    image_path = project_path(item["source_image"])
    context = page_context(run_dir, item["source_document"], int(item["page_num"]))
    operation = item["operation"]

    if operation == "image_description":
        result = image_processor.process_image(str(image_path), context)
        success = result.get("status") == "success" and bool(result.get("description"))
        if success and not result.get("review_required"):
            routes = description_routes(result.get("description"))
            specialized: Dict[str, Any] = {}
            if routes["code"]:
                specialized["code"] = client.recognize_code(str(image_path))
            elif routes["table"]:
                table_response = client.recognize_table(str(image_path), context=context)
                if table_response.get("has_table", True):
                    specialized["table"] = table_validator.validate_many([{
                        **table_response,
                        "source_image": str(image_path),
                        "source_type": "image_vlm_table",
                        "extraction_method": f"{client.provider}_api_table_retry",
                        "backend": "api",
                        "provider": client.provider,
                        "model": client.model,
                        "image_index": item.get("image_index"),
                    }])
                else:
                    specialized["table"] = table_response
            elif routes["formula"]:
                formula_response = client.recognize_formula(str(image_path), context=context)
                specialized["formula"] = formula_validator.validate_many([{
                    "source_image": str(image_path),
                    "source_type": "image_vlm",
                    "extraction_method": f"{client.provider}_api_formula_retry",
                    "backend": "api",
                    "provider": client.provider,
                    "model": client.model,
                    "image_index": item.get("image_index"),
                    **formula,
                } for formula in formula_response.get("formulas") or [] if isinstance(formula, dict)])
            result["specialized_retry"] = specialized
    elif operation == "formula":
        response = client.recognize_formula(str(image_path), context=context)
        formulas = response.get("formulas") or []
        enriched = [{
            "source_image": str(image_path),
            "source_type": "image_vlm",
            "extraction_method": f"{client.provider}_api_formula_retry",
            "backend": "api",
            "provider": client.provider,
            "model": client.model,
            "image_index": item.get("image_index"),
            **formula,
        } for formula in formulas if isinstance(formula, dict)]
        result = {**response, "formulas": formula_validator.validate_many(enriched)}
        success = bool(result.get("has_formula") is False or result.get("formulas"))
    elif operation == "table":
        response = client.recognize_table(str(image_path), context=context)
        if response.get("has_table", True):
            table = {
                **response,
                "source_image": str(image_path),
                "source_type": "image_vlm_table",
                "extraction_method": f"{client.provider}_api_table_retry",
                "backend": "api",
                "provider": client.provider,
                "model": client.model,
                "image_index": item.get("image_index"),
            }
            result = {"has_table": True, "tables": table_validator.validate_many([table])}
        else:
            result = response
        success = bool(result.get("has_table") is False or result.get("tables"))
    elif operation == "code":
        result = client.recognize_code(str(image_path))
        success = "has_code" in result
    else:
        raise ValueError(f"unsupported operation: {operation}")

    return {
        **item,
        "retry_status": "success" if success else "quality_review_required",
        "retried_at": datetime.now().isoformat(),
        "provider": client.provider,
        "model": client.model,
        "result": result,
    }


def count_log_events(run_dir: Path) -> Dict[str, int]:
    counts = {"attempt_failures": 0, "final_api_failures": 0}
    for path in (run_dir / "debug" / "logs").glob("*.log"):
        text = path.read_text(encoding="utf-8", errors="replace")
        counts["attempt_failures"] += text.count("VLM API attempt failed")
        counts["final_api_failures"] += text.count("VLM API request failed")
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="重试已完成抽取结果中的临时 API 失败项")
    parser.add_argument("--run-dir", help="指定 output 下的运行目录；默认选择最新完整运行")
    parser.add_argument("--dry-run", action="store_true", help="只扫描，不调用 API")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = project_path(args.run_dir) if args.run_dir else latest_completed_run()
    if not run_dir or not run_dir.is_dir():
        raise RuntimeError(f"运行目录不存在: {run_dir}")
    if not (run_dir / "kg_data" / "knowledge_export_summary.json").exists():
        raise RuntimeError("该运行尚未完成，请等待 knowledge_export_summary.json 生成后再补漏")

    candidates = collect_candidates(run_dir)
    log_events = count_log_events(run_dir)
    print(f"运行目录: {run_dir}")
    print(f"日志中的单次调用失败: {log_events['attempt_failures']}")
    print(f"可精确定位的临时 API 最终失败项: {len(candidates)}")
    if not candidates or args.dry_run:
        for item in candidates:
            print(f"  - {item['operation']}: {item['source_image']}")
        return 0

    logger = get_logger("ApiRecovery")
    client = create_vlm_client(logger=logger)
    if client is None:
        raise RuntimeError("VLM API 未配置，无法补漏")

    retry_dir = next_retry_dir(run_dir)
    image_processor = ImageProcessor(logger=logger, vlm_client=client)
    formula_validator = FormulaValidator()
    table_validator = TableValidator()
    results = []
    for index, item in enumerate(candidates, 1):
        print(f"[{index}/{len(candidates)}] 重试 {item['operation']}: {item['source_image']}")
        try:
            result = retry_one(
                item, client, image_processor, formula_validator, table_validator, run_dir
            )
        except Exception as exc:
            result = {
                **item,
                "retry_status": "api_failed",
                "retried_at": datetime.now().isoformat(),
                "api_error": str(exc),
            }
        item_path = retry_dir / "items" / f"{index:04d}_{item['operation']}.json"
        with item_path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2, default=str)
        results.append(result)

    manifest = {
        "source_run": str(run_dir.relative_to(PROJECT_ROOT)).replace(os.sep, "/"),
        "created_at": datetime.now().isoformat(),
        "provider": client.provider,
        "model": client.model,
        "log_events": log_events,
        "candidate_count": len(candidates),
        "success_count": sum(item.get("retry_status") == "success" for item in results),
        "quality_review_count": sum(
            item.get("retry_status") == "quality_review_required" for item in results
        ),
        "failed_count": sum(item.get("retry_status") == "api_failed" for item in results),
        "results": results,
        "merge_policy": "overlay_only_original_run_unchanged",
    }
    with (retry_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, default=str)
    print(f"补漏结果: {retry_dir}")
    print(
        f"成功={manifest['success_count']}，需质量复核={manifest['quality_review_count']}，"
        f"API仍失败={manifest['failed_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
