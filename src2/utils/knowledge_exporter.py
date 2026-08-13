"""
Unified knowledge export for multimodal GraphRAG.

The extractor still writes its original per-modality files. This exporter adds a
stable handoff layer for graph construction and vector retrieval:

- course_manifest.json
- documents.json
- pages.json
- content_list.json
- multimodal_nodes.json
- vectors/vector_index.json
- vectors/vector_matrix.npy
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from .vector_index import build_vector_index


def _clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return re.sub(r"\s+", " ", value).strip()
    return re.sub(r"\s+", " ", json.dumps(value, ensure_ascii=False, default=str)).strip()


def _short_hash(*parts: Any) -> str:
    raw = "|".join(_clean(p) for p in parts)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]


def _safe_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    if isinstance(value, set):
        return [str(v) for v in sorted(value) if str(v).strip()]
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [str(value)]


def _relpath(value: Any) -> str:
    if not value:
        return ""
    try:
        return os.path.relpath(str(value), os.getcwd()).replace(os.sep, "/")
    except Exception:
        return str(value).replace("\\", "/")


def _relpath_list(value: Any) -> List[str]:
    return [_relpath(item) for item in _safe_list(value)]


class KnowledgeExporter:
    def __init__(
        self,
        base_dir: str = "output",
        user_id: Optional[str] = None,
        course_id: Optional[str] = None,
        course_name: Optional[str] = None,
        logger=None,
    ):
        self.base_dir = base_dir
        self.logger = logger
        self.user_id = user_id or os.getenv("EXTRACTION_USER_ID", "default_user")
        self.course_id = course_id or os.getenv("EXTRACTION_COURSE_ID", "default_course")
        self.course_name = course_name or os.getenv("EXTRACTION_COURSE_NAME", "default_course")
        self.created_at = datetime.now().isoformat()

        self.documents: Dict[str, Dict[str, Any]] = {}
        self.pages: Dict[str, Dict[str, Any]] = {}
        self.content_items: Dict[str, Dict[str, Any]] = {}
        self.nodes: Dict[str, Dict[str, Any]] = {}

    def document_id(self, filename: str) -> str:
        return f"doc_{_short_hash(self.course_id, filename)}"

    def page_id(self, filename: str, page_num: int) -> str:
        return f"page_{_short_hash(self.course_id, filename, page_num + 1)}"

    def element_id(self, filename: str, page_num: int, kind: str, index: int) -> str:
        return f"{kind}_{_short_hash(self.course_id, filename, page_num + 1, kind, index + 1)}"

    def embedding_id(self, node_id: str) -> str:
        return f"emb_{node_id}"

    def register_document(self, filename: str, metadata: Optional[Dict[str, Any]] = None):
        metadata = metadata or {}
        doc_id = self.document_id(filename)
        total_pages = metadata.get("total_pages") or metadata.get("total_slides") or metadata.get("page_count")
        self.documents[doc_id] = {
            "document_id": doc_id,
            "user_id": self.user_id,
            "course_id": self.course_id,
            "file_name": metadata.get("file_name") or filename,
            "file_type": metadata.get("file_type") or metadata.get("extension") or "",
            "file_path": _relpath(metadata.get("file_path") or ""),
            "page_count": total_pages,
            "chapter_title": metadata.get("chapter_title") or filename,
            "document_order": metadata.get("document_order"),
            "metadata": self._normalize_paths(metadata),
        }
        return doc_id

    def register_page(
        self,
        filename: str,
        page_num: int,
        raw_text: str = "",
        images: Optional[List[str]] = None,
        page_image_path: str = "",
    ):
        doc_id = self.document_id(filename)
        page_id = self.page_id(filename, page_num)
        title = self._guess_title(raw_text)
        summary = self._summary(raw_text, 220)
        existing = self.pages.get(page_id, {})
        merged_raw_text = raw_text or existing.get("raw_text", "")
        merged_images = _relpath_list(images) if images else existing.get("image_paths", [])
        merged_title = title or existing.get("title", "")
        merged_summary = summary or existing.get("page_summary", "")
        merged_page_image_path = _relpath(page_image_path) or existing.get("page_image_path", "")
        self.pages[page_id] = {
            "page_id": page_id,
            "user_id": self.user_id,
            "course_id": self.course_id,
            "document_id": doc_id,
            "page_no": page_num + 1,
            "title": merged_title,
            "raw_text": merged_raw_text,
            "page_summary": merged_summary,
            "page_image_path": merged_page_image_path,
            "image_paths": merged_images,
            "prev_page_id": self.page_id(filename, page_num - 1) if page_num > 0 else existing.get("prev_page_id"),
            "next_page_id": self.page_id(filename, page_num + 1) if raw_text or images or page_image_path or not existing else existing.get("next_page_id"),
            "embedding_id": self.embedding_id(page_id),
            "embedding_text": " ".join([merged_title, merged_summary, merged_raw_text[:500]]).strip(),
        }
        self._add_node(
            node_id=page_id,
            node_type="Page",
            name=merged_title or f"{filename} page {page_num + 1}",
            summary=merged_summary,
            description=merged_raw_text[:1000],
            document_id=doc_id,
            page_id=page_id,
            source_path=merged_page_image_path,
            embedding_text=self.pages[page_id]["embedding_text"],
            extra={"page_no": page_num + 1},
        )
        return page_id

    def add_text(self, data: Dict[str, Any], filename: str, page_num: int):
        text = data.get("cleaned_text") or data.get("raw_text") or ""
        if not text.strip():
            return None
        page_id = self.register_page(
            filename,
            page_num,
            raw_text=data.get("raw_text", text),
            images=data.get("image_paths") or [],
        )
        element_id = self.element_id(filename, page_num, "text", 0)
        key_points = data.get("key_points") or []
        terms = data.get("technical_terms") or []
        embedding_text = " ".join([text, " ".join(key_points), " ".join(terms)]).strip()
        item = self._base_item(element_id, "text", filename, page_num, page_id)
        item.update(
            {
                "text": text,
                "raw_text": data.get("raw_text", ""),
                "key_points": key_points,
                "technical_terms": terms,
                "embedding_id": self.embedding_id(element_id),
                "embedding_text": embedding_text,
            }
        )
        self.content_items[element_id] = item
        self._add_node(
            element_id,
            "TextChunk",
            name=self._guess_title(text) or f"text page {page_num + 1}",
            summary=self._summary(text, 180),
            description=text,
            document_id=self.document_id(filename),
            page_id=page_id,
            source_path="",
            embedding_text=embedding_text,
            extra={"technical_terms": terms, "key_points": key_points},
        )
        return element_id

    def add_image(self, data: Dict[str, Any], filename: str, page_num: int, img_index: int):
        page_id = self.register_page(filename, page_num)
        element_id = self.element_id(filename, page_num, "figure", img_index)
        description = _clean(data.get("description"))
        if not description and str(data.get("method") or "").lower() in {"", "none", "unknown"}:
            if self.logger:
                self.logger.warning(
                    "Skip non-indexable image node: %s page=%s image=%s status=%s error=%s",
                    filename,
                    page_num + 1,
                    img_index + 1,
                    data.get("status"),
                    data.get("error") or data.get("api_error"),
                )
            return None
        image_path = _relpath(data.get("original_path") or data.get("image_path") or data.get("enhanced_path") or "")
        embedding_text = " ".join(
            [
                f"image figure page {page_num + 1}",
                description,
                _clean(data.get("method")),
            ]
        ).strip()
        item = self._base_item(element_id, "image", filename, page_num, page_id)
        item.update(
            {
                "img_path": image_path,
                "enhanced_path": _relpath(data.get("enhanced_path")),
                "caption": data.get("caption", ""),
                "description": description,
                "method": data.get("method"),
                "backend": data.get("backend"),
                "provider": data.get("provider"),
                "model": data.get("model"),
                "status": data.get("status"),
                "error": data.get("error"),
                "api_error": data.get("api_error"),
                "indexable": data.get("indexable", bool(description)),
                "embedding_id": self.embedding_id(element_id),
                "embedding_text": embedding_text,
            }
        )
        self.content_items[element_id] = item
        self._add_node(
            element_id,
            "Figure",
            name=data.get("caption") or f"figure page {page_num + 1}-{img_index + 1}",
            summary=self._summary(description, 160),
            description=description,
            document_id=self.document_id(filename),
            page_id=page_id,
            source_path=image_path,
            embedding_text=embedding_text,
            extra={
                "method": data.get("method"),
                "backend": data.get("backend"),
                "provider": data.get("provider"),
                "model": data.get("model"),
                "status": data.get("status"),
                "api_error": data.get("api_error"),
                "enhanced_path": _relpath(data.get("enhanced_path")),
                "indexable": data.get("indexable", bool(description)),
            },
        )
        return element_id

    def add_formulas(self, formulas: List[Dict[str, Any]], filename: str, page_num: int):
        ids = []
        page_id = self.register_page(filename, page_num)
        for idx, formula in enumerate(formulas or []):
            element_id = self.element_id(filename, page_num, "formula", idx)
            latex = _clean(formula.get("latex"))
            description = _clean(formula.get("description") or formula.get("text"))
            source_image = _relpath(formula.get("source_image"))
            embedding_text = " ".join([latex, description, _clean(source_image)]).strip()
            item = self._base_item(element_id, "equation", filename, page_num, page_id)
            item.update(
                {
                    "latex": latex,
                    "text": description,
                    "source_image": source_image,
                    "extraction_method": formula.get("extraction_method", ""),
                    "backend": formula.get("backend"),
                    "provider": formula.get("provider"),
                    "model": formula.get("model"),
                    "embedding_id": self.embedding_id(element_id),
                    "embedding_text": embedding_text,
                }
            )
            self.content_items[element_id] = item
            self._add_node(
                element_id,
                "Formula",
                name=latex[:80] or f"formula page {page_num + 1}-{idx + 1}",
                summary=self._summary(description or latex, 160),
                description=description,
                document_id=self.document_id(filename),
                page_id=page_id,
            source_path=source_image,
            embedding_text=embedding_text,
            extra={
                "latex": latex,
                "extraction_method": formula.get("extraction_method", ""),
                "backend": formula.get("backend"),
                "provider": formula.get("provider"),
                "model": formula.get("model"),
            },
            )
            ids.append(element_id)
        return ids

    def add_tables(self, tables: List[Dict[str, Any]], filename: str, page_num: int):
        ids = []
        page_id = self.register_page(filename, page_num)
        for idx, table in enumerate(tables or []):
            element_id = self.element_id(filename, page_num, "table", idx)
            table_body = table.get("csv") or table.get("content") or _clean(table.get("data") or table.get("json"))
            description = _clean(table.get("description"))
            headers = _clean(table.get("headers"))
            source_image = _relpath(table.get("source_image"))
            embedding_text = " ".join([headers, description, table_body[:1500]]).strip()
            item = self._base_item(element_id, "table", filename, page_num, page_id)
            item.update(
                {
                    "table_body": table_body,
                    "headers": table.get("headers", ""),
                    "description": description,
                    "source_image": source_image,
                    "extraction_method": table.get("extraction_method") or table.get("type", ""),
                    "backend": table.get("backend"),
                    "provider": table.get("provider"),
                    "model": table.get("model"),
                    "embedding_id": self.embedding_id(element_id),
                    "embedding_text": embedding_text,
                }
            )
            self.content_items[element_id] = item
            self._add_node(
                element_id,
                "Table",
                name=f"table page {page_num + 1}-{idx + 1}",
                summary=self._summary(description or table_body, 160),
                description=description or table_body,
                document_id=self.document_id(filename),
            page_id=page_id,
            source_path=source_image,
            embedding_text=embedding_text,
            extra={
                "headers": table.get("headers", ""),
                "rows": table.get("rows"),
                "cols": table.get("cols"),
                "extraction_method": table.get("extraction_method") or table.get("type", ""),
                "backend": table.get("backend"),
                "provider": table.get("provider"),
                "model": table.get("model"),
            },
            )
            ids.append(element_id)
        return ids

    def add_code(self, code_blocks: List[Dict[str, Any]], filename: str, page_num: int):
        ids = []
        page_id = self.register_page(filename, page_num)
        for idx, code in enumerate(code_blocks or []):
            element_id = self.element_id(filename, page_num, "code", idx)
            code_text = code.get("code", "")
            description = _clean(code.get("description"))
            language = code.get("language", "txt")
            source_image = _relpath(code.get("source_image"))
            embedding_text = " ".join([language, description, code_text[:1500]]).strip()
            item = self._base_item(element_id, "code", filename, page_num, page_id)
            item.update(
                {
                    "code": code_text,
                    "language": language,
                    "description": description,
                    "source_image": source_image,
                    "extraction_method": code.get("extraction_method", ""),
                    "backend": code.get("backend"),
                    "provider": code.get("provider"),
                    "model": code.get("model"),
                    "embedding_id": self.embedding_id(element_id),
                    "embedding_text": embedding_text,
                }
            )
            self.content_items[element_id] = item
            self._add_node(
                element_id,
                "CodeBlock",
                name=f"{language} code page {page_num + 1}-{idx + 1}",
                summary=self._summary(description or code_text, 160),
                description=description or code_text,
                document_id=self.document_id(filename),
            page_id=page_id,
            source_path=source_image,
            embedding_text=embedding_text,
            extra={
                "language": language,
                "extraction_method": code.get("extraction_method", ""),
                "backend": code.get("backend"),
                "provider": code.get("provider"),
                "model": code.get("model"),
            },
        )
            ids.append(element_id)
        return ids

    def finalize(self):
        self._fix_page_links()
        os.makedirs(self.base_dir, exist_ok=True)

        course_manifest = {
            "user_id": self.user_id,
            "course_id": self.course_id,
            "course_name": self.course_name,
            "created_at": self.created_at,
            "updated_at": datetime.now().isoformat(),
            "documents": sorted(self.documents.keys()),
            "output_schema": "multimodal_graphrag_extraction_v1",
        }

        self._write_json("course_manifest.json", course_manifest)
        self._write_json("documents.json", list(self.documents.values()))
        self._write_json("pages.json", sorted(self.pages.values(), key=lambda x: (x["document_id"], x["page_no"])))
        self._write_json("content_list.json", list(self.content_items.values()))
        self._write_json("multimodal_nodes.json", list(self.nodes.values()))
        self._write_json("relationships.json", self._build_relationships())
        self._write_json("schema.json", self._build_schema())

        vector_records = list(self.nodes.values())
        vector_summary = build_vector_index(vector_records, os.path.join(self.base_dir, "vectors"))
        self._write_json("knowledge_export_summary.json", {
            "documents": len(self.documents),
            "pages": len(self.pages),
            "content_items": len(self.content_items),
            "nodes": len(self.nodes),
            "vectors": vector_summary,
            "updated_at": datetime.now().isoformat(),
        })
        if self.logger:
            self.logger.info("Unified knowledge export completed")

    def _build_relationships(self) -> List[Dict[str, Any]]:
        relationships: List[Dict[str, Any]] = []

        for doc in self.documents.values():
            relationships.append(
                {
                    "relationship_id": f"rel_course_contains_{doc['document_id']}",
                    "source_id": self.course_id,
                    "target_id": doc["document_id"],
                    "relationship_type": "COURSE_CONTAINS_DOCUMENT",
                    "weight": 1.0,
                    "properties": {"document_order": doc.get("document_order")},
                }
            )

        for page in self.pages.values():
            relationships.append(
                {
                    "relationship_id": f"rel_doc_contains_{page['page_id']}",
                    "source_id": page["document_id"],
                    "target_id": page["page_id"],
                    "relationship_type": "DOCUMENT_CONTAINS_PAGE",
                    "weight": 1.0,
                    "properties": {"page_no": page.get("page_no")},
                }
            )
            if page.get("prev_page_id"):
                relationships.append(
                    {
                        "relationship_id": f"rel_prev_{page['page_id']}",
                        "source_id": page["prev_page_id"],
                        "target_id": page["page_id"],
                        "relationship_type": "NEXT_PAGE",
                        "weight": 1.0,
                        "properties": {},
                    }
                )

        for item in self.content_items.values():
            relationships.append(
                {
                    "relationship_id": f"rel_page_contains_{item['element_id']}",
                    "source_id": item["page_id"],
                    "target_id": item["element_id"],
                    "relationship_type": "PAGE_CONTAINS_CONTENT",
                    "weight": 1.0,
                    "properties": {"content_type": item.get("type")},
                }
            )

        return relationships

    def _build_schema(self) -> Dict[str, Any]:
        return {
            "schema_name": "multimodal_graphrag_extraction_v1",
            "node_files": {
                "documents": "documents.json",
                "pages": "pages.json",
                "content_items": "content_list.json",
                "graph_nodes": "multimodal_nodes.json",
            },
            "relationship_file": "relationships.json",
            "vector_files": {
                "metadata": "vectors/vector_index.json",
                "matrix": "vectors/vector_matrix.npy",
            },
            "node_types": [
                "Course",
                "Document",
                "Page",
                "TextChunk",
                "Figure",
                "Table",
                "Formula",
                "CodeBlock",
                "Entity",
                "Concept",
            ],
            "relationship_types": [
                "COURSE_CONTAINS_DOCUMENT",
                "DOCUMENT_CONTAINS_PAGE",
                "NEXT_PAGE",
                "PAGE_CONTAINS_CONTENT",
                "MENTIONS_CONCEPT",
                "RELATED_TO",
            ],
            "required_downstream_files": [
                "course_manifest.json",
                "documents.json",
                "pages.json",
                "content_list.json",
                "multimodal_nodes.json",
                "relationships.json",
                "vectors/vector_index.json",
                "vectors/vector_matrix.npy",
            ],
        }

    def _base_item(self, element_id: str, kind: str, filename: str, page_num: int, page_id: str) -> Dict[str, Any]:
        return {
            "element_id": element_id,
            "type": kind,
            "user_id": self.user_id,
            "course_id": self.course_id,
            "document_id": self.document_id(filename),
            "page_id": page_id,
            "page_idx": page_num,
            "page_no": page_num + 1,
        }

    def _add_node(
        self,
        node_id: str,
        node_type: str,
        name: str,
        summary: str,
        description: str,
        document_id: str,
        page_id: str,
        source_path: str,
        embedding_text: str,
        extra: Optional[Dict[str, Any]] = None,
    ):
        self.nodes[node_id] = {
            "node_id": node_id,
            "node_type": node_type,
            "user_id": self.user_id,
            "course_id": self.course_id,
            "document_id": document_id,
            "source_page_id": page_id,
            "name": name or node_id,
            "summary": summary or "",
            "description": description or "",
            "source_path": _relpath(source_path),
            "related_concepts": [],
            "embedding_id": self.embedding_id(node_id),
            "embedding_text": embedding_text or " ".join([name or "", summary or "", description or ""]).strip(),
            "created_at": self.created_at,
            **(extra or {}),
        }

    def _write_json(self, filename: str, data: Any):
        path = os.path.join(self.base_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._normalize_paths(data), f, ensure_ascii=False, indent=2, default=str)

    def _fix_page_links(self):
        by_doc: Dict[str, List[Dict[str, Any]]] = {}
        for page in self.pages.values():
            by_doc.setdefault(page["document_id"], []).append(page)
        for pages in by_doc.values():
            pages.sort(key=lambda x: x["page_no"])
            for idx, page in enumerate(pages):
                page["prev_page_id"] = pages[idx - 1]["page_id"] if idx > 0 else None
                page["next_page_id"] = pages[idx + 1]["page_id"] if idx < len(pages) - 1 else None

    @staticmethod
    def _guess_title(text: str) -> str:
        for line in (text or "").splitlines():
            line = _clean(line)
            if 2 <= len(line) <= 80:
                return line
        return ""

    @staticmethod
    def _summary(text: str, limit: int) -> str:
        text = _clean(text)
        return text[:limit]

    @staticmethod
    def _normalize_paths(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: KnowledgeExporter._normalize_paths(item) for key, item in value.items()}
        if isinstance(value, list):
            return [KnowledgeExporter._normalize_paths(item) for item in value]
        if isinstance(value, str):
            looks_like_path = (
                "\\" in value
                or value.startswith("input/")
                or value.startswith("output/")
                or value.startswith("input\\")
                or value.startswith("output\\")
            )
            return _relpath(value) if looks_like_path else value
        return value
