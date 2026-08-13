"""
Lightweight vector index writer for extraction outputs.

This is intentionally dependency-light. It gives the pipeline a stable vector
artifact now, while keeping the file format replaceable by a real embedding API
or model later.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from typing import Any, Dict, Iterable, List

import numpy as np


def _relpath(path: str) -> str:
    try:
        return os.path.relpath(path, os.getcwd()).replace(os.sep, "/")
    except Exception:
        return str(path).replace("\\", "/")


def _tokens(text: str) -> Iterable[str]:
    text = (text or "").strip().lower()
    if not text:
        return []

    ascii_words = re.findall(r"[a-z0-9_+\-./]+", text)
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
    chinese_bigrams = [
        "".join(chinese_chars[i : i + 2]) for i in range(max(0, len(chinese_chars) - 1))
    ]
    return ascii_words + chinese_chars + chinese_bigrams


def hash_embedding(text: str, dim: int = 384) -> List[float]:
    vector = np.zeros(dim, dtype=np.float32)
    for token in _tokens(text):
        digest = hashlib.md5(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign

    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector /= norm
    return vector.tolist()


def build_vector_index(
    records: List[Dict[str, Any]],
    output_dir: str,
    dim: int = 384,
    backend: str = "hash_text_v1",
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)

    metadata = []
    vectors = []
    for record in records:
        embedding_text = record.get("embedding_text") or record.get("text") or ""
        embedding_id = record.get("embedding_id")
        if not embedding_id:
            embedding_id = f"emb_{record.get('node_id') or record.get('element_id')}"

        vector = hash_embedding(embedding_text, dim=dim)
        vectors.append(vector)
        metadata.append(
            {
                "embedding_id": embedding_id,
                "node_id": record.get("node_id") or record.get("element_id"),
                "node_type": record.get("node_type") or record.get("type"),
                "user_id": record.get("user_id"),
                "course_id": record.get("course_id"),
                "document_id": record.get("document_id"),
                "page_id": record.get("page_id") or record.get("source_page_id"),
                "embedding_text": embedding_text,
                "backend": backend,
                "dim": dim,
            }
        )

    matrix = np.array(vectors, dtype=np.float32) if vectors else np.zeros((0, dim), dtype=np.float32)
    matrix_path = os.path.join(output_dir, "vector_matrix.npy")
    meta_path = os.path.join(output_dir, "vector_index.json")
    np.save(matrix_path, matrix)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "backend": backend,
                "dim": dim,
                "count": len(metadata),
                "matrix_file": os.path.basename(matrix_path),
                "items": metadata,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return {
        "backend": backend,
        "dim": dim,
        "count": len(metadata),
        "matrix_path": _relpath(matrix_path),
        "metadata_path": _relpath(meta_path),
    }


def cosine_search(query: str, index_dir: str, top_k: int = 5) -> List[Dict[str, Any]]:
    meta_path = os.path.join(index_dir, "vector_index.json")
    matrix_path = os.path.join(index_dir, "vector_matrix.npy")
    if not os.path.exists(meta_path) or not os.path.exists(matrix_path):
        return []

    with open(meta_path, "r", encoding="utf-8") as f:
        index = json.load(f)
    matrix = np.load(matrix_path)
    if matrix.size == 0:
        return []

    query_vec = np.array(hash_embedding(query, dim=index.get("dim", 384)), dtype=np.float32)
    scores = matrix @ query_vec
    order = np.argsort(-scores)[:top_k]
    results = []
    for rank, idx in enumerate(order, 1):
        item = dict(index["items"][int(idx)])
        item["rank"] = rank
        item["score"] = float(scores[int(idx)])
        results.append(item)
    return results
