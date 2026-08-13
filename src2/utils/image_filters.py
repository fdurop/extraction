from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from collections import Counter
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image


@dataclass
class ImageFilterDecision:
    keep: bool
    reason: str
    stats: Dict[str, float]


def analyze_image(image_path: str) -> ImageFilterDecision:
    try:
        if not os.path.exists(image_path):
            return ImageFilterDecision(False, "missing_file", {})

        with Image.open(image_path) as img:
            img = img.convert("RGB")
            width, height = img.size
            arr = np.asarray(img, dtype=np.uint8)

        if width <= 0 or height <= 0:
            return ImageFilterDecision(False, "invalid_size", {"width": width, "height": height})

        gray = arr.mean(axis=2)
        mean = float(gray.mean())
        std = float(gray.std())
        black_ratio = float((gray < 18).mean())
        white_ratio = float((gray > 245).mean())
        mid_ratio = float(((gray >= 18) & (gray <= 245)).mean())

        edge_x = float(np.abs(np.diff(gray, axis=1)).mean()) if width > 1 else 0.0
        edge_y = float(np.abs(np.diff(gray, axis=0)).mean()) if height > 1 else 0.0
        edge_score = (edge_x + edge_y) / 2.0

        non_white = 1.0 - white_ratio
        area = width * height
        min_dim = min(width, height)

        stats = {
            "width": float(width),
            "height": float(height),
            "area": float(area),
            "mean": mean,
            "std": std,
            "black_ratio": black_ratio,
            "white_ratio": white_ratio,
            "mid_ratio": mid_ratio,
            "edge_score": edge_score,
        }

        if area < 9000 or min_dim < 60:
            return ImageFilterDecision(False, "too_small", stats)

        if std < 5.0:
            return ImageFilterDecision(False, "flat_image", stats)

        if black_ratio > 0.92 or white_ratio > 0.97:
            return ImageFilterDecision(False, "mostly_black_or_white", stats)

        if non_white < 0.02 and area < 40000:
            return ImageFilterDecision(False, "almost_blank", stats)

        if edge_score < 1.2 and std < 18.0:
            return ImageFilterDecision(False, "low_information", stats)

        return ImageFilterDecision(True, "keep", stats)

    except Exception as exc:
        return ImageFilterDecision(False, f"analysis_failed:{exc}", {})


def should_keep_image(image_path: str) -> Tuple[bool, str, Dict[str, float]]:
    decision = analyze_image(image_path)
    return decision.keep, decision.reason, decision.stats


def image_signature(image_path: str) -> str:
    """Stable signature used for repeated logo/watermark removal."""
    try:
        suffix = os.path.splitext(image_path)[1].lower()
        if suffix in {".emf", ".wmf", ".svg"}:
            with open(image_path, "rb") as f:
                digest = hashlib.md5(f.read()).hexdigest()
            return f"vector:{suffix}:{digest}"

        with Image.open(image_path) as img:
            img = img.convert("RGB")
            width, height = img.size
            small = img.convert("L").resize((16, 16))
            values = np.asarray(small, dtype=np.float32).reshape(-1)
            avg = float(values.mean())
            bits = "".join("1" if value > avg else "0" for value in values)
            ahash = hex(int(bits, 2))[2:].zfill(64)
        file_size = os.path.getsize(image_path)
        return f"raster:{width}x{height}:{file_size}:{ahash}"
    except Exception:
        try:
            with open(image_path, "rb") as f:
                digest = hashlib.md5(f.read()).hexdigest()
            return f"fallback:{digest}"
        except Exception:
            return f"missing:{image_path}"


def repeated_image_paths(image_paths: Iterable[str], min_count: int = 3) -> Dict[str, str]:
    """Return {path: signature} for images repeated enough to be layout noise."""
    path_signature = {path: image_signature(path) for path in image_paths if os.path.exists(path)}
    counts = Counter(path_signature.values())
    return {
        path: signature
        for path, signature in path_signature.items()
        if counts[signature] >= min_count
    }
