from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

SUBTYPE_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("sleeveless", ("슬리브리스", "sleeveless", "나시", "tank top", "tanktop")),
    ("hoodie", ("후드", "hoodie", "후디")),
    ("sweatshirt", ("맨투맨", "스웨트", "sweatshirt", "sweat ")),
    ("knit", ("니트", "knit", "가디건", "cardigan", "스웨터", "sweater", "울 ")),
    ("blouse", ("블라우스", "blouse")),
    ("tshirt", ("티셔츠", "t-shirt", "tshirt", "tee", "반팔티", "긴팔티", "롱슬리브 티")),
    ("shirt", ("셔츠", "shirt", "남방")),
)

LAYERED_TOKENS = (
    "레이어드",
    "layered",
    "이너",
    "inner",
    "셋업",
    "set-up",
    "세트",
    "가디건",
    "조끼",
    "베스트",
    "베스트",
    "overshirt",
)


def resolve_subtype(product: str, category_label: str) -> str:
    text = f"{product} {category_label}".lower()
    for subtype, tokens in SUBTYPE_RULES:
        if any(token.lower() in text for token in tokens):
            return subtype
    if "니트" in category_label:
        return "knit"
    return "other"


def _skin_ratio(rgb: np.ndarray) -> float:
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
    skin = (cr > 133) & (cr < 178) & (cb > 77) & (cb < 132) & (y > 70)
    return float(skin.mean()) if skin.size else 0.0


def _hue_entropy(rgb: np.ndarray) -> float:
    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)
    maximum = np.maximum(np.maximum(r, g), b)
    minimum = np.minimum(np.minimum(r, g), b)
    delta = np.clip(maximum - minimum, 1e-6, None)
    hue = np.zeros_like(maximum)
    mask = maximum == r
    hue[mask] = ((g - b)[mask] / delta[mask]) % 6
    mask = maximum == g
    hue[mask] = (b - r)[mask] / delta[mask] + 2
    mask = maximum == b
    hue[mask] = (r - g)[mask] / delta[mask] + 4
    hue = (hue * 60) % 360
    sat = delta / np.clip(maximum, 1e-6, None)
    valid = sat > 0.12
    if not valid.any():
        return 0.0
    hist, _ = np.histogram(hue[valid], bins=16, range=(0, 360), density=True)
    hist = hist[hist > 0]
    return float(-(hist * np.log(hist + 1e-12)).sum())


def _edge_density(gray: np.ndarray) -> float:
    dx = np.abs(np.diff(gray.astype(np.float32), axis=1))
    dy = np.abs(np.diff(gray.astype(np.float32), axis=0))
    return float((dx.mean() + dy.mean()) / 2.0 / 255.0)


def image_features(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        rgb_image = image.convert("RGB")
        width, height = rgb_image.size
        small = rgb_image.resize((96, 96))
        rgb = np.asarray(small)
        gray = np.asarray(small.convert("L"))
        gray8 = np.asarray(small.convert("L").resize((8, 8)), dtype=np.int32).reshape(-1)
        mean = float(gray8.mean())
        phash = 0
        for pixel in gray8.tolist():
            phash = (phash << 1) | int(pixel >= mean)
    skin = _skin_ratio(rgb)
    entropy = _hue_entropy(rgb)
    edges = _edge_density(gray)
    if skin >= 0.09:
        shoot_type = "model"
    elif skin >= 0.035:
        shoot_type = "lookbook"
    else:
        shoot_type = "product_only"
    aspect = width / max(height, 1)
    return {
        "width": width,
        "height": height,
        "aspect": round(aspect, 4),
        "skin_ratio": round(skin, 4),
        "hue_entropy": round(entropy, 4),
        "edge_density": round(edges, 4),
        "shoot_type": shoot_type,
        "phash": phash,
    }


def complexity_score(item: dict[str, Any]) -> float:
    title = f"{item.get('product') or ''} {item.get('category_label') or ''}".lower()
    layered = any(token in title for token in LAYERED_TOKENS)
    features = item.get("heuristic") or {}
    skin = float(features.get("skin_ratio") or 0.0)
    entropy = float(features.get("hue_entropy") or 0.0)
    edges = float(features.get("edge_density") or 0.0)
    model_bonus = 1.0 if features.get("shoot_type") in {"model", "lookbook"} else 0.0
    return float(
        0.28 * model_bonus
        + 0.22 * min(skin / 0.2, 1.0)
        + 0.18 * min(entropy / 2.4, 1.0)
        + 0.14 * min(edges / 0.12, 1.0)
        + 0.18 * float(layered)
    )


def average_hash_from_features(features: dict[str, Any]) -> int:
    return int(features["phash"])
