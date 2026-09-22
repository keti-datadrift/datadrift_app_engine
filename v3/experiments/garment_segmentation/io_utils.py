from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


def load_attr_dataset():
    path = Path(__file__).resolve().parent.parent / "attribute_embedding" / "dataset.py"
    spec = importlib.util.spec_from_file_location("attr_emb_dataset", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def rle_encode(mask: np.ndarray) -> dict[str, Any]:
    binary = np.asarray(mask, dtype=bool)
    height, width = binary.shape
    flat = binary.astype(np.uint8).ravel(order="F")
    if flat.size == 0:
        return {"size": [int(height), int(width)], "counts": []}
    change_idx = np.flatnonzero(np.diff(flat)) + 1
    starts = np.concatenate(([0], change_idx))
    ends = np.concatenate((change_idx, [flat.size]))
    counts = (ends - starts).astype(int).tolist()
    if int(flat[0]) == 1:
        counts = [0] + counts
    return {"size": [int(height), int(width)], "counts": counts}


def rle_decode(rle: dict[str, Any]) -> np.ndarray:
    height, width = (int(rle["size"][0]), int(rle["size"][1]))
    counts = [int(c) for c in rle["counts"]]
    flat = np.zeros(height * width, dtype=np.uint8)
    pos = 0
    value = 0
    for count in counts:
        if value:
            flat[pos : pos + count] = 1
        pos += count
        value = 1 - value
    return flat.reshape((height, width), order="F").astype(bool)


def mask_from_box(height: int, width: int, box_xyxy: list[float] | tuple[float, ...]) -> np.ndarray:
    x0, y0, x1, y1 = [int(round(v)) for v in box_xyxy]
    x0 = min(max(x0, 0), width)
    x1 = min(max(x1, 0), width)
    y0 = min(max(y0, 0), height)
    y1 = min(max(y1, 0), height)
    mask = np.zeros((height, width), dtype=bool)
    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = True
    return mask


def box_from_mask(mask: np.ndarray) -> list[float]:
    ys, xs = np.where(mask)
    if not len(xs):
        return [0.0, 0.0, 0.0, 0.0]
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def centroid_from_mask(mask: np.ndarray) -> list[float]:
    ys, xs = np.where(mask)
    if not len(xs):
        height, width = mask.shape
        return [width / 2.0, height / 2.0]
    return [float(xs.mean()), float(ys.mean())]


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    inter = np.logical_and(left, right).sum()
    union = np.logical_or(left, right).sum()
    if union == 0:
        return 0.0
    return float(inter / union)


def connected_component_count(mask: np.ndarray) -> int:
    from scipy import ndimage

    _, count = ndimage.label(np.asarray(mask, dtype=np.uint8))
    return int(count)


def restore_mask(proposal: dict[str, Any], height: int, width: int) -> np.ndarray:
    rle = proposal.get("mask_rle") or {}
    size = rle.get("size") or [height, width]
    mask = rle_decode(rle)
    if tuple(size) != (height, width) or mask.shape != (height, width):
        from PIL import Image as PILImage

        resized = PILImage.fromarray(mask.astype(np.uint8) * 255).resize(
            (width, height), resample=PILImage.NEAREST
        )
        mask = np.asarray(resized) > 127
    return mask.astype(bool)


def save_overlay(
    image: Image.Image,
    mask: np.ndarray,
    path: Path,
    *,
    color: tuple[int, int, int] = (0, 180, 255),
    alpha: int = 96,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = image.convert("RGBA")
    binary = np.asarray(mask, dtype=bool)
    if binary.shape != (rgb.size[1], rgb.size[0]):
        binary = np.array(
            Image.fromarray(binary.astype(np.uint8) * 255).resize(rgb.size, resample=Image.NEAREST)
        ) > 127
    layer = np.zeros((rgb.size[1], rgb.size[0], 4), dtype=np.uint8)
    layer[binary] = (*color, alpha)
    composed = Image.alpha_composite(rgb, Image.fromarray(layer, mode="RGBA"))
    composed.convert("RGB").save(path, quality=88)


def save_thumbnail(image: Image.Image, path: Path, size: int = 256) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    thumb = image.convert("RGB").copy()
    thumb.thumbnail((size, size))
    thumb.save(path, format="JPEG", quality=85)
