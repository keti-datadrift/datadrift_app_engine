from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from PIL import Image

from io_utils import box_from_mask, centroid_from_mask, rle_encode
from schema import validate_proposal


class UnavailableError(RuntimeError):
    pass


@dataclass
class AdapterContext:
    prompts: list[str]
    box_threshold: float = 0.25
    text_threshold: float = 0.25
    mask_threshold: float = 0.5
    max_proposals: int = 12
    family: str = "tops"


class Segmentor(Protocol):
    key: str

    def available(self) -> bool: ...

    def load(self) -> None: ...

    def infer(self, image: Image.Image, context: AdapterContext) -> list[dict[str, Any]]: ...

    def metadata(self) -> dict[str, Any]: ...


def from_pretrained(loader, model_id: str, *, download: bool, **kwargs):
    try:
        return loader(model_id, local_files_only=not download, **kwargs)
    except Exception as exc:
        mode = "download" if download else "local cache"
        raise UnavailableError(f"{model_id} unavailable from {mode}: {exc}") from exc


def resolve_device(requested: str) -> str:
    import torch

    if requested and requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def proposals_from_mask(
    mask: np.ndarray,
    *,
    label: str,
    score: float,
    proposal_id: str,
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    binary = np.asarray(mask, dtype=bool)
    height, width = binary.shape
    area_ratio = float(binary.mean()) if binary.size else 0.0
    payload = {
        "proposal_id": proposal_id,
        "label": label,
        "score": float(score),
        "box_xyxy": box_from_mask(binary),
        "mask_rle": rle_encode(binary),
        "area_ratio": area_ratio,
        "centroid_xy": centroid_from_mask(binary),
        "extras": extras or {},
        "image_size": [int(width), int(height)],
    }
    return validate_proposal(payload)


def filter_proposals(rows: list[dict[str, Any]], max_proposals: int) -> list[dict[str, Any]]:
    ranked = sorted(rows, key=lambda row: (-float(row.get("score") or 0.0), -float(row.get("area_ratio") or 0)))
    return ranked[:max_proposals]
