from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
from PIL import Image

from io_utils import restore_mask
from schema import annotation_complete

RESOLVER_NAMES = ("max_area", "torso_prior", "metadata", "combined")


def torso_prior_score(proposal: dict[str, Any], width: int, height: int) -> float:
    cx, cy = proposal["centroid_xy"]
    target_x = width / 2.0
    target_y = height * 0.38
    dx = (cx - target_x) / max(width, 1)
    dy = (cy - target_y) / max(height, 1)
    dist = float(np.hypot(dx, dy))
    return float(np.clip(1.0 - dist / 0.75, 0.0, 1.0))


def area_score(proposal: dict[str, Any]) -> float:
    return float(np.clip(proposal.get("area_ratio") or 0.0, 0.0, 1.0))


def confidence_score(proposal: dict[str, Any]) -> float:
    return float(np.clip(proposal.get("score") or 0.0, 0.0, 1.0))


def crop_from_proposal(image: Image.Image, proposal: dict[str, Any]) -> Image.Image:
    width, height = image.size
    mask = restore_mask(proposal, height, width)
    box = [int(round(v)) for v in proposal["box_xyxy"]]
    x0, y0, x1, y1 = box
    x0 = min(max(x0, 0), width)
    x1 = min(max(x1, x0 + 1), width)
    y0 = min(max(y0, 0), height)
    y1 = min(max(y1, y0 + 1), height)
    rgb = np.asarray(image.convert("RGB"))
    crop = rgb[y0:y1, x0:x1].copy()
    local = mask[y0:y1, x0:x1]
    if local.shape[:2] == crop.shape[:2] and local.any():
        gray = np.full_like(crop, 240)
        crop = np.where(local[:, :, None], crop, gray)
    return Image.fromarray(crop)


class MetadataEncoder:
    def __init__(self, model_id: str, requested_device: str = "auto") -> None:
        self.model_id = model_id
        self.requested_device = requested_device
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self.device = "cpu"
        self.available = False
        self.error: str | None = None

    def load(self) -> None:
        import sys
        from pathlib import Path

        attr = Path(__file__).resolve().parent.parent / "attribute_embedding"
        if str(attr) not in sys.path:
            sys.path.insert(0, str(attr))
        try:
            from representations import _device, load_model

            self.model, self.preprocess, self.tokenizer, self.device = load_model(
                self.model_id, self.requested_device
            )
            self.available = True
        except Exception as exc:
            self.error = str(exc)
            self.available = False

    def embed_images(self, images: list[Image.Image]) -> np.ndarray:
        import torch

        if not self.available or not images:
            return np.zeros((len(images), 1), dtype=np.float32)
        tensors = [self.preprocess(image.convert("RGB")) for image in images]
        batch = torch.stack(tensors).to(self.device)
        with torch.inference_mode():
            encoded = self.model.encode_image(batch, normalize=True)
        return encoded.detach().float().cpu().numpy()

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        import torch

        if not self.available or not texts:
            return np.zeros((len(texts), 1), dtype=np.float32)
        tokens = self.tokenizer(texts).to(self.device)
        with torch.inference_mode():
            encoded = self.model.encode_text(tokens, normalize=True)
        return encoded.detach().float().cpu().numpy()


def metadata_text(item: dict[str, Any]) -> str:
    product = str(item.get("product") or "").strip()
    category = str(item.get("category_label") or "top")
    subtype = str(item.get("subtype") or "")
    return f"an ecommerce product photo of the sold {subtype or 'top'} garment: {product} ({category})"


def score_proposals(
    item: dict[str, Any],
    image: Image.Image,
    proposals: list[dict[str, Any]],
    *,
    encoder: MetadataEncoder | None,
    weights: dict[str, float],
) -> list[dict[str, Any]]:
    width, height = image.size
    meta_scores = np.zeros(len(proposals), dtype=np.float32)
    if encoder is not None and encoder.available and proposals:
        crops = [crop_from_proposal(image, proposal) for proposal in proposals]
        image_vectors = encoder.embed_images(crops)
        text_vector = encoder.embed_texts([metadata_text(item)])[0]
        denom = np.clip(np.linalg.norm(image_vectors, axis=1) * np.linalg.norm(text_vector), 1e-12, None)
        meta_scores = (image_vectors @ text_vector) / denom
        meta_scores = np.clip((meta_scores + 1.0) / 2.0, 0.0, 1.0)
    scored = []
    for index, proposal in enumerate(proposals):
        parts = {
            "area": area_score(proposal),
            "torso": torso_prior_score(proposal, width, height),
            "confidence": confidence_score(proposal),
            "metadata": float(meta_scores[index]),
        }
        combined = (
            weights.get("area", 0.25) * parts["area"]
            + weights.get("torso", 0.2) * parts["torso"]
            + weights.get("confidence", 0.15) * parts["confidence"]
            + weights.get("metadata", 0.4) * parts["metadata"]
        )
        scored.append(
            {
                **proposal,
                "resolver_scores": {**parts, "combined": float(combined)},
            }
        )
    return scored


def select_with_resolver(scored: list[dict[str, Any]], resolver: str) -> dict[str, Any] | None:
    if not scored:
        return None
    key_map = {
        "max_area": "area",
        "torso_prior": "torso",
        "metadata": "metadata",
        "combined": "combined",
    }
    if resolver not in key_map:
        raise ValueError(f"unknown resolver: {resolver}")
    field = key_map[resolver]
    return max(scored, key=lambda row: (float(row["resolver_scores"][field]), float(row["area_ratio"])))


def point_in_mask(point_xy: list[float], mask: np.ndarray) -> bool:
    if not point_xy or len(point_xy) != 2:
        return False
    x, y = float(point_xy[0]), float(point_xy[1])
    height, width = mask.shape
    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
        px = int(round(x * (width - 1)))
        py = int(round(y * (height - 1)))
    else:
        px, py = int(round(x)), int(round(y))
    if not (0 <= px < width and 0 <= py < height):
        return False
    return bool(mask[py, px])


def target_hit(item: dict[str, Any], proposal: dict[str, Any] | None, image_size: tuple[int, int]) -> dict[str, Any]:
    annotation = item.get("annotation") or {}
    result = {
        "annotated": annotation_complete(annotation),
        "target_present": annotation.get("target_present"),
        "has_proposal": proposal is not None,
        "top1_success": None,
        "absent_false_positive": None,
    }
    if not result["annotated"]:
        return result
    if annotation.get("target_present") is False:
        result["absent_false_positive"] = bool(proposal is not None)
        result["top1_success"] = proposal is None
        return result
    if proposal is None:
        result["top1_success"] = False
        return result
    point = annotation.get("target_point_xy")
    if point:
        width, height = image_size
        mask = restore_mask(proposal, height, width)
        result["top1_success"] = point_in_mask(point, mask)
        return result
    result["top1_success"] = None
    return result


def freeze_resolver_config(weights: dict[str, float], prompts: list[str]) -> dict[str, Any]:
    raw = json_ready({"weights": weights, "prompts": prompts})
    digest = hashlib.sha1(repr(raw).encode()).hexdigest()[:12]
    return {**raw, "config_hash": digest}


def json_ready(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {str(k): json_ready(v) for k, v in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [json_ready(v) for v in payload]
    if isinstance(payload, (np.floating, np.integer)):
        return payload.item()
    return payload
