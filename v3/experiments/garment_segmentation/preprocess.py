from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage

from io_utils import mask_from_box, restore_mask

ALL_ARMS = ("A", "B", "C", "D")
ARMS = ALL_ARMS


def resolve_arms(names: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    if not names:
        return ALL_ARMS
    ordered = tuple(str(name).strip().upper() for name in names)
    unknown = [name for name in ordered if name not in ALL_ARMS]
    if unknown:
        raise ValueError(f"unknown arms: {unknown}")
    if "A" not in ordered:
        raise ValueError("arm A is required as the unprocessed reference")
    return tuple(name for name in ALL_ARMS if name in ordered)

DEFAULT_GARMENT_LABELS = (
    "upper-clothes",
    "dress",
    "top",
    "shirt",
    "blouse",
    "knit",
    "hoodie",
    "sweater",
    "t-shirt",
    "tshirt",
)


def _as_box(box: list[float] | tuple[float, ...]) -> list[float]:
    if box is None or len(box) != 4:
        raise ValueError("box_xyxy must have 4 values")
    return [float(v) for v in box]


def pad_box(
    box: list[float] | tuple[float, ...],
    width: int,
    height: int,
    pad_ratio: float = 0.12,
) -> list[float]:
    x0, y0, x1, y1 = _as_box(box)
    bw = max(x1 - x0, 1.0)
    bh = max(y1 - y0, 1.0)
    px = bw * float(pad_ratio)
    py = bh * float(pad_ratio)
    nx0 = max(0.0, x0 - px)
    ny0 = max(0.0, y0 - py)
    nx1 = min(float(width), x1 + px)
    ny1 = min(float(height), y1 + py)
    if nx1 <= nx0:
        nx1 = min(float(width), nx0 + 1.0)
    if ny1 <= ny0:
        ny1 = min(float(height), ny0 + 1.0)
    return [nx0, ny0, nx1, ny1]


def crop_xyxy(box: list[float] | tuple[float, ...]) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = _as_box(box)
    ix0 = int(np.floor(x0))
    iy0 = int(np.floor(y0))
    ix1 = int(np.ceil(x1))
    iy1 = int(np.ceil(y1))
    if ix1 <= ix0:
        ix1 = ix0 + 1
    if iy1 <= iy0:
        iy1 = iy0 + 1
    return ix0, iy0, ix1, iy1


def _norm_label(label: str | None) -> str:
    return str(label or "").strip().lower().replace("_", "-")


def is_garment_label(label: str | None, garment_labels: tuple[str, ...] | list[str] = DEFAULT_GARMENT_LABELS) -> bool:
    text = _norm_label(label)
    allowed = {_norm_label(name) for name in garment_labels}
    if text in allowed:
        return True
    tokens = {part.strip() for part in text.replace(",", " ").split() if part.strip()}
    return bool(tokens & allowed)


def _proposal_mask(proposal: dict[str, Any] | None, height: int, width: int) -> np.ndarray:
    if not proposal:
        return np.zeros((height, width), dtype=bool)
    return restore_mask(proposal, height, width)


def resolve_mask_in_box(
    *,
    height: int,
    width: int,
    box_xyxy: list[float],
    proposals: list[dict[str, Any]] | None,
    combined_selected: dict[str, Any] | None,
    garment_labels: tuple[str, ...] | list[str] | None = None,
    min_mask_fill: float = 0.08,
) -> dict[str, Any]:
    garment_labels = tuple(garment_labels) if garment_labels else DEFAULT_GARMENT_LABELS
    box_mask = mask_from_box(height, width, box_xyxy)
    box_area = float(box_mask.sum())
    union = np.zeros((height, width), dtype=bool)
    used_ids: list[str] = []
    used_labels: list[str] = []
    for proposal in proposals or []:
        if not is_garment_label(proposal.get("label"), garment_labels):
            continue
        mask = _proposal_mask(proposal, height, width)
        overlap = np.logical_and(mask, box_mask)
        if not overlap.any():
            continue
        union = np.logical_or(union, overlap)
        used_ids.append(str(proposal.get("proposal_id") or ""))
        used_labels.append(str(proposal.get("label") or ""))
    clipped = np.logical_and(union, box_mask)
    fill = float(clipped.sum() / box_area) if box_area else 0.0
    source = "garment_in_box"
    if fill < float(min_mask_fill) and combined_selected:
        if is_garment_label(combined_selected.get("label"), garment_labels):
            fallback = np.logical_and(_proposal_mask(combined_selected, height, width), box_mask)
            fallback_fill = float(fallback.sum() / box_area) if box_area else 0.0
            if fallback_fill >= float(min_mask_fill):
                clipped = fallback
                fill = fallback_fill
                source = "combined_in_box"
                used_ids = [str(combined_selected.get("proposal_id") or "")]
                used_labels = [str(combined_selected.get("label") or "")]
    if fill < float(min_mask_fill) or not clipped.any():
        clipped = box_mask
        fill = 1.0 if box_area else 0.0
        source = "box_fallback"
        used_ids = []
        used_labels = []
    return {
        "mask": clipped,
        "source": source,
        "fill_ratio": fill,
        "proposal_ids": used_ids,
        "labels": used_labels,
        "combined_label": None if combined_selected is None else combined_selected.get("label"),
        "combined_is_garment": is_garment_label(
            None if combined_selected is None else combined_selected.get("label"),
            garment_labels,
        ),
    }


def crop_array(image: Image.Image | np.ndarray, box_xyxy: list[float]) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB") if isinstance(image, Image.Image) else image)
    height, width = rgb.shape[:2]
    x0, y0, x1, y1 = crop_xyxy(box_xyxy)
    x0 = min(max(x0, 0), width)
    x1 = min(max(x1, 0), width)
    y0 = min(max(y0, 0), height)
    y1 = min(max(y1, 0), height)
    if x1 <= x0 or y1 <= y0:
        return rgb
    return rgb[y0:y1, x0:x1]


def crop_mask(mask: np.ndarray, box_xyxy: list[float]) -> np.ndarray:
    height, width = mask.shape[:2]
    x0, y0, x1, y1 = crop_xyxy(box_xyxy)
    x0 = min(max(x0, 0), width)
    x1 = min(max(x1, 0), width)
    y0 = min(max(y0, 0), height)
    y1 = min(max(y1, 0), height)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((1, 1), dtype=bool)
    return np.asarray(mask[y0:y1, x0:x1], dtype=bool)


def _gaussian_alpha(mask: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return mask.astype(np.float32)
    return np.clip(ndimage.gaussian_filter(mask.astype(np.float32), sigma=float(sigma)), 0.0, 1.0)


def compose_soft(
    crop_rgb: np.ndarray,
    crop_mask: np.ndarray,
    *,
    dilate_px: int = 7,
    alpha_sigma: float = 5.0,
    background_blur_sigma: float = 21.0,
) -> Image.Image:
    rgb = np.asarray(crop_rgb, dtype=np.float32)
    mask = np.asarray(crop_mask, dtype=bool)
    if mask.shape[:2] != rgb.shape[:2]:
        raise ValueError("crop mask shape must match crop image")
    if dilate_px > 0:
        mask = ndimage.binary_dilation(mask, iterations=int(dilate_px))
    alpha = _gaussian_alpha(mask, alpha_sigma)[..., None]
    blurred = np.stack(
        [
            ndimage.gaussian_filter(rgb[:, :, channel], sigma=float(background_blur_sigma))
            for channel in range(3)
        ],
        axis=-1,
    )
    composed = rgb * alpha + blurred * (1.0 - alpha)
    return Image.fromarray(np.clip(composed, 0, 255).astype(np.uint8), mode="RGB")


def compose_hard(crop_rgb: np.ndarray, crop_mask: np.ndarray, fill: int = 0) -> Image.Image:
    rgb = np.asarray(crop_rgb, dtype=np.uint8).copy()
    mask = np.asarray(crop_mask, dtype=bool)
    if mask.shape[:2] != rgb.shape[:2]:
        raise ValueError("crop mask shape must match crop image")
    rgb[~mask] = int(fill)
    return Image.fromarray(rgb, mode="RGB")


def build_arm_images(
    image: Image.Image,
    *,
    box_xyxy: list[float],
    mask: np.ndarray,
    pad_ratio: float = 0.12,
    dilate_px: int = 7,
    alpha_sigma: float = 5.0,
    background_blur_sigma: float = 21.0,
    arms: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Image.Image]:
    wanted = resolve_arms(arms)
    rgb = image.convert("RGB")
    width, height = rgb.size
    out: dict[str, Image.Image] = {}
    if "A" in wanted:
        out["A"] = rgb
    if not any(name in wanted for name in ("B", "C", "D")):
        return out
    padded = pad_box(box_xyxy, width, height, pad_ratio=pad_ratio)
    crop = crop_array(rgb, padded)
    crop_binary = crop_mask(mask, padded)
    if crop_binary.shape[:2] != crop.shape[:2]:
        crop_binary = np.ones(crop.shape[:2], dtype=bool)
    if "B" in wanted:
        out["B"] = Image.fromarray(crop, mode="RGB")
    if "C" in wanted:
        out["C"] = compose_soft(
            crop,
            crop_binary,
            dilate_px=dilate_px,
            alpha_sigma=alpha_sigma,
            background_blur_sigma=background_blur_sigma,
        )
    if "D" in wanted:
        out["D"] = compose_hard(crop, crop_binary)
    return out
