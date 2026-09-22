from __future__ import annotations

from typing import Any, Iterable

SPLITS = ("dev", "primary", "stress")
SCENE_TYPES = ("product_only", "model", "lookbook", "still_life", "unknown")
OCCLUSION = ("none", "partial", "heavy", "unknown")
POSES = ("frontal", "three_quarter", "side", "back", "unknown")
VIEWS = ("closeup", "half", "full", "unknown")
SUBTYPES = (
    "tshirt",
    "shirt",
    "knit",
    "hoodie",
    "sweatshirt",
    "sleeveless",
    "blouse",
    "other",
)

PROPOSAL_KEYS = (
    "proposal_id",
    "label",
    "score",
    "box_xyxy",
    "mask_rle",
    "area_ratio",
    "centroid_xy",
)

EMPTY_ANNOTATION = {
    "status": "pending",
    "target_present": None,
    "target_point_xy": None,
    "target_box_xyxy": None,
    "scene_type": None,
    "n_people": None,
    "n_companion_garments": None,
    "occlusion": None,
    "pose": None,
    "view": None,
    "notes": "",
}

VISUAL_RUBRIC = {
    "target_match": "selected mask is the sold garment, not a companion item",
    "coverage": "mask covers the target garment body without large holes",
    "leakage": "mask avoids other garments, skin, and background",
    "boundary": "edges follow the garment silhouette closely enough to crop",
    "usable": "mask is good enough to feed an embedding encoder",
}


def validate_proposal(payload: dict[str, Any]) -> dict[str, Any]:
    missing = [key for key in PROPOSAL_KEYS if key not in payload]
    if missing:
        raise ValueError(f"proposal missing keys: {missing}")
    box = payload["box_xyxy"]
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise ValueError("box_xyxy must have 4 values")
    if any(not _is_number(v) for v in box):
        raise ValueError("box_xyxy must be numeric")
    centroid = payload["centroid_xy"]
    if not isinstance(centroid, (list, tuple)) or len(centroid) != 2:
        raise ValueError("centroid_xy must have 2 values")
    rle = payload["mask_rle"]
    if not isinstance(rle, dict) or "size" not in rle or "counts" not in rle:
        raise ValueError("mask_rle must contain size and counts")
    if len(rle["size"]) != 2:
        raise ValueError("mask_rle.size must be [height, width]")
    score = float(payload["score"])
    area_ratio = float(payload["area_ratio"])
    if not 0.0 <= area_ratio <= 1.0001:
        raise ValueError("area_ratio must be in [0, 1]")
    return {
        **payload,
        "score": score,
        "area_ratio": area_ratio,
        "box_xyxy": [float(v) for v in box],
        "centroid_xy": [float(v) for v in centroid],
        "label": str(payload["label"]),
        "proposal_id": str(payload["proposal_id"]),
    }


def validate_proposals(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [validate_proposal(row) for row in rows]


def annotation_complete(annotation: dict[str, Any] | None) -> bool:
    if not isinstance(annotation, dict):
        return False
    if annotation.get("status") != "complete":
        return False
    return annotation.get("target_present") is not None


def _is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False
