from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np
from sklearn.metrics import cohen_kappa_score

from io_utils import connected_component_count, mask_iou, rle_decode


def bootstrap_ci(values: list[float], *, iterations: int = 1000, seed: int = 42) -> list[float]:
    if not values:
        return [None, None]
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=np.float64)
    means = [float(rng.choice(array, size=len(array), replace=True).mean()) for _ in range(iterations)]
    low, high = np.percentile(means, [2.5, 97.5])
    return [round(float(low), 4), round(float(high), 4)]


def duplicate_rate(proposals: list[dict[str, Any]], iou_threshold: float = 0.5) -> float:
    if len(proposals) < 2:
        return 0.0
    masks = [rle_decode(row["mask_rle"]) for row in proposals]
    pairs = 0
    dupes = 0
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            pairs += 1
            if mask_iou(masks[i], masks[j]) >= iou_threshold:
                dupes += 1
    return dupes / pairs if pairs else 0.0


def fragmentation(proposal: dict[str, Any] | None) -> float | None:
    if proposal is None:
        return None
    return float(connected_component_count(rle_decode(proposal["mask_rle"])))


def summarize_latencies(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"p50": None, "p95": None, "mean": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "p50": round(float(np.percentile(array, 50)), 4),
        "p95": round(float(np.percentile(array, 95)), 4),
        "mean": round(float(array.mean()), 4),
    }


def auto_metrics_for_split(
    rows: list[dict[str, Any]],
    *,
    split: str,
    weight_mode: str = "uniform",
) -> dict[str, Any]:
    subset = [row for row in rows if row.get("split") == split]
    if not subset:
        return {"n": 0}
    weights = np.array(_weights_for_mode(subset, weight_mode), dtype=np.float64)
    weights = weights / weights.sum()
    proposal_counts = np.array([row["proposal_count"] for row in subset], dtype=np.float64)
    has_proposal = np.array([1.0 if row["proposal_count"] > 0 else 0.0 for row in subset])
    failures = np.array([1.0 if row.get("failed") else 0.0 for row in subset])
    area = np.array([row.get("selected_area_ratio") or 0.0 for row in subset])
    dupes = np.array([row.get("duplicate_rate") or 0.0 for row in subset])
    frags = [row.get("fragmentation") for row in subset if row.get("fragmentation") is not None]
    success_values = [
        1.0 if row["target"]["top1_success"] else 0.0
        for row in subset
        if row.get("target") and row["target"].get("top1_success") is not None
    ]
    absent_fp = [
        1.0 if row["target"]["absent_false_positive"] else 0.0
        for row in subset
        if row.get("target") and row["target"].get("absent_false_positive") is not None
    ]
    usable = [
        1.0 if row.get("visual", {}).get("usable") in {1, True, "1", "true", "yes"} else 0.0
        for row in subset
        if row.get("visual") and row["visual"].get("usable") not in {None, ""}
    ]
    result = {
        "n": len(subset),
        "weight_mode": weight_mode,
        "proposal_rate": _weighted_mean(has_proposal, weights),
        "mean_proposal_count": _weighted_mean(proposal_counts, weights),
        "failure_rate": _weighted_mean(failures, weights),
        "mean_selected_area_ratio": _weighted_mean(area, weights),
        "mean_duplicate_rate": _weighted_mean(dupes, weights),
        "mean_fragmentation": round(float(np.mean(frags)), 4) if frags else None,
        "latency_sec": summarize_latencies([row["latency_sec"] for row in subset if row.get("latency_sec") is not None]),
        "peak_memory_mb": summarize_latencies(
            [row["peak_memory_mb"] for row in subset if row.get("peak_memory_mb") is not None]
        ),
        "top1_success": _mean_and_ci(success_values),
        "absent_false_positive": _mean_and_ci(absent_fp),
        "visual_usable": _mean_and_ci(usable),
        "by_source": _group_rate(subset, "source", "target"),
        "by_shoot_type": _group_rate(
            subset,
            lambda row: str((row.get("heuristic") or {}).get("shoot_type")),
            "target",
        ),
        "by_occlusion": _group_rate(
            subset,
            lambda row: str((row.get("annotation") or {}).get("occlusion") or "unknown"),
            "target",
        ),
        "by_multi_garment": _group_rate(
            subset,
            lambda row: "multi" if int((row.get("annotation") or {}).get("n_companion_garments") or 0) > 0 else "single",
            "target",
        ),
    }
    return result


def paired_success_delta(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> dict[str, Any]:
    left_map = {row["item_id"]: row for row in left}
    right_map = {row["item_id"]: row for row in right}
    deltas = []
    for item_id, left_row in left_map.items():
        right_row = right_map.get(item_id)
        if not right_row:
            continue
        left_hit = left_row.get("target", {}).get("top1_success")
        right_hit = right_row.get("target", {}).get("top1_success")
        if left_hit is None or right_hit is None:
            continue
        deltas.append(float(left_hit) - float(right_hit))
    return {"n": len(deltas), "mean": round(float(np.mean(deltas)), 4) if deltas else None, "ci95": bootstrap_ci(deltas)}


def reviewer_kappa(rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]], field: str = "usable") -> dict[str, Any]:
    map_a = {(row["item_id"], row.get("model_key"), row.get("resolver")): row for row in rows_a}
    labels_a = []
    labels_b = []
    for row in rows_b:
        key = (row["item_id"], row.get("model_key"), row.get("resolver"))
        other = map_a.get(key)
        if not other:
            continue
        if row.get(field) in {None, ""} or other.get(field) in {None, ""}:
            continue
        labels_a.append(str(other[field]))
        labels_b.append(str(row[field]))
    if len(labels_a) < 2:
        return {"n": len(labels_a), "kappa": None}
    return {
        "n": len(labels_a),
        "kappa": round(float(cohen_kappa_score(labels_a, labels_b, weights="quadratic")), 4),
        "field": field,
    }


def _weights_for_mode(rows: list[dict[str, Any]], mode: str) -> list[float]:
    if mode == "population":
        return [float(row.get("sampling_weight") or 1.0) for row in rows]
    if mode == "source_balanced":
        counts = Counter(str(row.get("source") or "unknown") for row in rows)
        return [1.0 / counts[str(row.get("source") or "unknown")] for row in rows]
    return [1.0 for _ in rows]


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    if not len(values):
        return 0.0
    return round(float(np.sum(values * weights)), 4)


def _mean_and_ci(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "rate": None, "ci95": [None, None]}
    rate = float(np.mean(values))
    return {"n": len(values), "rate": round(rate, 4), "ci95": bootstrap_ci(values)}


def _group_rate(rows: list[dict[str, Any]], key: Any, target_field: str) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = key(row) if callable(key) else str(row.get(key))
        hit = (row.get(target_field) or {}).get("top1_success")
        if hit is None:
            continue
        grouped[str(value)].append(1.0 if hit else 0.0)
    return {name: _mean_and_ci(vals) for name, vals in grouped.items()}


def merge_visual_scores(rows: list[dict[str, Any]], reviews: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {(row["item_id"], row.get("model_key"), row.get("resolver")): row for row in reviews}
    merged = []
    for row in rows:
        key = (row["item_id"], row.get("model_key"), row.get("resolver"))
        review = lookup.get(key)
        payload = dict(row)
        if review:
            payload["visual"] = {
                "target_match": _to_score(review.get("target_match")),
                "coverage": _to_score(review.get("coverage")),
                "leakage": _to_score(review.get("leakage")),
                "boundary": _to_score(review.get("boundary")),
                "usable": review.get("usable"),
                "reviewer": review.get("reviewer"),
            }
        merged.append(payload)
    return merged


def _to_score(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
