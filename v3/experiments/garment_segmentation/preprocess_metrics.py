from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.manifold import trustworthiness
from sklearn.metrics import normalized_mutual_info_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

from preprocess import ALL_ARMS, ARMS


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)


def bootstrap_ci(values: list[float] | np.ndarray, iterations: int, seed: int) -> tuple[float, float]:
    if len(values) == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    means = [float(rng.choice(arr, len(arr), replace=True).mean()) for _ in range(int(iterations))]
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def linear_cka(left: np.ndarray, right: np.ndarray) -> float:
    """Linear Centered Kernel Alignment between two embedding matrices."""

    x = left - left.mean(axis=0, keepdims=True)
    y = right - right.mean(axis=0, keepdims=True)
    xy = x.T @ y
    xx = x.T @ x
    yy = y.T @ y
    hsic_xy = float(np.linalg.norm(xy, ord="fro") ** 2)
    hsic_xx = float(np.linalg.norm(xx, ord="fro") ** 2)
    hsic_yy = float(np.linalg.norm(yy, ord="fro") ** 2)
    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom <= 1e-12:
        return 0.0
    return float(hsic_xy / denom)


def mean_item_cosine(left: np.ndarray, right: np.ndarray) -> float:
    left_n = l2_normalize(left)
    right_n = l2_normalize(right)
    return float(np.mean(np.sum(left_n * right_n, axis=1)))


def distance_spearman(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 3:
        return float("nan")
    stat = spearmanr(pdist(left, metric="cosine"), pdist(right, metric="cosine")).statistic
    return float(stat)


def _neighbor_sets(matrix: np.ndarray, k: int) -> list[set[int]]:
    count = min(int(k) + 1, len(matrix))
    neighbors = NearestNeighbors(n_neighbors=count, metric="cosine", algorithm="brute").fit(matrix)
    indices = neighbors.kneighbors(return_distance=False)
    result: list[set[int]] = []
    for index, row in enumerate(indices):
        result.append({int(value) for value in row if int(value) != index})
    return result


def neighborhood_jaccard(left: np.ndarray, right: np.ndarray, k: int) -> float:
    if len(left) < 3:
        return float("nan")
    left_sets = _neighbor_sets(left, k)
    right_sets = _neighbor_sets(right, k)
    scores = [
        len(a & b) / max(1, len(a | b))
        for a, b in zip(left_sets, right_sets, strict=True)
    ]
    return float(np.mean(scores))


def intra_inter_gap(matrix: np.ndarray, labels: list[set[str] | None]) -> dict[str, Any]:
    labeled = [i for i, label in enumerate(labels) if label]
    if len(labeled) < 4:
        return {"gap": None, "intra": None, "inter": None, "pairs_intra": 0, "pairs_inter": 0}
    vectors = l2_normalize(matrix[labeled])
    groups = [labels[i] or set() for i in labeled]
    intra: list[float] = []
    inter: list[float] = []
    for i in range(len(labeled)):
        for j in range(i + 1, len(labeled)):
            sim = float(np.dot(vectors[i], vectors[j]))
            if groups[i] & groups[j]:
                intra.append(sim)
            else:
                inter.append(sim)
    if not intra or not inter:
        return {
            "gap": None,
            "intra": None if not intra else round(float(np.mean(intra)), 4),
            "inter": None if not inter else round(float(np.mean(inter)), 4),
            "pairs_intra": len(intra),
            "pairs_inter": len(inter),
        }
    intra_mean = float(np.mean(intra))
    inter_mean = float(np.mean(inter))
    return {
        "gap": round(intra_mean - inter_mean, 4),
        "intra": round(intra_mean, 4),
        "inter": round(inter_mean, 4),
        "pairs_intra": len(intra),
        "pairs_inter": len(inter),
    }


def attribute_precision_at_k(
    matrix: np.ndarray,
    items: list[dict[str, Any]],
    *,
    k: int,
    bootstrap_iterations: int,
    seed: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    n_neighbors = min(int(k) + 1, len(items))
    if n_neighbors < 2:
        return {"macro": 0.0}
    neighbors = NearestNeighbors(
        n_neighbors=n_neighbors, metric="cosine", algorithm="brute"
    ).fit(matrix)
    indices = neighbors.kneighbors(return_distance=False)
    for label_type in ("color", "pattern", "material", "shape"):
        scores: list[float] = []
        for index, item in enumerate(items):
            expected = set(item.get("weak_labels", {}).get(label_type) or [])
            if not expected:
                continue
            hits: list[float] = []
            for neighbor in indices[index]:
                if int(neighbor) == index:
                    continue
                actual = set(items[int(neighbor)].get("weak_labels", {}).get(label_type) or [])
                hits.append(float(bool(expected & actual)))
                if len(hits) >= k:
                    break
            if hits:
                scores.append(float(np.mean(hits)))
        low, high = bootstrap_ci(scores, bootstrap_iterations, seed)
        result[label_type] = {
            "precision_at_k": round(float(np.mean(scores)) if scores else 0.0, 4),
            "ci95": [round(low, 4), round(high, 4)],
            "queries": len(scores),
            "k": int(k),
            "label_source": "product_title_weak_labels",
        }
    values = [row["precision_at_k"] for row in result.values() if row["queries"]]
    result["macro"] = round(float(np.mean(values)) if values else 0.0, 4)
    return result


def silhouette_for_labels(matrix: np.ndarray, labels: list[str | None]) -> float | None:
    valid = [(i, label) for i, label in enumerate(labels) if label]
    unique = {label for _, label in valid}
    if len(valid) < 6 or len(unique) < 2:
        return None
    idx = np.array([i for i, _ in valid], dtype=int)
    y = np.array([label for _, label in valid])
    if min(int((y == value).sum()) for value in unique) < 2:
        return None
    return round(float(silhouette_score(matrix[idx], y, metric="cosine")), 4)


def projection_trustworthiness(matrix: np.ndarray, coords: np.ndarray, n_neighbors: int = 10) -> float | None:
    if len(matrix) < 8:
        return None
    eval_n = min(800, len(matrix))
    k = min(int(n_neighbors), max(1, eval_n // 3))
    return round(
        float(trustworthiness(matrix[:eval_n], coords[:eval_n], n_neighbors=k, metric="cosine")),
        4,
    )


def cluster_label_nmi(cluster_labels: np.ndarray, item_labels: list[str | None]) -> float | None:
    valid = [
        (int(cluster), label)
        for cluster, label in zip(cluster_labels, item_labels, strict=True)
        if cluster >= 0 and label
    ]
    if len(valid) < 8:
        return None
    clusters = [row[0] for row in valid]
    labels = [row[1] for row in valid]
    if len(set(clusters)) < 2 or len(set(labels)) < 2:
        return None
    return round(float(normalized_mutual_info_score(labels, [str(v) for v in clusters])), 4)


def representation_shift(reference: np.ndarray, other: np.ndarray, k: int) -> dict[str, Any]:
    return {
        "mean_item_cosine": round(mean_item_cosine(reference, other), 4),
        "linear_cka": round(linear_cka(reference, other), 4),
        "distance_spearman": round(distance_spearman(reference, other), 4)
        if len(reference) >= 3
        else None,
        "neighborhood_jaccard": round(neighborhood_jaccard(reference, other, k), 4)
        if len(reference) >= 3
        else None,
    }


def summarize_mask_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    fills = [float(row.get("fill_ratio") or 0.0) for row in rows]
    sources = {}
    combined_garment = 0
    for row in rows:
        source = str(row.get("mask_source") or row.get("source") or "unknown")
        sources[source] = sources.get(source, 0) + 1
        if row.get("combined_is_garment"):
            combined_garment += 1
    return {
        "n": len(rows),
        "mean_fill_ratio": round(float(np.mean(fills)), 4),
        "p50_fill_ratio": round(float(np.quantile(fills, 0.5)), 4),
        "fallback_rate": round(sources.get("box_fallback", 0) / len(rows), 4),
        "source_counts": sources,
        "combined_garment_rate": round(combined_garment / len(rows), 4),
    }


def compare_arms(
    embeddings: dict[str, np.ndarray],
    items: list[dict[str, Any]],
    *,
    k: int,
    bootstrap_iterations: int,
    seed: int,
) -> dict[str, Any]:
    reference = embeddings["A"]
    weak_sets = {
        kind: [set(item.get("weak_labels", {}).get(kind) or []) or None for item in items]
        for kind in ("color", "pattern", "material", "shape")
    }
    color_first = [
        (item.get("weak_labels", {}).get("color") or [None])[0] for item in items
    ]
    subtype = [str(item.get("subtype") or "other") for item in items]
    payload: dict[str, Any] = {"arms": {}, "pairwise_vs_A": {}, "deltas_vs_A": {}}
    for arm in (name for name in ALL_ARMS if name in embeddings):
        matrix = embeddings[arm]
        attr = attribute_precision_at_k(
            matrix,
            items,
            k=k,
            bootstrap_iterations=bootstrap_iterations,
            seed=seed,
        )
        gaps = {kind: intra_inter_gap(matrix, labels) for kind, labels in weak_sets.items()}
        payload["arms"][arm] = {
            "attribute_neighbors": attr,
            "intra_inter": gaps,
            "color_silhouette": silhouette_for_labels(matrix, color_first),
            "subtype_silhouette": silhouette_for_labels(matrix, subtype),
        }
        if arm != "A":
            payload["pairwise_vs_A"][arm] = representation_shift(reference, matrix, k)
            payload["deltas_vs_A"][arm] = {
                "macro_precision_at_k": round(
                    float(attr["macro"]) - float(payload["arms"]["A"]["attribute_neighbors"]["macro"]),
                    4,
                ),
                "color_gap": _gap_delta(payload["arms"]["A"]["intra_inter"]["color"], gaps["color"]),
                "pattern_gap": _gap_delta(payload["arms"]["A"]["intra_inter"]["pattern"], gaps["pattern"]),
                "shape_gap": _gap_delta(payload["arms"]["A"]["intra_inter"]["shape"], gaps["shape"]),
            }
    return payload


def _gap_delta(left: dict[str, Any], right: dict[str, Any]) -> float | None:
    if left.get("gap") is None or right.get("gap") is None:
        return None
    return round(float(right["gap"]) - float(left["gap"]), 4)


def _round(value: float | None) -> float | None:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    return round(float(value), 4)


def centroid_radii(matrix: np.ndarray) -> np.ndarray:
    vectors = l2_normalize(np.asarray(matrix, dtype=np.float32))
    center = l2_normalize(vectors.mean(axis=0, keepdims=True))[0]
    return (1.0 - vectors @ center).astype(np.float64)


def mean_knn_cosine_distance(matrix: np.ndarray, k: int = 10) -> float | None:
    vectors = l2_normalize(np.asarray(matrix, dtype=np.float32))
    if len(vectors) < 3:
        return None
    count = min(int(k) + 1, len(vectors) - 1)
    if count < 2:
        return None
    distances = (
        NearestNeighbors(n_neighbors=count, metric="cosine", algorithm="brute")
        .fit(vectors)
        .kneighbors()[0]
    )
    return float(np.mean(distances[:, 1:]))


def mean_pairwise_cosine_distance(matrix: np.ndarray) -> float | None:
    vectors = l2_normalize(np.asarray(matrix, dtype=np.float32))
    if len(vectors) < 2:
        return None
    sims = vectors @ vectors.T
    tri = np.triu_indices(len(vectors), k=1)
    return float(np.mean(1.0 - sims[tri]))


def hull_area_2d(points: np.ndarray) -> float | None:
    if points is None or len(points) < 3:
        return None
    from scipy.spatial import ConvexHull

    unique = np.unique(np.round(np.asarray(points, dtype=float), 5), axis=0)
    if len(unique) < 3:
        return None
    try:
        return float(ConvexHull(unique).volume)
    except Exception:
        return None


def knn_label_purity(matrix: np.ndarray, labels: list[str], k: int) -> float | None:
    if len(matrix) < 3:
        return None
    count = min(int(k) + 1, len(matrix) - 1)
    if count < 2:
        return None
    neighbors = (
        NearestNeighbors(n_neighbors=count, metric="cosine", algorithm="brute")
        .fit(matrix)
        .kneighbors(return_distance=False)
    )
    scores: list[float] = []
    for index, row in enumerate(neighbors):
        hits = []
        for neighbor in row:
            if int(neighbor) == index:
                continue
            hits.append(float(labels[int(neighbor)] == labels[index]))
            if len(hits) >= k:
                break
        if hits:
            scores.append(float(np.mean(hits)))
    if not scores:
        return None
    return float(np.mean(scores))


def _subset_compactness(
    matrix: np.ndarray,
    idx: np.ndarray,
    *,
    k: int,
    coords: np.ndarray | None,
) -> dict[str, Any]:
    subset = matrix[idx]
    radii = centroid_radii(subset)
    p50 = float(np.quantile(radii, 0.5))
    p95 = float(np.quantile(radii, 0.95))
    core_idx = idx[radii <= p50] if np.any(radii <= p50) else idx
    local_k = min(int(k), max(1, len(subset) - 1))
    core_k = min(int(k), max(1, len(core_idx) - 1))
    hull = None
    core_hull = None
    if coords is not None:
        hull = hull_area_2d(coords[idx])
        core_hull = hull_area_2d(coords[core_idx])
    return {
        "n": int(len(idx)),
        "centroid_radius_p50": _round(p50),
        "centroid_radius_p95": _round(p95),
        "outlier_span_ratio": _round(p95 / max(p50, 1e-8)),
        "mean_knn_cosine": _round(mean_knn_cosine_distance(subset, local_k)),
        "core_knn_cosine": _round(mean_knn_cosine_distance(matrix[core_idx], core_k)),
        "within_pairwise_cosine": _round(mean_pairwise_cosine_distance(subset)),
        "hull_area_2d": _round(hull),
        "core_hull_area_2d": _round(core_hull),
    }


def density_alignment_metrics(
    embeddings: dict[str, np.ndarray],
    items: list[dict[str, Any]],
    aligned: dict[str, np.ndarray] | None,
    *,
    k: int,
    min_group: int = 8,
) -> dict[str, Any]:
    subtypes = [str(item.get("subtype") or "other") for item in items]
    groups: dict[str, np.ndarray] = {}
    for name in sorted(set(subtypes)):
        idx = np.array([i for i, value in enumerate(subtypes) if value == name], dtype=int)
        if len(idx) >= min_group:
            groups[name] = idx
    payload: dict[str, Any] = {"arms": {}, "interpretation": {
        "better_core_density": "core_knn_cosine 과 centroid_radius_p50 가 작을수록 핵심이 더 밀집",
        "outlier_inflated_extent": "outlier_span_ratio = p95/p50 가 크면 먼 점 때문에 범위만 넓어짐",
        "better_alignment": "subtype_knn_purity 와 subtype_separation 이 클수록 제품군 관계가 더 잘 갈림",
    }}
    for arm in (name for name in ALL_ARMS if name in embeddings):
        matrix = embeddings[arm]
        coords = None if aligned is None else aligned[arm]
        overall = _subset_compactness(
            matrix, np.arange(len(items)), k=k, coords=coords
        )
        per_group = {
            name: _subset_compactness(matrix, idx, k=k, coords=coords)
            for name, idx in groups.items()
        }
        within_vals = [row["within_pairwise_cosine"] for row in per_group.values() if row["within_pairwise_cosine"] is not None]
        knn_vals = [row["mean_knn_cosine"] for row in per_group.values() if row["mean_knn_cosine"] is not None]
        p50_vals = [row["centroid_radius_p50"] for row in per_group.values() if row["centroid_radius_p50"] is not None]
        core_vals = [row["core_knn_cosine"] for row in per_group.values() if row["core_knn_cosine"] is not None]
        hull_vals = [row["hull_area_2d"] for row in per_group.values() if row["hull_area_2d"] is not None]
        core_hull_vals = [row["core_hull_area_2d"] for row in per_group.values() if row["core_hull_area_2d"] is not None]
        centroids = []
        for idx in groups.values():
            vectors = l2_normalize(matrix[idx])
            centroids.append(l2_normalize(vectors.mean(axis=0, keepdims=True))[0])
        between = None
        if len(centroids) >= 2:
            stacked = np.stack(centroids, axis=0)
            between = mean_pairwise_cosine_distance(stacked)
        within_macro = float(np.mean(within_vals)) if within_vals else None
        payload["arms"][arm] = {
            "overall": overall,
            "subtype_macro": {
                "centroid_radius_p50": _round(float(np.mean(p50_vals)) if p50_vals else None),
                "mean_knn_cosine": _round(float(np.mean(knn_vals)) if knn_vals else None),
                "core_knn_cosine": _round(float(np.mean(core_vals)) if core_vals else None),
                "within_pairwise_cosine": _round(within_macro),
                "hull_area_2d": _round(float(np.mean(hull_vals)) if hull_vals else None),
                "core_hull_area_2d": _round(float(np.mean(core_hull_vals)) if core_hull_vals else None),
            },
            "subtype_separation": _round(
                None if between is None or within_macro is None else float(between) - float(within_macro)
            ),
            "between_centroid_cosine": _round(between),
            "subtype_knn_purity": _round(knn_label_purity(matrix, subtypes, k)),
            "by_subtype": per_group,
        }
    return payload
