from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import hdbscan
import numpy as np
import umap
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import trustworthiness
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import NearestNeighbors


def _reduce_for_analysis(matrix: np.ndarray, seed: int, max_dim: int = 64) -> np.ndarray:
    n_components = min(max_dim, matrix.shape[0] - 1, matrix.shape[1])
    if matrix.shape[1] <= n_components:
        return matrix
    return PCA(n_components=n_components, random_state=seed).fit_transform(matrix)


def project(
    matrix: np.ndarray,
    *,
    seed: int,
    n_neighbors: int,
    min_dist: float,
) -> tuple[np.ndarray, dict[str, float]]:
    reduced = _reduce_for_analysis(matrix, seed)
    reducer = umap.UMAP(
        n_components=2,
        metric="cosine",
        n_neighbors=min(n_neighbors, max(2, len(matrix) - 1)),
        min_dist=min_dist,
        random_state=seed,
        n_jobs=1,
    )
    coords = reducer.fit_transform(reduced)
    eval_n = min(800, len(matrix))
    quality = {
        "trustworthiness": round(
            float(
                trustworthiness(
                    matrix[:eval_n],
                    coords[:eval_n],
                    n_neighbors=min(10, max(1, eval_n // 3)),
                    metric="cosine",
                )
            ),
            4,
        )
    }
    return coords.astype(np.float32), quality


def _fit_cluster(matrix: np.ndarray, min_cluster_size: int, min_samples: int) -> tuple[np.ndarray, np.ndarray]:
    reduced = _reduce_for_analysis(matrix, seed=42)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min(min_cluster_size, max(5, len(matrix) // 3)),
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=False,
    )
    labels = clusterer.fit_predict(reduced)
    probabilities = getattr(clusterer, "probabilities_", np.ones(len(labels)))
    return labels.astype(int), np.asarray(probabilities, dtype=np.float32)


def cluster_arm(
    arm: str,
    matrix: np.ndarray,
    items: list[dict[str, Any]],
    *,
    min_cluster_size: int,
    min_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    if arm != "B":
        return _fit_cluster(matrix, min_cluster_size, min_samples)
    labels = np.full(len(items), -1, dtype=int)
    probabilities = np.zeros(len(items), dtype=np.float32)
    next_cluster = 0
    for family in sorted({str(item["family"]) for item in items}):
        idx = np.array([i for i, item in enumerate(items) if item["family"] == family])
        local_labels, local_probabilities = _fit_cluster(
            matrix[idx], min_cluster_size, min_samples
        )
        for local in sorted(set(local_labels) - {-1}):
            labels[idx[local_labels == local]] = next_cluster
            next_cluster += 1
        probabilities[idx] = local_probabilities
    return labels, probabilities


def _bootstrap_ci(values: list[float], iterations: int, seed: int) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    means = [float(rng.choice(arr, len(arr), replace=True).mean()) for _ in range(iterations)]
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def attribute_neighbor_metrics(
    matrix: np.ndarray,
    items: list[dict[str, Any]],
    *,
    k: int,
    bootstrap_iterations: int,
    seed: int,
    scope: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    families = np.asarray([str(item["family"]) for item in items])
    groups = (
        [("all", np.arange(len(items), dtype=int))]
        if scope == "global"
        else [
            (family, np.flatnonzero(families == family))
            for family in sorted(set(families))
        ]
    )
    for label_type in ("color", "pattern", "material", "shape"):
        query_scores: list[float] = []
        query_count = 0
        for _group, idx in groups:
            if len(idx) <= k:
                continue
            neighbors = NearestNeighbors(
                n_neighbors=min(k + 1, len(idx)), metric="cosine", algorithm="brute"
            ).fit(matrix[idx])
            local_neighbors = neighbors.kneighbors(return_distance=False)
            for local_i, global_i in enumerate(idx):
                expected = set(items[global_i].get("weak_labels", {}).get(label_type) or [])
                if not expected:
                    continue
                hits = []
                for local_j in local_neighbors[local_i]:
                    candidate = int(idx[local_j])
                    if candidate == global_i:
                        continue
                    actual = set(
                        items[candidate].get("weak_labels", {}).get(label_type) or []
                    )
                    hits.append(float(bool(expected & actual)))
                    if len(hits) >= k:
                        break
                if hits:
                    query_scores.append(float(np.mean(hits)))
                    query_count += 1
        low, high = _bootstrap_ci(query_scores, bootstrap_iterations, seed)
        result[label_type] = {
            "precision_at_k": round(float(np.mean(query_scores)) if query_scores else 0.0, 4),
            "ci95": [round(low, 4), round(high, 4)],
            "queries": query_count,
            "k": k,
            "scope": scope,
            "label_source": "product_title_weak_labels",
        }
    values = [row["precision_at_k"] for row in result.values() if row["queries"]]
    result["macro"] = round(float(np.mean(values)) if values else 0.0, 4)
    return result


def category_probe(matrix: np.ndarray, items: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    labels = np.asarray([str(item["family"]) for item in items])
    reduced = _reduce_for_analysis(matrix, seed)
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    model = LogisticRegression(max_iter=500, class_weight="balanced")
    scores = cross_val_score(model, reduced, labels, cv=folds, scoring="accuracy")
    return {
        "accuracy_mean": round(float(scores.mean()), 4),
        "accuracy_std": round(float(scores.std()), 4),
        "chance": round(1.0 / len(set(labels)), 4),
    }


def cluster_metrics(
    matrix: np.ndarray,
    labels: np.ndarray,
    items: list[dict[str, Any]],
    seed: int,
) -> dict[str, Any]:
    non_noise = labels >= 0
    cluster_ids = sorted(set(labels.tolist()) - {-1})
    families = [str(item["family"]) for item in items]
    silhouette = None
    if len(cluster_ids) >= 2 and int(non_noise.sum()) > len(cluster_ids):
        rng = np.random.default_rng(seed)
        valid_idx = np.flatnonzero(non_noise)
        if len(valid_idx) > 1200:
            valid_idx = rng.choice(valid_idx, 1200, replace=False)
        silhouette = float(
            silhouette_score(matrix[valid_idx], labels[valid_idx], metric="cosine")
        )
    enrichment: dict[str, Any] = {}
    for cluster_id in cluster_ids:
        idx = np.flatnonzero(labels == cluster_id)
        label_counts: dict[str, Counter[str]] = defaultdict(Counter)
        for i in idx:
            for kind, values in items[int(i)].get("weak_labels", {}).items():
                label_counts[kind].update(values)
        top_labels = {
            kind: [{"label": label, "count": count} for label, count in counts.most_common(3)]
            for kind, counts in label_counts.items()
        }
        enrichment[str(cluster_id)] = {
            "size": int(len(idx)),
            "family_counts": dict(Counter(families[int(i)] for i in idx)),
            "top_weak_labels": top_labels,
        }
    return {
        "cluster_count": len(cluster_ids),
        "noise_rate": round(float(np.mean(labels < 0)), 4),
        "silhouette_cosine": round(silhouette, 4) if silhouette is not None else None,
        "family_nmi": round(
            float(normalized_mutual_info_score(families, labels.astype(str))), 4
        ),
        "clusters": enrichment,
    }


def analyze_arms(
    arms: dict[str, np.ndarray],
    items: list[dict[str, Any]],
    *,
    seed: int,
    umap_cfg: dict[str, Any],
    cluster_cfg: dict[str, Any],
    evaluation_cfg: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    projections: dict[str, np.ndarray] = {}
    assignments: dict[str, np.ndarray] = {}
    probabilities: dict[str, np.ndarray] = {}
    metrics: dict[str, Any] = {}
    for arm, matrix in arms.items():
        print(f"analysis arm {arm}", flush=True)
        coords, projection_quality = project(
            matrix,
            seed=seed,
            n_neighbors=int(umap_cfg["n_neighbors"]),
            min_dist=float(umap_cfg["min_dist"]),
        )
        labels, probs = cluster_arm(
            arm,
            matrix,
            items,
            min_cluster_size=int(cluster_cfg["min_cluster_size"]),
            min_samples=int(cluster_cfg["min_samples"]),
        )
        projections[arm] = coords
        assignments[arm] = labels
        probabilities[arm] = probs
        global_neighbors = attribute_neighbor_metrics(
            matrix,
            items,
            k=int(evaluation_cfg["neighbor_k"]),
            bootstrap_iterations=int(evaluation_cfg["bootstrap_iterations"]),
            seed=seed,
            scope="global",
        )
        within_family_neighbors = attribute_neighbor_metrics(
            matrix,
            items,
            k=int(evaluation_cfg["neighbor_k"]),
            bootstrap_iterations=int(evaluation_cfg["bootstrap_iterations"]),
            seed=seed,
            scope="within_family",
        )
        metrics[arm] = {
            "projection": projection_quality,
            "attribute_neighbors": (
                within_family_neighbors if arm == "B" else global_neighbors
            ),
            "attribute_neighbors_global": global_neighbors,
            "attribute_neighbors_within_family": within_family_neighbors,
            "category_probe": category_probe(matrix, items, seed),
            "clustering": cluster_metrics(matrix, labels, items, seed),
        }
    arms_order = sorted(arms)
    metrics["coassignment"] = {
        left: {
            right: round(float(adjusted_rand_score(assignments[left], assignments[right])), 4)
            for right in arms_order
        }
        for left in arms_order
    }
    return projections, assignments, probabilities, metrics
