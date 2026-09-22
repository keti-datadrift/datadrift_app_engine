"""Run global-vs-family DR and target-attribute subspace experiments.

Date: 2026-09-14

The image encoder is never fine-tuned. Existing image embeddings are reused;
only text directions and lightweight linear probes are fitted.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import html
import io
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import hdbscan
import numpy as np
import plotly.graph_objects as go
import umap
from PIL import Image, ImageOps
from plotly.subplots import make_subplots
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import trustworthiness
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors

from dataset import WEAK_LABEL_ALIASES
from representations import (
    ATTRIBUTE_DESCRIPTIONS,
    FAMILY_PROMPT_NAMES,
    l2_normalize,
    load_model,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

TARGET_ATTRIBUTES = {
    "shape": ("oversized", "cropped", "slim"),
    "pattern": ("solid", "stripe", "check", "graphic"),
}

PALETTE = (
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#E45756",
    "#72B7B2",
    "#B279A2",
    "#FF9DA6",
    "#9D755D",
    "#BAB0AC",
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _stable_id(item: dict[str, Any]) -> str:
    raw = f"{item.get('source')}|{item.get('product_id')}".encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def _source_aliases(items: list[dict[str, Any]]) -> dict[str, str]:
    names = sorted({str(item.get("source") or "") for item in items})
    return {name: f"retail_{chr(97 + index)}" for index, name in enumerate(names)}


def _thumb_data_uri(path: str, size: int = 384, quality: int = 90) -> str:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        if max(rgb.size) > size:
            rgb = ImageOps.contain(rgb, (size, size))
    buffer = io.BytesIO()
    rgb.save(buffer, format="JPEG", quality=quality, optimize=True, subsampling=0)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _fit_pca_umap(
    matrix: np.ndarray,
    *,
    seed: int,
    n_neighbors: int,
    min_dist: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    n_components = min(64, len(matrix) - 1, matrix.shape[1])
    pca = PCA(n_components=n_components, random_state=seed)
    reduced = pca.fit_transform(matrix)
    reducer = umap.UMAP(
        n_components=2,
        metric="cosine",
        n_neighbors=min(n_neighbors, len(matrix) - 1),
        min_dist=min_dist,
        random_state=seed,
        n_jobs=1,
    )
    coords = reducer.fit_transform(reduced).astype(np.float32)
    return coords, {
        "pca_dimensions": int(n_components),
        "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
        "n_neighbors": int(reducer.n_neighbors),
        "min_dist": float(reducer.min_dist),
    }


def procrustes_align(
    reference: np.ndarray, moving: np.ndarray
) -> tuple[np.ndarray, float]:
    """Align moving coordinates to reference without changing local geometry."""

    ref_center = reference.mean(axis=0, keepdims=True)
    mov_center = moving.mean(axis=0, keepdims=True)
    ref_norm = float(np.linalg.norm(reference - ref_center))
    mov_norm = float(np.linalg.norm(moving - mov_center))
    ref_unit = (reference - ref_center) / max(ref_norm, 1e-12)
    mov_unit = (moving - mov_center) / max(mov_norm, 1e-12)
    left, _, right = np.linalg.svd(mov_unit.T @ ref_unit)
    rotation = left @ right
    aligned_unit = mov_unit @ rotation
    disparity = float(np.square(ref_unit - aligned_unit).sum())
    aligned = aligned_unit * ref_norm + ref_center
    return aligned.astype(np.float32), disparity


def _neighbor_sets(coords: np.ndarray, k: int) -> list[set[int]]:
    count = min(k + 1, len(coords))
    neighbors = NearestNeighbors(n_neighbors=count).fit(coords).kneighbors(
        return_distance=False
    )
    result = []
    for index, row in enumerate(neighbors):
        result.append({int(value) for value in row if int(value) != index})
    return result


def neighborhood_jaccard(
    left: np.ndarray, right: np.ndarray, k: int
) -> float:
    left_sets = _neighbor_sets(left, k)
    right_sets = _neighbor_sets(right, k)
    values = [
        len(a & b) / max(1, len(a | b))
        for a, b in zip(left_sets, right_sets, strict=True)
    ]
    return float(np.mean(values))


def _geometry_metrics(
    matrix: np.ndarray,
    global_family: np.ndarray,
    local: np.ndarray,
    aligned: np.ndarray,
    disparity: float,
) -> dict[str, Any]:
    distance_correlation = spearmanr(
        pdist(global_family), pdist(local)
    ).statistic
    return {
        "global_family_trustworthiness": round(
            float(
                trustworthiness(
                    matrix,
                    global_family,
                    n_neighbors=10,
                    metric="cosine",
                )
            ),
            4,
        ),
        "local_trustworthiness": round(
            float(trustworthiness(matrix, local, n_neighbors=10, metric="cosine")),
            4,
        ),
        "neighbor_jaccard_at_10": round(
            neighborhood_jaccard(global_family, local, 10), 4
        ),
        "neighbor_jaccard_at_20": round(
            neighborhood_jaccard(global_family, local, 20), 4
        ),
        "pairwise_distance_spearman": round(float(distance_correlation), 4),
        "procrustes_disparity": round(disparity, 6),
        "aligned_neighbor_jaccard_at_10": round(
            neighborhood_jaccard(global_family, aligned, 10), 4
        ),
    }


def _encode_text_directions(
    *,
    model_id: str,
    requested_device: str,
    family: str,
    targets: dict[str, tuple[str, ...]],
) -> tuple[dict[tuple[str, str], np.ndarray], dict[str, Any], str]:
    import torch

    model, _preprocess, tokenizer, device = load_model(model_id, requested_device)
    product_type = FAMILY_PROMPT_NAMES[family]
    directions: dict[tuple[str, str], np.ndarray] = {}
    prompts: dict[str, Any] = {}
    for kind, labels in targets.items():
        for label in labels:
            positive = [
                f"a clean ecommerce product photo of a {product_type} showing {ATTRIBUTE_DESCRIPTIONS[label]}",
                f"a studio product image of a {ATTRIBUTE_DESCRIPTIONS[label]} {product_type}",
                f"a {product_type} with {ATTRIBUTE_DESCRIPTIONS[label]}",
            ]
            other = [candidate for candidate in labels if candidate != label]
            negative = [
                f"a clean ecommerce product photo of a {product_type} showing {ATTRIBUTE_DESCRIPTIONS[candidate]}"
                for candidate in other
            ]
            tokens = tokenizer(positive + negative).to(device)
            with torch.inference_mode():
                vectors = model.encode_text(tokens, normalize=True).detach().float().cpu().numpy()
            raw = vectors[: len(positive)].mean(axis=0) - vectors[len(positive) :].mean(
                axis=0
            )
            directions[(kind, label)] = l2_normalize(raw[None, :])[0]
            prompts[f"{kind}:{label}"] = {
                "positive": positive,
                "negative": negative,
            }
    return directions, prompts, device


def _label_masks(
    items: list[dict[str, Any]], kind: str, label: str
) -> tuple[np.ndarray, np.ndarray]:
    known = np.asarray(
        [bool(item.get("weak_labels", {}).get(kind)) for item in items],
        dtype=bool,
    )
    positive = np.asarray(
        [label in (item.get("weak_labels", {}).get(kind) or []) for item in items],
        dtype=bool,
    )
    return known, positive


def _fit_data_directions(
    matrix: np.ndarray,
    items: list[dict[str, Any]],
    targets: dict[str, tuple[str, ...]],
) -> tuple[
    dict[tuple[str, str], np.ndarray],
    dict[tuple[str, str], np.ndarray],
    dict[str, Any],
]:
    centroid: dict[tuple[str, str], np.ndarray] = {}
    linear: dict[tuple[str, str], np.ndarray] = {}
    diagnostics: dict[str, Any] = {}
    train = np.asarray([str(item.get("split")) == "train" for item in items])
    for kind, labels in targets.items():
        for label in labels:
            known, positive = _label_masks(items, kind, label)
            pos_idx = np.flatnonzero(train & known & positive)
            neg_idx = np.flatnonzero(train & known & ~positive)
            key = f"{kind}:{label}"
            diagnostics[key] = {
                "train_positive": int(len(pos_idx)),
                "train_negative": int(len(neg_idx)),
            }
            if not len(pos_idx) or not len(neg_idx):
                continue
            contrast = matrix[pos_idx].mean(axis=0) - matrix[neg_idx].mean(axis=0)
            centroid[(kind, label)] = l2_normalize(contrast[None, :])[0]
            if len(pos_idx) >= 3 and len(neg_idx) >= 3:
                fit_idx = np.concatenate([pos_idx, neg_idx])
                y = positive[fit_idx].astype(int)
                probe = LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=2000,
                    solver="liblinear",
                    random_state=42,
                ).fit(matrix[fit_idx], y)
                direction = probe.coef_[0].astype(np.float32)
                linear[(kind, label)] = l2_normalize(direction[None, :])[0]
    return centroid, linear, diagnostics


def _ranking_metrics(
    scores: np.ndarray,
    items: list[dict[str, Any]],
    kind: str,
    label: str,
    k: int = 10,
) -> dict[str, Any]:
    known, positive = _label_masks(items, kind, label)
    test = np.asarray([str(item.get("split")) == "test" for item in items])
    idx = np.flatnonzero(known & test)
    y = positive[idx].astype(int)
    result: dict[str, Any] = {
        "test_known": int(len(idx)),
        "test_positive": int(y.sum()),
    }
    if len(idx) < 4 or len(np.unique(y)) < 2:
        result["status"] = "insufficient_test_labels"
        return result
    local_scores = scores[idx]
    order = np.argsort(-local_scores)
    top = y[order[: min(k, len(order))]]
    discounts = 1.0 / np.log2(np.arange(2, len(order) + 2))
    dcg = float(np.sum(y[order] * discounts))
    ideal = float(np.sum(np.sort(y)[::-1] * discounts))
    result.update(
        {
            "status": "ok",
            "roc_auc": round(float(roc_auc_score(y, local_scores)), 4),
            "average_precision": round(
                float(average_precision_score(y, local_scores)), 4
            ),
            "precision_at_10": round(float(top.mean()), 4),
            "ndcg": round(dcg / max(ideal, 1e-12), 4),
        }
    )
    return result


def _direction_experiment(
    matrix: np.ndarray,
    items: list[dict[str, Any]],
    *,
    model_id: str,
    requested_device: str,
    family: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    zero, prompts, device = _encode_text_directions(
        model_id=model_id,
        requested_device=requested_device,
        family=family,
        targets=TARGET_ATTRIBUTES,
    )
    centroid, linear, diagnostics = _fit_data_directions(
        matrix, items, TARGET_ATTRIBUTES
    )
    methods = {"zero_shot": zero, "centroid": centroid, "linear": linear}
    scores: dict[str, np.ndarray] = {}
    metrics: dict[str, Any] = {}
    attribute_columns: list[dict[str, str]] = []
    vectors_for_npz: dict[str, np.ndarray] = {}
    for kind, labels in TARGET_ATTRIBUTES.items():
        for label in labels:
            key = (kind, label)
            metric_key = f"{kind}:{label}"
            attribute_columns.append({"kind": kind, "label": label})
            metrics[metric_key] = {"diagnostics": diagnostics.get(metric_key, {})}
            for method, directions in methods.items():
                if key not in directions:
                    continue
                vector = directions[key]
                values = matrix @ vector
                scores[f"{method}:{metric_key}"] = values.astype(np.float32)
                vectors_for_npz[f"{method}__{kind}__{label}"] = vector.astype(
                    np.float32
                )
                metrics[metric_key][method] = _ranking_metrics(
                    values, items, kind, label
                )

    train = np.asarray([str(item.get("split")) == "train" for item in items])
    standardization: dict[str, Any] = {}
    available_methods: list[str] = []
    for method in methods:
        columns = [
            scores.get(f"{method}:{column['kind']}:{column['label']}")
            for column in attribute_columns
        ]
        if any(column is None for column in columns):
            continue
        raw_subspace = np.column_stack(columns).astype(np.float32)
        means = raw_subspace[train].mean(axis=0)
        stds = np.clip(raw_subspace[train].std(axis=0), 1e-6, None)
        subspace = ((raw_subspace - means) / stds).astype(np.float32)
        vectors_for_npz[f"{method}_subspace_scores"] = subspace
        standardization[method] = {
            "train_means": means.tolist(),
            "train_stds": stds.tolist(),
        }
        available_methods.append(method)
    output_meta = {
        "device": device,
        "prompts": prompts,
        "attribute_columns": attribute_columns,
        "subspace_methods": available_methods,
        "standardization": standardization,
    }
    return metrics, {**scores, **vectors_for_npz}, output_meta


def _cluster_subspaces(
    subspaces: dict[str, np.ndarray],
    raw_matrix: np.ndarray,
    items: list[dict[str, Any]],
    attribute_columns: list[dict[str, str]],
    *,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    kinds = {
        kind: [
            index
            for index, column in enumerate(attribute_columns)
            if column["kind"] == kind
        ]
        for kind in TARGET_ATTRIBUTES
    }
    kinds["combined"] = list(range(len(attribute_columns)))
    blocks: dict[str, tuple[np.ndarray, list[dict[str, str]]]] = {}
    for method, subspace in subspaces.items():
        for kind, columns in kinds.items():
            blocks[f"{method}:{kind}"] = (
                subspace[:, columns],
                [attribute_columns[index] for index in columns],
            )
    raw_components = min(64, len(raw_matrix) - 1, raw_matrix.shape[1])
    blocks["raw:combined"] = (
        PCA(n_components=raw_components, random_state=seed).fit_transform(raw_matrix),
        [],
    )
    outputs: dict[str, np.ndarray] = {}
    summaries: dict[str, Any] = {}
    baseline_counts = {
        attr_kind: Counter(
            value
            for item in items
            for value in item.get("weak_labels", {}).get(attr_kind, [])
        )
        for attr_kind in TARGET_ATTRIBUTES
    }
    for name, (block, dimensions) in blocks.items():
        method, kind = name.split(":", 1)
        coords = umap.UMAP(
            n_components=2,
            metric="euclidean",
            n_neighbors=min(20, len(block) - 1),
            min_dist=0.15,
            random_state=seed,
            n_jobs=1,
        ).fit_transform(block).astype(np.float32)
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=max(8, len(block) // 20),
            min_samples=3,
            metric="euclidean",
        )
        labels = clusterer.fit_predict(block).astype(int)
        outputs[f"{method}_{kind}_coords"] = coords
        outputs[f"{method}_{kind}_clusters"] = labels
        cluster_summary: dict[str, Any] = {}
        for cluster_id in sorted(set(labels.tolist()) - {-1}):
            idx = np.flatnonzero(labels == cluster_id)
            enrichment: dict[str, Any] = {}
            for attr_kind in TARGET_ATTRIBUTES:
                counts = Counter(
                    value
                    for index in idx
                    for value in items[int(index)]
                    .get("weak_labels", {})
                    .get(attr_kind, [])
                )
                values = []
                for label, count in counts.most_common(5):
                    baseline_rate = baseline_counts[attr_kind][label] / len(items)
                    cluster_rate = count / len(idx)
                    values.append(
                        {
                            "label": label,
                            "count": count,
                            "rate": round(cluster_rate, 4),
                            "lift_vs_all": round(
                                cluster_rate / max(baseline_rate, 1e-12), 3
                            ),
                        }
                    )
                enrichment[attr_kind] = values
            cluster_summary[str(cluster_id)] = {
                "size": int(len(idx)),
                "top_weak_labels": enrichment,
            }
        summaries[name] = {
            "method": method,
            "attribute_scope": kind,
            "dimensions": dimensions,
            "cluster_count": len(set(labels.tolist()) - {-1}),
            "noise_rate": round(float(np.mean(labels < 0)), 4),
            "clusters": cluster_summary,
        }
    return outputs, summaries


def _customdata(
    items: list[dict[str, Any]], aliases: dict[str, str]
) -> list[list[Any]]:
    rows = []
    for index, item in enumerate(items):
        weak = item.get("weak_labels") or {}
        rows.append(
            [
                index,
                _stable_id(item),
                str(item.get("brand") or ""),
                str(item.get("product") or ""),
                aliases[str(item.get("source") or "")],
                ", ".join(weak.get("shape") or []) or "-",
                ", ".join(weak.get("pattern") or []) or "-",
            ]
        )
    return rows


def _comparison_figure(
    all_items: list[dict[str, Any]],
    family_items: list[dict[str, Any]],
    global_coords: np.ndarray,
    family_indices: np.ndarray,
    local_coords: np.ndarray,
    aligned_coords: np.ndarray,
    aliases: dict[str, str],
    family: str,
) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            f"Global map (all; {family} highlighted)",
            f"Global map: {family} zoom",
            f"{family}-only refit",
            "Local map aligned to global-family",
        ),
        horizontal_spacing=0.06,
        vertical_spacing=0.12,
    )
    family_set = set(family_indices.tolist())
    global_custom = []
    for global_index, item in enumerate(all_items):
        weak = item.get("weak_labels") or {}
        family_local_index = (
            int(np.flatnonzero(family_indices == global_index)[0])
            if global_index in family_set
            else -1
        )
        global_custom.append(
            [
                family_local_index,
                _stable_id(item),
                str(item.get("brand") or ""),
                str(item.get("product") or ""),
                aliases[str(item.get("source") or "")],
                ", ".join(weak.get("shape") or []) or "-",
                ", ".join(weak.get("pattern") or []) or "-",
            ]
        )
    marker_colors = [
        "#E45756" if index in family_set else "#D3D3D3"
        for index in range(len(all_items))
    ]
    fig.add_trace(
        go.Scatter(
            x=global_coords[:, 0],
            y=global_coords[:, 1],
            mode="markers",
            marker={"size": 6, "color": marker_colors, "opacity": 0.75},
            customdata=global_custom,
            hovertemplate=(
                "<b>%{customdata[2]}</b><br>%{customdata[3]}<br>"
                "%{customdata[4]} · id=%{customdata[1]}<br>"
                "shape=%{customdata[5]} | pattern=%{customdata[6]}<extra></extra>"
            ),
            name="global",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    family_custom = _customdata(family_items, aliases)
    labels = [
        (item.get("weak_labels", {}).get("shape") or ["unknown"])[0]
        for item in family_items
    ]
    label_names = sorted(set(labels))
    color_map = {
        label: PALETTE[index % len(PALETTE)]
        for index, label in enumerate(label_names)
    }
    colors = [color_map[label] for label in labels]
    hover = (
        "<b>%{customdata[2]}</b><br>%{customdata[3]}<br>"
        "%{customdata[4]} · id=%{customdata[1]}<br>"
        "shape=%{customdata[5]} | pattern=%{customdata[6]}<extra></extra>"
    )
    for panel, coords in enumerate(
        (global_coords[family_indices], local_coords, aligned_coords), start=1
    ):
        row, col = ((1, 2), (2, 1), (2, 2))[panel - 1]
        fig.add_trace(
            go.Scatter(
                x=coords[:, 0],
                y=coords[:, 1],
                mode="markers",
                marker={
                    "size": 9,
                    "color": colors,
                    "line": {"width": 0.4, "color": "#555"},
                },
                customdata=family_custom,
                hovertemplate=hover,
                name=("global_family", "local", "aligned")[panel - 1],
                showlegend=False,
            ),
            row=row,
            col=col,
        )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(
        height=940,
        template="plotly_white",
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        hovermode="closest",
    )
    return fig


def _subspace_figure(
    family_items: list[dict[str, Any]],
    clusters: dict[str, np.ndarray],
    aliases: dict[str, str],
) -> go.Figure:
    methods = ("zero_shot", "linear")
    fig = make_subplots(
        rows=len(methods),
        cols=3,
        subplot_titles=tuple(
            f"{method}: {kind}"
            for method in methods
            for kind in ("shape", "pattern", "combined")
        ),
        horizontal_spacing=0.05,
        vertical_spacing=0.12,
    )
    custom = _customdata(family_items, aliases)
    hover = (
        "<b>%{customdata[2]}</b><br>%{customdata[3]}<br>"
        "%{customdata[4]} · id=%{customdata[1]}<br>"
        "shape=%{customdata[5]} | pattern=%{customdata[6]}<extra></extra>"
    )
    for row, method in enumerate(methods, start=1):
        for col, kind in enumerate(("shape", "pattern", "combined"), start=1):
            coords = clusters[f"{method}_{kind}_coords"]
            labels = clusters[f"{method}_{kind}_clusters"]
            colors = [
                "#BDBDBD" if value < 0 else PALETTE[int(value) % len(PALETTE)]
                for value in labels
            ]
            fig.add_trace(
                go.Scatter(
                    x=coords[:, 0],
                    y=coords[:, 1],
                    mode="markers",
                    marker={"size": 9, "color": colors},
                    customdata=custom,
                    hovertemplate=hover,
                    name=f"{method}:{kind}",
                    showlegend=False,
                ),
                row=row,
                col=col,
            )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(
        height=900,
        template="plotly_white",
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        hovermode="closest",
    )
    return fig


def _metric_cards(direction_metrics: dict[str, Any]) -> str:
    cards = []
    for attribute, payload in direction_metrics.items():
        rows = []
        for method in ("zero_shot", "centroid", "linear"):
            metric = payload.get(method)
            if not metric:
                continue
            if metric.get("status") != "ok":
                text = f"test n={metric.get('test_known', 0)} · insufficient"
            else:
                text = (
                    f"AP={metric['average_precision']:.4f} · "
                    f"AUC={metric['roc_auc']:.4f} · "
                    f"P@10={metric['precision_at_10']:.4f} · "
                    f"nDCG={metric['ndcg']:.4f}"
                )
            rows.append(f"<li><b>{html.escape(method)}</b>: {text}</li>")
        diagnostics = html.escape(json.dumps(payload.get("diagnostics", {})))
        cards.append(
            f"<section class='card'><h3>{html.escape(attribute)}</h3>"
            f"<code>{diagnostics}</code><ul>{''.join(rows)}</ul></section>"
        )
    return "".join(cards)


def _ranking_galleries(
    family_items: list[dict[str, Any]],
    scores: dict[str, np.ndarray],
    attribute_columns: list[dict[str, str]],
    thumbs: list[str],
) -> str:
    sections = []
    for column in attribute_columns:
        kind = column["kind"]
        label = column["label"]
        method_groups = []
        for method in ("zero_shot", "linear"):
            values = scores[f"{method}:{kind}:{label}"]
            order = np.argsort(-values)[:6]
            tiles = []
            for index in order:
                item = family_items[int(index)]
                title = html.escape(str(item.get("product") or ""))
                tiles.append(
                    "<div class='tile'>"
                    f"<img src='{thumbs[int(index)]}' alt='{title}'>"
                    f"<div>{float(values[int(index)]):.3f}</div>"
                    f"<small>{title}</small></div>"
                )
            method_groups.append(
                f"<h4>{html.escape(method)}</h4>"
                f"<div class='gallery'>{''.join(tiles)}</div>"
            )
        sections.append(
            f"<h3>{html.escape(kind)} / {html.escape(label)}</h3>"
            + "".join(method_groups)
        )
    return "".join(sections)


def _build_report(
    *,
    output_dir: Path,
    family: str,
    source_run: str,
    family_items: list[dict[str, Any]],
    comparison_figure: go.Figure,
    subspace_figure: go.Figure,
    geometry_metrics: dict[str, Any],
    direction_metrics: dict[str, Any],
    cluster_metrics: dict[str, Any],
    score_outputs: dict[str, np.ndarray],
    attribute_columns: list[dict[str, str]],
    aliases: dict[str, str],
) -> Path:
    thumbs = [_thumb_data_uri(str(item["image_path"])) for item in family_items]
    compare_div = comparison_figure.to_html(
        full_html=False, include_plotlyjs="cdn", div_id="geometry-comparison"
    )
    subspace_div = subspace_figure.to_html(
        full_html=False, include_plotlyjs=False, div_id="target-subspaces"
    )
    galleries = _ranking_galleries(
        family_items, score_outputs, attribute_columns, thumbs
    )
    thumb_json = json.dumps(thumbs)
    body = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<title>Attribute direction follow-up · {html.escape(family)}</title>
<style>
html{{color-scheme:light;background:#fff}}body{{font-family:-apple-system,BlinkMacSystemFont,sans-serif;max-width:1560px;margin:auto;padding:24px;color:#222;background:#fff}}
.warning{{background:#fff4ce;border-left:5px solid #d89b00;padding:14px;margin:16px 0}}.layout{{display:grid;grid-template-columns:minmax(0,1fr) 300px;gap:16px;align-items:start}}
.preview{{position:sticky;top:16px;border:1px solid #ddd;background:#fafafa;padding:12px;min-height:410px}}.preview img{{display:block;width:100%;aspect-ratio:1/1;object-fit:contain;background:#fff}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:10px}}.card{{border:1px solid #ddd;padding:10px;background:#fafafa}}.card ul{{padding-left:20px}}
.gallery{{display:grid;grid-template-columns:repeat(8,minmax(0,1fr));gap:8px}}.tile{{border:1px solid #ddd;padding:6px;overflow:hidden}}.tile img{{width:100%;aspect-ratio:1/1;object-fit:contain}}.tile small{{display:block;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
pre{{background:#f6f6f6;padding:12px;overflow:auto}}code{{font-size:12px}}@media(max-width:1000px){{.layout{{grid-template-columns:1fr}}.gallery{{grid-template-columns:repeat(4,1fr)}}}}
</style></head><body>
<h1>전역/Family 축소 비교 + 타겟 속성 공간 군집화</h1>
<p>source_run=<code>{html.escape(source_run)}</code> · family=<code>{html.escape(family)}</code> · n=<code>{len(family_items)}</code></p>
<div class="warning"><b>해석 경계:</b> 독립 UMAP의 절대 위치·회전·축은 비교 대상이 아닙니다. 동일 제품의 이웃 보존, 거리 순위, 정렬 후 변형을 보십시오. 속성 평가는 현재 상품명 weak label 기반이며 수동 라벨 검증 전에는 예비 결과입니다.</div>
<h2>1. 전역 축소와 family-local 축소</h2>
<p>색상은 tops의 첫 번째 weak shape label이며 회색은 미표기입니다. 어느 패널에서든 점을 hover하면 동일 상품 미리보기가 표시됩니다.</p>
<div class="layout"><div>{compare_div}</div><aside class="preview"><p>상품 미리보기</p><img id="geometry-preview-image" hidden><div id="geometry-preview-meta"></div></aside></div>
<h3>기하 비교 지표</h3><pre>{html.escape(json.dumps(geometry_metrics, ensure_ascii=False, indent=2))}</pre>
<h2>2. 타겟 속성 방향 평가</h2>
<p>zero-shot 텍스트 방향, train weak-label centroid 방향, L2 선형 probe 방향을 held-out split에서 비교합니다.</p>
<div class="cards">{_metric_cards(direction_metrics)}</div>
<h2>3. 형태·패턴 부분공간 군집</h2>
<p>군집 색상은 HDBSCAN cluster이며 회색은 noise입니다. 점에 커서를 올리면 오른쪽에서 해당 상품 이미지를 확인할 수 있습니다.</p>
<div class="layout"><div>{subspace_div}</div><aside class="preview"><p>군집 상품 미리보기</p><img id="cluster-preview-image" hidden><div id="cluster-preview-meta"></div></aside></div>
<pre>{html.escape(json.dumps(cluster_metrics, ensure_ascii=False, indent=2))}</pre>
<h2>4. 속성 방향 정렬 상위 상품</h2>{galleries}
<script>
const THUMBS={thumb_json};
function showPoint(point,imageId,metaId){{const c=point.customdata||[];const i=Number(c[0]);const img=document.getElementById(imageId);const meta=document.getElementById(metaId);if(i>=0&&THUMBS[i]){{img.hidden=false;img.src=THUMBS[i];}}else{{img.hidden=true;img.removeAttribute('src');}}meta.innerHTML=['<b>'+(c[2]||'')+'</b>',c[3]||'',(c[4]||'')+' · '+(c[1]||''),'shape='+(c[5]||'-'),'pattern='+(c[6]||'-')].join('<br>');}}
function bind(id,imageId,metaId){{const plot=document.getElementById(id);if(!plot||!plot.on){{setTimeout(()=>bind(id,imageId,metaId),50);return;}}plot.on('plotly_hover',e=>{{if(e?.points?.length)showPoint(e.points[0],imageId,metaId);}});}}
bind('geometry-comparison','geometry-preview-image','geometry-preview-meta');
bind('target-subspaces','cluster-preview-image','cluster-preview-meta');
</script></body></html>"""
    path = output_dir / "index.html"
    path.write_text(body, encoding="utf-8")
    return path


def _write_annotation_template(
    path: Path,
    items: list[dict[str, Any]],
    aliases: dict[str, str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "stable_id",
                "source_alias",
                "product",
                "weak_shape",
                "weak_pattern",
                "human_shape",
                "human_pattern",
                "reviewer",
                "notes",
            ),
        )
        writer.writeheader()
        for item in items:
            weak = item.get("weak_labels") or {}
            writer.writerow(
                {
                    "stable_id": _stable_id(item),
                    "source_alias": aliases[str(item.get("source") or "")],
                    "product": str(item.get("product") or ""),
                    "weak_shape": "|".join(weak.get("shape") or []),
                    "weak_pattern": "|".join(weak.get("pattern") or []),
                    "human_shape": "",
                    "human_pattern": "",
                    "reviewer": "",
                    "notes": "",
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", default="mvp-20260911-v2")
    parser.add_argument("--family", default="tops")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--min-dist", type=float, default=0.12)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "artifacts" / "attribute_embedding",
    )
    parser.add_argument("--output-id", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_dir = args.output_root / args.source_run
    all_items = _load_jsonl(source_dir / "dataset" / "items.jsonl")
    raw_cache = np.load(source_dir / "representations" / "A_raw.npz")
    matrix = raw_cache["matrix"].astype(np.float32)
    model_id = str(raw_cache["model_id"].item())
    family_indices = np.asarray(
        [
            index
            for index, item in enumerate(all_items)
            if str(item.get("family")) == args.family
        ],
        dtype=int,
    )
    if len(family_indices) < 30:
        raise RuntimeError(f"family={args.family} has only {len(family_indices)} items")
    family_items = [all_items[int(index)] for index in family_indices]
    family_matrix = matrix[family_indices]
    aliases = _source_aliases(all_items)
    output_id = args.output_id or (
        f"followup-{args.family}-"
        + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    )
    output_dir = args.output_root / output_id
    output_dir.mkdir(parents=True, exist_ok=False)

    print("step 1/2: fitting global and family-local PCA64→UMAP", flush=True)
    global_coords, global_notes = _fit_pca_umap(
        matrix,
        seed=args.seed,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
    )
    local_coords, local_notes = _fit_pca_umap(
        family_matrix,
        seed=args.seed,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
    )
    aligned_coords, disparity = procrustes_align(
        global_coords[family_indices], local_coords
    )
    geometry_metrics = _geometry_metrics(
        family_matrix,
        global_coords[family_indices],
        local_coords,
        aligned_coords,
        disparity,
    )
    geometry_metrics["global_fit"] = global_notes
    geometry_metrics["local_fit"] = local_notes

    print("step 2/2: extracting target directions and clustering subspaces", flush=True)
    direction_metrics, score_outputs, direction_meta = _direction_experiment(
        family_matrix,
        family_items,
        model_id=model_id,
        requested_device=args.device,
        family=args.family,
    )
    subspaces = {
        method: score_outputs[f"{method}_subspace_scores"]
        for method in direction_meta["subspace_methods"]
    }
    cluster_outputs, cluster_metrics = _cluster_subspaces(
        subspaces,
        family_matrix,
        family_items,
        direction_meta["attribute_columns"],
        seed=args.seed,
    )

    comparison_figure = _comparison_figure(
        all_items,
        family_items,
        global_coords,
        family_indices,
        local_coords,
        aligned_coords,
        aliases,
        args.family,
    )
    subspace_figure = _subspace_figure(family_items, cluster_outputs, aliases)
    np.savez_compressed(
        output_dir / "geometry.npz",
        global_coords=global_coords,
        family_indices=family_indices,
        local_coords=local_coords,
        aligned_coords=aligned_coords,
    )
    direction_vectors = {
        key: value
        for key, value in score_outputs.items()
        if "__" in key or key.endswith("_subspace_scores")
    }
    np.savez_compressed(output_dir / "directions_and_scores.npz", **direction_vectors)
    np.savez_compressed(output_dir / "target_clusters.npz", **cluster_outputs)
    _write_json(output_dir / "geometry_metrics.json", geometry_metrics)
    _write_json(output_dir / "direction_metrics.json", direction_metrics)
    _write_json(output_dir / "cluster_metrics.json", cluster_metrics)
    _write_json(output_dir / "direction_metadata.json", direction_meta)
    _write_annotation_template(
        output_dir / "annotation_template.csv", family_items, aliases
    )
    report = _build_report(
        output_dir=output_dir,
        family=args.family,
        source_run=args.source_run,
        family_items=family_items,
        comparison_figure=comparison_figure,
        subspace_figure=subspace_figure,
        geometry_metrics=geometry_metrics,
        direction_metrics=direction_metrics,
        cluster_metrics=cluster_metrics,
        score_outputs=score_outputs,
        attribute_columns=direction_meta["attribute_columns"],
        aliases=aliases,
    )
    manifest = {
        "schema_version": 1,
        "experiment": "global_family_and_attribute_subspace_followup",
        "status": "complete",
        "created_at": datetime.now(UTC).isoformat(),
        "source_run": args.source_run,
        "family": args.family,
        "item_count": len(family_items),
        "model_id": model_id,
        "image_embeddings_reused": True,
        "image_encoder_fine_tuned": False,
        "label_source": "product_title_weak_labels",
        "target_attributes": TARGET_ATTRIBUTES,
        "outputs": [path.name for path in sorted(output_dir.iterdir())],
    }
    _write_json(output_dir / "manifest.json", manifest)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
