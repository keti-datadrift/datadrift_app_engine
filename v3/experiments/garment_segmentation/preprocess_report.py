from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA

from preprocess import ALL_ARMS, ARMS, resolve_arms

ARM_PATH_OFFSET = 11

PALETTE = {
    "black": "#222222",
    "white": "#d9d9d9",
    "gray": "#8a8a8a",
    "navy": "#1f3a93",
    "blue": "#4c78a8",
    "red": "#e45756",
    "green": "#54a24b",
    "brown": "#9d755d",
    "yellow": "#eeca3b",
    "pink": "#f58518",
    "purple": "#b279a2",
    "orange": "#ff9d4a",
    "unknown": "#b0b0b0",
}

METRIC_DOCS = [
    {
        "id": "precision_at_k",
        "title": "속성 이웃 Precision@k",
        "formula": "P@k(i) = (1/k) Σ_{j ∈ N_k(i)} 1[weak(i) ∩ weak(j) ≠ ∅],  보고값 = 질의 평균",
        "meaning": (
            "임베딩에서 가까운 k개가 같은 색·패턴·소재·실루엣 weak label을 공유하는 비율입니다. "
            "상품명에서 추출한 약한 라벨이라 상한은 1이 아니고, A 대비 상승이 개선의 핵심 신호입니다. "
            "95% 구간은 질의 점수 부트스트랩입니다."
        ),
    },
    {
        "id": "intra_inter",
        "title": "라벨 내부-외부 코사인 간격",
        "formula": "gap = mean{cos(z_i,z_j) | 라벨 공유} − mean{cos(z_i,z_j) | 라벨 비공유}",
        "meaning": (
            "같은 속성은 가깝고 다른 속성은 멀수록 값이 커집니다. "
            "전처리가 배경을 지워 색·패턴이 드러나면 이 간격이 커져야 합니다. "
            "간격이 줄면 마스크 아티팩트가 속성보다 강해진 것입니다."
        ),
    },
    {
        "id": "mean_item_cosine",
        "title": "동일 상품 코사인 (암 간)",
        "formula": "(1/n) Σ_i cos(z_i^A, z_i^X)",
        "meaning": (
            "같은 이미지의 A와 X 표현이 얼마나 비슷한지입니다. "
            "1에 가까우면 전처리가 거의 무효이고, 너무 낮으면 입력이 다른 분포가 된 것입니다. "
            "개선은 이 값이 아니라 속성 이웃으로 판단합니다."
        ),
    },
    {
        "id": "linear_cka",
        "title": "Linear CKA",
        "formula": "CKA(X,Y) = ||X_c^T Y_c||_F^2 / (||X_c^T X_c||_F ||Y_c^T Y_c||_F),  열 평균을 뺀 뒤 계산",
        "meaning": (
            "두 임베딩 공간의 상대 기하가 얼마나 같은지입니다. "
            "개별 벡터 회전과 무관하게 관계 구조를 비교합니다. "
            "CKA가 높고 속성 지표가 안 오르면 전처리가 정렬을 바꾸지 못한 것입니다."
        ),
    },
    {
        "id": "distance_spearman",
        "title": "쌍거리 Spearman 상관",
        "formula": "Spearman( pdist_cos(A), pdist_cos(X) )",
        "meaning": (
            "모든 상품 쌍의 거리 순위가 유지되는 정도입니다. "
            "이웃 Jaccard보다 전역적이고, CKA보다 순위만 봅니다."
        ),
    },
    {
        "id": "neighborhood_jaccard",
        "title": "k-이웃 Jaccard",
        "formula": "(1/n) Σ_i |N_k^A(i) ∩ N_k^X(i)| / |N_k^A(i) ∪ N_k^X(i)|",
        "meaning": (
            "각 점의 로컬 이웃이 전처리 후 얼마나 바뀌는지입니다. "
            "낮으면 검색 결과가 달라진다는 뜻이고, 좋거나 나쁜지는 속성 P@k와 같이 봐야 합니다."
        ),
    },
    {
        "id": "trustworthiness",
        "title": "UMAP Trustworthiness",
        "formula": "고차원 이웃이 2D에서도 가깝게 남는지 sklearn.trustworthiness로 측정",
        "meaning": (
            "산점도가 임베딩 이웃을 얼마나 믿을 수 있게 그렸는지입니다. "
            "패널 간 점의 절대 위치·회전은 비교하지 말고, 이 값과 정렬 패널을 보십시오."
        ),
    },
    {
        "id": "procrustes",
        "title": "Procrustes disparity",
        "formula": "X를 A에 맞춰 중심·스케일·회전만 맞춘 뒤 Σ ||A_unit − X_aligned||^2",
        "meaning": (
            "로컬 모양은 유지한 채 두 UMAP을 겹칠 때 남는 전역 차이입니다. "
            "값이 크면 군집 배치 자체가 바뀐 것입니다."
        ),
    },
    {
        "id": "silhouette",
        "title": "약한 라벨 Silhouette",
        "formula": "cosine 거리 기준 (b−a)/max(a,b), a=같은 라벨 평균거리, b=가장 가까운 다른 라벨 거리",
        "meaning": (
            "색 또는 subtype이 임베딩에서 뭉치는 정도입니다. "
            "음수면 같은 라벨보다 다른 라벨이 더 가깝습니다."
        ),
    },
    {
        "id": "mask_fill",
        "title": "박스 채움 비율 / fallback",
        "formula": "fill = |mask ∩ box| / |box|,  fallback_rate = 1{source=box_fallback}의 평균",
        "meaning": (
            "SegFormer 상의 픽셀이 SAM 박스를 얼마나 채우는지입니다. "
            "fallback이 높으면 C·D가 B와 거의 같아집니다. "
            "combined_garment_rate는 combined resolver가 실제로 상의를 골랐는지입니다."
        ),
    },
    {
        "id": "centroid_radius",
        "title": "중심 반지름 p50 / p95",
        "formula": "c = normalize(mean z),  r_i = 1 − cos(z_i, c),  보고값 = percentile(r)",
        "meaning": (
            "한 집합이 자신의 중심에 얼마나 붙어 있는지입니다. p50이 작으면 핵심이 밀집입니다. "
            "p95가 크면 먼 점이 범위를 벌립니다. 면적 가설은 p50으로, 이상치 팽창은 p95와 비율로 봅니다."
        ),
    },
    {
        "id": "core_knn",
        "title": "평균 k-NN 코사인 / 핵심 k-NN",
        "formula": "d_k(i) = 평균 cosine 거리 to k nearest,  core는 r_i ≤ r_p50 인 점만",
        "meaning": (
            "로컬 밀도입니다. 값이 작을수록 가까이 붙어 있습니다. "
            "hull이 넓은데 core_knn만 작으면, 면적은 튀는 점 때문이고 알맹이는 더 뭉친 것입니다."
        ),
    },
    {
        "id": "outlier_span",
        "title": "이상치 팽창비 p95/p50",
        "formula": "outlier_span_ratio = r_p95 / r_p50",
        "meaning": (
            "1에 가까우면 고르게 퍼진 것이고, 크면 소수 점이 범위를 키운 것입니다. "
            "C의 hull이 커 보여도 이 비율이 크고 core_knn이 작으면 가설과 맞습니다."
        ),
    },
    {
        "id": "subtype_separation",
        "title": "제품군 분리도와 이웃 순도",
        "formula": "separation = mean_{s≠t}(1−cos(c_s,c_t)) − mean_s within(s),  purity = k-NN 중 같은 subtype 비율",
        "meaning": (
            "밀집만 보면 모든 옷이 비슷해져도 좋아 보입니다. 분리도와 순도는 제품군 관계가 실제로 갈리는지 봅니다. "
            "추출 품질은 핵심 밀도, 정렬 품질은 분리도·순도·속성 P@k를 같이 봐야 합니다."
        ),
    },
]


def procrustes_align(reference: np.ndarray, moving: np.ndarray) -> tuple[np.ndarray, float]:
    ref_center = reference.mean(axis=0, keepdims=True)
    mov_center = moving.mean(axis=0, keepdims=True)
    ref_norm = float(np.linalg.norm(reference - ref_center))
    mov_norm = float(np.linalg.norm(moving - mov_center))
    ref_unit = (reference - ref_center) / max(ref_norm, 1e-12)
    mov_unit = (moving - mov_center) / max(mov_norm, 1e-12)
    left, _, right = np.linalg.svd(mov_unit.T @ ref_unit)
    aligned_unit = mov_unit @ (left @ right)
    disparity = float(np.square(ref_unit - aligned_unit).sum())
    aligned = aligned_unit * ref_norm + ref_center
    return aligned.astype(np.float32), disparity


def fit_umap(matrix: np.ndarray, *, seed: int, n_neighbors: int, min_dist: float) -> tuple[np.ndarray, dict[str, Any]]:
    import umap

    n_components = min(64, max(2, len(matrix) - 1), matrix.shape[1])
    reduced = PCA(n_components=n_components, random_state=seed).fit_transform(matrix)
    neighbors = min(max(2, int(n_neighbors)), max(2, len(matrix) - 1))
    reducer = umap.UMAP(
        n_components=2,
        metric="cosine",
        n_neighbors=neighbors,
        min_dist=float(min_dist),
        random_state=seed,
        n_jobs=1,
    )
    coords = reducer.fit_transform(reduced).astype(np.float32)
    return coords, {
        "pca_dimensions": int(n_components),
        "n_neighbors": int(neighbors),
        "min_dist": float(min_dist),
    }


def _color_label(item: dict[str, Any]) -> str:
    colors = item.get("weak_labels", {}).get("color") or []
    return str(colors[0]) if colors else "unknown"


def _customdata(
    items: list[dict[str, Any]],
    preview_rel: dict[str, list[str]],
    arms: tuple[str, ...] = ARMS,
) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for index, item in enumerate(items):
        weak = item.get("weak_labels") or {}
        row = [
            index,
            item.get("item_id"),
            item.get("brand") or "",
            item.get("product") or "",
            item.get("split") or "",
            item.get("subtype") or "",
            ", ".join(weak.get("color") or []) or "-",
            ", ".join(weak.get("pattern") or []) or "-",
            ", ".join(weak.get("shape") or []) or "-",
            item.get("mask_source") or "",
            f"{float(item.get('fill_ratio') or 0):.3f}",
        ]
        for arm in arms:
            paths = preview_rel.get(arm) or []
            row.append(paths[index] if index < len(paths) else "")
        rows.append(row)
    return rows


def scatter_figure(
    coords_by_arm: dict[str, np.ndarray],
    items: list[dict[str, Any]],
    preview_rel: dict[str, list[str]],
    *,
    title: str,
    arms: tuple[str, ...] | None = None,
) -> go.Figure:
    arms = tuple(arms) if arms else tuple(name for name in ALL_ARMS if name in coords_by_arm)
    n_cols = min(2, max(1, len(arms)))
    n_rows = int(np.ceil(len(arms) / n_cols))
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[ARM_TITLES.get(arm, arm) for arm in arms],
    )
    custom = _customdata(items, preview_rel, arms)
    hover = (
        "<b>%{customdata[3]}</b><br>%{customdata[2]}<br>"
        "%{customdata[4]} · %{customdata[5]} · %{customdata[1]}<br>"
        "color=%{customdata[6]} | pattern=%{customdata[7]} | shape=%{customdata[8]}<br>"
        "mask=%{customdata[9]} fill=%{customdata[10]}<extra></extra>"
    )
    colors = [PALETTE.get(_color_label(item), PALETTE["unknown"]) for item in items]
    for index, arm in enumerate(arms):
        row, col = divmod(index, n_cols)
        fig.add_trace(
            go.Scattergl(
                x=coords_by_arm[arm][:, 0],
                y=coords_by_arm[arm][:, 1],
                mode="markers",
                marker={"size": 8, "color": colors, "line": {"width": 0.4, "color": "#555"}},
                customdata=custom,
                hovertemplate=hover,
                name=arm,
                showlegend=False,
            ),
            row=row + 1,
            col=col + 1,
        )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(
        title=title,
        height=max(480, 420 * n_rows),
        template="plotly_white",
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        hovermode="closest",
    )
    return fig


SUBTYPE_LABELS = {
    "tshirt": "티셔츠",
    "shirt": "셔츠",
    "knit": "니트",
    "hoodie": "후드",
    "sweatshirt": "맨투맨",
    "sleeveless": "슬리브리스",
    "blouse": "블라우스",
    "other": "기타",
}

ARM_COLORS = {
    "A": "#4c78a8",
    "B": "#f58518",
    "C": "#54a24b",
    "D": "#e45756",
}

ARM_TITLES = {
    "A": "A 원본",
    "B": "B 패딩 박스",
    "C": "C soft+blur",
    "D": "D hard mask",
}


def _subtype_groups(items: list[dict[str, Any]], min_count: int = 8) -> list[tuple[str, np.ndarray]]:
    names = [str(item.get("subtype") or "other") for item in items]
    counts: dict[str, int] = {}
    for name in names:
        counts[name] = counts.get(name, 0) + 1
    ranked = sorted(counts.items(), key=lambda row: (-row[1], row[0]))
    groups: list[tuple[str, np.ndarray]] = []
    for name, count in ranked:
        if count < min_count:
            continue
        idx = np.array([i for i, value in enumerate(names) if value == name], dtype=int)
        groups.append((name, idx))
    return groups


def _hull_xy(points: np.ndarray) -> np.ndarray | None:
    if len(points) < 3:
        return None
    from scipy.spatial import ConvexHull

    unique = np.unique(np.round(points, 5), axis=0)
    if len(unique) < 3:
        return None
    try:
        hull = ConvexHull(unique)
    except Exception:
        return None
    verts = unique[hull.vertices]
    return np.vstack([verts, verts[:1]])


def _figure_arms(
    coords_by_arm: dict[str, np.ndarray],
    arms: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    if arms:
        return resolve_arms(arms)
    found = tuple(name for name in ALL_ARMS if name in coords_by_arm)
    return found or ALL_ARMS


def _axis_range(
    coords_by_arm: dict[str, np.ndarray],
    pad: float = 0.08,
    arms: tuple[str, ...] | None = None,
) -> tuple[list[float], list[float]]:
    names = _figure_arms(coords_by_arm, arms)
    stacked = np.concatenate([coords_by_arm[arm] for arm in names], axis=0)
    mins = stacked.min(axis=0)
    maxs = stacked.max(axis=0)
    span = np.clip(maxs - mins, 1e-6, None)
    lo = mins - span * pad
    hi = maxs + span * pad
    return [float(lo[0]), float(hi[0])], [float(lo[1]), float(hi[1])]


def subtype_grid_figure(
    coords_by_arm: dict[str, np.ndarray],
    items: list[dict[str, Any]],
    preview_rel: dict[str, list[str]],
    *,
    arms: tuple[str, ...] | None = None,
) -> go.Figure:
    arms = _figure_arms(coords_by_arm, arms)
    groups = _subtype_groups(items)
    custom = _customdata(items, preview_rel, arms)
    colors = [PALETTE.get(_color_label(item), PALETTE["unknown"]) for item in items]
    hover = (
        "<b>%{customdata[3]}</b><br>%{customdata[2]}<br>"
        "%{customdata[4]} · %{customdata[5]} · %{customdata[1]}<br>"
        "color=%{customdata[6]} | pattern=%{customdata[7]}<extra></extra>"
    )
    fig = make_subplots(
        rows=max(1, len(groups)),
        cols=len(arms),
        shared_xaxes=True,
        shared_yaxes=True,
        row_titles=[f"{SUBTYPE_LABELS.get(name, name)} n={len(idx)}" for name, idx in groups],
        column_titles=[ARM_TITLES.get(arm, arm) for arm in arms],
        horizontal_spacing=0.015,
        vertical_spacing=0.02,
    )
    x_range, y_range = _axis_range(coords_by_arm, arms=arms)
    for row_i, (_name, idx) in enumerate(groups, start=1):
        rest = np.setdiff1d(np.arange(len(items)), idx, assume_unique=False)
        for col_i, arm in enumerate(arms, start=1):
            pts = coords_by_arm[arm]
            if len(rest):
                fig.add_trace(
                    go.Scattergl(
                        x=pts[rest, 0],
                        y=pts[rest, 1],
                        mode="markers",
                        marker={"size": 4, "color": "#d8d8d8", "opacity": 0.45},
                        hoverinfo="skip",
                        showlegend=False,
                    ),
                    row=row_i,
                    col=col_i,
                )
            hull = _hull_xy(pts[idx])
            if hull is not None:
                fig.add_trace(
                    go.Scatter(
                        x=hull[:, 0],
                        y=hull[:, 1],
                        mode="lines",
                        line={"color": "#222", "width": 1.2},
                        hoverinfo="skip",
                        showlegend=False,
                    ),
                    row=row_i,
                    col=col_i,
                )
            fig.add_trace(
                go.Scattergl(
                    x=pts[idx, 0],
                    y=pts[idx, 1],
                    mode="markers",
                    marker={
                        "size": 8,
                        "color": [colors[i] for i in idx],
                        "line": {"width": 0.4, "color": "#555"},
                    },
                    customdata=[custom[i] for i in idx],
                    hovertemplate=hover,
                    showlegend=False,
                ),
                row=row_i,
                col=col_i,
            )
            centroid = pts[idx].mean(axis=0)
            fig.add_trace(
                go.Scatter(
                    x=[float(centroid[0])],
                    y=[float(centroid[1])],
                    mode="markers",
                    marker={"size": 11, "symbol": "x", "color": "#111", "line": {"width": 1.5}},
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=row_i,
                col=col_i,
            )
    fig.update_xaxes(range=x_range, visible=False)
    fig.update_yaxes(range=y_range, visible=False)
    fig.update_layout(
        height=max(420, 210 * max(1, len(groups))),
        template="plotly_white",
        margin={"l": 90, "r": 20, "t": 50, "b": 20},
        hovermode="closest",
    )
    return fig


def subtype_overlay_figure(
    coords_by_arm: dict[str, np.ndarray],
    items: list[dict[str, Any]],
    preview_rel: dict[str, list[str]],
    *,
    arms: tuple[str, ...] | None = None,
) -> go.Figure:
    arms = _figure_arms(coords_by_arm, arms)
    groups = _subtype_groups(items)
    custom = _customdata(items, preview_rel, arms)
    hover = (
        "<b>%{customdata[3]}</b><br>%{customdata[2]} · %{customdata[5]}<br>"
        "%{fullData.name}<extra></extra>"
    )
    n_cols = min(4, max(1, len(groups) or 1))
    n_rows = max(1, int(np.ceil(len(groups) / n_cols)))
    titles = [f"{SUBTYPE_LABELS.get(name, name)} n={len(idx)}" for name, idx in groups]
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=titles,
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=0.03,
        vertical_spacing=0.08,
    )
    x_range, y_range = _axis_range(coords_by_arm, arms=arms)
    for panel, (_name, idx) in enumerate(groups):
        row, col = divmod(panel, n_cols)
        for arm in arms:
            pts = coords_by_arm[arm][idx]
            hull = _hull_xy(pts)
            if hull is not None:
                fig.add_trace(
                    go.Scatter(
                        x=hull[:, 0],
                        y=hull[:, 1],
                        mode="lines",
                        line={"color": ARM_COLORS[arm], "width": 2},
                        name=arm,
                        legendgroup=arm,
                        showlegend=panel == 0,
                        hoverinfo="skip",
                    ),
                    row=row + 1,
                    col=col + 1,
                )
            fig.add_trace(
                go.Scattergl(
                    x=pts[:, 0],
                    y=pts[:, 1],
                    mode="markers",
                    marker={"size": 7, "color": ARM_COLORS[arm], "opacity": 0.7},
                    customdata=[custom[i] for i in idx],
                    hovertemplate=hover,
                    name=arm,
                    legendgroup=arm,
                    showlegend=False,
                ),
                row=row + 1,
                col=col + 1,
            )
    fig.update_xaxes(range=x_range, visible=False)
    fig.update_yaxes(range=y_range, visible=False)
    fig.update_layout(
        height=max(420, 380 * n_rows),
        template="plotly_white",
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        hovermode="closest",
        legend={"orientation": "h", "y": 1.08},
    )
    return fig


def _preview_grid_html(prefix: str, arms: tuple[str, ...]) -> str:
    figures = "".join(
        f"<figure><img id='{html.escape(prefix)}-{html.escape(arm)}' alt='{html.escape(arm)}'>"
        f"<figcaption>{html.escape(ARM_TITLES.get(arm, arm))}</figcaption></figure>"
        for arm in arms
    )
    return f'<div class="preview-grid">{figures}</div>'


def _arm_label(arms: tuple[str, ...]) -> str:
    return "/".join(arms)


def _metric_cards(metrics: dict[str, Any], arms: tuple[str, ...]) -> str:
    cards = []
    for arm in arms:
        arm_metrics = metrics.get("arms", {}).get(arm) or {}
        neighbors = arm_metrics.get("attribute_neighbors") or {}
        gaps = arm_metrics.get("intra_inter") or {}
        delta = (metrics.get("deltas_vs_A") or {}).get(arm)
        shift = (metrics.get("pairwise_vs_A") or {}).get(arm)
        lines = [
            f"<h3>암 {html.escape(arm)}</h3>",
            f"<p>속성 P@k macro: <b>{neighbors.get('macro')}</b></p>",
            "<ul>",
        ]
        for kind in ("color", "pattern", "material", "shape"):
            row = neighbors.get(kind) or {}
            lines.append(
                f"<li>{kind}: {row.get('precision_at_k')} "
                f"(CI {row.get('ci95')}, n={row.get('queries')})</li>"
            )
        lines.append("</ul><p>intra-inter gap</p><ul>")
        for kind, gap in gaps.items():
            lines.append(
                f"<li>{html.escape(kind)}: gap={gap.get('gap')} "
                f"intra={gap.get('intra')} inter={gap.get('inter')}</li>"
            )
        lines.append("</ul>")
        if delta:
            lines.append(
                "<p>A 대비 Δ macro P@k = "
                f"<b>{delta.get('macro_precision_at_k')}</b>, "
                f"Δ color gap = {delta.get('color_gap')}</p>"
            )
        if shift:
            lines.append(
                "<p>vs A: cosine="
                f"{shift.get('mean_item_cosine')}, CKA={shift.get('linear_cka')}, "
                f"Spearman={shift.get('distance_spearman')}, "
                f"Jaccard={shift.get('neighborhood_jaccard')}</p>"
            )
        cards.append(f"<div class='card'>{''.join(lines)}</div>")
    return "".join(cards)


def _density_table(metrics: dict[str, Any], arms: tuple[str, ...]) -> str:
    density = metrics.get("density") or {}
    by_arm = density.get("arms") or {}
    if not by_arm:
        return "<p class='note'>밀도·분리 지표가 아직 없습니다.</p>"
    keys = [
        ("centroid_radius_p50", "subtype p50 반지름 ↓"),
        ("core_knn_cosine", "subtype 핵심 k-NN ↓"),
        ("mean_knn_cosine", "subtype 평균 k-NN ↓"),
        ("within_pairwise_cosine", "subtype 내부 쌍거리 ↓"),
        ("hull_area_2d", "정렬 2D hull 면적 ↓"),
        ("core_hull_area_2d", "핵심 2D hull 면적 ↓"),
    ]

    def _cell(arm: str, key: str) -> str:
        row = (by_arm.get(arm) or {}).get("subtype_macro") or {}
        value = row.get(key)
        return "-" if value is None else f"{value:.4f}"

    header = "<tr><th>지표</th>" + "".join(f"<th>{arm}</th>" for arm in arms) + "</tr>"
    rows = []
    for key, label in keys:
        rows.append(
            "<tr><td>"
            + html.escape(label)
            + "</td>"
            + "".join(f"<td>{_cell(arm, key)}</td>" for arm in arms)
            + "</tr>"
        )
    extra = [
        ("outlier_span_ratio", "overall p95/p50", lambda arm: ((by_arm.get(arm) or {}).get("overall") or {}).get("outlier_span_ratio")),
        ("subtype_separation", "제품군 분리도 ↑", lambda arm: (by_arm.get(arm) or {}).get("subtype_separation")),
        ("subtype_knn_purity", "subtype k-NN 순도 ↑", lambda arm: (by_arm.get(arm) or {}).get("subtype_knn_purity")),
    ]
    for _key, label, getter in extra:
        rows.append(
            "<tr><td>"
            + html.escape(label)
            + "</td>"
            + "".join(
                f"<td>{'-' if getter(arm) is None else f'{getter(arm):.4f}'}</td>" for arm in arms
            )
            + "</tr>"
        )
    group_names = sorted({name for arm in arms for name in ((by_arm.get(arm) or {}).get("by_subtype") or {})})
    group_rows = []
    for name in group_names:
        label = SUBTYPE_LABELS.get(name, name)
        for key, suffix in (
            ("core_knn_cosine", "핵심 k-NN"),
            ("centroid_radius_p50", "p50"),
            ("hull_area_2d", "hull"),
        ):
            cells = []
            for arm in arms:
                value = (((by_arm.get(arm) or {}).get("by_subtype") or {}).get(name) or {}).get(key)
                cells.append(f"<td>{'-' if value is None else value}</td>")
            group_rows.append(
                "<tr><td>"
                + html.escape(f"{label} · {suffix}")
                + "</td>"
                + "".join(cells)
                + "</tr>"
            )
    return (
        "<table class='metrics'><thead>"
        + header
        + "</thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
        + "<h3>제품군별 밀도</h3><table class='metrics'><thead>"
        + header
        + "</thead><tbody>"
        + "".join(group_rows)
        + "</tbody></table>"
    )


def _doc_blocks() -> str:
    blocks = []
    for doc in METRIC_DOCS:
        blocks.append(
            "<article class='doc'>"
            f"<h3>{html.escape(doc['title'])}</h3>"
            f"<pre>{html.escape(doc['formula'])}</pre>"
            f"<p>{html.escape(doc['meaning'])}</p>"
            "</article>"
        )
    return "".join(blocks)


def write_report(
    path: Path,
    *,
    run_id: str,
    items: list[dict[str, Any]],
    metrics: dict[str, Any],
    mask_stats: dict[str, Any],
    coverage: dict[str, Any],
    independent_fig: go.Figure,
    aligned_fig: go.Figure,
    subtype_grid_fig: go.Figure,
    subtype_overlay_fig: go.Figure,
    projection_quality: dict[str, Any],
    arms: tuple[str, ...] | None = None,
) -> Path:
    arms = resolve_arms(arms or coverage.get("arms") or tuple((metrics.get("arms") or {}).keys()) or ALL_ARMS)
    arm_label = _arm_label(arms)
    box_spec = str(coverage.get("box_spec") or "grounded_sam21/torso_prior")
    mask_spec = str(coverage.get("mask_spec") or "segformer garment ∩ padded box")
    independent_div = independent_fig.to_html(
        full_html=False, include_plotlyjs="cdn", div_id="umap-independent"
    )
    aligned_div = aligned_fig.to_html(
        full_html=False, include_plotlyjs=False, div_id="umap-aligned"
    )
    subtype_grid_div = subtype_grid_fig.to_html(
        full_html=False, include_plotlyjs=False, div_id="umap-subtype-grid"
    )
    subtype_overlay_div = subtype_overlay_fig.to_html(
        full_html=False, include_plotlyjs=False, div_id="umap-subtype-overlay"
    )
    warning = ""
    if int(coverage.get("n_used") or 0) < int(coverage.get("n_dataset") or 0):
        skipped = coverage.get("skipped") or []
        empty = sum(1 for row in skipped if row.get("reason") == "empty_proposals")
        incomplete = sum(1 for row in skipped if row.get("reason") == "incomplete_cache")
        if empty and not incomplete:
            warning = (
                "<div class='warning'><b>탐지 실패 제외:</b> "
                f"데이터셋 {coverage.get('n_dataset')}장 중 SAM3가 박스를 못 찾은 "
                f"{empty}장을 빼고 {coverage.get('n_used')}장으로 비교했습니다. "
                "프롬프트에 안 잡힌 sleeveless/tank 등이 대부분입니다.</div>"
            )
        else:
            warning = (
                "<div class='warning'><b>부분 표본:</b> "
                f"데이터셋 {coverage.get('n_dataset')}장 중 박스·마스크가 모두 있는 "
                f"{coverage.get('n_used')}장만 사용했습니다. "
                "전체 표본 cache 이후에 결론을 내십시오.</div>"
            )
    overlay_note = ", ".join({"A": "파랑 A", "B": "주황 B", "C": "초록 C", "D": "빨강 D"}.get(arm, arm) for arm in arms)
    if set(arms) == {"A", "C"}:
        decision = (
            "개선의 1차 판정은 속성 P@k와 intra-inter gap입니다. "
            "CKA·코사인·Jaccard는 “얼마나 달라졌는지”이지 “좋아졌는지”가 아닙니다. "
            "C가 A를 이기면 SAM3 soft mask를 채택하고, A가 이기면 원본을 유지합니다."
        )
    else:
        decision = (
            "개선의 1차 판정은 속성 P@k와 intra-inter gap입니다. "
            "CKA·코사인·Jaccard는 “얼마나 달라졌는지”이지 “좋아졌는지”가 아닙니다. "
            "C가 A·B를 이기고 D를 이기면 soft mask를 채택합니다. "
            "D가 더 좋으면 마스크가 충분히 깨끗합니다. B가 더 좋으면 마스크를 아직 쓰지 않습니다."
        )
    arms_js = json.dumps(list(arms))
    body = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<title>전처리 {html.escape(arm_label)} 임베딩 비교</title>
<style>
html{{color-scheme:light;background:#fff}}
body{{font-family:-apple-system,BlinkMacSystemFont,sans-serif;max-width:1600px;margin:auto;padding:24px;color:#222}}
.warning{{background:#fff4ce;border-left:5px solid #d89b00;padding:14px;margin:16px 0}}
.layout{{display:grid;grid-template-columns:minmax(0,1fr) 340px;gap:16px;align-items:start}}
.preview{{position:sticky;top:16px;border:1px solid #ddd;background:#fafafa;padding:12px;min-height:520px}}
.preview-grid{{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:8px}}
.preview-grid figure{{margin:0;border:1px solid #eee;background:#fff;padding:4px}}
.preview-grid img{{display:block;width:100%;aspect-ratio:1/1;object-fit:contain;background:#111}}
.preview-grid figcaption{{font-size:12px;text-align:center;padding-top:4px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:10px}}
.card,.doc{{border:1px solid #ddd;padding:12px;background:#fafafa}}
.docs{{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:10px}}
pre{{background:#f6f6f6;padding:10px;overflow:auto;white-space:pre-wrap}}
.note{{color:#555;font-size:14px}}
table.metrics{{border-collapse:collapse;width:100%;margin:12px 0 24px;font-size:13px}}
table.metrics th,table.metrics td{{border:1px solid #ddd;padding:6px 8px;text-align:right}}
table.metrics th:first-child,table.metrics td:first-child{{text-align:left}}
</style></head><body>
<h1>전처리 {html.escape(arm_label)}가 FashionSigLIP 정렬을 바꾸는가</h1>
<p>run=<code>{html.escape(run_id)}</code> · n=<code>{len(items)}</code> ·
박스=<code>{html.escape(box_spec)}</code> ·
마스크=<code>{html.escape(mask_spec)}</code></p>
{warning}
<p class="note">점 색은 상품명에서 뽑은 첫 번째 색 weak label입니다. 어느 점을 가리켜도 오른쪽에서 {html.escape(arm_label)} 입력을 같이 봅니다.
독립 UMAP의 축·회전은 비교하지 마십시오. 이웃과 아래 정렬 패널을 보십시오.</p>
<h2>1. 독립 UMAP</h2>
<div class="layout">
  <div>{independent_div}</div>
  <aside class="preview">
    <p id="preview-meta">상품 미리보기</p>
    {_preview_grid_html("img", arms)}
  </aside>
</div>
<h2>2. A에 맞춘 Procrustes 정렬 UMAP</h2>
<div class="layout">
  <div>{aligned_div}</div>
  <aside class="preview">
    <p id="aligned-meta">정렬 패널 미리보기</p>
    {_preview_grid_html("img2", arms)}
  </aside>
</div>
<h2>3. 제품군별 {html.escape(arm_label)} 분포</h2>
<p class="note">행은 subtype, 열은 전처리 암입니다. 축 범위는 선택한 암·모든 제품군에 동일합니다. 회색은 다른 제품군, 색점은 해당 제품군, 검은 선은 convex hull, X는 중심입니다. 점이 한쪽으로 모이거나 hull이 작아지면 그 제품군 표현이 더 뭉친 것입니다.</p>
<div class="layout">
  <div>{subtype_grid_div}</div>
  <aside class="preview">
    <p id="subtype-meta">제품군 패널 미리보기</p>
    {_preview_grid_html("img3", arms)}
  </aside>
</div>
<h3>같은 제품군 위에 {html.escape(arm_label)} 겹치기</h3>
<p class="note">{html.escape(overlay_note)}입니다. hull이 겹치면 전처리가 그 제품군 배치를 거의 안 바꾼 것이고, C hull만 이동·축소되면 마스크가 그 제품군 구조를 바꾼 것입니다.</p>
<div>{subtype_overlay_div}</div>
<h2>4. 정량 비교</h2>
<div class="cards">{_metric_cards(metrics, arms)}</div>
<h3>밀집도와 제품군 정렬</h3>
<p class="note">↓는 작을수록 핵심이 밀집, ↑는 클수록 제품군이 잘 갈립니다. 2D hull은 겹침 그림의 면적에 해당하고, p50·핵심 k-NN은 고차원 임베딩에서 이상치에 덜 흔들립니다.</p>
{_density_table(metrics, arms)}
<h3>마스크 통계</h3>
<pre>{html.escape(json.dumps(mask_stats, ensure_ascii=False, indent=2))}</pre>
<h3>투영 품질</h3>
<pre>{html.escape(json.dumps(projection_quality, ensure_ascii=False, indent=2))}</pre>
<h2>5. 수치의 정의</h2>
<p class="note">{html.escape(decision)}</p>
<div class="docs">{_doc_blocks()}</div>
<script>
const REPORT_ARMS = {arms_js};
const ARM_PATH_OFFSET = {ARM_PATH_OFFSET};
function showPoint(point, prefix, metaId) {{
  const c = point.customdata || [];
  const meta = document.getElementById(metaId);
  if (meta) meta.innerHTML = ['<b>' + (c[3] || '') + '</b>', c[2] || '', (c[4] || '') + ' · ' + (c[5] || '') + ' · ' + (c[1] || ''),
    'color=' + (c[6] || '-') + ' | pattern=' + (c[7] || '-'), 'mask=' + (c[9] || '') + ' fill=' + (c[10] || '')].join('<br>');
  REPORT_ARMS.forEach((arm, i) => {{
    const el = document.getElementById(prefix + '-' + arm);
    const src = c[ARM_PATH_OFFSET + i];
    if (el && src) el.src = src;
  }});
}}
function bind(id, prefix, metaId) {{
  const plot = document.getElementById(id);
  if (!plot || !plot.on) {{ setTimeout(() => bind(id, prefix, metaId), 50); return; }}
  plot.on('plotly_hover', e => {{ if (e && e.points && e.points.length) showPoint(e.points[0], prefix, metaId); }});
}}
bind('umap-independent', 'img', 'preview-meta');
bind('umap-aligned', 'img2', 'aligned-meta');
bind('umap-subtype-grid', 'img3', 'subtype-meta');
bind('umap-subtype-overlay', 'img3', 'subtype-meta');
</script>
</body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def write_preview_gallery(
    path: Path,
    *,
    run_id: str,
    items: list[dict[str, Any]],
    preview_rel: dict[str, list[str]],
    mask_stats: dict[str, Any],
    coverage: dict[str, Any],
    arms: tuple[str, ...] | None = None,
) -> Path:
    arms = resolve_arms(arms or coverage.get("arms") or tuple(preview_rel.keys()) or ALL_ARMS)
    tiles = []
    for index, item in enumerate(items):
        tiles.append(
            "<article class='tile'>"
            f"<h3>{html.escape(str(item.get('product') or item.get('item_id')))}</h3>"
            f"<p>{html.escape(str(item.get('subtype') or ''))} · "
            f"{html.escape(str(item.get('mask_source') or ''))} · fill="
            f"{float(item.get('fill_ratio') or 0):.3f}</p>"
            "<div class='row'>"
            + "".join(
                f"<figure><img src='{html.escape(preview_rel[arm][index])}' alt='{arm}'>"
                f"<figcaption>{arm}</figcaption></figure>"
                for arm in arms
            )
            + "</div></article>"
        )
    body = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<title>전처리 미리보기</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,sans-serif;max-width:1400px;margin:auto;padding:24px}}
.row{{display:grid;grid-template-columns:repeat({len(arms)},1fr);gap:8px}}
.tile{{border:1px solid #ddd;padding:12px;margin:12px 0}}
img{{width:100%;aspect-ratio:1/1;object-fit:contain;background:#111}}
.warning{{background:#fff4ce;border-left:5px solid #d89b00;padding:14px}}
</style></head><body>
<h1>{html.escape(_arm_label(arms))} 전처리 미리보기</h1>
<p>run=<code>{html.escape(run_id)}</code> · n={len(items)} · dataset={coverage.get('n_dataset')} · used={coverage.get('n_used')}</p>
<pre>{html.escape(json.dumps(mask_stats, ensure_ascii=False, indent=2))}</pre>
{''.join(tiles)}
</body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path
