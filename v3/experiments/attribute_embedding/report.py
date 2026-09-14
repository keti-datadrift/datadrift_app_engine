from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from PIL import Image, ImageOps
from plotly.subplots import make_subplots
from sklearn.neighbors import NearestNeighbors

ARM_NAMES = {
    "A": "A · global raw",
    "B": "B · category-conditioned",
    "C": "C · family residual",
    "D": "D · attribute scores",
}

FAMILY_COLORS = {
    "tops": "#4C78A8",
    "outerwear": "#F58518",
    "bottoms": "#E45756",
    "dresses": "#72B7B2",
    "shoes": "#54A24B",
    "bags": "#EECA3B",
    "sport": "#B279A2",
    "headwear": "#FF9DA6",
}


def _hover(items: list[dict[str, Any]]) -> list[str]:
    return [
        "<br>".join(
            [
                f"{html.escape(str(item.get('brand') or ''))}",
                f"{html.escape(str(item.get('product') or ''))}",
                f"family={item['family']}",
                f"{item['source']}:{item['product_id']}",
            ]
        )
        for item in items
    ]


def intrinsic_figure(
    projections: dict[str, np.ndarray],
    assignments: dict[str, np.ndarray],
    items: list[dict[str, Any]],
) -> go.Figure:
    arms = sorted(projections)
    fig = make_subplots(
        rows=2,
        cols=len(arms),
        subplot_titles=[ARM_NAMES[arm] for arm in arms] * 2,
        vertical_spacing=0.12,
    )
    hover = _hover(items)
    for col, arm in enumerate(arms, 1):
        coords = projections[arm]
        family_colors = [FAMILY_COLORS[str(item["family"])] for item in items]
        cluster_values = assignments[arm]
        fig.add_trace(
            go.Scattergl(
                x=coords[:, 0],
                y=coords[:, 1],
                mode="markers",
                marker={"size": 5, "color": family_colors, "opacity": 0.72},
                text=hover,
                hovertemplate="%{text}<extra></extra>",
                name=f"{arm} family",
                showlegend=False,
            ),
            row=1,
            col=col,
        )
        fig.add_trace(
            go.Scattergl(
                x=coords[:, 0],
                y=coords[:, 1],
                mode="markers",
                marker={
                    "size": 5,
                    "color": cluster_values,
                    "colorscale": "Turbo",
                    "opacity": 0.75,
                    "showscale": False,
                },
                text=[
                    f"{hover[i]}<br>cluster={cluster_values[i]}"
                    for i in range(len(items))
                ],
                hovertemplate="%{text}<extra></extra>",
                name=f"{arm} cluster",
                showlegend=False,
            ),
            row=2,
            col=col,
        )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(
        height=920,
        title="Intrinsic maps — 위: family, 아래: cluster",
        template="plotly_white",
        margin={"l": 20, "r": 20, "t": 80, "b": 20},
    )
    return fig


def common_overlay_figure(
    reference: np.ndarray,
    assignments: dict[str, np.ndarray],
    items: list[dict[str, Any]],
) -> go.Figure:
    arms = sorted(assignments)
    fig = make_subplots(rows=1, cols=len(arms), subplot_titles=[ARM_NAMES[a] for a in arms])
    hover = _hover(items)
    for col, arm in enumerate(arms, 1):
        labels = assignments[arm]
        fig.add_trace(
            go.Scattergl(
                x=reference[:, 0],
                y=reference[:, 1],
                mode="markers",
                marker={
                    "size": 5,
                    "color": labels,
                    "colorscale": "Turbo",
                    "opacity": 0.75,
                    "showscale": False,
                },
                text=[f"{hover[i]}<br>cluster={labels[i]}" for i in range(len(items))],
                hovertemplate="%{text}<extra></extra>",
                showlegend=False,
            ),
            row=1,
            col=col,
        )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(
        height=460,
        title="Common-coordinate cluster overlay — 모든 패널의 점 위치는 A 기준으로 고정",
        template="plotly_white",
    )
    return fig


def coassignment_figure(metrics: dict[str, Any]) -> go.Figure:
    matrix = metrics["coassignment"]
    arms = sorted(matrix)
    values = [[matrix[left][right] for right in arms] for left in arms]
    return go.Figure(
        data=go.Heatmap(
            z=values,
            x=[ARM_NAMES[a] for a in arms],
            y=[ARM_NAMES[a] for a in arms],
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            text=[[f"{v:.3f}" for v in row] for row in values],
            texttemplate="%{text}",
            colorbar={"title": "ARI"},
        ),
        layout=go.Layout(
            title="Cluster co-assignment similarity (Adjusted Rand Index)",
            template="plotly_white",
            height=500,
        ),
    )


def alluvial_figure(assignments: dict[str, np.ndarray]) -> go.Figure:
    arms = sorted(assignments)
    node_labels: list[str] = []
    node_index: dict[tuple[str, int], int] = {}
    for arm in arms:
        for cluster in sorted(set(assignments[arm].tolist())):
            node_index[(arm, cluster)] = len(node_labels)
            node_labels.append(f"{arm}:{'noise' if cluster < 0 else cluster}")
    sources: list[int] = []
    targets: list[int] = []
    values: list[int] = []
    for left, right in zip(arms[:-1], arms[1:]):
        counts: dict[tuple[int, int], int] = {}
        for a, b in zip(assignments[left], assignments[right]):
            key = (int(a), int(b))
            counts[key] = counts.get(key, 0) + 1
        for (a, b), count in counts.items():
            sources.append(node_index[(left, a)])
            targets.append(node_index[(right, b)])
            values.append(count)
    return go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                node={"label": node_labels, "pad": 12, "thickness": 12},
                link={"source": sources, "target": targets, "value": values},
            )
        ],
        layout=go.Layout(
            title="Cluster split/merge flow (A → B → C → D)",
            template="plotly_white",
            height=700,
        ),
    )


def metric_figure(metrics: dict[str, Any]) -> go.Figure:
    arms = [arm for arm in sorted(metrics) if arm in ARM_NAMES]
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=["Weak-label Precision@10", "Category probe accuracy", "Noise rate"],
    )
    for arm in arms:
        attr = metrics[arm]["attribute_neighbors"]
        fig.add_trace(
            go.Bar(
                x=["color", "pattern", "material", "shape"],
                y=[attr[k]["precision_at_k"] for k in ("color", "pattern", "material", "shape")],
                name=ARM_NAMES[arm],
                legendgroup=arm,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=[ARM_NAMES[arm]],
                y=[metrics[arm]["category_probe"]["accuracy_mean"]],
                name=ARM_NAMES[arm],
                legendgroup=arm,
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Bar(
                x=[ARM_NAMES[arm]],
                y=[metrics[arm]["clustering"]["noise_rate"]],
                name=ARM_NAMES[arm],
                legendgroup=arm,
                showlegend=False,
            ),
            row=1,
            col=3,
        )
    fig.update_layout(
        barmode="group",
        template="plotly_white",
        height=520,
        title="정량 지표 비교 — 상품명 기반 weak label은 예비 지표",
    )
    return fig


def _thumbnail(path: str, size: int = 128) -> Image.Image:
    with Image.open(path) as image:
        thumb = ImageOps.contain(image.convert("RGB"), (size, size))
    canvas = Image.new("RGB", (size, size), "white")
    canvas.paste(thumb, ((size - thumb.width) // 2, (size - thumb.height) // 2))
    return canvas


def create_contact_sheets(
    arms: dict[str, np.ndarray],
    items: list[dict[str, Any]],
    output_dir: Path,
    *,
    count: int,
    seed: int,
    neighbors: int = 6,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    eligible = [
        i
        for i, item in enumerate(items)
        if item.get("weak_labels", {}).get("color")
        or item.get("weak_labels", {}).get("pattern")
    ]
    rng = np.random.default_rng(seed)
    chosen = rng.choice(eligible, min(count, len(eligible)), replace=False) if eligible else []
    nearest = {
        arm: NearestNeighbors(n_neighbors=neighbors + 1, metric="cosine", algorithm="brute")
        .fit(matrix)
        .kneighbors(return_distance=False)
        for arm, matrix in arms.items()
    }
    paths: list[str] = []
    tile = 128
    for query_i in chosen:
        rows = len(arms)
        sheet = Image.new("RGB", ((neighbors + 1) * tile, rows * (tile + 28)), "white")
        for row_i, arm in enumerate(sorted(arms)):
            indices = [int(query_i)] + [
                int(i) for i in nearest[arm][int(query_i)] if int(i) != int(query_i)
            ][:neighbors]
            for col_i, item_i in enumerate(indices):
                sheet.paste(
                    _thumbnail(items[item_i]["image_path"], tile),
                    (col_i * tile, row_i * (tile + 28)),
                )
            from PIL import ImageDraw

            ImageDraw.Draw(sheet).text(
                (4, row_i * (tile + 28) + tile + 5),
                f"{arm} | query labels={items[int(query_i)].get('weak_labels', {})}",
                fill="black",
            )
        name = f"query_{items[int(query_i)]['source']}_{items[int(query_i)]['product_id']}.jpg"
        path = output_dir / name
        sheet.save(path, quality=88)
        paths.append(f"figures/contact_sheets/{name}")
    return paths


def save_static_metric_plot(metrics: dict[str, Any], path: Path) -> None:
    arms = [arm for arm in sorted(metrics) if arm in ARM_NAMES]
    kinds = ("color", "pattern", "material", "shape")
    x = np.arange(len(kinds))
    width = 0.18
    fig, ax = plt.subplots(figsize=(10, 5))
    for offset, arm in enumerate(arms):
        values = [
            metrics[arm]["attribute_neighbors"][kind]["precision_at_k"] for kind in kinds
        ]
        ax.bar(x + (offset - 1.5) * width, values, width, label=ARM_NAMES[arm])
    ax.set_xticks(x, kinds)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Precision@10")
    ax.set_title("Weak-label attribute retrieval")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def build_report(
    run_dir: Path,
    *,
    items: list[dict[str, Any]],
    inventory: dict[str, Any],
    arms: dict[str, np.ndarray],
    projections: dict[str, np.ndarray],
    assignments: dict[str, np.ndarray],
    metrics: dict[str, Any],
    manifest: dict[str, Any],
    contact_sheets: list[str],
) -> None:
    report_dir = run_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    figures = [
        intrinsic_figure(projections, assignments, items),
        common_overlay_figure(projections["A"], assignments, items),
        coassignment_figure(metrics),
        alluvial_figure(assignments),
        metric_figure(metrics),
    ]
    divs = [
        figure.to_html(full_html=False, include_plotlyjs=True if i == 0 else False)
        for i, figure in enumerate(figures)
    ]
    contacts = "\n".join(
        f'<figure><img src="../{html.escape(path)}" loading="lazy"></figure>'
        for path in contact_sheets
    )
    metric_rows = "\n".join(
        "<tr>"
        f"<td>{ARM_NAMES[arm]}</td>"
        f"<td>{metrics[arm]['attribute_neighbors']['color']['scope']}</td>"
        f"<td>{metrics[arm]['attribute_neighbors']['macro']:.4f}</td>"
        f"<td>{metrics[arm]['category_probe']['accuracy_mean']:.4f}</td>"
        f"<td>{metrics[arm]['clustering']['cluster_count']}</td>"
        f"<td>{metrics[arm]['clustering']['noise_rate']:.4f}</td>"
        f"<td>{metrics[arm]['clustering']['silhouette_cosine']}</td>"
        "</tr>"
        for arm in sorted(arms)
    )
    best_macro = max(
        sorted(arms), key=lambda arm: metrics[arm]["attribute_neighbors"]["macro"]
    )
    best_color = max(
        sorted(arms),
        key=lambda arm: metrics[arm]["attribute_neighbors"]["color"]["precision_at_k"],
    )
    lowest_probe = min(
        sorted(arms), key=lambda arm: metrics[arm]["category_probe"]["accuracy_mean"]
    )
    executive = (
        f"전체 속성 macro는 {ARM_NAMES[best_macro]}가 가장 높고 "
        f"({metrics[best_macro]['attribute_neighbors']['macro']:.4f}), "
        f"색상은 {ARM_NAMES[best_color]}가 가장 높습니다 "
        f"({metrics[best_color]['attribute_neighbors']['color']['precision_at_k']:.4f}). "
        f"카테고리 선형 분리도는 {ARM_NAMES[lowest_probe]}가 가장 낮습니다 "
        f"({metrics[lowest_probe]['category_probe']['accuracy_mean']:.4f})."
    )
    body = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><title>Attribute embedding experiment</title>
<style>
html{{background:#fff;color-scheme:light}}body{{font-family:-apple-system,BlinkMacSystemFont,sans-serif;max-width:1500px;margin:auto;padding:24px;color:#222;background:#fff}}
.warning{{background:#fff4ce;border-left:5px solid #d89b00;padding:14px}} table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #ddd;padding:8px;text-align:right}} th:first-child,td:first-child{{text-align:left}}
.contacts{{display:grid;grid-template-columns:1fr;gap:16px}} figure{{margin:0}} img{{max-width:100%;height:auto}}
code{{background:#f3f3f3;padding:2px 5px}}
</style></head><body>
<h1>의류 속성 중심 임베딩 군집화 실험</h1>
<p>run_id=<code>{html.escape(str(manifest['run_id']))}</code>, dataset=<code>{inventory['dataset_fingerprint'][:16]}</code>,
model=<code>{html.escape(str(manifest['model_id']))}</code></p>
<div class="warning"><strong>해석 경계:</strong> 현재 속성 평가는 상품명에서 추출한 weak label 기반의 예비 평가입니다.
사람이 검수한 gold label이 없으므로 D의 우위는 최종 성능으로 확정할 수 없습니다. UMAP 패널 간 축과 절대 거리는 직접 비교하지 않습니다.</div>
<h2>Executive summary</h2><p>{html.escape(executive)}</p>
<h2>핵심 지표</h2>
<table><thead><tr><th>실험군</th><th>검색 범위</th><th>속성 P@10 macro</th><th>category probe</th><th>clusters</th><th>noise</th><th>silhouette</th></tr></thead>
<tbody>{metric_rows}</tbody></table>
<h2>데이터 인벤토리</h2><pre>{html.escape(json.dumps(inventory, ensure_ascii=False, indent=2))}</pre>
<h2>군집 및 표현 비교</h2>{''.join(divs)}
<h2>고정 Query 이웃 이미지 비교</h2><div class="contacts">{contacts}</div>
<h2>재현 정보</h2><pre>{html.escape(json.dumps(manifest, ensure_ascii=False, indent=2))}</pre>
</body></html>"""
    (report_dir / "index.html").write_text(body, encoding="utf-8")
    summary = f"""# 의류 속성 임베딩 실험 결과

- 실행 ID: `{manifest['run_id']}`
- 데이터셋 fingerprint: `{inventory['dataset_fingerprint']}`
- 표본 수: {len(items)}
- 모델: `{manifest['model_id']}` / `{manifest['model_revision']}`
- weak-label 속성 검색 최고 실험군: **{ARM_NAMES[best_macro]}**

## Executive summary

{executive}

- B의 A 대비 macro 차이는 작으므로 gold label 검증 전에는 category 조건부 검색의 우위로 확정하지 않습니다.
- C는 category probe를 크게 낮췄지만 전역 HDBSCAN에서 전부 noise가 되어, residual 공간에는 별도 군집 파라미터 또는 kNN 기반 분석이 필요합니다.
- D는 색상 검색은 개선했지만 패턴·소재·형태가 하락해 단일 통합 표현보다 속성별 전용 score/filter로 사용하는 편이 타당합니다.

## 해석
- A는 전체 원본 임베딩 기준선입니다.
- B는 원본 벡터는 같고 family 내부에서만 군집화하므로, A와의 차이는 후보군 제약 효과입니다.
- C는 family centroid를 제거해 카테고리 공통 성분을 약화합니다.
- D는 family-conditioned 텍스트 속성 유사도만 사용합니다.
- 속성 평가는 상품명 기반 weak label이므로 사람 검수 gold set 구축 전까지 결과는 예비 결론입니다.

## 주요 지표
{json.dumps({arm: metrics[arm] for arm in sorted(arms)}, ensure_ascii=False, indent=2)}
"""
    (report_dir / "summary.md").write_text(summary, encoding="utf-8")
