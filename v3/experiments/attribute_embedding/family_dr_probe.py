"""Family-local dimensionality reduction probe on raw FashionSigLIP embeddings.

Date: 2026-09-11

This does NOT re-encode images. It takes a single family subset of already
computed raw embeddings and fits several 2D projections so secondary visual
structure can be inspected qualitatively.
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
from PIL import Image, ImageOps
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE, trustworthiness

REPO_ROOT = Path(__file__).resolve().parents[2]

METHOD_ORDER = (
    "pca2",
    "svd2",
    "pca64_umap",
    "pca50_tsne",
)

METHOD_TITLES = {
    "pca2": "PCA → 2D",
    "svd2": "TruncatedSVD → 2D",
    "pca64_umap": "PCA64 → UMAP",
    "pca50_tsne": "PCA50 → t-SNE",
}

COLOR_PALETTE = {
    "black": "#222222",
    "white": "#D9D9D9",
    "gray": "#8A8A8A",
    "navy": "#1F3A5F",
    "blue": "#4C78A8",
    "red": "#E45756",
    "green": "#54A24B",
    "brown": "#9C755F",
    "yellow": "#EECA3B",
    "pink": "#FF9DA6",
    "purple": "#B279A2",
    "orange": "#F58518",
    "unknown": "#B0B0B0",
}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _thumb_data_uri(path: str, size: int = 384, quality: int = 90) -> str:
    """Build a preview-grade JPEG data URI.

    Source assets are already ~384px WebP thumbs; keep near-native size and
    high JPEG quality so the report preview is not re-compressed into mush.
    """

    with Image.open(path) as image:
        rgb = image.convert("RGB")
        # Never upscale tiny sources; only downscale when larger than target.
        if max(rgb.size) > size:
            rgb = ImageOps.contain(rgb, (size, size))
    buffer = io.BytesIO()
    rgb.save(buffer, format="JPEG", quality=quality, optimize=True, subsampling=0)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _primary_color(item: dict[str, Any]) -> str:
    colors = item.get("weak_labels", {}).get("color") or []
    return str(colors[0]) if colors else "unknown"


def _fit_methods(matrix: np.ndarray, seed: int) -> dict[str, dict[str, Any]]:
    import umap

    results: dict[str, dict[str, Any]] = {}
    n = len(matrix)

    pca2 = PCA(n_components=2, random_state=seed)
    coords_pca = pca2.fit_transform(matrix)
    results["pca2"] = {
        "coords": coords_pca.astype(np.float32),
        "notes": {
            "explained_variance_ratio": [float(x) for x in pca2.explained_variance_ratio_],
            "explained_variance_sum": float(sum(pca2.explained_variance_ratio_)),
        },
    }

    svd2 = TruncatedSVD(n_components=2, random_state=seed)
    coords_svd = svd2.fit_transform(matrix)
    results["svd2"] = {
        "coords": coords_svd.astype(np.float32),
        "notes": {
            "explained_variance_ratio": [float(x) for x in svd2.explained_variance_ratio_],
            "explained_variance_sum": float(sum(svd2.explained_variance_ratio_)),
        },
    }

    pca64 = PCA(n_components=min(64, n - 1, matrix.shape[1]), random_state=seed)
    reduced64 = pca64.fit_transform(matrix)
    umap_model = umap.UMAP(
        n_components=2,
        metric="cosine",
        n_neighbors=min(30, max(5, n // 6)),
        min_dist=0.12,
        random_state=seed,
        n_jobs=1,
    )
    coords_umap = umap_model.fit_transform(reduced64)
    results["pca64_umap"] = {
        "coords": coords_umap.astype(np.float32),
        "notes": {
            "pca_dim": int(pca64.n_components_),
            "pca64_variance_sum": float(sum(pca64.explained_variance_ratio_)),
            "n_neighbors": int(umap_model.n_neighbors),
            "min_dist": float(umap_model.min_dist),
        },
    }

    pca50 = PCA(n_components=min(50, n - 1, matrix.shape[1]), random_state=seed)
    reduced50 = pca50.fit_transform(matrix)
    tsne = TSNE(
        n_components=2,
        perplexity=min(30, max(5, (n - 1) // 3)),
        metric="euclidean",
        init="pca",
        learning_rate="auto",
        random_state=seed,
    )
    coords_tsne = tsne.fit_transform(reduced50)
    results["pca50_tsne"] = {
        "coords": coords_tsne.astype(np.float32),
        "notes": {
            "pca_dim": int(pca50.n_components_),
            "pca50_variance_sum": float(sum(pca50.explained_variance_ratio_)),
            "perplexity": float(tsne.perplexity),
        },
    }

    for name, payload in results.items():
        coords = payload["coords"]
        k = min(10, max(2, n // 10))
        payload["notes"]["trustworthiness"] = round(
            float(trustworthiness(matrix, coords, n_neighbors=k, metric="cosine")),
            4,
        )
    return results


def _point_customdata(items: list[dict[str, Any]]) -> list[list[Any]]:
    customdata: list[list[Any]] = []
    for index, item in enumerate(items):
        weak = item.get("weak_labels") or {}
        customdata.append(
            [
                index,
                str(item.get("brand") or ""),
                str(item.get("product") or ""),
                str(item.get("source") or ""),
                str(item.get("product_id") or ""),
                str(item.get("category_label") or ""),
                ", ".join(weak.get("color") or []) or "-",
                ", ".join(weak.get("pattern") or []) or "-",
                ", ".join(weak.get("material") or []) or "-",
                ", ".join(weak.get("shape") or []) or "-",
            ]
        )
    return customdata


def _scatter_figure(
    items: list[dict[str, Any]],
    methods: dict[str, dict[str, Any]],
    *,
    family: str,
) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[METHOD_TITLES[name] for name in METHOD_ORDER],
        horizontal_spacing=0.06,
        vertical_spacing=0.12,
    )
    customdata = _point_customdata(items)
    hover = (
        "<b>%{customdata[1]}</b><br>"
        "%{customdata[2]}<br>"
        "%{customdata[3]}:%{customdata[4]}<br>"
        "category=%{customdata[5]}<br>"
        "color=%{customdata[6]} | pattern=%{customdata[7]}<br>"
        "material=%{customdata[8]} | shape=%{customdata[9]}<br>"
        "<extra></extra>"
    )
    for index, name in enumerate(METHOD_ORDER):
        row, col = divmod(index, 2)
        coords = methods[name]["coords"]
        colors = [COLOR_PALETTE[_primary_color(item)] for item in items]
        fig.add_trace(
            go.Scatter(
                x=coords[:, 0],
                y=coords[:, 1],
                mode="markers",
                marker={"size": 9, "color": colors, "line": {"width": 0.5, "color": "#666"}},
                customdata=customdata,
                hovertemplate=hover,
                name=METHOD_TITLES[name],
                showlegend=False,
            ),
            row=row + 1,
            col=col + 1,
        )
    fig.update_xaxes(visible=False, showgrid=False, zeroline=False)
    fig.update_yaxes(visible=False, showgrid=False, zeroline=False)
    for axis_index in range(1, 5):
        x_ref = "x" if axis_index == 1 else f"x{axis_index}"
        yaxis = "yaxis" if axis_index == 1 else f"yaxis{axis_index}"
        fig.layout[yaxis].update(scaleanchor=x_ref, scaleratio=1)
    fig.update_layout(
        height=980,
        title=(
            f"Family-local raw embedding projections · family={family} · "
            f"n={len(items)} · color=weak color label"
        ),
        template="plotly_white",
        margin={"l": 30, "r": 30, "t": 80, "b": 30},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        hovermode="closest",
    )
    return fig


def build_report(
    *,
    family: str,
    items: list[dict[str, Any]],
    methods: dict[str, dict[str, Any]],
    source_run: str,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure = _scatter_figure(items, methods, family=family)
    # Plotly hovertemplate cannot reliably render <img> with large data-URI customdata.
    # Keep text hover, and drive a side preview panel from plotly_hover instead.
    plot_div = figure.to_html(
        full_html=False,
        include_plotlyjs="cdn",
        div_id="family-dr-plot",
    )
    thumbs_json = json.dumps(
        [str(item.get("thumb_data_uri") or "") for item in items],
        ensure_ascii=False,
    )

    color_counts = Counter(_primary_color(item) for item in items)
    method_cards = []
    for name in METHOD_ORDER:
        notes = methods[name]["notes"]
        method_cards.append(
            "<div class='card'>"
            f"<h3>{html.escape(METHOD_TITLES[name])}</h3>"
            f"<pre>{html.escape(json.dumps(notes, ensure_ascii=False, indent=2))}</pre>"
            "</div>"
        )

    caveats = """
    <ul>
      <li><b>이 화면은 가설 검증이지 성능 증명 도구가 아닙니다.</b> family를 고정한 뒤 2D로 접으면
      색상·패턴이 더 잘 보일 수 있지만, 그것이 원본 768D 검색이 실제로 색상 중심으로 바뀌었다는 뜻은 아닙니다.</li>
      <li><b>방법마다 목적함수가 다릅니다.</b> PCA/SVD는 전역 분산, UMAP/t-SNE는 지역 이웃 보존을 우선합니다.
      네 패널을 같은 품질 순위표처럼 읽으면 안 됩니다.</li>
      <li><b>UMAP/t-SNE는 허위 군집을 만들 수 있습니다.</b> 떨어져 보여도 고차원에서는 가깝거나,
      붙어 보여도 의미가 약한 경우가 있습니다. trustworthiness는 보조 지표일 뿐입니다.</li>
      <li><b>점 색은 상품명 weak label입니다.</b> 제목에 색이 없으면 unknown이 되고,
      제목 색과 실제 픽셀 색이 다를 수 있어 시각적 확증 편향이 생깁니다.</li>
      <li><b>표본이 family당 200개로 작습니다.</b> 탐색용으로는 충분하지만, 안정적인 구조 주장에는 부족합니다.</li>
      <li><b>미리보기 이미지는 정성 검사용입니다.</b> 눈에 띄는 몇 개 이웃으로 전체 구조를 일반화하면 안 됩니다.</li>
    </ul>
    """

    body = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<title>Family DR probe · {html.escape(family)}</title>
<style>
html{{color-scheme:light;background:#fff}}
body{{font-family:-apple-system,BlinkMacSystemFont,sans-serif;max-width:1500px;margin:auto;padding:24px;color:#222;background:#fff}}
.warning{{background:#fff4ce;border-left:5px solid #d89b00;padding:14px;margin:16px 0}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
.card{{border:1px solid #ddd;padding:12px;background:#fafafa}}
.plot-layout{{display:grid;grid-template-columns:minmax(0,1fr) 280px;gap:16px;align-items:start}}
.preview{{position:sticky;top:16px;border:1px solid #ddd;background:#fafafa;padding:12px;min-height:420px}}
.preview img{{display:block;width:100%;max-width:360px;height:auto;aspect-ratio:1/1;object-fit:contain;background:#fff;border:1px solid #eee;margin:0 auto 12px;image-rendering:auto}}
.preview .meta{{font-size:13px;line-height:1.45;word-break:break-word}}
.preview .muted{{color:#777}}
pre{{white-space:pre-wrap;word-break:break-word;font-size:12px}}
code{{background:#f3f3f3;padding:2px 5px}}
@media (max-width: 980px){{.plot-layout{{grid-template-columns:1fr}}}}
</style></head><body>
<h1>Family-local 차원압축 프로브</h1>
<p>family=<code>{html.escape(family)}</code> · n=<code>{len(items)}</code> ·
source_run=<code>{html.escape(source_run)}</code> · representation=<code>A_raw</code></p>
<div class="warning">
<strong>해석 경계</strong>
{caveats}
</div>
<h2>스캐터 비교</h2>
<p>점 색은 상품명 weak color label입니다. 점에 커서를 올리면 오른쪽 미리보기에 이미지와 상품 정보가 표시됩니다.</p>
<div class="plot-layout">
  <div>{plot_div}</div>
  <aside class="preview" id="product-preview">
    <div class="muted">점에 커서를 올리면 이미지가 표시됩니다.</div>
    <img id="preview-image" alt="product preview" hidden>
    <div class="meta" id="preview-meta"></div>
  </aside>
</div>
<h2>방법별 요약</h2>
<div class="grid">{''.join(method_cards)}</div>
<h2>표본 메모</h2>
<pre>{html.escape(json.dumps({
    "family": family,
    "count": len(items),
    "sources": dict(Counter(str(item.get("source")) for item in items)),
    "weak_color_counts": dict(color_counts),
    "generated_at": datetime.now(UTC).isoformat(),
}, ensure_ascii=False, indent=2))}</pre>
<script>
const THUMBS = {thumbs_json};
const previewImage = document.getElementById('preview-image');
const previewMeta = document.getElementById('preview-meta');
function showPreview(point) {{
  const cd = point.customdata || [];
  const idx = Number(cd[0]);
  const thumb = THUMBS[idx] || '';
  if (thumb) {{
    previewImage.hidden = false;
    previewImage.src = thumb;
  }} else {{
    previewImage.hidden = true;
    previewImage.removeAttribute('src');
  }}
  previewMeta.innerHTML = [
    '<b>' + (cd[1] || '') + '</b>',
    (cd[2] || ''),
    (cd[3] || '') + ':' + (cd[4] || ''),
    'category=' + (cd[5] || '-'),
    'color=' + (cd[6] || '-') + ' | pattern=' + (cd[7] || '-'),
    'material=' + (cd[8] || '-') + ' | shape=' + (cd[9] || '-'),
    'panel=' + (point.data && point.data.name ? point.data.name : '')
  ].join('<br>');
}}
function bindPreview() {{
  const plot = document.getElementById('family-dr-plot');
  if (!plot || !plot.on) {{
    setTimeout(bindPreview, 50);
    return;
  }}
  plot.on('plotly_hover', function(event) {{
    if (!event || !event.points || !event.points.length) return;
    showPreview(event.points[0]);
  }});
}}
bindPreview();
</script>
</body></html>"""

    report_path = output_dir / "index.html"
    report_path.write_text(body, encoding="utf-8")

    coords_payload = {
        name: {
            "coords": methods[name]["coords"].tolist(),
            "notes": methods[name]["notes"],
        }
        for name in METHOD_ORDER
    }
    (output_dir / "coords.json").write_text(
        json.dumps(
            {
                "family": family,
                "source_run": source_run,
                "item_ids": [f"{item['source']}:{item['product_id']}" for item in items],
                "methods": coords_payload,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", default="mvp-20260911-v2")
    parser.add_argument("--family", default="tops")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "artifacts" / "attribute_embedding",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.output_root / args.source_run
    items = _load_jsonl(run_dir / "dataset" / "items.jsonl")
    matrix = np.load(run_dir / "representations" / "A_raw.npz")["matrix"].astype(np.float32)
    selected = [
        (index, item)
        for index, item in enumerate(items)
        if str(item.get("family")) == args.family
    ]
    if len(selected) < 30:
        raise RuntimeError(f"family={args.family} has only {len(selected)} items")

    family_items = []
    for _, item in selected:
        row = dict(item)
        row["thumb_data_uri"] = _thumb_data_uri(str(item["image_path"]))
        family_items.append(row)
    family_matrix = matrix[[index for index, _ in selected]]

    print(f"fitting DR for family={args.family} n={len(family_items)}", flush=True)
    methods = _fit_methods(family_matrix, seed=args.seed)
    out_dir = args.output_root / f"family-dr-{args.family}-{args.source_run}"
    report = build_report(
        family=args.family,
        items=family_items,
        methods=methods,
        source_run=args.source_run,
        output_dir=out_dir,
    )
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
