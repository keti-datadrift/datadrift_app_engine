from __future__ import annotations

import csv
import html
import json
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from io_utils import read_jsonl, restore_mask, save_overlay, write_json, write_jsonl
from schema import VISUAL_RUBRIC

FAILURE_REASONS = {
    "empty_proposals": "제안 0건 (프롬프트 미검출)",
    "infer_error": "추론 예외",
    "missing_proposal_row": "캐시 행 없음",
}


def write_reviewer_sheet(
    path: Path,
    items: list[dict[str, Any]],
    selections: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "item_id",
        "split",
        "model_key",
        "resolver",
        "product",
        "overlay_path",
        "target_match",
        "coverage",
        "leakage",
        "boundary",
        "usable",
        "reviewer",
        "notes",
    ]
    by_id = {item["item_id"]: item for item in items}
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in selections:
            item = by_id.get(row["item_id"], {})
            writer.writerow(
                {
                    "item_id": row["item_id"],
                    "split": item.get("split") or row.get("split") or "",
                    "model_key": row.get("model_key") or "",
                    "resolver": row.get("resolver") or "",
                    "product": item.get("product") or "",
                    "overlay_path": row.get("overlay_path") or "",
                    "target_match": "",
                    "coverage": "",
                    "leakage": "",
                    "boundary": "",
                    "usable": "",
                    "reviewer": "",
                    "notes": "",
                }
            )


def write_annotation_html(path: Path, items: list[dict[str, Any]]) -> None:
    cards = []
    for item in items:
        thumb = f"thumbs/{item['item_id']}.jpg"
        cards.append(
            f"""
            <article class="card" data-id="{html.escape(item['item_id'])}">
              <img src="{html.escape(thumb)}" alt="" />
              <div>
                <strong>{html.escape(str(item.get('product') or ''))}</strong>
                <p>{html.escape(item['split'])} · {html.escape(str(item.get('subtype') or ''))}</p>
                <label>target present <select name="target_present">
                  <option value=""></option><option value="true">true</option><option value="false">false</option>
                </select></label>
                <label>scene <input name="scene_type" placeholder="product_only|model|lookbook" /></label>
                <label>people <input name="n_people" type="number" min="0" /></label>
                <label>companion garments <input name="n_companion_garments" type="number" min="0" /></label>
                <label>occlusion <input name="occlusion" placeholder="none|partial|heavy" /></label>
                <label>pose <input name="pose" placeholder="frontal|side|back" /></label>
                <label>view <input name="view" placeholder="closeup|half|full" /></label>
                <p>click image to set target point: <span class="point">unset</span></p>
              </div>
            </article>
            """
        )
    path.write_text(
        f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8" />
<title>Garment target annotation</title>
<style>
body {{ font-family: sans-serif; margin: 24px; }}
.card {{ display:grid; grid-template-columns: 256px 1fr; gap: 16px; margin-bottom: 18px; border-bottom:1px solid #ddd; padding-bottom:12px; }}
img {{ width:256px; cursor:crosshair; background:#f3f3f3; }}
label {{ display:block; margin: 4px 0; }}
</style></head><body>
<h1>Target garment annotation</h1>
<p>Click the sold garment. Coordinates are stored as normalized values in [0, 1]. Pixel masks are not collected.</p>
<button id="export">Export JSON</button>
<section>{''.join(cards)}</section>
<script>
const cards = [...document.querySelectorAll('.card')];
cards.forEach(card => {{
  const img = card.querySelector('img');
  img.addEventListener('click', (event) => {{
    const rect = img.getBoundingClientRect();
    const x = (event.clientX - rect.left) / rect.width;
    const y = (event.clientY - rect.top) / rect.height;
    card.dataset.x = x.toFixed(4);
    card.dataset.y = y.toFixed(4);
    card.querySelector('.point').textContent = x.toFixed(3) + ', ' + y.toFixed(3);
  }});
}});
document.getElementById('export').onclick = () => {{
  const rows = cards.map(card => {{
    const val = name => card.querySelector('[name="'+name+'"]').value;
    return {{
      item_id: card.dataset.id,
      target_present: val('target_present'),
      target_point_x: card.dataset.x || '',
      target_point_y: card.dataset.y || '',
      scene_type: val('scene_type'),
      n_people: val('n_people'),
      n_companion_garments: val('n_companion_garments'),
      occlusion: val('occlusion'),
      pose: val('pose'),
      view: val('view'),
    }};
  }});
  const blob = new Blob([JSON.stringify(rows, null, 2)], {{type:'application/json'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'annotations.json';
  a.click();
}};
</script></body></html>
""",
        encoding="utf-8",
    )


def save_selected_overlay(item: dict[str, Any], proposal: dict[str, Any] | None, path: Path) -> str | None:
    if proposal is None:
        return None
    with Image.open(item["image_path"]) as image:
        rgb = image.convert("RGB")
        mask = restore_mask(proposal, rgb.size[1], rgb.size[0])
        save_overlay(rgb, mask, path)
        _draw_box(path, proposal["box_xyxy"], rgb.size)
    return str(path)


def _draw_box(path: Path, box: list[float], size: tuple[int, int]) -> None:
    with Image.open(path) as image:
        draw = ImageDraw.Draw(image)
        x0, y0, x1, y1 = box
        draw.rectangle([x0, y0, x1, y1], outline=(255, 80, 0), width=2)
        image.save(path, quality=88)


def build_contact_sheet(
    items: list[dict[str, Any]],
    overlays: dict[str, Path],
    path: Path,
    *,
    columns: int = 4,
    limit: int = 16,
) -> None:
    chosen = items[:limit]
    if not chosen:
        return
    thumbs: list[Image.Image] = []
    for item in chosen:
        overlay = overlays.get(item["item_id"])
        source = overlay if overlay and overlay.exists() else Path(item["image_path"])
        with Image.open(source) as image:
            thumb = image.convert("RGB")
            thumb.thumbnail((192, 192))
            canvas = Image.new("RGB", (192, 192), (245, 245, 245))
            canvas.paste(thumb, ((192 - thumb.size[0]) // 2, (192 - thumb.size[1]) // 2))
            thumbs.append(canvas)
    rows = (len(thumbs) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * 192, rows * 192), (255, 255, 255))
    for index, thumb in enumerate(thumbs):
        r, c = divmod(index, columns)
        sheet.paste(thumb, (c * 192, r * 192))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def collect_detection_failures(run_dir: Path, items: list[dict[str, Any]]) -> dict[str, Any]:
    proposals_root = run_dir / "proposals"
    models: list[dict[str, Any]] = []
    if not proposals_root.exists():
        return {"models": models}
    for model_dir in sorted(path for path in proposals_root.iterdir() if path.is_dir()):
        cache_path = model_dir / "proposals.jsonl"
        rows = read_jsonl(cache_path) if cache_path.exists() else []
        by_prop = {str(row.get("item_id")): row for row in rows if row.get("item_id")}
        failures: list[dict[str, Any]] = []
        for item in items:
            item_id = str(item["item_id"])
            row = by_prop.get(item_id)
            if row is None:
                reason = "missing_proposal_row"
                n_proposals = 0
                error = None
            elif row.get("failed"):
                reason = "infer_error"
                n_proposals = len(row.get("proposals") or [])
                error = row.get("error")
            elif not (row.get("proposals") or []):
                reason = "empty_proposals"
                n_proposals = 0
                error = row.get("error")
            else:
                continue
            failures.append(
                {
                    "item_id": item_id,
                    "split": item.get("split"),
                    "subtype": item.get("subtype"),
                    "product": item.get("product"),
                    "brand": item.get("brand"),
                    "reason": reason,
                    "n_proposals": n_proposals,
                    "error": error,
                    "thumb_rel": f"../dataset/thumbs/{item_id}.jpg",
                }
            )
        failures.sort(
            key=lambda row: (
                str(row.get("subtype") or ""),
                str(row.get("split") or ""),
                str(row.get("product") or ""),
                str(row.get("item_id") or ""),
            )
        )
        models.append(
            {
                "model_key": model_dir.name,
                "n_items": len(items),
                "n_failed": len(failures),
                "by_reason": dict(Counter(row["reason"] for row in failures)),
                "by_subtype": dict(Counter(str(row.get("subtype") or "unknown") for row in failures)),
                "by_split": dict(Counter(str(row.get("split") or "unknown") for row in failures)),
                "failures": failures,
            }
        )
    return {"models": models}


def _count_cells(counts: dict[str, int]) -> str:
    if not counts:
        return "<span class='muted'>없음</span>"
    return ", ".join(f"{html.escape(name)} {count}" for name, count in sorted(counts.items(), key=lambda row: (-row[1], row[0])))


def _failure_gallery_html(detection: dict[str, Any], *, thumb_prefix: str = "") -> str:
    blocks = []
    for model in detection.get("models") or []:
        failures = model.get("failures") or []
        summary_rows = []
        keys = sorted(
            {(str(row.get("split") or ""), str(row.get("subtype") or "")) for row in failures}
        )
        for split, subtype in keys:
            n = sum(1 for row in failures if str(row.get("split") or "") == split and str(row.get("subtype") or "") == subtype)
            summary_rows.append(
                f"<tr><td>{html.escape(split)}</td><td>{html.escape(subtype)}</td><td>{n}</td></tr>"
            )
        table_rows = []
        cards = []
        for row in failures:
            thumb = html.escape(thumb_prefix + str(row.get("thumb_rel") or ""))
            reason = FAILURE_REASONS.get(str(row.get("reason")), str(row.get("reason")))
            table_rows.append(
                "<tr>"
                f"<td><code>{html.escape(str(row.get('item_id')))}</code></td>"
                f"<td>{html.escape(str(row.get('split') or ''))}</td>"
                f"<td>{html.escape(str(row.get('subtype') or ''))}</td>"
                f"<td>{html.escape(reason)}</td>"
                f"<td>{html.escape(str(row.get('product') or ''))}</td>"
                "</tr>"
            )
            cards.append(
                "<article class='fail-card'>"
                f"<img src='{thumb}' alt=''>"
                f"<p><b>{html.escape(str(row.get('product') or row.get('item_id')))}</b></p>"
                f"<p class='muted'>{html.escape(str(row.get('split') or ''))} · "
                f"{html.escape(str(row.get('subtype') or ''))} · {html.escape(reason)}</p>"
                f"<p class='muted'><code>{html.escape(str(row.get('item_id')))}</code></p>"
                "</article>"
            )
        empty = "<p class='muted'>탐지 실패가 없습니다.</p>" if not cards else ""
        summary_table = (
            "<table><tr><th>split</th><th>subtype</th><th>n</th></tr>"
            + "".join(summary_rows)
            + "</table>"
            if summary_rows
            else ""
        )
        detail_table = (
            "<table><tr><th>item_id</th><th>split</th><th>subtype</th><th>이유</th><th>상품명</th></tr>"
            + "".join(table_rows)
            + "</table>"
            if table_rows
            else ""
        )
        blocks.append(
            f"<h3><code>{html.escape(str(model.get('model_key')))}</code> "
            f"{model.get('n_failed')}/{model.get('n_items')} 실패</h3>"
            f"<p>이유: {_count_cells(model.get('by_reason') or {})} · "
            f"split: {_count_cells(model.get('by_split') or {})} · "
            f"subtype: {_count_cells(model.get('by_subtype') or {})}</p>"
            f"{summary_table}{detail_table}{empty}"
            f"<div class='fail-grid'>{''.join(cards)}</div>"
        )
    if not blocks:
        return "<p class='muted'>proposal cache가 없어 실패 목록을 만들 수 없습니다.</p>"
    return "".join(blocks)


def _write_failures_page(path: Path, *, run_id: str, detection: dict[str, Any]) -> None:
    body = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8" /><title>SAM3 탐지 실패</title>
<style>
body {{ font-family: sans-serif; margin: 24px; max-width: 1400px; }}
table {{ border-collapse: collapse; margin: 12px 0 24px; }}
td, th {{ border: 1px solid #ccc; padding: 6px 10px; text-align: left; }}
.muted {{ color: #555; }}
.fail-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 12px; }}
.fail-card {{ border: 1px solid #ddd; padding: 8px; background: #fafafa; }}
.fail-card img {{ width: 100%; aspect-ratio: 1 / 1; object-fit: contain; background: #111; }}
.fail-card p {{ margin: 6px 0 0; font-size: 13px; }}
</style></head><body>
<h1>탐지 실패만 보기</h1>
<p class="muted">run {html.escape(run_id)} · SAM3 프롬프트 <code>shirt</code>/<code>hoodie</code>/<code>sweater</code>/<code>blouse</code></p>
<p><a href="index.html">벤치마크 리포트로</a></p>
{_failure_gallery_html(detection)}
</body></html>
"""
    path.write_text(body, encoding="utf-8")


def build_report(
    run_dir: Path,
    *,
    items: list[dict[str, Any]],
    inventory: dict[str, Any],
    licenses: dict[str, Any],
    availability: dict[str, Any],
    metrics: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    report_dir = run_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    detection = collect_detection_failures(run_dir, items)
    write_json(report_dir / "failures.json", detection)
    write_jsonl(
        report_dir / "failures.jsonl",
        [
            {"model_key": model["model_key"], **row}
            for model in detection.get("models") or []
            for row in model.get("failures") or []
        ],
    )
    _write_failures_page(
        report_dir / "failures.html",
        run_id=str(manifest.get("run_id") or run_dir.name),
        detection=detection,
    )
    summary_lines = [
        "# 의류 세그멘테이션 벤치마크",
        "",
        f"- 실행 ID: `{manifest.get('run_id')}`",
        f"- fingerprint: `{inventory.get('dataset_fingerprint')}`",
        f"- 표본: {inventory.get('selected')} (dev/primary/stress = {inventory.get('selected_by_split')})",
        "",
        "## 라이선스 게이트",
        "",
    ]
    for key, record in licenses.items():
        summary_lines.append(
            f"- `{key}`: {record.get('eligibility')} · {record.get('display_name')}"
        )
    summary_lines.extend(["", "## 모델 가용성", ""])
    for key, payload in availability.items():
        summary_lines.append(f"- `{key}`: {payload.get('status')} {payload.get('error') or ''}".rstrip())
    summary_lines.extend(["", "## 자동 지표", "", "픽셀 정답이 없으므로 mIoU/AP/Boundary F1은 보고하지 않습니다.", ""])
    summary_lines.append("```json")
    summary_lines.append(json.dumps(metrics, ensure_ascii=False, indent=2, default=str))
    summary_lines.append("```")
    summary_lines.extend(["", "## 탐지 실패", ""])
    for model in detection.get("models") or []:
        summary_lines.append(
            f"- `{model.get('model_key')}`: {model.get('n_failed')}/{model.get('n_items')} "
            f"(이유 {model.get('by_reason')}, subtype {model.get('by_subtype')})"
        )
        for row in model.get("failures") or []:
            summary_lines.append(
                f"  - `{row.get('item_id')}` {row.get('split')} {row.get('subtype')} "
                f"{FAILURE_REASONS.get(str(row.get('reason')), row.get('reason'))} — {row.get('product')}"
            )
    if not detection.get("models"):
        summary_lines.append("- proposal cache가 없습니다.")
    summary_lines.extend(
        [
            "",
            "## 육안 평가 rubric",
            "",
        ]
    )
    for key, text in VISUAL_RUBRIC.items():
        summary_lines.append(f"- **{key}**: {text}")
    summary_lines.extend(
        [
            "",
            "## 판정 규칙",
            "",
            "- 타깃 선택 Top-1과 육안 usable rate를 우선한다.",
            "- 라이선스가 research-only이면 채택 후보에서 제외하고 연구 기준선으로만 둔다.",
            "- 실패율·지연시간을 별도 gate로 적용한다.",
            "- 상위 후보 차이가 5%p 이내이거나 경계 품질이 불안정하면 50~100장 pixel-gold 2차 평가를 권고한다.",
        ]
    )
    (report_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    tables = []
    for model_key, resolvers in metrics.get("models", {}).items():
        for resolver, splits in resolvers.items():
            tables.append(
                "<tr>"
                f"<td>{html.escape(model_key)}</td><td>{html.escape(resolver)}</td>"
                f"<td>{_split_rate(splits, 'dev', 'top1_success')}</td>"
                f"<td>{_split_rate(splits, 'primary', 'top1_success')}</td>"
                f"<td>{_split_rate(splits, 'primary', 'visual_usable')}</td>"
                f"<td>{_split_rate(splits, 'stress', 'top1_success')}</td>"
                f"<td>{_split_latency(splits, 'primary') or _split_latency(splits, 'dev')}</td></tr>"
            )
    sheets = []
    figures = run_dir / "figures" / "contact_sheets"
    if figures.exists():
        for path in sorted(figures.glob("*.jpg")):
            rel = f"../figures/contact_sheets/{path.name}"
            sheets.append(
                f'<p>{html.escape(path.stem)}<br /><img src="{html.escape(rel)}" alt="" width="768" /></p>'
            )
    html_doc = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8" /><title>Garment segmentation benchmark</title>
<style>
body {{ font-family: sans-serif; margin: 24px; max-width: 1400px; }}
table {{ border-collapse: collapse; }}
td, th {{ border: 1px solid #ccc; padding: 6px 10px; text-align: left; }}
.muted {{ color: #555; }}
.fail-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 12px; }}
.fail-card {{ border: 1px solid #ddd; padding: 8px; background: #fafafa; }}
.fail-card img {{ width: 100%; aspect-ratio: 1 / 1; object-fit: contain; background: #111; }}
.fail-card p {{ margin: 6px 0 0; font-size: 13px; }}
</style></head><body>
<h1>의류 세그멘테이션 벤치마크</h1>
<p class="muted">run {html.escape(str(manifest.get('run_id')))} · {inventory.get('selected')} tops</p>
<p>픽셀 정답 없이 타깃 선택과 블라인드 육안 품질만 비교합니다. mIoU/AP는 보고하지 않습니다.</p>
<p><a href="failures.html">탐지 실패만 보기</a></p>
<h2>모델 가용성</h2>
<ul>
{''.join(f"<li><code>{html.escape(k)}</code> {html.escape(str(v.get('status')))} {html.escape(str(v.get('error') or ''))}</li>" for k,v in availability.items())}
</ul>
<h2>비교표</h2>
<table>
<tr><th>model</th><th>resolver</th><th>dev top-1</th><th>primary top-1</th><th>primary usable</th><th>stress top-1</th><th>p50 sec</th></tr>
{''.join(tables) or '<tr><td colspan="7">아직 추론 결과가 없습니다.</td></tr>'}
</table>
<h2>탐지 실패</h2>
<p class="muted">proposal이 없거나 비어 타깃 박스를 고를 수 없는 항목입니다. SAM3는 <code>shirt</code>/<code>hoodie</code>/<code>sweater</code>/<code>blouse</code> 프롬프트만 씁니다.</p>
{_failure_gallery_html(detection)}
<h2>라이선스</h2>
<ul>
{''.join(f"<li><code>{html.escape(k)}</code> {html.escape(str(v.get('eligibility')))} — {html.escape(str(v.get('notes') or ''))}</li>" for k,v in licenses.items())}
</ul>
<h2>Contact sheets</h2>
{''.join(sheets) or '<p class="muted">아직 overlay contact sheet가 없습니다.</p>'}
<p><a href="summary.md">markdown summary</a> · <a href="failures.html">탐지 실패 페이지</a> · <a href="failures.json">failures.json</a></p>
</body></html>
"""
    (report_dir / "index.html").write_text(html_doc, encoding="utf-8")
    write_json(report_dir / "metrics.json", metrics)


def _split_rate(splits: dict[str, Any], split: str, field: str) -> Any:
    payload = ((splits.get(split) or {}).get("uniform") or {}).get(field) or {}
    return payload.get("rate") if isinstance(payload, dict) else None


def _split_latency(splits: dict[str, Any], split: str) -> Any:
    payload = ((splits.get(split) or {}).get("uniform") or {}).get("latency_sec") or {}
    return payload.get("p50")
