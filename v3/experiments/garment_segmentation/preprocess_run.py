from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from io_utils import read_json, read_jsonl, write_json, write_jsonl
from preprocess import ALL_ARMS, DEFAULT_GARMENT_LABELS, build_arm_images, pad_box, resolve_arms, resolve_mask_in_box
from preprocess_metrics import (
    cluster_label_nmi,
    compare_arms,
    density_alignment_metrics,
    projection_trustworthiness,
    summarize_mask_stats,
)
from preprocess_report import (
    fit_umap,
    procrustes_align,
    scatter_figure,
    subtype_grid_figure,
    subtype_overlay_figure,
    write_preview_gallery,
    write_report,
)
from run import extra_artifact_roots, resolve_run_dir


def _load_attr_repr():
    attr_dir = HERE.parent / "attribute_embedding"
    if str(attr_dir) not in sys.path:
        sys.path.insert(0, str(attr_dir))
    path = attr_dir / "representations.py"
    spec = importlib.util.spec_from_file_location("attr_emb_repr", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return None


def _index_jsonl(path: Path, key: str = "item_id") -> dict[str, dict[str, Any]]:
    return {str(row[key]): row for row in read_jsonl(path) if row.get(key)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare FashionSigLIP embeddings under preprocess A/B/C/D")
    parser.add_argument("--config", type=Path, default=HERE / "config.preprocess.yaml")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--box-run", default=None)
    parser.add_argument("--mask-run", default=None)
    parser.add_argument("--splits", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--previews-only", action="store_true")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Rebuild HTML from an existing run's embeddings/projections without re-encoding",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Proceed when box/mask caches cover fewer than the frozen 450 items",
    )
    parser.add_argument(
        "--arms",
        nargs="*",
        default=None,
        help="Subset of A B C D. Default comes from config arms, else all four.",
    )
    return parser.parse_args()


def _save_preview(image: Image.Image, path: Path, size: int = 384) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = image.convert("RGB")
    if max(rgb.size) > size:
        rgb = ImageOps.contain(rgb, (size, size))
    rgb.save(path, format="JPEG", quality=85, optimize=True)


def _embed_images(images: list[Image.Image], *, model, preprocess, device: str, batch_size: int) -> np.ndarray:
    import torch

    vectors: list[np.ndarray] = []
    total = len(images)
    for start in range(0, total, batch_size):
        batch = images[start : start + batch_size]
        tensors = torch.stack([preprocess(image.convert("RGB")) for image in batch]).to(device)
        with torch.inference_mode():
            encoded = model.encode_image(tensors, normalize=True)
        vectors.append(encoded.detach().float().cpu().numpy())
        print(f"image embedding {min(start + batch_size, total)}/{total}", flush=True)
    from preprocess_metrics import l2_normalize

    return l2_normalize(np.concatenate(vectors, axis=0).astype(np.float32))


def _cap_splits(items: list[dict[str, Any]], splits: set[str] | None, limit: int | None) -> list[dict[str, Any]]:
    selected = [item for item in items if splits is None or item.get("split") in splits]
    if not limit:
        return selected
    counts: dict[str, int] = {}
    out: list[dict[str, Any]] = []
    for item in selected:
        split = str(item.get("split") or "unknown")
        if counts.get(split, 0) >= limit:
            continue
        out.append(item)
        counts[split] = counts.get(split, 0) + 1
    return out


def _preview_rel(records: list[dict[str, Any]], arms: tuple[str, ...]) -> dict[str, list[str]]:
    return {arm: [row["preview_rel"][arm] for row in records] for arm in arms}


def _detect_arms(run_dir: Path, coverage: dict[str, Any] | None = None) -> tuple[str, ...]:
    coverage = coverage or {}
    if coverage.get("arms"):
        return resolve_arms(coverage["arms"])
    found = [name for name in ALL_ARMS if (run_dir / "embeddings" / f"{name}.npy").exists()]
    if found:
        return resolve_arms(found)
    found = [name for name in ALL_ARMS if (run_dir / "projections" / f"{name}.npy").exists()]
    if found:
        return resolve_arms(found)
    records = read_jsonl(run_dir / "dataset" / "records.jsonl")
    rel = (records[0].get("preview_rel") or {}) if records else {}
    found = [name for name in ALL_ARMS if name in rel]
    return resolve_arms(found) if found else ALL_ARMS


def write_figures(
    *,
    run_dir: Path,
    run_id: str,
    records: list[dict[str, Any]],
    coords: dict[str, np.ndarray],
    aligned: dict[str, np.ndarray],
    metrics: dict[str, Any],
    mask_stats: dict[str, Any],
    coverage: dict[str, Any],
    projection_quality: dict[str, Any],
    arms: tuple[str, ...] | None = None,
) -> Path:
    arms = resolve_arms(arms or coverage.get("arms") or tuple(coords.keys()))
    preview_rel = _preview_rel(records, arms)
    independent = scatter_figure(coords, records, preview_rel, title="독립 UMAP (축 비교 금지)", arms=arms)
    aligned_fig = scatter_figure(aligned, records, preview_rel, title="A에 Procrustes 정렬한 UMAP", arms=arms)
    subtype_grid = subtype_grid_figure(aligned, records, preview_rel, arms=arms)
    subtype_overlay = subtype_overlay_figure(aligned, records, preview_rel, arms=arms)
    return write_report(
        run_dir / "report" / "index.html",
        run_id=run_id,
        items=records,
        metrics=metrics,
        mask_stats=mask_stats,
        coverage=coverage,
        independent_fig=independent,
        aligned_fig=aligned_fig,
        subtype_grid_fig=subtype_grid,
        subtype_overlay_fig=subtype_overlay,
        projection_quality=projection_quality,
        arms=arms,
    )


def rebuild_report(run_dir: Path) -> Path:
    records = read_jsonl(run_dir / "dataset" / "records.jsonl")
    if not records:
        raise RuntimeError(f"missing records in {run_dir}")
    coverage = read_json(run_dir / "coverage.json")
    arms = _detect_arms(run_dir, coverage)
    coords = {arm: np.load(run_dir / "projections" / f"{arm}.npy") for arm in arms}
    aligned_path = run_dir / "projections" / "aligned_A.npy"
    if aligned_path.exists():
        aligned = {arm: np.load(run_dir / "projections" / f"aligned_{arm}.npy") for arm in arms}
        disparities = {}
    else:
        aligned = {"A": coords["A"]}
        disparities = {}
        for arm in arms:
            if arm == "A":
                continue
            aligned[arm], disparity = procrustes_align(coords["A"], coords[arm])
            disparities[arm] = round(disparity, 4)
            np.save(run_dir / "projections" / f"aligned_{arm}.npy", aligned[arm])
        np.save(run_dir / "projections" / "aligned_A.npy", aligned["A"])
    metrics = read_json(run_dir / "metrics" / "summary.json")
    mask_stats = read_json(run_dir / "metrics" / "mask_stats.json")
    quality: dict[str, Any] = {}
    embed_dir = run_dir / "embeddings"
    embeddings: dict[str, np.ndarray] = {}
    if (embed_dir / "A.npy").exists():
        for arm in arms:
            embeddings[arm] = np.load(embed_dir / f"{arm}.npy")
            quality[arm] = {
                "trustworthiness": projection_trustworthiness(embeddings[arm], coords[arm])
            }
        metrics["density"] = density_alignment_metrics(embeddings, records, aligned, k=10)
        write_json(run_dir / "metrics" / "summary.json", metrics)
    if disparities:
        quality["procrustes_disparity_vs_A"] = disparities
    elif (run_dir / "metrics" / "projection_quality.json").exists():
        quality.update(read_json(run_dir / "metrics" / "projection_quality.json"))
    return write_figures(
        run_dir=run_dir,
        run_id=run_dir.name,
        records=records,
        coords=coords,
        aligned=aligned,
        metrics=metrics,
        mask_stats=mask_stats,
        coverage=coverage,
        projection_quality=quality,
        arms=arms,
    )


def build_records(
    items: list[dict[str, Any]],
    *,
    box_rows: dict[str, dict[str, Any]],
    mask_selected: dict[str, dict[str, Any]],
    mask_proposals: dict[str, list[dict[str, Any]]],
    preprocess_cfg: dict[str, Any],
    run_dir: Path,
    arms: tuple[str, ...],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    pad_ratio = float(preprocess_cfg.get("pad_ratio") or 0.12)
    garment_labels = tuple(preprocess_cfg.get("garment_labels") or DEFAULT_GARMENT_LABELS)
    min_mask_fill = float(preprocess_cfg.get("min_mask_fill") or 0.08)
    for item in items:
        item_id = str(item["item_id"])
        box_row = box_rows.get(item_id) or {}
        selected_box = box_row.get("selected")
        if not selected_box or not selected_box.get("box_xyxy"):
            continue
        image_path = Path(item["image_path"])
        if not image_path.exists():
            continue
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            width, height = rgb.size
            padded = pad_box(selected_box["box_xyxy"], width, height, pad_ratio=pad_ratio)
            resolved = resolve_mask_in_box(
                height=height,
                width=width,
                box_xyxy=padded,
                proposals=mask_proposals.get(item_id) or [],
                combined_selected=(mask_selected.get(item_id) or {}).get("selected"),
                garment_labels=garment_labels,
                min_mask_fill=min_mask_fill,
            )
            arms_images = build_arm_images(
                rgb,
                box_xyxy=selected_box["box_xyxy"],
                mask=resolved["mask"],
                pad_ratio=pad_ratio,
                dilate_px=int(preprocess_cfg.get("dilate_px") or 7),
                alpha_sigma=float(preprocess_cfg.get("alpha_sigma") or 5.0),
                background_blur_sigma=float(preprocess_cfg.get("background_blur_sigma") or 21.0),
                arms=arms,
            )
            rel: dict[str, str] = {}
            for arm, rendered in arms_images.items():
                preview_path = run_dir / "previews" / arm / f"{item_id}.jpg"
                _save_preview(rendered, preview_path)
                rel[arm] = f"../previews/{arm}/{item_id}.jpg"
            record = {
                **item,
                "box_xyxy": selected_box["box_xyxy"],
                "padded_box_xyxy": padded,
                "mask_source": resolved["source"],
                "fill_ratio": resolved["fill_ratio"],
                "mask_labels": resolved["labels"],
                "combined_label": resolved["combined_label"],
                "combined_is_garment": resolved["combined_is_garment"],
                "source_model_box": box_row.get("model_key"),
                "preview_rel": rel,
                "_images": arms_images,
            }
            records.append(record)
            print(f"preprocessed {len(records)}/{len(items)} {item_id} {resolved['source']}", flush=True)
    return records


def main() -> int:
    args = parse_args()
    config_path = args.config
    if not config_path.exists():
        example = HERE / "config.preprocess.example.yaml"
        raise FileNotFoundError(f"missing {config_path}. Copy {example} and fill box_run / mask_run.")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_root = (REPO_ROOT / config.get("output_root", "artifacts/garment_segmentation")).resolve()
    if args.report_only:
        run_id = args.run_id
        if not run_id:
            raise ValueError("--report-only requires --run-id")
        report_path = rebuild_report(output_root / run_id)
        print(report_path)
        return 0
    extra_roots = extra_artifact_roots(output_root)
    dataset_name = args.dataset or config["dataset_run"]
    box_name = args.box_run or config["box_run"]
    mask_name = args.mask_run or config["mask_run"]
    arms = resolve_arms(args.arms if args.arms else config.get("arms"))
    dataset_dir = resolve_run_dir(dataset_name, output_root, extra_roots)
    box_dir = resolve_run_dir(box_name, output_root, extra_roots)
    mask_dir = resolve_run_dir(mask_name, output_root, extra_roots)
    default_run_id = "preprocess-ac-sam3-%Y%m%d-%H%M%S" if arms == ("A", "C") else "preprocess-abcd-%Y%m%d-%H%M%S"
    run_id = args.run_id or datetime.now(UTC).strftime(default_run_id)
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    box_model = str(config.get("box_model") or "grounded_sam21")
    box_resolver = str(config.get("box_resolver") or "torso_prior")
    mask_model = str(config.get("mask_model") or "segformer")
    mask_resolver = str(config.get("mask_resolver") or "combined")
    box_path = box_dir / "selected" / box_model / box_resolver / "assignments.jsonl"
    mask_sel_path = mask_dir / "selected" / mask_model / mask_resolver / "assignments.jsonl"
    mask_prop_path = mask_dir / "proposals" / mask_model / "proposals.jsonl"
    for path in (box_path, mask_sel_path, mask_prop_path):
        if not path.exists():
            raise FileNotFoundError(path)

    items, inventory = read_jsonl(dataset_dir / "dataset" / "items.jsonl"), read_json(
        dataset_dir / "dataset" / "inventory.json"
    )
    splits = set(args.splits) if args.splits else None
    items = _cap_splits(items, splits, args.limit)
    box_rows = _index_jsonl(box_path)
    mask_selected = _index_jsonl(mask_sel_path)
    mask_proposals = {
        str(row["item_id"]): row.get("proposals") or []
        for row in read_jsonl(mask_prop_path)
        if row.get("item_id")
    }
    eligible: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for item in items:
        item_id = item["item_id"]
        selected = (box_rows.get(item_id) or {}).get("selected") or {}
        has_rows = item_id in box_rows and item_id in mask_proposals
        has_box = bool(selected.get("box_xyxy"))
        if has_rows and has_box:
            eligible.append(item)
            continue
        skipped.append(
            {
                "item_id": item_id,
                "split": item.get("split"),
                "subtype": item.get("subtype"),
                "product": item.get("product"),
                "n_proposals": len(mask_proposals.get(item_id) or []),
                "reason": "empty_proposals" if has_rows else "incomplete_cache",
            }
        )
    coverage = {
        "n_dataset": len(items),
        "n_box": sum(1 for item in items if item["item_id"] in box_rows),
        "n_mask": sum(1 for item in items if item["item_id"] in mask_proposals),
        "n_box_selected": len(eligible),
        "n_used": len(eligible),
        "n_skipped": len(skipped),
        "skipped": skipped,
        "dataset_run": str(dataset_dir),
        "box_run": str(box_dir),
        "mask_run": str(mask_dir),
        "box_spec": f"{box_model}/{box_resolver}",
        "mask_spec": f"{mask_model}/{mask_resolver}",
        "arms": list(arms),
        "dataset_fingerprint": inventory.get("dataset_fingerprint"),
    }
    if coverage["n_used"] == 0:
        raise RuntimeError("no items have both box assignments and mask proposals")
    cache_incomplete = any(row["reason"] == "incomplete_cache" for row in skipped)
    if cache_incomplete and not args.allow_partial and args.limit is None:
        raise RuntimeError(
            f"only {coverage['n_used']}/{coverage['n_dataset']} items have box+mask caches. "
            "Run the box/mask jobs first, or pass --allow-partial / --limit."
        )
    if skipped:
        print(
            f"using {coverage['n_used']}/{coverage['n_dataset']} items; "
            f"skipped {len(skipped)} with no SAM3 box",
            flush=True,
        )

    preprocess_cfg = dict(config.get("preprocess") or {})
    records = build_records(
        eligible,
        box_rows=box_rows,
        mask_selected=mask_selected,
        mask_proposals=mask_proposals,
        preprocess_cfg=preprocess_cfg,
        run_dir=run_dir,
        arms=arms,
    )
    mask_stats = summarize_mask_stats(records)
    preview_rel = _preview_rel(records, arms)
    write_json(run_dir / "coverage.json", coverage)
    write_json(run_dir / "metrics" / "mask_stats.json", mask_stats)
    write_jsonl(
        run_dir / "dataset" / "records.jsonl",
        [{k: v for k, v in row.items() if k != "_images"} for row in records],
    )
    write_preview_gallery(
        run_dir / "report" / "previews.html",
        run_id=run_id,
        items=records,
        preview_rel=preview_rel,
        mask_stats=mask_stats,
        coverage=coverage,
        arms=arms,
    )
    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "coverage": coverage,
        "arms": list(arms),
        "status": "previews",
    }
    write_json(run_dir / "manifest.json", manifest)
    if args.previews_only:
        print(run_dir / "report" / "previews.html")
        return 0

    repr_mod = _load_attr_repr()
    device = args.device or config.get("device") or "auto"
    model, preprocess, _tokenizer, device = repr_mod.load_model(str(config.get("encoder_id")), device)
    embed_dir = run_dir / "embeddings"
    proj_dir = run_dir / "projections"
    embed_dir.mkdir(parents=True, exist_ok=True)
    proj_dir.mkdir(parents=True, exist_ok=True)
    embeddings: dict[str, np.ndarray] = {}
    for arm in arms:
        print(f"embedding arm {arm}", flush=True)
        embeddings[arm] = _embed_images(
            [row["_images"][arm] for row in records],
            model=model,
            preprocess=preprocess,
            device=device,
            batch_size=int(config.get("batch_size") or 8),
        )
        np.save(embed_dir / f"{arm}.npy", embeddings[arm])

    eval_cfg = dict(config.get("evaluation") or {})
    metrics = compare_arms(
        embeddings,
        records,
        k=int(eval_cfg.get("neighbor_k") or 10),
        bootstrap_iterations=int(eval_cfg.get("bootstrap_iterations") or 300),
        seed=int(config.get("seed") or 42),
    )
    umap_cfg = dict(config.get("umap") or {})
    coords: dict[str, np.ndarray] = {}
    projection_quality: dict[str, Any] = {}
    for arm in arms:
        arm_coords, meta = fit_umap(
            embeddings[arm],
            seed=int(config.get("seed") or 42),
            n_neighbors=int(umap_cfg.get("n_neighbors") or 30),
            min_dist=float(umap_cfg.get("min_dist") or 0.12),
        )
        coords[arm] = arm_coords
        projection_quality[arm] = {
            **meta,
            "trustworthiness": projection_trustworthiness(embeddings[arm], arm_coords),
        }
        np.save(proj_dir / f"{arm}.npy", arm_coords)
    aligned = {"A": coords["A"]}
    disparities: dict[str, float] = {}
    for arm in arms:
        if arm == "A":
            continue
        aligned[arm], disparity = procrustes_align(coords["A"], coords[arm])
        disparities[arm] = round(disparity, 4)
        np.save(proj_dir / f"aligned_{arm}.npy", aligned[arm])
    np.save(proj_dir / "aligned_A.npy", aligned["A"])
    projection_quality["procrustes_disparity_vs_A"] = disparities
    color_first = [
        ((row.get("weak_labels") or {}).get("color") or [None])[0] for row in records
    ]
    subtype = [str(row.get("subtype") or "other") for row in records]
    try:
        import hdbscan

        if len(records) >= 30:
            for arm in arms:
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=min(int(eval_cfg.get("cluster_min_size") or 18), max(5, len(records) // 4)),
                    min_samples=5,
                    metric="euclidean",
                )
                labels = clusterer.fit_predict(coords[arm])
                metrics["arms"][arm]["cluster_nmi_color"] = cluster_label_nmi(labels, color_first)
                metrics["arms"][arm]["cluster_nmi_subtype"] = cluster_label_nmi(labels, subtype)
    except Exception as exc:
        projection_quality["cluster_error"] = f"{type(exc).__name__}: {exc}"

    metrics["density"] = density_alignment_metrics(
        embeddings,
        records,
        aligned,
        k=int(eval_cfg.get("neighbor_k") or 10),
    )

    write_json(run_dir / "metrics" / "summary.json", metrics)
    write_json(run_dir / "metrics" / "projection_quality.json", projection_quality)
    report_path = write_figures(
        run_dir=run_dir,
        run_id=run_id,
        records=records,
        coords=coords,
        aligned=aligned,
        metrics=metrics,
        mask_stats=mask_stats,
        coverage=coverage,
        projection_quality=projection_quality,
        arms=arms,
    )
    manifest["status"] = "complete"
    manifest["completed_at"] = datetime.now(UTC).isoformat()
    write_json(run_dir / "manifest.json", manifest)
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
