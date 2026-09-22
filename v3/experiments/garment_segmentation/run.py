from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
LIGHT_ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "garment_segmentation"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from adapters import DEFAULT_MODELS, OPTIONAL_MODELS, create_adapter
from adapters.base import AdapterContext, UnavailableError
from evaluation import auto_metrics_for_split, merge_visual_scores, paired_success_delta, reviewer_kappa
from freeze import freeze_dataset, merge_annotations, publicize
from io_utils import load_attr_dataset, read_json, read_jsonl, restore_mask, write_json, write_jsonl
from licenses import catalog_payload
from report import build_contact_sheet, build_report, save_selected_overlay, write_annotation_html, write_reviewer_sheet
from resolver import (
    RESOLVER_NAMES,
    MetadataEncoder,
    freeze_resolver_config,
    score_proposals,
    select_with_resolver,
    target_hit,
)
from evaluation import duplicate_rate, fragmentation


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return None


def resolve_run_dir(name: str, output_root: Path, extra_roots: list[Path] | None = None) -> Path:
    """Resolve a run id or filesystem path to an existing experiment directory."""

    raw = Path(name).expanduser()
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append((REPO_ROOT / raw).resolve())
        candidates.append((output_root / name).resolve())
        for root in extra_roots or []:
            candidates.append((Path(root) / name).resolve())
    seen: set[Path] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if (path / "dataset" / "items.jsonl").exists() or (path / "proposals").is_dir():
            return path
    raise FileNotFoundError(f"missing experiment run: {name}")


def extra_artifact_roots(output_root: Path) -> list[Path]:
    roots = [LIGHT_ARTIFACT_ROOT]
    if output_root.resolve() != LIGHT_ARTIFACT_ROOT.resolve():
        roots.append(output_root)
    return roots


def _public_config(config: dict[str, Any]) -> dict[str, Any]:
    payload = json.loads(json.dumps(config, default=str))
    sources = payload.get("sources")
    if isinstance(sources, dict):
        for meta in sources.values():
            if isinstance(meta, dict) and "disk_name" in meta:
                meta["disk_name"] = "<local>"
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Garment segmentation benchmark")
    parser.add_argument("--config", type=Path, default=HERE / "config.yaml")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument(
        "--reuse-dataset",
        default=None,
        help="Frozen run_id or path. Also searches artifacts/garment_segmentation/",
    )
    parser.add_argument(
        "--reuse-proposals",
        default=None,
        help="Proposal cache run_id or path. Also searches artifacts/garment_segmentation/",
    )
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--splits", nargs="*", default=None, help="Subset of dev/primary/stress")
    parser.add_argument("--limit", type=int, default=None, help="Optional per-split cap for smoke runs")
    parser.add_argument("--device", default=None, help="Override config device, e.g. cpu or mps")
    parser.add_argument(
        "--download-weights",
        action="store_true",
        help="Allow Hugging Face checkpoint download when the local cache is empty",
    )
    parser.add_argument("--skip-encoder", action="store_true")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Rebuild HTML from an existing run without inference",
    )
    parser.add_argument("--annotations", type=Path, default=None)
    parser.add_argument("--reviews-a", type=Path, default=None)
    parser.add_argument("--reviews-b", type=Path, default=None)
    return parser.parse_args()


def _peak_memory_mb() -> float | None:
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return round(usage / (1024 * 1024), 2)
        return round(usage / 1024, 2)
    except Exception:
        return None


def load_items(run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    items = read_jsonl(run_dir / "dataset" / "items.jsonl")
    inventory = read_json(run_dir / "dataset" / "inventory.json")
    return items, inventory


def maybe_merge_csv_annotations(items: list[dict[str, Any]], path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return items
    if path.suffix.lower() == ".json":
        rows = json.loads(path.read_text(encoding="utf-8"))
    else:
        with path.open("r", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    return merge_annotations(items, rows)


def load_reviews(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def infer_model(
    adapter,
    items: list[dict[str, Any]],
    *,
    context: AdapterContext,
    cache_dir: Path,
    reuse: bool,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any], list[dict[str, Any]]]:
    cache_path = cache_dir / "proposals.jsonl"
    meta_path = cache_dir / "metadata.json"
    if reuse and cache_path.exists():
        cached = read_jsonl(cache_path)
        by_item = {row["item_id"]: row.get("proposals") or [] for row in cached}
        timing = [row for row in cached]
        meta = read_json(meta_path) if meta_path.exists() else adapter.metadata()
        return by_item, meta, timing

    by_item: dict[str, list[dict[str, Any]]] = {}
    timing_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, item in enumerate(items, 1):
        t0 = time.perf_counter()
        failed = False
        error = None
        proposals: list[dict[str, Any]] = []
        try:
            with Image.open(item["image_path"]) as image:
                proposals = adapter.infer(image.convert("RGB"), context)
        except Exception as exc:
            failed = True
            error = f"{type(exc).__name__}: {exc}"
        elapsed = time.perf_counter() - t0
        by_item[item["item_id"]] = proposals
        timing_rows.append(
            {
                "item_id": item["item_id"],
                "split": item["split"],
                "proposals": proposals,
                "proposal_count": len(proposals),
                "latency_sec": round(elapsed, 4),
                "peak_memory_mb": _peak_memory_mb(),
                "failed": failed,
                "error": error,
            }
        )
        if index % 10 == 0 or index == len(items):
            print(f"{adapter.key} {index}/{len(items)}", flush=True)
    metadata = {
        **adapter.metadata(),
        "items": len(items),
        "wall_sec": round(time.perf_counter() - started, 3),
        "cold_start_sec": timing_rows[0]["latency_sec"] if timing_rows else None,
    }
    write_jsonl(cache_path, timing_rows)
    write_json(meta_path, metadata)
    return by_item, metadata, timing_rows


def rebuild_report(run_dir: Path) -> int:
    items, inventory = load_items(run_dir)
    manifest = read_json(run_dir / "manifest.json")
    availability = read_json(run_dir / "availability.json") if (run_dir / "availability.json").exists() else {}
    metrics_path = run_dir / "metrics" / "summary.json"
    metrics = read_json(metrics_path) if metrics_path.exists() else {}
    model_keys = list(manifest.get("models") or availability.keys())
    build_report(
        run_dir,
        items=items,
        inventory=inventory,
        licenses=catalog_payload(model_keys),
        availability=availability,
        metrics=metrics,
        manifest=manifest,
    )
    print(run_dir / "report" / "index.html")
    return 0


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.device:
        config["device"] = args.device
    if args.download_weights:
        config["download_weights"] = True
    output_root = Path(config["output_root"])
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    if args.report_only:
        if not args.run_id:
            raise ValueError("--report-only requires --run-id")
        run_dir = resolve_run_dir(args.run_id, output_root, extra_artifact_roots(output_root))
        return rebuild_report(run_dir)
    run_id = args.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / run_id
    reuse_dataset = args.reuse_dataset or config.get("reuse_dataset")
    if reuse_dataset:
        source = resolve_run_dir(str(reuse_dataset), output_root, extra_artifact_roots(output_root))
        if not (source / "dataset" / "items.jsonl").exists():
            raise FileNotFoundError(f"missing frozen dataset in {source}")
        run_dir.mkdir(parents=True, exist_ok=True)
        if not (run_dir / "dataset").exists():
            import shutil

            shutil.copytree(source / "dataset", run_dir / "dataset")
    else:
        run_dir.mkdir(parents=True, exist_ok=False)

    model_keys = args.models or list(config.get("models") or DEFAULT_MODELS)
    if config.get("include_optional"):
        for key in OPTIONAL_MODELS:
            if key not in model_keys:
                model_keys.append(key)
    if config.get("include_dummy") and "dummy" not in model_keys:
        model_keys.append("dummy")

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "experiment": str(config.get("experiment") or "garment_segmentation_benchmark"),
        "run_id": run_id,
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "config": _public_config(config),
        "models": model_keys,
    }
    write_json(run_dir / "manifest.json", manifest)

    dataset_mod = load_attr_dataset()
    sources = dataset_mod.normalize_sources(config.get("sources"))
    input_root = Path(config["input_root"]).expanduser().resolve()

    if reuse_dataset:
        items, inventory = load_items(run_dir)
    else:
        items, inventory = freeze_dataset(
            input_root,
            sources,
            run_dir,
            seed=int(config["seed"]),
            n_dev=int(config["n_dev"]),
            n_primary=int(config["n_primary"]),
            n_stress=int(config["n_stress"]),
            pool_size=int(config["pool_size"]),
            candidate_multiplier=int(config.get("candidate_multiplier") or 3),
            write_thumbs=True,
        )
    items = maybe_merge_csv_annotations(items, args.annotations)
    write_jsonl(run_dir / "dataset" / "items.jsonl", [publicize(item) for item in items])
    write_annotation_html(run_dir / "dataset" / "annotate.html", items)
    manifest["dataset_fingerprint"] = inventory.get("dataset_fingerprint")
    write_json(run_dir / "manifest.json", manifest)
    write_json(run_dir / "licenses.json", catalog_payload(model_keys))

    if args.freeze_only:
        manifest["status"] = "dataset_frozen"
        manifest["completed_at"] = datetime.now(UTC).isoformat()
        write_json(run_dir / "manifest.json", manifest)
        print(run_dir)
        return 0

    split_filter = set(args.splits or ["dev", "primary", "stress"])
    work_items = [item for item in items if item["split"] in split_filter]
    if args.limit:
        capped: list[dict[str, Any]] = []
        counts: dict[str, int] = {}
        for item in work_items:
            split = item["split"]
            if counts.get(split, 0) >= args.limit:
                continue
            capped.append(item)
            counts[split] = counts.get(split, 0) + 1
        work_items = capped

    prompts = list(config.get("prompts") or ["shirt", "t-shirt", "top", "blouse", "knit", "hoodie"])
    context = AdapterContext(
        prompts=prompts,
        box_threshold=float(config.get("box_threshold") or 0.25),
        text_threshold=float(config.get("text_threshold") or 0.25),
        mask_threshold=float(config.get("mask_threshold") or 0.5),
        max_proposals=int(config.get("max_proposals") or 12),
        family="tops",
    )
    weights = dict(config.get("resolver_weights") or {"area": 0.25, "torso": 0.2, "confidence": 0.15, "metadata": 0.4})
    resolver_cfg = freeze_resolver_config(weights, prompts)
    write_json(run_dir / "resolver_config.json", resolver_cfg)

    encoder = None
    if not args.skip_encoder:
        encoder = MetadataEncoder(str(config.get("encoder_id") or "Marqo/marqo-fashionSigLIP"), str(config.get("device") or "auto"))
        encoder.load()

    availability: dict[str, Any] = {}
    metrics: dict[str, Any] = {"models": {}, "paired": {}, "kappa": None, "resolver": resolver_cfg}
    all_selection_rows: list[dict[str, Any]] = []
    reviews_a = load_reviews(args.reviews_a)
    reviews_b = load_reviews(args.reviews_b)

    reuse_root = (
        resolve_run_dir(args.reuse_proposals, output_root, extra_artifact_roots(output_root))
        if args.reuse_proposals
        else None
    )

    for model_key in model_keys:
        adapter = create_adapter(
            model_key,
            requested_device=str(config.get("device") or "auto"),
            checkpoints=config.get("checkpoints") or {},
            download_weights=bool(config.get("download_weights") or False),
        )
        cache_dir = run_dir / "proposals" / model_key
        cache_dir.mkdir(parents=True, exist_ok=True)
        reuse = False
        if reuse_root and (reuse_root / "proposals" / model_key / "proposals.jsonl").exists():
            import shutil

            src = reuse_root / "proposals" / model_key
            if src.resolve() != cache_dir.resolve():
                shutil.copytree(src, cache_dir, dirs_exist_ok=True)
            reuse = True
        status = "ready"
        error = None
        try:
            if not reuse:
                if not adapter.available():
                    raise UnavailableError("adapter import unavailable")
                adapter.load()
        except Exception as exc:
            status = "unavailable"
            error = f"{type(exc).__name__}: {exc}"
            availability[model_key] = {"status": status, "error": error, "metadata": {}}
            continue

        by_item, metadata, timing_rows = infer_model(
            adapter, work_items, context=context, cache_dir=cache_dir, reuse=reuse
        )
        availability[model_key] = {"status": "ok", "error": None, "metadata": metadata}
        timing_by_id = {row["item_id"]: row for row in timing_rows}

        metrics["models"][model_key] = {}
        for resolver in RESOLVER_NAMES:
            selection_rows: list[dict[str, Any]] = []
            overlays: dict[str, Path] = {}
            overlay_dir = run_dir / "selected" / model_key / resolver / "overlays"
            for item in work_items:
                with Image.open(item["image_path"]) as image:
                    rgb = image.convert("RGB")
                    proposals = by_item.get(item["item_id"]) or []
                    scored = score_proposals(item, rgb, proposals, encoder=encoder, weights=weights)
                    selected = select_with_resolver(scored, resolver)
                    hit = target_hit(item, selected, rgb.size)
                    overlay_path = overlay_dir / f"{item['item_id']}.jpg"
                    saved = save_selected_overlay(item, selected, overlay_path)
                    if saved:
                        overlays[item["item_id"]] = Path(saved)
                    timing = timing_by_id.get(item["item_id"]) or {}
                    selection_rows.append(
                        {
                            "item_id": item["item_id"],
                            "split": item["split"],
                            "source": item["source"],
                            "sampling_weight": item.get("sampling_weight") or 1.0,
                            "heuristic": item.get("heuristic"),
                            "annotation": item.get("annotation"),
                            "model_key": model_key,
                            "resolver": resolver,
                            "proposal_count": len(proposals),
                            "duplicate_rate": round(duplicate_rate(proposals), 4),
                            "fragmentation": fragmentation(selected),
                            "selected_area_ratio": None if selected is None else selected.get("area_ratio"),
                            "selected_proposal_id": None if selected is None else selected.get("proposal_id"),
                            "selected": selected,
                            "target": hit,
                            "latency_sec": timing.get("latency_sec"),
                            "peak_memory_mb": timing.get("peak_memory_mb"),
                            "failed": timing.get("failed", False),
                            "overlay_path": saved,
                        }
                    )
            if reviews_a:
                selection_rows = merge_visual_scores(selection_rows, reviews_a)
            write_jsonl(run_dir / "selected" / model_key / resolver / "assignments.jsonl", selection_rows)
            build_contact_sheet(
                work_items,
                overlays,
                run_dir / "figures" / "contact_sheets" / f"{model_key}_{resolver}.jpg",
            )
            split_metrics = {}
            for split in ("dev", "primary", "stress"):
                split_metrics[split] = {
                    mode: auto_metrics_for_split(selection_rows, split=split, weight_mode=mode)
                    for mode in ("uniform", "population", "source_balanced")
                }
            metrics["models"][model_key][resolver] = split_metrics
            all_selection_rows.extend(selection_rows)
        write_reviewer_sheet(
            run_dir / "dataset" / f"reviewer_sheet_{model_key}.csv",
            items,
            [row for row in all_selection_rows if row["model_key"] == model_key and row["resolver"] == "combined"],
        )
        write_reviewer_sheet(
            run_dir / "dataset" / f"reviewer_sheet_{model_key}_b.csv",
            items,
            [row for row in all_selection_rows if row["model_key"] == model_key and row["resolver"] == "combined"],
        )

    # Paired deltas on combined resolver, primary split.
    model_ids = [key for key, payload in availability.items() if payload.get("status") == "ok"]
    for i, left in enumerate(model_ids):
        for right in model_ids[i + 1 :]:
            left_rows = [
                row
                for row in all_selection_rows
                if row["model_key"] == left and row["resolver"] == "combined" and row["split"] == "primary"
            ]
            right_rows = [
                row
                for row in all_selection_rows
                if row["model_key"] == right and row["resolver"] == "combined" and row["split"] == "primary"
            ]
            metrics["paired"][f"{left}__minus__{right}"] = paired_success_delta(left_rows, right_rows)

    if reviews_a and reviews_b:
        metrics["kappa"] = reviewer_kappa(reviews_a, reviews_b, field="usable")

    write_json(run_dir / "metrics" / "summary.json", metrics)
    write_json(run_dir / "availability.json", availability)
    manifest["status"] = "complete"
    manifest["completed_at"] = datetime.now(UTC).isoformat()
    manifest["availability"] = availability
    write_json(run_dir / "manifest.json", manifest)
    build_report(
        run_dir,
        items=items,
        inventory=inventory,
        licenses=catalog_payload(model_keys),
        availability=availability,
        metrics=metrics,
        manifest=manifest,
    )
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
