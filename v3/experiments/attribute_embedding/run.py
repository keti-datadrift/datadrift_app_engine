from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from analysis import analyze_arms
from dataset import (
    FAMILIES,
    collect_candidates,
    load_registry,
    normalize_sources,
    sample_items,
    write_jsonl,
)
from report import build_report, create_contact_sheets, save_static_metric_plot
from representations import create_representations


def _public_config(config: dict[str, Any]) -> dict[str, Any]:
    """Drop on-disk channel folder names from persisted run metadata."""

    payload = json.loads(json.dumps(config, default=str))
    sources = payload.get("sources")
    if isinstance(sources, dict):
        for meta in sources.values():
            if isinstance(meta, dict) and "disk_name" in meta:
                meta["disk_name"] = "<local>"
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return None


def _preflight(config: dict[str, Any]) -> dict[str, Any]:
    input_root = Path(config["input_root"]).expanduser().resolve()
    registry = input_root / "data" / "retail_images" / "ingest_registry.sqlite"
    webp_root = input_root / "data" / "retail_images" / "webp"
    survey_root = input_root / "runs" / "survey_days"
    missing = [
        str(path)
        for path in (input_root, registry, webp_root, survey_root)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"missing experiment inputs: {missing}")
    import hdbscan  # noqa: F401
    import open_clip
    import sklearn
    import torch
    import umap  # noqa: F401

    return {
        "input_root": str(input_root),
        "python": sys.version,
        "executable": sys.executable,
        "torch": torch.__version__,
        "open_clip": open_clip.__version__,
        "sklearn": sklearn.__version__,
        "device": "mps" if torch.backends.mps.is_available() else "cpu",
        "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fashion attribute embedding experiment")
    parser.add_argument("--config", type=Path, default=HERE / "config.yaml")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--sample-per-family", type=int, default=None)
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument(
        "--reuse-run",
        default=None,
        help="Reuse representation NPZ files from a run with the same dataset fingerprint",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.sample_per_family is not None:
        config["sample_per_family"] = args.sample_per_family
    run_id = args.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_root = Path(config["output_root"])
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "model_id": config["model_id"],
        "model_revision": config["model_revision"],
        "preprocess_contract": "open_clip_hf_transform",
        "config": _public_config(config),
    }
    _write_json(run_dir / "manifest.json", manifest)
    try:
        preflight = _preflight(config)
        manifest["preflight"] = preflight
        _write_json(run_dir / "manifest.json", manifest)
        input_root = Path(config["input_root"]).expanduser().resolve()
        sources = normalize_sources(config.get("sources"))
        registry = load_registry(input_root, sources)
        families = tuple(config.get("families") or FAMILIES)
        target_candidates = int(config["sample_per_family"]) * int(
            config["candidate_multiplier"]
        )
        pools, scanned_days = collect_candidates(
            input_root,
            registry,
            sources=sources,
            families=families,
            target_per_family=target_candidates,
        )
        items, inventory = sample_items(
            input_root,
            pools,
            sources=sources,
            families=families,
            sample_per_family=int(config["sample_per_family"]),
            seed=int(config["seed"]),
        )
        for item in items:
            item.pop("disk_source", None)
        inventory.update(
            {
                "registry_embedded": len(registry),
                "survey_days_scanned": len(scanned_days),
                "newest_scanned_day": scanned_days[0] if scanned_days else None,
                "oldest_scanned_day": scanned_days[-1] if scanned_days else None,
            }
        )
        write_jsonl(run_dir / "dataset" / "items.jsonl", items)
        _write_json(run_dir / "dataset" / "inventory.json", inventory)
        manifest["dataset_fingerprint"] = inventory["dataset_fingerprint"]
        manifest["item_count"] = len(items)
        if any(inventory["selected_by_family"].get(family, 0) < 30 for family in families):
            raise RuntimeError(
                f"insufficient family sample: {inventory['selected_by_family']}"
            )
        if args.inventory_only:
            manifest["status"] = "inventory_complete"
            manifest["completed_at"] = datetime.now(UTC).isoformat()
            _write_json(run_dir / "manifest.json", manifest)
            print(run_dir)
            return 0

        if args.reuse_run:
            source_run = output_root / args.reuse_run
            source_manifest = json.loads(
                (source_run / "manifest.json").read_text(encoding="utf-8")
            )
            if source_manifest.get("dataset_fingerprint") != inventory["dataset_fingerprint"]:
                raise RuntimeError("reuse run dataset fingerprint mismatch")
            shutil.copytree(
                source_run / "representations",
                run_dir / "representations",
                dirs_exist_ok=True,
            )
            arms = {
                "A": np.load(run_dir / "representations" / "A_raw.npz")["matrix"].astype(np.float32),
                "B": np.load(run_dir / "representations" / "B_category_conditioned.npz")["matrix"].astype(np.float32),
                "C": np.load(run_dir / "representations" / "C_family_residual.npz")["matrix"].astype(np.float32),
                "D": np.load(run_dir / "representations" / "D_attribute_scores.npz")["matrix"].astype(np.float32),
            }
            representation_meta = json.loads(
                (run_dir / "representations" / "metadata.json").read_text(encoding="utf-8")
            )
            manifest["representations_reused_from"] = args.reuse_run
        else:
            arms, representation_meta = create_representations(
                items,
                run_dir=run_dir,
                model_id=str(config["model_id"]),
                model_revision=str(config["model_revision"]),
                requested_device=str(config["device"]),
                batch_size=int(config["batch_size"]),
            )
        _write_json(run_dir / "representations" / "metadata.json", representation_meta)
        projections, assignments, probabilities, metrics = analyze_arms(
            arms,
            items,
            seed=int(config["seed"]),
            umap_cfg=dict(config["umap"]),
            cluster_cfg=dict(config["clustering"]),
            evaluation_cfg=dict(config["evaluation"]),
        )
        (run_dir / "projections").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            run_dir / "projections" / "intrinsic_umap.npz",
            **{arm: coords for arm, coords in projections.items()},
        )
        cluster_rows = []
        for arm in sorted(assignments):
            for index, item in enumerate(items):
                cluster_rows.append(
                    {
                        "schema_version": 1,
                        "run_id": run_id,
                        "representation_id": arm,
                        "dataset_fingerprint": inventory["dataset_fingerprint"],
                        "source": item["source"],
                        "product_id": item["product_id"],
                        "family": item["family"],
                        "cluster_id": int(assignments[arm][index]),
                        "membership_probability": round(
                            float(probabilities[arm][index]), 6
                        ),
                        "x": float(projections[arm][index, 0]),
                        "y": float(projections[arm][index, 1]),
                    }
                )
        write_jsonl(run_dir / "clusters" / "assignments.jsonl", cluster_rows)
        _write_json(run_dir / "metrics" / "summary.json", metrics)
        contact_sheets = create_contact_sheets(
            arms,
            items,
            run_dir / "figures" / "contact_sheets",
            count=int(config["evaluation"]["query_contact_sheets"]),
            seed=int(config["seed"]),
        )
        save_static_metric_plot(metrics, run_dir / "figures" / "metric_comparison.png")
        manifest["status"] = "complete"
        manifest["completed_at"] = datetime.now(UTC).isoformat()
        _write_json(run_dir / "manifest.json", manifest)
        build_report(
            run_dir,
            items=items,
            inventory=inventory,
            arms=arms,
            projections=projections,
            assignments=assignments,
            metrics=metrics,
            manifest=manifest,
            contact_sheets=contact_sheets,
        )
        print(run_dir)
        return 0
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["failed_at"] = datetime.now(UTC).isoformat()
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        _write_json(run_dir / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
