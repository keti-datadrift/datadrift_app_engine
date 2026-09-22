from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from PIL import Image

from adapters.base import AdapterContext
from adapters.sam3 import Sam3Segmentor
from io_utils import read_json, read_jsonl, write_json, write_jsonl
from run import REPO_ROOT, extra_artifact_roots, resolve_run_dir


DEFAULT_FALLBACK_PROMPTS = (
    "t-shirt",
    "short-sleeve top",
    "long-sleeve top",
    "sweatshirt",
    "sleeveless top",
    "tank top",
    "sports bra",
    "dress",
)
LOW_THRESHOLD_PROMPTS = (
    "shirt",
    "blouse",
    "top",
    *DEFAULT_FALLBACK_PROMPTS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retry only zero-proposal SAM3 items and create a merged proposal cache"
    )
    parser.add_argument("--source-run", default="sam3-mps-450")
    parser.add_argument("--output-run", default="sam3-mps-450-fallback-cache")
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/garment_segmentation"))
    parser.add_argument("--device", default="mps")
    parser.add_argument("--model-id", default="facebook/sam3")
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--retry-threshold", type=float, default=0.18)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--max-proposals", type=int, default=12)
    parser.add_argument("--download-weights", action="store_true")
    return parser.parse_args()


def _infer(
    adapter: Sam3Segmentor,
    item: dict[str, Any],
    *,
    threshold: float,
    prompts: tuple[str, ...],
    mask_threshold: float,
    max_proposals: int,
) -> tuple[list[dict[str, Any]], float, str | None]:
    adapter.prompt_list = list(prompts)
    context = AdapterContext(
        prompts=list(prompts),
        box_threshold=threshold,
        text_threshold=threshold,
        mask_threshold=mask_threshold,
        max_proposals=max_proposals,
        family="tops",
    )
    started = time.perf_counter()
    error = None
    proposals: list[dict[str, Any]] = []
    try:
        with Image.open(item["image_path"]) as image:
            proposals = adapter.infer(image.convert("RGB"), context)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    return proposals, time.perf_counter() - started, error


def main() -> int:
    args = parse_args()
    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = (REPO_ROOT / output_root).resolve()
    source = resolve_run_dir(
        args.source_run,
        output_root,
        extra_artifact_roots(output_root),
    )
    source_rows = read_jsonl(source / "proposals" / "sam3" / "proposals.jsonl")
    items = read_jsonl(source / "dataset" / "items.jsonl")
    item_by_id = {str(item["item_id"]): item for item in items}
    failures = [row for row in source_rows if not (row.get("proposals") or [])]
    if not failures:
        print("no zero-proposal items")
        return 0

    adapter = Sam3Segmentor(
        model_id=args.model_id,
        requested_device=args.device,
        download_weights=args.download_weights,
        prompt_list=list(DEFAULT_FALLBACK_PROMPTS),
    )
    adapter.load()

    replacements: dict[str, dict[str, Any]] = {}
    attempts: list[dict[str, Any]] = []
    for index, original in enumerate(failures, 1):
        item_id = str(original["item_id"])
        item = item_by_id[item_id]
        proposals, elapsed, error = _infer(
            adapter,
            item,
            threshold=args.box_threshold,
            prompts=DEFAULT_FALLBACK_PROMPTS,
            mask_threshold=args.mask_threshold,
            max_proposals=args.max_proposals,
        )
        stages = [
            {
                "threshold": args.box_threshold,
                "prompts": list(DEFAULT_FALLBACK_PROMPTS),
                "proposal_count": len(proposals),
                "latency_sec": round(elapsed, 4),
                "error": error,
            }
        ]
        total_elapsed = elapsed
        if not proposals and error is None:
            proposals, elapsed, error = _infer(
                adapter,
                item,
                threshold=args.retry_threshold,
                prompts=LOW_THRESHOLD_PROMPTS,
                mask_threshold=args.mask_threshold,
                max_proposals=args.max_proposals,
            )
            total_elapsed += elapsed
            stages.append(
                {
                    "threshold": args.retry_threshold,
                    "prompts": list(LOW_THRESHOLD_PROMPTS),
                    "proposal_count": len(proposals),
                    "latency_sec": round(elapsed, 4),
                    "error": error,
                }
            )
        replacements[item_id] = {
            **original,
            "proposals": proposals,
            "proposal_count": len(proposals),
            "latency_sec": round(total_elapsed, 4),
            "failed": error is not None,
            "error": error,
            "fallback_retry": True,
            "fallback_stages": stages,
        }
        attempts.append(
            {
                "item_id": item_id,
                "split": item.get("split"),
                "subtype": item.get("subtype"),
                "product": item.get("product"),
                "recovered": bool(proposals),
                "proposal_count": len(proposals),
                "labels": sorted({str(row.get("label")) for row in proposals}),
                "stages": stages,
            }
        )
        print(
            f"fallback {index}/{len(failures)} {item_id} "
            f"proposals={len(proposals)} labels={attempts[-1]['labels']}",
            flush=True,
        )

    merged = [replacements.get(str(row["item_id"]), row) for row in source_rows]
    output = output_root / args.output_run
    cache_dir = output / "proposals" / "sam3"
    write_jsonl(cache_dir / "proposals.jsonl", merged)
    source_meta = read_json(source / "proposals" / "sam3" / "metadata.json")
    recovered = sum(bool(row["recovered"]) for row in attempts)
    write_json(
        cache_dir / "metadata.json",
        {
            **source_meta,
            "fallback_source_run": str(source),
            "fallback_items": len(attempts),
            "fallback_recovered": recovered,
            "fallback_remaining": len(attempts) - recovered,
            "fallback_prompts": list(DEFAULT_FALLBACK_PROMPTS),
            "fallback_retry_threshold": args.retry_threshold,
        },
    )
    write_json(
        output / "fallback_summary.json",
        {
            "source_run": str(source),
            "items": len(attempts),
            "recovered": recovered,
            "remaining": len(attempts) - recovered,
            "attempts": attempts,
        },
    )
    print(output / "fallback_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
