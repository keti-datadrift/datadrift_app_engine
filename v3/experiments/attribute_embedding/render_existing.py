from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from report import build_report


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate an existing experiment report")
    parser.add_argument("run_id")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "artifacts" / "attribute_embedding",
    )
    args = parser.parse_args()
    run_dir = args.output_root / args.run_id
    items = _jsonl(run_dir / "dataset" / "items.jsonl")
    inventory = _json(run_dir / "dataset" / "inventory.json")
    metrics = _json(run_dir / "metrics" / "summary.json")
    manifest = _json(run_dir / "manifest.json")
    arms = {
        "A": np.load(run_dir / "representations" / "A_raw.npz")["matrix"].astype(np.float32),
        "B": np.load(run_dir / "representations" / "B_category_conditioned.npz")["matrix"].astype(np.float32),
        "C": np.load(run_dir / "representations" / "C_family_residual.npz")["matrix"].astype(np.float32),
        "D": np.load(run_dir / "representations" / "D_attribute_scores.npz")["matrix"].astype(np.float32),
    }
    projection_npz = np.load(run_dir / "projections" / "intrinsic_umap.npz")
    projections = {arm: projection_npz[arm] for arm in arms}
    cluster_rows = _jsonl(run_dir / "clusters" / "assignments.jsonl")
    assignments = {arm: np.full(len(items), -1, dtype=int) for arm in arms}
    item_index = {
        (str(item["source"]), str(item["product_id"])): i for i, item in enumerate(items)
    }
    for row in cluster_rows:
        index = item_index[(str(row["source"]), str(row["product_id"]))]
        assignments[str(row["representation_id"])][index] = int(row["cluster_id"])
    contacts = [
        f"figures/contact_sheets/{path.name}"
        for path in sorted((run_dir / "figures" / "contact_sheets").glob("*.jpg"))
    ]
    build_report(
        run_dir,
        items=items,
        inventory=inventory,
        arms=arms,
        projections=projections,
        assignments=assignments,
        metrics=metrics,
        manifest=manifest,
        contact_sheets=contacts,
    )
    print(run_dir / "report" / "index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
