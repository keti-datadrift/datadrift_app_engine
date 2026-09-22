from __future__ import annotations

import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from heuristics import (
    average_hash_from_features,
    complexity_score,
    image_features,
    resolve_subtype,
)
from io_utils import load_attr_dataset, save_thumbnail, write_json, write_jsonl
from schema import EMPTY_ANNOTATION, SPLITS


def rank_key(item: dict[str, Any], seed: int) -> str:
    raw = f"{seed}|{item['source']}|{item['product_id']}".encode()
    return hashlib.sha1(raw).hexdigest()


def item_id(item: dict[str, Any]) -> str:
    raw = f"{item['source']}|{item['product_id']}|{item.get('content_sha256') or ''}".encode()
    return hashlib.sha1(raw).hexdigest()[:16]


def collect_tops_pool(
    input_root: Path,
    sources: dict[str, dict[str, str]],
    *,
    seed: int,
    pool_size: int,
    candidate_multiplier: int,
    near_duplicate_distance: int = 4,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset = load_attr_dataset()
    registry = dataset.load_registry(input_root, sources)
    target_candidates = max(pool_size * candidate_multiplier, pool_size)
    pools, scanned_days = dataset.collect_candidates(
        input_root,
        registry,
        sources=sources,
        families=("tops",),
        target_per_family=target_candidates,
    )
    selected: list[dict[str, Any]] = []
    hashes: list[int] = []
    seen_digests: set[str] = set()
    skipped = Counter()
    candidates = sorted(pools["tops"].values(), key=lambda row: rank_key(row, seed))
    for item in candidates:
        digest = str(item.get("content_sha256") or "")
        if digest in seen_digests:
            skipped["duplicate_digest"] += 1
            continue
        path = dataset._image_path(input_root, item, sources=sources)
        if path is None:
            skipped["missing_image"] += 1
            continue
        try:
            features = image_features(path)
        except Exception:
            skipped["invalid_image"] += 1
            continue
        phash = average_hash_from_features(features)
        if any((phash ^ other).bit_count() <= near_duplicate_distance for other in hashes):
            skipped["near_duplicate"] += 1
            continue
        hashes.append(phash)
        seen_digests.add(digest)
        enriched = {
            **item,
            "family": "tops",
            "image_path": str(path.resolve()),
            "subtype": resolve_subtype(str(item.get("product") or ""), str(item.get("category_label") or "")),
            "heuristic": features,
            "weak_labels": dataset.weak_labels(str(item.get("product") or "")),
        }
        enriched["complexity"] = round(complexity_score(enriched), 4)
        selected.append(enriched)
        if len(selected) >= pool_size:
            break
    if len(selected) < 450:
        raise RuntimeError(
            f"need at least 450 unique tops, got {len(selected)} skipped={dict(skipped)} "
            f"candidates={len(pools['tops'])}"
        )
    inventory = {
        "family": "tops",
        "pool_size": len(selected),
        "candidate_pool": len(pools["tops"]),
        "skipped": dict(skipped),
        "registry_embedded": len(registry),
        "survey_days_scanned": len(scanned_days),
        "newest_scanned_day": scanned_days[0] if scanned_days else None,
        "oldest_scanned_day": scanned_days[-1] if scanned_days else None,
        "source_aliases": sorted(sources),
    }
    return selected, inventory


def _stratum(item: dict[str, Any]) -> tuple[str, str, str]:
    heuristic = item.get("heuristic") or {}
    return (
        str(item.get("source") or "unknown"),
        str(item.get("subtype") or "other"),
        str(heuristic.get("shoot_type") or "unknown"),
    )


def assign_splits(
    pool: list[dict[str, Any]],
    *,
    seed: int,
    n_dev: int,
    n_primary: int,
    n_stress: int,
    max_source_share: float = 0.72,
) -> list[dict[str, Any]]:
    needed = n_dev + n_primary + n_stress
    if len(pool) < needed:
        raise RuntimeError(f"need {needed} unique tops, got {len(pool)}")
    ordered = sorted(pool, key=lambda row: rank_key(row, seed))
    remaining = list(ordered)
    stress_ranked = sorted(
        remaining,
        key=lambda row: (-float(row.get("complexity") or 0.0), rank_key(row, seed)),
    )
    stress: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    max_per_source = max(1, int(n_stress * max_source_share))
    used: set[str] = set()
    for item in stress_ranked:
        key = item_id(item)
        source = str(item.get("source") or "unknown")
        if source_counts[source] >= max_per_source:
            continue
        stress.append(item)
        used.add(key)
        source_counts[source] += 1
        if len(stress) >= n_stress:
            break
    if len(stress) < n_stress:
        for item in stress_ranked:
            key = item_id(item)
            if key in used:
                continue
            stress.append(item)
            used.add(key)
            if len(stress) >= n_stress:
                break

    leftover = [item for item in remaining if item_id(item) not in used]
    primary = _stratified_take(leftover, n_primary, seed)
    used.update(item_id(item) for item in primary)
    leftover = [item for item in leftover if item_id(item) not in used]
    dev = _stratified_take(leftover, n_dev, seed)

    assigned = []
    pool_stratum_counts = Counter(_stratum(item) for item in ordered)
    for split, rows in (("stress", stress), ("primary", primary), ("dev", dev)):
        split_stratum_counts = Counter(_stratum(item) for item in rows)
        for item in rows:
            row = dict(item)
            row["item_id"] = item_id(row)
            row["split"] = split
            stratum = _stratum(row)
            row["stratum"] = {"source": stratum[0], "subtype": stratum[1], "shoot_type": stratum[2]}
            pool_n = pool_stratum_counts[stratum] or 1
            split_n = split_stratum_counts[stratum] or 1
            row["sampling_weight"] = round(pool_n / split_n, 6)
            row["annotation"] = dict(EMPTY_ANNOTATION)
            assigned.append(row)
    assigned.sort(key=lambda row: (SPLITS.index(row["split"]), row["source"], row["product_id"]))
    return assigned


def _stratified_take(items: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    if count <= 0:
        return []
    if len(items) < count:
        raise RuntimeError(f"need {count} leftover items, got {len(items)}")
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in sorted(items, key=lambda row: rank_key(row, seed)):
        grouped[_stratum(item)].append(item)
    quotas: dict[tuple[str, str, str], int] = {}
    remaining = count
    keys = sorted(grouped)
    base_total = len(items)
    for key in keys:
        share = max(1, round(count * len(grouped[key]) / base_total)) if grouped[key] else 0
        quotas[key] = min(len(grouped[key]), share)
    # Adjust to exact count.
    current = sum(quotas.values())
    if current < count:
        extras = sorted(keys, key=lambda key: (-(len(grouped[key]) - quotas[key]), key))
        idx = 0
        while current < count and extras:
            key = extras[idx % len(extras)]
            if quotas[key] < len(grouped[key]):
                quotas[key] += 1
                current += 1
            idx += 1
            if idx > count * 8:
                break
    elif current > count:
        extras = sorted(keys, key=lambda key: (-quotas[key], key))
        idx = 0
        while current > count and extras:
            key = extras[idx % len(extras)]
            if quotas[key] > 0:
                quotas[key] -= 1
                current -= 1
            idx += 1
    chosen: list[dict[str, Any]] = []
    used: set[str] = set()
    for key in keys:
        for item in grouped[key][: quotas[key]]:
            chosen.append(item)
            used.add(item_id(item))
    if len(chosen) < count:
        for item in sorted(items, key=lambda row: rank_key(row, seed)):
            if item_id(item) in used:
                continue
            chosen.append(item)
            used.add(item_id(item))
            if len(chosen) >= count:
                break
    return chosen[:count]


def publicize(item: dict[str, Any]) -> dict[str, Any]:
    row = dict(item)
    row.pop("disk_source", None)
    return row


def write_annotation_template(path: Path, items: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "item_id",
        "split",
        "source",
        "product_id",
        "product",
        "subtype",
        "heuristic_shoot_type",
        "target_present",
        "target_point_x",
        "target_point_y",
        "scene_type",
        "n_people",
        "n_companion_garments",
        "occlusion",
        "pose",
        "view",
        "reviewer",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in items:
            heuristic = item.get("heuristic") or {}
            writer.writerow(
                {
                    "item_id": item["item_id"],
                    "split": item["split"],
                    "source": item["source"],
                    "product_id": item["product_id"],
                    "product": item.get("product") or "",
                    "subtype": item.get("subtype") or "",
                    "heuristic_shoot_type": heuristic.get("shoot_type") or "",
                    "target_present": "",
                    "target_point_x": "",
                    "target_point_y": "",
                    "scene_type": "",
                    "n_people": "",
                    "n_companion_garments": "",
                    "occlusion": "",
                    "pose": "",
                    "view": "",
                    "reviewer": "",
                    "notes": "",
                }
            )


def merge_annotations(items: list[dict[str, Any]], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {row["item_id"]: row for row in rows if row.get("item_id")}
    merged = []
    for item in items:
        row = dict(item)
        payload = by_id.get(item["item_id"])
        if not payload:
            merged.append(row)
            continue
        annotation = dict(row.get("annotation") or EMPTY_ANNOTATION)

        def _maybe(name: str) -> str | None:
            value = str(payload.get(name) or "").strip()
            return value or None

        present = _maybe("target_present")
        if present is not None:
            annotation["target_present"] = present.lower() in {"1", "true", "yes", "y"}
        x_raw, y_raw = _maybe("target_point_x"), _maybe("target_point_y")
        if x_raw is not None and y_raw is not None:
            annotation["target_point_xy"] = [float(x_raw), float(y_raw)]
        for field in ("scene_type", "occlusion", "pose", "view", "notes"):
            value = _maybe(field)
            if value is not None:
                annotation[field] = value
        for field in ("n_people", "n_companion_garments"):
            value = _maybe(field)
            if value is not None:
                annotation[field] = int(value)
        if present is not None:
            annotation["status"] = "complete"
        row["annotation"] = annotation
        merged.append(row)
    return merged


def freeze_dataset(
    input_root: Path,
    sources: dict[str, dict[str, str]],
    output_dir: Path,
    *,
    seed: int,
    n_dev: int,
    n_primary: int,
    n_stress: int,
    pool_size: int,
    candidate_multiplier: int,
    write_thumbs: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pool, inventory = collect_tops_pool(
        input_root,
        sources,
        seed=seed,
        pool_size=pool_size,
        candidate_multiplier=candidate_multiplier,
    )
    assigned = assign_splits(
        pool,
        seed=seed,
        n_dev=n_dev,
        n_primary=n_primary,
        n_stress=n_stress,
    )
    public_items = [publicize(item) for item in assigned]
    fingerprint = hashlib.sha256(
        "\n".join(f"{row['item_id']}|{row['split']}|{row['content_sha256']}" for row in public_items).encode()
    ).hexdigest()
    split_counts = dict(Counter(row["split"] for row in public_items))
    source_counts = dict(Counter(row["source"] for row in public_items))
    inventory.update(
        {
            "selected": len(public_items),
            "selected_by_split": split_counts,
            "selected_by_source": source_counts,
            "selected_by_subtype": dict(Counter(str(row.get("subtype")) for row in public_items)),
            "selected_by_shoot_type": dict(
                Counter(str((row.get("heuristic") or {}).get("shoot_type")) for row in public_items)
            ),
            "dataset_fingerprint": fingerprint,
        }
    )
    dataset_dir = output_dir / "dataset"
    write_jsonl(dataset_dir / "items.jsonl", public_items)
    write_json(dataset_dir / "inventory.json", inventory)
    write_annotation_template(dataset_dir / "annotation_template.csv", public_items)
    if write_thumbs:
        from PIL import Image

        for item in public_items:
            thumb = dataset_dir / "thumbs" / f"{item['item_id']}.jpg"
            with Image.open(item["image_path"]) as image:
                save_thumbnail(image, thumb)
    return public_items, inventory
