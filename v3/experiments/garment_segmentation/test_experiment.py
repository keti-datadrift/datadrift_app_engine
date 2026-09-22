from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from adapters.base import AdapterContext, proposals_from_mask
from adapters.dummy import DummySegmentor
from evaluation import auto_metrics_for_split, bootstrap_ci, duplicate_rate, reviewer_kappa
from freeze import assign_splits, item_id, merge_annotations
from heuristics import resolve_subtype
from io_utils import mask_from_box, rle_decode, rle_encode, restore_mask
from licenses import license_record
from resolver import point_in_mask, score_proposals, select_with_resolver, target_hit, torso_prior_score
from report import collect_detection_failures
from run import resolve_run_dir
from schema import validate_proposal, validate_proposals


def _item(index: int, source: str, subtype: str, shoot: str, complexity: float) -> dict:
    payload = {
        "source": source,
        "product_id": f"p{index}",
        "content_sha256": f"{index:064x}",
        "product": f"{subtype} item {index}",
        "category_label": "상의",
        "subtype": subtype,
        "heuristic": {"shoot_type": shoot, "skin_ratio": 0.1},
        "complexity": complexity,
        "family": "tops",
    }
    payload["item_id"] = item_id(payload)
    return payload


def test_resolve_run_dir_accepts_path_and_run_id(tmp_path: Path) -> None:
    frozen = tmp_path / "artifacts" / "garment_segmentation" / "tops-450-20260918"
    (frozen / "dataset").mkdir(parents=True)
    (frozen / "dataset" / "items.jsonl").write_text("{}\n", encoding="utf-8")
    other_root = tmp_path / "artifacts" / "garment_segmentation_sam"
    other_root.mkdir(parents=True)
    found = resolve_run_dir("tops-450-20260918", other_root, [frozen.parent])
    assert found == frozen
    assert resolve_run_dir(str(frozen), other_root) == frozen


def test_collect_detection_failures_lists_empty_proposals(tmp_path: Path) -> None:
    run_dir = tmp_path / "sam3-run"
    (run_dir / "proposals" / "sam3").mkdir(parents=True)
    items = [
        {"item_id": "ok", "split": "dev", "subtype": "shirt", "product": "ok shirt"},
        {"item_id": "miss", "split": "primary", "subtype": "sleeveless", "product": "tank"},
    ]
    (run_dir / "proposals" / "sam3" / "proposals.jsonl").write_text(
        json.dumps({"item_id": "ok", "proposals": [{"label": "shirt"}]})
        + "\n"
        + json.dumps({"item_id": "miss", "proposals": [], "failed": False})
        + "\n",
        encoding="utf-8",
    )
    payload = collect_detection_failures(run_dir, items)
    assert payload["models"][0]["n_failed"] == 1
    assert payload["models"][0]["failures"][0]["item_id"] == "miss"
    assert payload["models"][0]["failures"][0]["reason"] == "empty_proposals"


def test_rle_roundtrip_and_restore() -> None:
    mask = np.zeros((20, 30), dtype=bool)
    mask[2:8, 4:11] = True
    encoded = rle_encode(mask)
    decoded = rle_decode(encoded)
    assert decoded.shape == mask.shape
    assert np.array_equal(decoded, mask)
    proposal = proposals_from_mask(mask, label="top", score=0.8, proposal_id="a")
    restored = restore_mask(proposal, 20, 30)
    assert np.array_equal(restored, mask)
    restored_resized = restore_mask(proposal, 40, 60)
    assert restored_resized.shape == (40, 60)
    assert restored_resized.any()


def test_proposal_schema_and_empty_lists() -> None:
    mask = mask_from_box(10, 12, [1, 2, 6, 8])
    proposal = proposals_from_mask(mask, label="shirt", score=0.4, proposal_id="p0")
    assert validate_proposal(proposal)["area_ratio"] > 0
    assert validate_proposals([]) == []


def test_duplicate_rate_detects_overlap() -> None:
    first = proposals_from_mask(mask_from_box(8, 8, [0, 0, 8, 8]), label="a", score=1, proposal_id="1")
    second = proposals_from_mask(mask_from_box(8, 8, [0, 0, 8, 8]), label="b", score=1, proposal_id="2")
    third = proposals_from_mask(mask_from_box(8, 8, [6, 6, 8, 8]), label="c", score=1, proposal_id="3")
    assert duplicate_rate([first, second]) == 1.0
    assert duplicate_rate([first, third]) == 0.0
    assert duplicate_rate([]) == 0.0


def test_deterministic_split_assignment() -> None:
    pool = []
    index = 0
    for source in ("retail_a", "retail_b"):
        for subtype in ("tshirt", "shirt", "knit"):
            for shoot in ("model", "product_only"):
                for _ in range(40):
                    pool.append(_item(index, source, subtype, shoot, complexity=(index % 17) / 17))
                    index += 1
    first = assign_splits(pool, seed=42, n_dev=50, n_primary=300, n_stress=100)
    second = assign_splits(pool, seed=42, n_dev=50, n_primary=300, n_stress=100)
    assert [row["item_id"] for row in first] == [row["item_id"] for row in second]
    counts = {split: sum(row["split"] == split for row in first) for split in ("dev", "primary", "stress")}
    assert counts == {"dev": 50, "primary": 300, "stress": 100}
    ids = [row["item_id"] for row in first]
    assert len(ids) == len(set(ids))
    stress = [row for row in first if row["split"] == "stress"]
    assert len({row["source"] for row in stress}) >= 2


def test_target_resolver_prefers_torso_and_metadata_parts() -> None:
    image = Image.new("RGB", (100, 100), (20, 20, 20))
    far = proposals_from_mask(mask_from_box(100, 100, [80, 80, 98, 98]), label="bag", score=0.9, proposal_id="far")
    torso = proposals_from_mask(mask_from_box(100, 100, [30, 20, 70, 55]), label="shirt", score=0.4, proposal_id="torso")
    item = {"product": "gray tee", "subtype": "tshirt", "category_label": "상의"}
    scored = score_proposals(
        item,
        image,
        [far, torso],
        encoder=None,
        weights={"area": 0.2, "torso": 0.8, "confidence": 0.0, "metadata": 0.0},
    )
    selected = select_with_resolver(scored, "torso_prior")
    assert selected["proposal_id"] == "torso"
    assert torso_prior_score(torso, 100, 100) > torso_prior_score(far, 100, 100)
    area_selected = select_with_resolver(scored, "max_area")
    assert area_selected["proposal_id"] == "torso"


def test_empty_proposals_do_not_select() -> None:
    image = Image.new("RGB", (16, 16), (10, 10, 10))
    scored = score_proposals({"product": "x"}, image, [], encoder=None, weights={})
    assert scored == []
    assert select_with_resolver(scored, "combined") is None


def test_target_hit_with_normalized_point() -> None:
    mask = mask_from_box(50, 50, [10, 10, 30, 30])
    proposal = proposals_from_mask(mask, label="top", score=1, proposal_id="t")
    item = {
        "annotation": {
            "status": "complete",
            "target_present": True,
            "target_point_xy": [0.4, 0.4],
        }
    }
    hit = target_hit(item, proposal, (50, 50))
    assert hit["top1_success"] is True
    miss_item = {
        "annotation": {"status": "complete", "target_present": True, "target_point_xy": [0.95, 0.95]}
    }
    assert target_hit(miss_item, proposal, (50, 50))["top1_success"] is False
    absent = {"annotation": {"status": "complete", "target_present": False}}
    assert target_hit(absent, proposal, (50, 50))["absent_false_positive"] is True
    assert target_hit(absent, None, (50, 50))["top1_success"] is True
    assert point_in_mask([0.0, 0.0], mask) in {True, False}


def test_dummy_adapter_returns_valid_proposals() -> None:
    adapter = DummySegmentor()
    adapter.load()
    image = Image.new("RGB", (64, 80), (30, 30, 30))
    image.paste((200, 200, 200), (8, 8, 56, 50))
    proposals = adapter.infer(image, AdapterContext(prompts=["shirt"]))
    assert proposals
    for proposal in proposals:
        validate_proposal(proposal)
        decoded = rle_decode(proposal["mask_rle"])
        assert decoded.shape == (80, 64)


def test_clipseg_heatmap_components() -> None:
    from adapters.clipseg import heatmaps_to_proposals

    heat = np.zeros((2, 32, 32), dtype=np.float32)
    heat[0, 4:20, 6:22] = 0.9
    heat[1, 22:30, 22:30] = 0.8
    proposals = heatmaps_to_proposals(heat, ["shirt", "bag"], threshold=0.5, min_area_ratio=0.01)
    labels = {row["label"] for row in proposals}
    assert "shirt" in labels
    assert proposals[0]["area_ratio"] > 0


def test_subtype_rules_and_license_gate() -> None:
    assert resolve_subtype("오버핏 셔츠", "상의") == "shirt"
    assert resolve_subtype("헤비웨이트 크루 넥 티셔츠", "상의") == "tshirt"
    assert resolve_subtype("Yak wool V-neck knit", "니트웨어") == "knit"
    record = license_record("sam3")
    assert record["eligibility"] == "research-only"
    adoptable = license_record("clipseg")
    assert adoptable["eligibility"] == "adoptable"
    assert license_record("yolo")["eligibility"] == "research-only"
    assert license_record("segformer")["display_name"].startswith("SegFormer")


def test_annotation_merge_and_metrics() -> None:
    items = assign_splits(
        [_item(i, "retail_a" if i % 2 == 0 else "retail_b", "tshirt", "model", i / 500) for i in range(500)],
        seed=1,
        n_dev=50,
        n_primary=300,
        n_stress=100,
    )
    merged = merge_annotations(
        items[:2],
        [
            {
                "item_id": items[0]["item_id"],
                "target_present": "true",
                "target_point_x": "0.4",
                "target_point_y": "0.4",
                "scene_type": "model",
                "n_people": "1",
                "n_companion_garments": "2",
                "occlusion": "partial",
            }
        ],
    )
    assert merged[0]["annotation"]["status"] == "complete"
    assert merged[0]["annotation"]["n_companion_garments"] == 2
    rows = [
        {
            "item_id": "a",
            "split": "primary",
            "source": "retail_a",
            "sampling_weight": 1.0,
            "proposal_count": 2,
            "duplicate_rate": 0.0,
            "fragmentation": 1,
            "selected_area_ratio": 0.2,
            "target": {"top1_success": True, "absent_false_positive": None},
            "latency_sec": 0.2,
            "peak_memory_mb": 100,
            "failed": False,
            "heuristic": {"shoot_type": "model"},
            "annotation": {"occlusion": "none", "n_companion_garments": 0},
        },
        {
            "item_id": "b",
            "split": "primary",
            "source": "retail_b",
            "sampling_weight": 2.0,
            "proposal_count": 0,
            "duplicate_rate": 0.0,
            "fragmentation": None,
            "selected_area_ratio": 0.0,
            "target": {"top1_success": False, "absent_false_positive": None},
            "latency_sec": 0.4,
            "peak_memory_mb": 120,
            "failed": False,
            "heuristic": {"shoot_type": "product_only"},
            "annotation": {"occlusion": "heavy", "n_companion_garments": 3},
        },
    ]
    summary = auto_metrics_for_split(rows, split="primary", weight_mode="uniform")
    assert summary["n"] == 2
    assert summary["top1_success"]["rate"] == 0.5
    ci = bootstrap_ci([0.0, 1.0, 1.0, 0.0], iterations=200, seed=0)
    assert ci[0] <= ci[1]


def test_reviewer_kappa() -> None:
    left = [
        {"item_id": "1", "model_key": "dummy", "resolver": "combined", "usable": 1},
        {"item_id": "2", "model_key": "dummy", "resolver": "combined", "usable": 0},
        {"item_id": "3", "model_key": "dummy", "resolver": "combined", "usable": 1},
        {"item_id": "4", "model_key": "dummy", "resolver": "combined", "usable": 1},
    ]
    right = [
        {"item_id": "1", "model_key": "dummy", "resolver": "combined", "usable": 1},
        {"item_id": "2", "model_key": "dummy", "resolver": "combined", "usable": 0},
        {"item_id": "3", "model_key": "dummy", "resolver": "combined", "usable": 1},
        {"item_id": "4", "model_key": "dummy", "resolver": "combined", "usable": 0},
    ]
    result = reviewer_kappa(left, right, field="usable")
    assert result["n"] == 4
    assert result["kappa"] is not None
    assert 0.0 <= result["kappa"] <= 1.0


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
