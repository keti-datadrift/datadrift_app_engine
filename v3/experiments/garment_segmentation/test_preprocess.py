from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from adapters.base import proposals_from_mask
from io_utils import mask_from_box
from preprocess import (
    build_arm_images,
    compose_hard,
    compose_soft,
    is_garment_label,
    pad_box,
    resolve_arms,
    resolve_mask_in_box,
)
from preprocess_metrics import intra_inter_gap, linear_cka, mean_item_cosine, neighborhood_jaccard


def test_pad_box_expands_and_clips() -> None:
    padded = pad_box([10, 20, 30, 40], width=40, height=50, pad_ratio=0.5)
    assert padded[0] == 0.0
    assert padded[1] == 10.0
    assert padded[2] == 40.0
    assert padded[3] == 50.0


def test_is_garment_label_accepts_segformer_names() -> None:
    assert is_garment_label("Upper-clothes")
    assert is_garment_label("Dress")
    assert not is_garment_label("Background")
    assert not is_garment_label("Pants")
    assert is_garment_label("shirt, blouse")


def test_mask_in_box_ignores_background_and_uses_upper_clothes() -> None:
    height, width = 40, 30
    box = [5, 4, 20, 22]
    background = proposals_from_mask(
        mask_from_box(height, width, [0, 0, width, height]),
        label="Background",
        score=0.9,
        proposal_id="bg",
    )
    pants = proposals_from_mask(
        mask_from_box(height, width, [6, 24, 18, 38]),
        label="Pants",
        score=0.8,
        proposal_id="pants",
    )
    top = proposals_from_mask(
        mask_from_box(height, width, [6, 6, 18, 20]),
        label="Upper-clothes",
        score=0.4,
        proposal_id="top",
    )
    resolved = resolve_mask_in_box(
        height=height,
        width=width,
        box_xyxy=box,
        proposals=[background, pants, top],
        combined_selected=background,
        min_mask_fill=0.08,
    )
    assert resolved["source"] == "garment_in_box"
    assert resolved["labels"] == ["Upper-clothes"]
    assert not resolved["combined_is_garment"]
    ys, xs = np.where(resolved["mask"])
    assert ys.min() >= 4 and ys.max() < 22
    assert xs.min() >= 5 and xs.max() < 20


def test_mask_falls_back_to_box_when_no_garment() -> None:
    height, width = 16, 16
    box = [2, 2, 10, 10]
    background = proposals_from_mask(
        mask_from_box(height, width, [0, 0, 16, 16]),
        label="Background",
        score=1.0,
        proposal_id="bg",
    )
    resolved = resolve_mask_in_box(
        height=height,
        width=width,
        box_xyxy=box,
        proposals=[background],
        combined_selected=background,
        min_mask_fill=0.08,
    )
    assert resolved["source"] == "box_fallback"
    assert np.array_equal(resolved["mask"], mask_from_box(height, width, box))


def test_soft_keeps_background_texture_hard_zeros_it() -> None:
    crop = np.zeros((24, 24, 3), dtype=np.uint8)
    crop[:, :] = (30, 180, 40)
    crop[6:18, 6:18] = (200, 20, 20)
    mask = np.zeros((24, 24), dtype=bool)
    mask[6:18, 6:18] = True
    soft = np.asarray(compose_soft(crop, mask, dilate_px=1, alpha_sigma=1.0, background_blur_sigma=3.0))
    hard = np.asarray(compose_hard(crop, mask, fill=0))
    assert hard[0, 0].tolist() == [0, 0, 0]
    assert hard[12, 12].tolist() == [200, 20, 20]
    assert soft[0, 0].sum() > 0
    assert not np.array_equal(soft, hard)


def test_build_arm_images_shapes() -> None:
    image = Image.new("RGB", (40, 50), (12, 34, 56))
    pixels = np.array(image)
    pixels[8:28, 10:30] = (200, 10, 10)
    image = Image.fromarray(pixels)
    mask = np.zeros((50, 40), dtype=bool)
    mask[8:28, 10:30] = True
    arms = build_arm_images(image, box_xyxy=[10, 8, 30, 28], mask=mask, pad_ratio=0.1)
    assert arms["A"].size == (40, 50)
    assert arms["B"].size == arms["C"].size == arms["D"].size
    assert arms["B"].size[0] < 40 or arms["B"].size[1] < 50
    subset = build_arm_images(image, box_xyxy=[10, 8, 30, 28], mask=mask, pad_ratio=0.1, arms=("C", "A"))
    assert set(subset) == {"A", "C"}


def test_resolve_arms_orders_and_requires_a() -> None:
    assert resolve_arms(["C", "A"]) == ("A", "C")
    assert resolve_arms(None) == ("A", "B", "C", "D")
    try:
        resolve_arms(["C"])
    except ValueError:
        return
    raise AssertionError("expected ValueError when A is missing")


def test_mask_in_box_accepts_sam3_shirt() -> None:
    height, width = 40, 30
    box = [5, 4, 20, 22]
    sam3_labels = ("shirt", "hoodie", "sweater", "blouse")
    assert is_garment_label("shirt", sam3_labels)
    assert is_garment_label("hoodie", sam3_labels)
    assert not is_garment_label("Background", sam3_labels)
    shirt = proposals_from_mask(
        mask_from_box(height, width, [6, 6, 18, 20]),
        label="shirt",
        score=0.7,
        proposal_id="s",
    )
    resolved = resolve_mask_in_box(
        height=height,
        width=width,
        box_xyxy=box,
        proposals=[shirt],
        combined_selected=None,
        garment_labels=sam3_labels,
        min_mask_fill=0.08,
    )
    assert resolved["source"] == "garment_in_box"
    assert resolved["labels"] == ["shirt"]


def test_compare_arms_ac_only() -> None:
    from preprocess_metrics import compare_arms

    rng = np.random.default_rng(2)
    matrix_a = rng.normal(size=(20, 6)).astype(np.float32)
    matrix_c = matrix_a + rng.normal(0, 0.05, size=matrix_a.shape).astype(np.float32)
    items = [{"weak_labels": {"color": ["black"], "pattern": [], "material": [], "shape": []}, "subtype": "tshirt"}] * 20
    payload = compare_arms({"A": matrix_a, "C": matrix_c}, items, k=3, bootstrap_iterations=20, seed=0)
    assert set(payload["arms"]) == {"A", "C"}
    assert "C" in payload["pairwise_vs_A"]
    assert "B" not in payload["arms"]
    assert "D" not in payload["pairwise_vs_A"]


def test_ac_preview_gallery_omits_bd() -> None:
    import tempfile

    from preprocess_report import write_preview_gallery

    items = [
        {
            "item_id": "x",
            "product": "테스트",
            "subtype": "tshirt",
            "mask_source": "garment_in_box",
            "fill_ratio": 0.4,
        }
    ]
    preview_rel = {"A": ["../previews/A/x.jpg"], "C": ["../previews/C/x.jpg"]}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "previews.html"
        write_preview_gallery(
            path,
            run_id="preprocess-ac-sam3",
            items=items,
            preview_rel=preview_rel,
            mask_stats={},
            coverage={"n_dataset": 1, "n_used": 1, "arms": ["A", "C"]},
            arms=("A", "C"),
        )
        html = path.read_text(encoding="utf-8")
        assert "../previews/A/x.jpg" in html
        assert "../previews/C/x.jpg" in html
        assert "../previews/B/" not in html
        assert "../previews/D/" not in html
        assert "A/C 전처리 미리보기" in html


def test_linear_cka_identity_and_gap_sign() -> None:
    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(20, 8)).astype(np.float32)
    assert abs(linear_cka(matrix, matrix) - 1.0) < 1e-5
    assert mean_item_cosine(matrix, matrix) > 0.99
    left = np.concatenate(
        [
            rng.normal(0, 0.1, size=(10, 4)) + np.array([1, 0, 0, 0]),
            rng.normal(0, 0.1, size=(10, 4)) + np.array([-1, 0, 0, 0]),
        ]
    )
    labels = [ {"red"} if i < 10 else {"blue"} for i in range(20) ]
    gap = intra_inter_gap(left.astype(np.float32), labels)
    assert gap["gap"] is not None and gap["gap"] > 0
    shuffled = left.copy()
    rng.shuffle(shuffled)
    jaccard = neighborhood_jaccard(left.astype(np.float32), shuffled.astype(np.float32), k=3)
    assert 0.0 <= jaccard <= 1.0


def test_density_prefers_tight_cluster() -> None:
    from preprocess_metrics import density_alignment_metrics

    rng = np.random.default_rng(1)
    tight = rng.normal(0, 0.05, size=(40, 8)).astype(np.float32)
    tight[20:] += np.array([3, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    wide = rng.normal(0, 1.2, size=(40, 8)).astype(np.float32)
    wide[20:] += np.array([3, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    items = [{"subtype": "tshirt"}] * 20 + [{"subtype": "knit"}] * 20
    coords = {"A": rng.normal(size=(40, 2)).astype(np.float32), "B": rng.normal(size=(40, 2)).astype(np.float32), "C": rng.normal(size=(40, 2)).astype(np.float32), "D": rng.normal(size=(40, 2)).astype(np.float32)}
    dummy = {arm: wide for arm in ("A", "B", "D")}
    dummy["C"] = tight
    payload = density_alignment_metrics(dummy, items, coords, k=5, min_group=8)
    assert payload["arms"]["C"]["subtype_macro"]["core_knn_cosine"] < payload["arms"]["A"]["subtype_macro"]["core_knn_cosine"]
    assert payload["arms"]["C"]["subtype_separation"] > payload["arms"]["A"]["subtype_separation"]


def test_hull_and_subtype_groups() -> None:
    from preprocess_report import _hull_xy, _subtype_groups

    triangle = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    hull = _hull_xy(triangle)
    assert hull is not None and len(hull) >= 4
    items = [{"subtype": "tshirt"}] * 10 + [{"subtype": "knit"}] * 3
    groups = _subtype_groups(items, min_count=8)
    assert [name for name, _ in groups] == ["tshirt"]


if __name__ == "__main__":
    tests = [
        test_pad_box_expands_and_clips,
        test_is_garment_label_accepts_segformer_names,
        test_mask_in_box_ignores_background_and_uses_upper_clothes,
        test_mask_falls_back_to_box_when_no_garment,
        test_soft_keeps_background_texture_hard_zeros_it,
        test_build_arm_images_shapes,
        test_resolve_arms_orders_and_requires_a,
        test_mask_in_box_accepts_sam3_shirt,
        test_compare_arms_ac_only,
        test_ac_preview_gallery_omits_bd,
        test_linear_cka_identity_and_gap_sign,
        test_density_prefers_tight_cluster,
        test_hull_and_subtype_groups,
    ]
    for test in tests:
        test()
        print(f"ok {test.__name__}")
    print("all preprocess tests passed")
