from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from dataset import resolve_family, weak_labels, normalize_sources
from representations import residualize_by_family


def test_resolve_family_for_both_sources() -> None:
    assert resolve_family("001000", "상의") == "tops"
    assert resolve_family("268102100", "아우터") == "outerwear"
    assert resolve_family("273104100", "백팩") == "bags"
    assert resolve_family("311107100", "볼캡") == "headwear"
    assert resolve_family("104", "뷰티") == "skip"


def test_normalize_sources_uses_public_aliases() -> None:
    sources = normalize_sources(
        {
            "retail_a": {"disk_name": "channel_a", "format": "ranking_items"},
            "retail_b": {"disk_name": "channel_b", "format": "survey_signals"},
        }
    )
    assert set(sources) == {"retail_a", "retail_b"}
    assert sources["retail_a"]["format"] == "ranking_items"
    assert sources["retail_b"]["format"] == "survey_signals"


def test_weak_labels_are_multilabel() -> None:
    labels = weak_labels("블랙 데님 스트라이프 와이드 팬츠")
    assert labels["color"] == ["black"]
    assert labels["material"] == ["denim"]
    assert labels["pattern"] == ["stripe"]
    assert labels["shape"] == ["wide"]


def test_residualization_uses_train_centroid_and_normalizes() -> None:
    raw = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.8, 0.2, 0.0],
            [0.9, 0.1, 0.1],
            [0.0, 1.0, 0.0],
            [0.2, 0.8, 0.0],
            [0.1, 0.9, 0.1],
        ],
        dtype=np.float32,
    )
    raw /= np.linalg.norm(raw, axis=1, keepdims=True)
    items = [
        {"family": "tops", "split": "train"},
        {"family": "tops", "split": "train"},
        {"family": "tops", "split": "test"},
        {"family": "bags", "split": "train"},
        {"family": "bags", "split": "train"},
        {"family": "bags", "split": "test"},
    ]
    residual = residualize_by_family(raw, items)
    assert residual.shape == raw.shape
    assert np.allclose(np.linalg.norm(residual, axis=1), 1.0, atol=1e-5)
    assert not np.allclose(residual, raw)
