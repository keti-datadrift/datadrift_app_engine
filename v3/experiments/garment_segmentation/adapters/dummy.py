from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from adapters.base import AdapterContext, proposals_from_mask


class DummySegmentor:
    key = "dummy"

    def __init__(self, **_: Any) -> None:
        self._loaded = False

    def available(self) -> bool:
        return True

    def load(self) -> None:
        self._loaded = True

    def infer(self, image: Image.Image, context: AdapterContext) -> list[dict[str, Any]]:
        rgb = np.asarray(image.convert("RGB"))
        height, width = rgb.shape[:2]
        gray = rgb.mean(axis=2)
        threshold = float(np.median(gray))
        foreground = gray < threshold
        if foreground.mean() < 0.05 or foreground.mean() > 0.95:
            foreground = np.ones((height, width), dtype=bool)
            foreground[: height // 8] = False
            foreground[-height // 10 :] = False
        upper = np.zeros((height, width), dtype=bool)
        upper[: int(height * 0.55), int(width * 0.12) : int(width * 0.88)] = True
        center = np.zeros((height, width), dtype=bool)
        center[int(height * 0.18) : int(height * 0.82), int(width * 0.2) : int(width * 0.8)] = True
        rows = [
            proposals_from_mask(
                np.logical_and(foreground, upper),
                label="upper-clothes",
                score=0.9,
                proposal_id="dummy-upper",
                extras={"kind": "upper"},
            ),
            proposals_from_mask(
                np.logical_and(foreground, center),
                label="garment",
                score=0.6,
                proposal_id="dummy-center",
                extras={"kind": "center"},
            ),
            proposals_from_mask(
                foreground,
                label="foreground",
                score=0.35,
                proposal_id="dummy-full",
                extras={"kind": "full"},
            ),
        ]
        return [row for row in rows if row["area_ratio"] > 0.01][: context.max_proposals]

    def metadata(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "loaded": self._loaded,
            "checkpoint": None,
            "revision": None,
            "device": "cpu",
            "dtype": "uint8",
            "kind": "synthetic",
        }
