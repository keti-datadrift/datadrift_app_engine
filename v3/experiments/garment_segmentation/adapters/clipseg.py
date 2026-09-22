from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage

from adapters.base import (
    AdapterContext,
    UnavailableError,
    filter_proposals,
    from_pretrained,
    proposals_from_mask,
    resolve_device,
)


def heatmaps_to_proposals(
    heatmaps: np.ndarray,
    labels: list[str],
    *,
    threshold: float,
    min_area_ratio: float = 0.008,
    max_proposals: int = 12,
) -> list[dict[str, Any]]:
    if heatmaps.ndim == 2:
        heatmaps = heatmaps[None, ...]
    proposals: list[dict[str, Any]] = []
    for index, heatmap in enumerate(heatmaps):
        label = labels[index] if index < len(labels) else "garment"
        binary = np.asarray(heatmap, dtype=np.float32) >= threshold
        if binary.mean() < min_area_ratio:
            continue
        components, count = ndimage.label(binary.astype(np.uint8))
        for component_id in range(1, count + 1):
            component = components == component_id
            area = float(component.mean())
            if area < min_area_ratio:
                continue
            score = float(np.clip(heatmap[component].mean(), 0.0, 1.0))
            proposals.append(
                proposals_from_mask(
                    component,
                    label=label,
                    score=score,
                    proposal_id=f"clipseg-{index}-{component_id}",
                    extras={"prompt": label, "threshold": threshold},
                )
            )
    return filter_proposals(proposals, max_proposals)


class ClipSegSegmentor:
    key = "clipseg"

    def __init__(
        self,
        model_id: str = "CIDAS/clipseg-rd64-refined",
        requested_device: str = "auto",
        download_weights: bool = False,
    ) -> None:
        self.model_id = model_id
        self.requested_device = requested_device
        self.download_weights = download_weights
        self.device = "cpu"
        self.model = None
        self.processor = None

    def available(self) -> bool:
        try:
            from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor  # noqa: F401
        except Exception:
            return False
        return True

    def load(self) -> None:
        from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

        self.device = resolve_device(self.requested_device)
        try:
            self.processor = from_pretrained(
                CLIPSegProcessor.from_pretrained, self.model_id, download=self.download_weights
            )
            self.model = from_pretrained(
                CLIPSegForImageSegmentation.from_pretrained, self.model_id, download=self.download_weights
            )
            self.model.to(self.device)
            self.model.eval()
        except Exception as exc:
            raise UnavailableError(f"clipseg load failed: {exc}") from exc

    def infer(self, image: Image.Image, context: AdapterContext) -> list[dict[str, Any]]:
        if self.model is None or self.processor is None:
            raise UnavailableError("clipseg is not loaded")
        import torch
        import torch.nn.functional as F

        rgb = image.convert("RGB")
        width, height = rgb.size
        prompts = context.prompts or ["shirt", "top"]
        inputs = self.processor(
            text=prompts,
            images=[rgb] * len(prompts),
            padding=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()}
        with torch.inference_mode():
            logits = self.model(**inputs).logits
        if logits.ndim == 2:
            logits = logits.unsqueeze(0)
        probs = torch.sigmoid(logits)
        if probs.ndim == 3:
            probs = probs.unsqueeze(1)
        upsampled = F.interpolate(probs, size=(height, width), mode="bilinear", align_corners=False)
        heatmaps = upsampled.squeeze(1).detach().cpu().numpy()
        threshold = max(0.25, float(context.mask_threshold))
        proposals = heatmaps_to_proposals(
            heatmaps,
            prompts,
            threshold=threshold,
            max_proposals=context.max_proposals,
        )
        if not proposals:
            proposals = heatmaps_to_proposals(
                heatmaps,
                prompts,
                threshold=max(0.12, threshold * 0.5),
                max_proposals=context.max_proposals,
            )
        return proposals

    def metadata(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "checkpoint": self.model_id,
            "device": self.device,
            "dtype": "float32",
            "kind": "clip_text_segmentation",
        }
