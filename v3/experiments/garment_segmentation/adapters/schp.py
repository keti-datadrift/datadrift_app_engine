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

ATR_LABELS = {
    0: "background",
    1: "hat",
    2: "hair",
    3: "sunglasses",
    4: "upper-clothes",
    5: "skirt",
    6: "pants",
    7: "dress",
    8: "belt",
    9: "left-shoe",
    10: "right-shoe",
    11: "face",
    12: "left-leg",
    13: "right-leg",
    14: "left-arm",
    15: "right-arm",
    16: "bag",
    17: "scarf",
}

TOP_CLASS_IDS = {4, 7}


class SchpSegmentor:
    key = "schp"

    def __init__(
        self,
        model_id: str = "mattmdjaga/segformer_b2_clothes",
        requested_device: str = "auto",
        download_weights: bool = False,
    ) -> None:
        self.model_id = model_id
        self.requested_device = requested_device
        self.download_weights = download_weights
        self.device = "cpu"
        self.model = None
        self.processor = None
        self.id2label = dict(ATR_LABELS)

    def available(self) -> bool:
        try:
            from transformers import AutoModelForSemanticSegmentation, SegformerImageProcessor
        except Exception:
            return False
        return True

    def load(self) -> None:
        from transformers import AutoModelForSemanticSegmentation, AutoImageProcessor

        self.device = resolve_device(self.requested_device)
        try:
            self.processor = from_pretrained(
                AutoImageProcessor.from_pretrained, self.model_id, download=self.download_weights
            )
            self.model = from_pretrained(
                AutoModelForSemanticSegmentation.from_pretrained, self.model_id, download=self.download_weights
            )
            self.model.to(self.device)
            self.model.eval()
            config_labels = getattr(self.model.config, "id2label", None)
            if config_labels:
                self.id2label = {int(k): str(v) for k, v in config_labels.items()}
        except Exception as exc:
            raise UnavailableError(f"schp load failed: {exc}") from exc

    def infer(self, image: Image.Image, context: AdapterContext) -> list[dict[str, Any]]:
        if self.model is None or self.processor is None:
            raise UnavailableError("schp is not loaded")
        import torch

        rgb = image.convert("RGB")
        width, height = rgb.size
        inputs = self.processor(images=rgb, return_tensors="pt")
        inputs = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()}
        with torch.inference_mode():
            logits = self.model(**inputs).logits
        upsampled = torch.nn.functional.interpolate(
            logits, size=(height, width), mode="bilinear", align_corners=False
        )
        labels = upsampled.argmax(dim=1)[0].detach().cpu().numpy()
        proposals: list[dict[str, Any]] = []
        for class_id, class_name in sorted(self.id2label.items()):
            if class_name in {"background", "hair", "face", "left-arm", "right-arm", "left-leg", "right-leg"}:
                continue
            binary = labels == int(class_id)
            if binary.mean() < 0.002:
                continue
            components, count = ndimage.label(binary.astype(np.uint8))
            for component_id in range(1, count + 1):
                component = components == component_id
                if component.mean() < 0.002:
                    continue
                score = 0.85 if int(class_id) in TOP_CLASS_IDS or class_name in {"upper-clothes", "dress"} else 0.55
                proposals.append(
                    proposals_from_mask(
                        component,
                        label=str(class_name),
                        score=score,
                        proposal_id=f"schp-{class_id}-{component_id}",
                        extras={"checkpoint": self.model_id, "semantic_class": class_name},
                    )
                )
        return filter_proposals(proposals, context.max_proposals)

    def metadata(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "checkpoint": self.model_id,
            "device": self.device,
            "dtype": "float32",
            "kind": "semantic_human_parsing",
        }
