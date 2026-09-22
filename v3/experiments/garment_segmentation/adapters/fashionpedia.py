from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from adapters.base import (
    AdapterContext,
    UnavailableError,
    filter_proposals,
    from_pretrained,
    proposals_from_mask,
    resolve_device,
)
from io_utils import mask_from_box

FASHIONPEDIA_TOP_LABELS = {
    "shirt, blouse",
    "top, t-shirt, sweatshirt",
    "sweater",
    "cardigan",
    "jacket",
    "vest",
    "coat",
    "cape",
    "jumpsuit",
    "dress",
    "hood",
    "collar",
    "lapel",
    "sleeve",
}

TOPS_FAMILY_LABELS = {
    "shirt, blouse",
    "top, t-shirt, sweatshirt",
    "sweater",
    "cardigan",
    "vest",
    "hood",
    "collar",
    "sleeve",
}


class FashionpediaSegmentor:
    key = "fashionpedia"

    def __init__(
        self,
        model_id: str = "valentinafeve/yolos-fashionpedia",
        requested_device: str = "auto",
        score_threshold: float = 0.3,
        download_weights: bool = False,
    ) -> None:
        self.model_id = model_id
        self.requested_device = requested_device
        self.score_threshold = score_threshold
        self.download_weights = download_weights
        self.device = "cpu"
        self.model = None
        self.processor = None
        self.id2label: dict[int, str] = {}

    def available(self) -> bool:
        try:
            from transformers import AutoImageProcessor, AutoModelForObjectDetection
        except Exception:
            return False
        return True

    def load(self) -> None:
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        self.device = resolve_device(self.requested_device)
        try:
            self.processor = from_pretrained(
                AutoImageProcessor.from_pretrained, self.model_id, download=self.download_weights
            )
            self.model = from_pretrained(
                AutoModelForObjectDetection.from_pretrained, self.model_id, download=self.download_weights
            )
            self.model.to(self.device)
            self.model.eval()
            self.id2label = {int(k): str(v) for k, v in getattr(self.model.config, "id2label", {}).items()}
        except Exception as exc:
            raise UnavailableError(f"fashionpedia load failed: {exc}") from exc

    def infer(self, image: Image.Image, context: AdapterContext) -> list[dict[str, Any]]:
        if self.model is None or self.processor is None:
            raise UnavailableError("fashionpedia is not loaded")
        import torch

        rgb = image.convert("RGB")
        width, height = rgb.size
        inputs = self.processor(images=rgb, return_tensors="pt")
        inputs = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()}
        with torch.inference_mode():
            outputs = self.model(**inputs)
        target_sizes = torch.tensor([(height, width)], device=self.device)
        try:
            results = self.processor.post_process_object_detection(
                outputs, threshold=max(context.box_threshold, self.score_threshold), target_sizes=target_sizes
            )[0]
        except Exception:
            logits = outputs.logits[0]
            boxes = outputs.pred_boxes[0]
            probs = logits.softmax(-1)
            scores, labels = probs.max(-1)
            results = {
                "scores": scores,
                "labels": labels,
                "boxes": boxes,
            }
        proposals: list[dict[str, Any]] = []
        boxes = results.get("boxes")
        scores = results.get("scores")
        labels = results.get("labels")
        if boxes is None:
            return []
        for index, box in enumerate(boxes.detach().cpu().numpy()):
            if box.max() <= 1.5:
                box = np.array(
                    [box[0] * width, box[1] * height, box[2] * width, box[3] * height],
                    dtype=np.float32,
                )
            score = float(scores[index].detach().cpu()) if scores is not None else 0.0
            label_id = int(labels[index].detach().cpu()) if labels is not None else -1
            label = self.id2label.get(label_id, str(label_id))
            if context.family == "tops" and label in self.id2label.values():
                if label not in TOPS_FAMILY_LABELS and label not in FASHIONPEDIA_TOP_LABELS:
                    # Keep non-top detections; resolver decides. Mild downweight.
                    score *= 0.65
            mask = mask_from_box(height, width, box)
            proposals.append(
                proposals_from_mask(
                    mask,
                    label=label,
                    score=score,
                    proposal_id=f"fashionpedia-{index}",
                    extras={"checkpoint": self.model_id, "mask_source": "box"},
                )
            )
        return filter_proposals(proposals, context.max_proposals)

    def metadata(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "checkpoint": self.model_id,
            "device": self.device,
            "dtype": "float32",
            "kind": "fashion_instance",
            "mask_source": "detection_box",
        }
