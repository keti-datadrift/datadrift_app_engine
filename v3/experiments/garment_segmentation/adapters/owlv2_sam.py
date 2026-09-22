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


class Owlv2SamSegmentor:
    key = "owlv2_sam"

    def __init__(
        self,
        detector_id: str = "google/owlv2-base-patch16-ensemble",
        segmentor_id: str = "facebook/sam2.1-hiera-tiny",
        requested_device: str = "auto",
        download_weights: bool = False,
    ) -> None:
        self.detector_id = detector_id
        self.segmentor_id = segmentor_id
        self.requested_device = requested_device
        self.download_weights = download_weights
        self.device = "cpu"
        self.detector = None
        self.detector_processor = None
        self.sam = None
        self.sam_processor = None

    def available(self) -> bool:
        try:
            from transformers import Owlv2ForObjectDetection, Owlv2Processor
            from transformers import Sam2Model, Sam2Processor  # noqa: F401
        except Exception:
            return False
        return True

    def load(self) -> None:
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
        from transformers import Sam2Model, Sam2Processor

        self.device = resolve_device(self.requested_device)
        try:
            self.detector_processor = from_pretrained(
                Owlv2Processor.from_pretrained, self.detector_id, download=self.download_weights
            )
            self.detector = from_pretrained(
                Owlv2ForObjectDetection.from_pretrained, self.detector_id, download=self.download_weights
            )
            self.detector.to(self.device)
            self.detector.eval()
            self.sam_processor = from_pretrained(
                Sam2Processor.from_pretrained, self.segmentor_id, download=self.download_weights
            )
            self.sam = from_pretrained(
                Sam2Model.from_pretrained, self.segmentor_id, download=self.download_weights
            )
            self.sam.to(self.device)
            self.sam.eval()
        except Exception as exc:
            raise UnavailableError(f"owlv2_sam load failed: {exc}") from exc

    def infer(self, image: Image.Image, context: AdapterContext) -> list[dict[str, Any]]:
        if self.detector is None:
            raise UnavailableError("owlv2_sam is not loaded")
        import torch

        rgb = image.convert("RGB")
        width, height = rgb.size
        texts = [context.prompts or ["shirt", "top", "blouse"]]
        inputs = self.detector_processor(text=texts, images=rgb, return_tensors="pt")
        inputs = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()}
        with torch.inference_mode():
            outputs = self.detector(**inputs)
        target_sizes = torch.tensor([(height, width)], device=self.device)
        results = self.detector_processor.post_process_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=context.box_threshold
        )[0]
        proposals: list[dict[str, Any]] = []
        boxes = results.get("boxes")
        scores = results.get("scores")
        labels = results.get("labels")
        if boxes is None:
            return []
        prompt_names = texts[0]
        for index, box in enumerate(boxes.detach().cpu().numpy()):
            score = float(scores[index].detach().cpu()) if scores is not None else 0.0
            label_id = int(labels[index].detach().cpu()) if labels is not None else 0
            label = prompt_names[label_id] if 0 <= label_id < len(prompt_names) else "garment"
            mask = self._segment_box(rgb, box)
            if mask is None:
                mask = mask_from_box(height, width, box)
            proposals.append(
                proposals_from_mask(
                    mask,
                    label=label,
                    score=score,
                    proposal_id=f"owlv2-{index}",
                    extras={"detector": self.detector_id},
                )
            )
        return filter_proposals(proposals, context.max_proposals)

    def _segment_box(self, image: Image.Image, box: np.ndarray) -> np.ndarray | None:
        if self.sam is None or self.sam_processor is None:
            return None
        import torch

        try:
            inputs = self.sam_processor(images=image, input_boxes=[[[float(v) for v in box]]], return_tensors="pt")
            inputs = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()}
            with torch.inference_mode():
                outputs = self.sam(**inputs)
            masks = self.sam_processor.post_process_masks(
                outputs.pred_masks, inputs["original_sizes"], inputs["original_sizes"]
            )[0]
            return (masks[0, 0].detach().cpu().numpy() > 0.5).astype(bool)
        except Exception:
            return None

    def metadata(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "checkpoint": {"detector": self.detector_id, "segmentor": self.segmentor_id},
            "device": self.device,
            "dtype": "float32",
            "kind": "open_vocabulary_instance",
        }
