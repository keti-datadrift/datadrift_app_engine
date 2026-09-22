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


class GroundedSamSegmentor:
    key = "grounded_sam21"

    def __init__(
        self,
        detector_id: str = "IDEA-Research/grounding-dino-tiny",
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
        self.dtype_name = "float32"

    def available(self) -> bool:
        try:
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
            from transformers import Sam2Model, Sam2Processor  # noqa: F401
        except Exception:
            return False
        return True

    def load(self) -> None:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        from transformers import Sam2Model, Sam2Processor

        self.device = resolve_device(self.requested_device)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.dtype_name = str(dtype).replace("torch.", "")
        try:
            self.detector_processor = from_pretrained(
                AutoProcessor.from_pretrained, self.detector_id, download=self.download_weights
            )
            self.detector = from_pretrained(
                AutoModelForZeroShotObjectDetection.from_pretrained,
                self.detector_id,
                download=self.download_weights,
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
            raise UnavailableError(f"grounded_sam21 load failed: {exc}") from exc

    def infer(self, image: Image.Image, context: AdapterContext) -> list[dict[str, Any]]:
        if self.detector is None or self.sam is None:
            raise UnavailableError("grounded_sam21 is not loaded")
        import torch

        rgb = image.convert("RGB")
        width, height = rgb.size
        text = _grounding_text(context.prompts)
        inputs = self.detector_processor(images=rgb, text=text, return_tensors="pt")
        inputs = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()}
        with torch.inference_mode():
            outputs = self.detector(**inputs)
        try:
            results = self.detector_processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                threshold=context.box_threshold,
                text_threshold=context.text_threshold,
                target_sizes=[(height, width)],
            )[0]
        except TypeError:
            results = self.detector_processor.post_process_grounded_object_detection(
                outputs,
                threshold=context.box_threshold,
                target_sizes=[(height, width)],
            )[0]
        boxes = results.get("boxes")
        scores = results.get("scores")
        labels = results.get("labels") or results.get("text_labels") or []
        proposals: list[dict[str, Any]] = []
        if boxes is None:
            return proposals
        for index, box in enumerate(boxes.detach().cpu().numpy()):
            score = float(scores[index].detach().cpu()) if scores is not None else 0.0
            label = str(labels[index] if index < len(labels) else "garment")
            mask = self._segment_box(rgb, box)
            if mask is None:
                mask = mask_from_box(height, width, box)
            proposals.append(
                proposals_from_mask(
                    mask,
                    label=label,
                    score=score,
                    proposal_id=f"gdino-{index}",
                    extras={"detector": self.detector_id, "segmentor": self.segmentor_id},
                )
            )
        return filter_proposals(proposals, context.max_proposals)

    def _segment_box(self, image: Image.Image, box: np.ndarray) -> np.ndarray | None:
        import torch

        try:
            inputs = self.sam_processor(
                images=image,
                input_boxes=[[[float(v) for v in box]]],
                return_tensors="pt",
            )
            inputs = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()}
            with torch.inference_mode():
                outputs = self.sam(**inputs)
            masks = self.sam_processor.post_process_masks(
                outputs.pred_masks,
                inputs["original_sizes"],
                inputs.get("reshaped_input_sizes") or inputs.get("reshaped_input_sizes", inputs["original_sizes"]),
            )[0]
            mask = masks[0, 0].detach().cpu().numpy() > 0.5
            return mask.astype(bool)
        except Exception:
            return None

    def metadata(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "checkpoint": {"detector": self.detector_id, "segmentor": self.segmentor_id},
            "device": self.device,
            "dtype": self.dtype_name,
            "kind": "open_vocabulary_instance",
        }


def _grounding_text(prompts: list[str]) -> str:
    cleaned = [prompt.strip().rstrip(".") for prompt in prompts if prompt.strip()]
    if not cleaned:
        cleaned = ["shirt", "top"]
    return " . ".join(cleaned) + " ."
