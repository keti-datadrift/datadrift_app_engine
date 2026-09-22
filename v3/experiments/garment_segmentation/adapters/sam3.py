from __future__ import annotations

from typing import Any

from PIL import Image

from adapters.base import (
    AdapterContext,
    UnavailableError,
    filter_proposals,
    from_pretrained,
    proposals_from_mask,
    resolve_device,
)


class Sam3Segmentor:
    key = "sam3"

    def __init__(
        self,
        model_id: str = "facebook/sam3",
        requested_device: str = "auto",
        download_weights: bool = False,
        prompt_list: list[str] | None = None,
    ) -> None:
        self.model_id = model_id
        self.requested_device = requested_device
        self.download_weights = download_weights
        self.prompt_list = [str(p).strip() for p in (prompt_list or []) if str(p).strip()]
        self.device = "cpu"
        self.model = None
        self.processor = None
        self.dtype_name = "float32"

    def available(self) -> bool:
        try:
            from transformers import Sam3Model, Sam3Processor  # noqa: F401
        except Exception:
            return False
        return True

    def load(self) -> None:
        import torch
        from transformers import Sam3Model, Sam3Processor

        self.device = resolve_device(self.requested_device)
        try:
            self.processor = from_pretrained(
                Sam3Processor.from_pretrained, self.model_id, download=self.download_weights
            )
            self.model = from_pretrained(Sam3Model.from_pretrained, self.model_id, download=self.download_weights)
            self.model.to(self.device)
            self.model.eval()
            self.dtype_name = str(next(self.model.parameters()).dtype).replace("torch.", "")
        except Exception as exc:
            raise UnavailableError(f"sam3 load failed: {exc}") from exc

    def infer(self, image: Image.Image, context: AdapterContext) -> list[dict[str, Any]]:
        if self.model is None or self.processor is None:
            raise UnavailableError("sam3 is not loaded")
        import numpy as np
        import torch

        rgb = image.convert("RGB")
        width, height = rgb.size
        phrases = self.prompt_list or list(context.prompts or ["top"])
        proposals: list[dict[str, Any]] = []
        for phrase in phrases:
            inputs = self.processor(images=rgb, text=phrase, return_tensors="pt")
            inputs = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()}
            with torch.inference_mode():
                outputs = self.model(**inputs)
            try:
                processed = self.processor.post_process_instance_segmentation(
                    outputs,
                    threshold=context.box_threshold,
                    mask_threshold=context.mask_threshold,
                    target_sizes=[(height, width)],
                )[0]
            except Exception:
                processed = None
            if not processed:
                continue
            masks = processed.get("masks")
            scores = processed.get("scores")
            if masks is None:
                continue
            if hasattr(masks, "detach"):
                masks = masks.detach().cpu().numpy()
            score_row = None
            if scores is not None and hasattr(scores, "detach"):
                score_row = scores.detach().cpu().numpy().reshape(-1)
            elif scores is not None:
                score_row = np.asarray(scores).reshape(-1)
            mask_list = list(masks) if getattr(masks, "ndim", 1) != 0 else []
            for index, mask in enumerate(mask_list):
                array = np.asarray(mask)
                if array.ndim == 3:
                    array = array[0]
                if not array.any():
                    continue
                score = float(score_row[index]) if score_row is not None and index < len(score_row) else 0.0
                proposals.append(
                    proposals_from_mask(
                        array > 0,
                        label=phrase,
                        score=score,
                        proposal_id=f"sam3-{phrase}-{index}",
                        extras={"checkpoint": self.model_id, "text": phrase},
                    )
                )
        return filter_proposals(proposals, context.max_proposals)

    def metadata(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "checkpoint": self.model_id,
            "device": self.device,
            "dtype": self.dtype_name,
            "kind": "open_vocabulary_instance",
            "prompts": self.prompt_list,
        }
