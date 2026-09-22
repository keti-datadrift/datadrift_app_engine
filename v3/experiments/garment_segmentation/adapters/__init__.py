from __future__ import annotations

from adapters.clipseg import ClipSegSegmentor
from adapters.dummy import DummySegmentor
from adapters.fashionpedia import FashionpediaSegmentor
from adapters.grounded_sam import GroundedSamSegmentor
from adapters.owlv2_sam import Owlv2SamSegmentor
from adapters.sam3 import Sam3Segmentor
from adapters.schp import SchpSegmentor

ADAPTERS = {
    "dummy": DummySegmentor,
    "grounded_sam21": GroundedSamSegmentor,
    "sam3": Sam3Segmentor,
    "fashionpedia": FashionpediaSegmentor,
    "yolo": FashionpediaSegmentor,
    "schp": SchpSegmentor,
    "segformer": SchpSegmentor,
    "clipseg": ClipSegSegmentor,
    "owlv2_sam": Owlv2SamSegmentor,
}

DEFAULT_MODELS = ["yolo", "segformer", "clipseg"]
OPTIONAL_MODELS = ["owlv2_sam"]
SAM_FAMILY_MODELS = ["grounded_sam21", "sam3", "owlv2_sam"]
CHECKPOINT_ALIASES = {
    "yolo": "fashionpedia",
    "segformer": "schp",
}


def create_adapter(
    key: str,
    requested_device: str = "auto",
    checkpoints: dict | None = None,
    download_weights: bool = False,
):
    if key not in ADAPTERS:
        raise KeyError(f"unknown adapter: {key}")
    mapping = checkpoints or {}
    spec = mapping.get(key)
    if spec is None and key in CHECKPOINT_ALIASES:
        spec = mapping.get(CHECKPOINT_ALIASES[key])
    if key == "dummy":
        adapter = DummySegmentor()
        adapter.key = key
        return adapter
    if key == "grounded_sam21":
        kwargs = {"requested_device": requested_device, "download_weights": download_weights}
        if isinstance(spec, dict):
            if spec.get("detector"):
                kwargs["detector_id"] = spec["detector"]
            if spec.get("segmentor"):
                kwargs["segmentor_id"] = spec["segmentor"]
        adapter = GroundedSamSegmentor(**kwargs)
        adapter.key = key
        return adapter
    if key == "owlv2_sam":
        kwargs = {"requested_device": requested_device, "download_weights": download_weights}
        if isinstance(spec, dict):
            if spec.get("detector"):
                kwargs["detector_id"] = spec["detector"]
            if spec.get("segmentor"):
                kwargs["segmentor_id"] = spec["segmentor"]
        adapter = Owlv2SamSegmentor(**kwargs)
        adapter.key = key
        return adapter
    kwargs = {"requested_device": requested_device, "download_weights": download_weights}
    if isinstance(spec, str):
        kwargs["model_id"] = spec
    elif isinstance(spec, dict) and key == "sam3":
        if spec.get("model_id"):
            kwargs["model_id"] = spec["model_id"]
        if spec.get("prompts"):
            kwargs["prompt_list"] = list(spec["prompts"])
    adapter = ADAPTERS[key](**kwargs)
    adapter.key = key
    return adapter
