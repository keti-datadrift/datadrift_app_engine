from __future__ import annotations

from typing import Any

# License classes are recorded from official model cards / repositories as of 2026-09-18.
# Code license and checkpoint license are stored separately. Public weights are not
# automatically treated as OSI open source.

LICENSE_CATALOG: dict[str, dict[str, Any]] = {
    "grounded_sam21": {
        "display_name": "Grounding DINO + SAM 2.1",
        "components": [
            {
                "name": "Grounding DINO",
                "code_license": "Apache-2.0",
                "weight_license": "Apache-2.0",
                "source": "https://github.com/IDEA-Research/GroundingDINO",
            },
            {
                "name": "SAM 2.1",
                "code_license": "Apache-2.0",
                "weight_license": "Apache-2.0",
                "source": "https://github.com/facebookresearch/sam2",
            },
        ],
        "eligibility": "adoptable",
        "notes": "Local Grounding DINO + SAM 2.1 weights are Apache 2.0. API-only DINO-X / Grounding DINO 1.5 paths are out of scope.",
    },
    "sam3": {
        "display_name": "SAM 3.1",
        "components": [
            {
                "name": "SAM 3 / 3.1",
                "code_license": "SAM License (custom)",
                "weight_license": "SAM License (custom, gated checkpoint)",
                "source": "https://github.com/facebookresearch/sam3/blob/main/LICENSE",
            }
        ],
        "eligibility": "research-only",
        "notes": "Not OSI open source. Redistribution and use-case restrictions apply; do not treat as Apache/MIT.",
    },
    "fashionpedia": {
        "display_name": "Fashionpedia instance segmentation",
        "components": [
            {
                "name": "Fashionpedia Attribute-Mask R-CNN (original TPU)",
                "code_license": "Apache-2.0",
                "weight_license": "unspecified / verify before commercial use",
                "source": "https://github.com/tensorflow/tpu/tree/master/models/official/detection/projects/fashionpedia",
            },
            {
                "name": "YOLOS-Fashionpedia (transformers adapter)",
                "code_license": "Apache-2.0",
                "weight_license": "verify model card",
                "source": "https://huggingface.co/valentinafeve/yolos-fashionpedia",
            },
        ],
        "eligibility": "research-only",
        "notes": "Taxonomy is Fashionpedia. Original checkpoint license is not clearly OSI; keep research-only until legal review.",
    },
    "schp": {
        "display_name": "Human parsing (SCHP-ATR family)",
        "components": [
            {
                "name": "SCHP",
                "code_license": "MIT",
                "weight_license": "unspecified Google Drive weights",
                "source": "https://github.com/PeikeLi/Self-Correction-Human-Parsing",
            },
            {
                "name": "SegFormer B2 clothes (ATR-like labels)",
                "code_license": "verify model card",
                "weight_license": "verify model card",
                "source": "https://huggingface.co/mattmdjaga/segformer_b2_clothes",
            },
        ],
        "eligibility": "research-only",
        "notes": "Semantic parsing, not native instance AP. Compare only on end-to-end target mask quality.",
    },
    "owlv2_sam": {
        "display_name": "OWLv2 + SAM 2.1",
        "components": [
            {
                "name": "OWLv2",
                "code_license": "Apache-2.0",
                "weight_license": "Apache-2.0",
                "source": "https://huggingface.co/google/owlv2-base-patch16-ensemble",
            },
            {
                "name": "SAM 2.1",
                "code_license": "Apache-2.0",
                "weight_license": "Apache-2.0",
                "source": "https://github.com/facebookresearch/sam2",
            },
        ],
        "eligibility": "adoptable",
        "notes": "CUDA-friendly open-vocabulary fallback when Grounding DINO or SAM 3 is unavailable.",
    },
    "dummy": {
        "display_name": "Deterministic dummy segmentor",
        "components": [],
        "eligibility": "adoptable",
        "notes": "Synthetic boxes for pipeline tests. Not a benchmark candidate.",
    },
    "yolo": {
        "display_name": "YOLOS-Fashionpedia",
        "components": [
            {
                "name": "YOLOS-Fashionpedia",
                "code_license": "Apache-2.0",
                "weight_license": "verify model card",
                "source": "https://huggingface.co/valentinafeve/yolos-fashionpedia",
            }
        ],
        "eligibility": "research-only",
        "notes": "ViT 기반 패션 검출기(YOLOS). Ultralytics YOLOv8이 아니라 Fashionpedia 46-class detector이며, 마스크는 박스 근사다.",
    },
    "segformer": {
        "display_name": "SegFormer B2 clothes",
        "components": [
            {
                "name": "SegFormer B2 clothes (ATR-like labels)",
                "code_license": "verify model card",
                "weight_license": "verify model card",
                "source": "https://huggingface.co/mattmdjaga/segformer_b2_clothes",
            }
        ],
        "eligibility": "research-only",
        "notes": "의류 semantic parsing. 같은 클래스의 여러 벌은 connected component로 나눈다.",
    },
    "clipseg": {
        "display_name": "CLIPSeg",
        "components": [
            {
                "name": "CLIPSeg rd64-refined",
                "code_license": "Apache-2.0",
                "weight_license": "Apache-2.0 / model card",
                "source": "https://huggingface.co/CIDAS/clipseg-rd64-refined",
            }
        ],
        "eligibility": "adoptable",
        "notes": "텍스트 프롬프트 CLIP 세그멘테이션. SAM 없이 히트맵 마스크를 만든다.",
    },
}


def license_record(model_key: str) -> dict[str, Any]:
    if model_key not in LICENSE_CATALOG:
        raise KeyError(f"unknown model key: {model_key}")
    record = LICENSE_CATALOG[model_key]
    return {
        "model_key": model_key,
        "display_name": record["display_name"],
        "eligibility": record["eligibility"],
        "components": record["components"],
        "notes": record["notes"],
        "code_licenses": sorted({c.get("code_license", "") for c in record["components"] if c}),
        "weight_licenses": sorted({c.get("weight_license", "") for c in record["components"] if c}),
    }


def catalog_payload(model_keys: list[str]) -> dict[str, Any]:
    return {key: license_record(key) for key in model_keys}
