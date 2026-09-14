from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from dataset import WEAK_LABEL_ALIASES

MODEL_PREFIX = "hf-hub:"

FAMILY_PROMPT_NAMES = {
    "tops": "top",
    "outerwear": "outerwear garment",
    "bottoms": "bottom garment",
    "dresses": "dress or skirt",
    "shoes": "shoe",
    "bags": "bag",
    "sport": "sportswear product",
    "headwear": "headwear product",
}

ATTRIBUTE_DESCRIPTIONS = {
    "black": "black",
    "white": "white or ivory",
    "gray": "gray or charcoal",
    "navy": "navy",
    "blue": "blue",
    "red": "red or burgundy",
    "green": "green or khaki",
    "brown": "brown, beige, or tan",
    "yellow": "yellow",
    "pink": "pink",
    "purple": "purple",
    "orange": "orange",
    "solid": "a solid plain pattern",
    "stripe": "a striped pattern",
    "check": "a check, plaid, or gingham pattern",
    "floral": "a floral pattern",
    "dot": "a polka dot pattern",
    "graphic": "a graphic print",
    "logo": "a visible logo pattern",
    "animal": "an animal print",
    "camouflage": "a camouflage pattern",
    "denim": "denim material",
    "leather": "leather material",
    "knit": "knitted material",
    "linen": "linen material",
    "wool": "wool material",
    "suede": "suede material",
    "corduroy": "corduroy material",
    "fleece": "fleece material",
    "oversized": "an oversized silhouette",
    "cropped": "a cropped silhouette",
    "wide": "a wide silhouette",
    "slim": "a slim silhouette",
    "straight": "a straight silhouette",
    "bootcut": "a bootcut silhouette",
    "baggy": "a baggy or balloon silhouette",
    "mini": "a mini length",
    "maxi": "a maxi or long length",
}


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)


def attribute_columns() -> list[dict[str, str]]:
    return [
        {"type": kind, "label": label}
        for kind, labels in WEAK_LABEL_ALIASES.items()
        for label in labels
    ]


def _device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_model(model_id: str, requested_device: str) -> tuple[Any, Any, Any, str]:
    import open_clip

    device = _device(requested_device)
    hub_id = model_id if model_id.startswith(MODEL_PREFIX) else f"{MODEL_PREFIX}{model_id}"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    model, preprocess = open_clip.create_model_from_pretrained(hub_id, device=device)
    tokenizer = open_clip.get_tokenizer(hub_id)
    model.eval()
    return model, preprocess, tokenizer, device


def embed_images(
    items: list[dict[str, Any]],
    *,
    model: Any,
    preprocess: Any,
    device: str,
    batch_size: int,
) -> np.ndarray:
    import torch

    vectors: list[np.ndarray] = []
    total = len(items)
    for start in range(0, total, batch_size):
        batch_items = items[start : start + batch_size]
        tensors = []
        for item in batch_items:
            with Image.open(item["image_path"]) as image:
                tensors.append(preprocess(image.convert("RGB")))
        batch = torch.stack(tensors).to(device)
        with torch.inference_mode():
            encoded = model.encode_image(batch, normalize=True)
        vectors.append(encoded.detach().float().cpu().numpy())
        done = min(start + batch_size, total)
        print(f"image embedding {done}/{total}", flush=True)
    return l2_normalize(np.concatenate(vectors, axis=0).astype(np.float32))


def embed_attribute_prompts(
    *,
    model: Any,
    tokenizer: Any,
    device: str,
    families: list[str],
) -> tuple[dict[str, np.ndarray], list[dict[str, str]], dict[str, list[str]]]:
    import torch

    columns = attribute_columns()
    vectors: dict[str, np.ndarray] = {}
    rendered: dict[str, list[str]] = {}
    for family in families:
        product_type = FAMILY_PROMPT_NAMES[family]
        prompts = [
            f"a clean ecommerce product photo of a {product_type} showing {ATTRIBUTE_DESCRIPTIONS[col['label']]}"
            for col in columns
        ]
        tokens = tokenizer(prompts).to(device)
        with torch.inference_mode():
            encoded = model.encode_text(tokens, normalize=True)
        vectors[family] = encoded.detach().float().cpu().numpy()
        rendered[family] = prompts
    return vectors, columns, rendered


def score_attributes(
    raw: np.ndarray,
    items: list[dict[str, Any]],
    text_vectors: dict[str, np.ndarray],
) -> np.ndarray:
    scores = np.empty((len(items), next(iter(text_vectors.values())).shape[0]), dtype=np.float32)
    for index, item in enumerate(items):
        scores[index] = raw[index] @ text_vectors[str(item["family"])].T
    return scores


def residualize_by_family(raw: np.ndarray, items: list[dict[str, Any]]) -> np.ndarray:
    result = raw.copy()
    families = sorted({str(item["family"]) for item in items})
    for family in families:
        family_idx = np.array([i for i, item in enumerate(items) if item["family"] == family])
        train_idx = np.array(
            [i for i in family_idx if str(items[i].get("split")) == "train"], dtype=int
        )
        if not len(train_idx):
            train_idx = family_idx
        centroid = raw[train_idx].mean(axis=0, keepdims=True)
        result[family_idx] -= centroid
    return l2_normalize(result)


def standardize_prompt_scores(scores: np.ndarray, items: list[dict[str, Any]]) -> np.ndarray:
    result = np.zeros_like(scores, dtype=np.float32)
    for family in sorted({str(item["family"]) for item in items}):
        idx = np.array([i for i, item in enumerate(items) if item["family"] == family])
        block = scores[idx]
        result[idx] = (block - block.mean(axis=0)) / np.clip(block.std(axis=0), 1e-6, None)
    return l2_normalize(result)


def create_representations(
    items: list[dict[str, Any]],
    *,
    run_dir: Path,
    model_id: str,
    model_revision: str,
    requested_device: str,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    cache_dir = run_dir / "representations"
    cache_dir.mkdir(parents=True, exist_ok=True)
    raw_path = cache_dir / "A_raw.npz"
    keys = np.asarray([f"{row['source']}:{row['product_id']}" for row in items])
    model = preprocess = tokenizer = None
    device = _device(requested_device)
    if raw_path.is_file():
        cached = np.load(raw_path)
        if np.array_equal(cached["keys"], keys):
            raw = cached["matrix"].astype(np.float32)
        else:
            raw_path.unlink()
            raw = None
    else:
        raw = None
    if raw is None:
        model, preprocess, tokenizer, device = load_model(model_id, requested_device)
        raw = embed_images(
            items,
            model=model,
            preprocess=preprocess,
            device=device,
            batch_size=batch_size,
        )
        np.savez_compressed(
            raw_path,
            matrix=raw.astype(np.float16),
            keys=keys,
            model_id=model_id,
            model_revision=model_revision,
        )

    prompt_path = cache_dir / "D_prompt_scores.npz"
    columns = attribute_columns()
    prompts: dict[str, list[str]]
    if prompt_path.is_file():
        cached = np.load(prompt_path)
        if np.array_equal(cached["keys"], keys):
            prompt_scores = cached["raw_scores"].astype(np.float32)
            prompts = json.loads(str(cached["prompts"].item()))
        else:
            prompt_path.unlink()
            prompt_scores = None
            prompts = {}
    else:
        prompt_scores = None
        prompts = {}
    if prompt_scores is None:
        if model is None:
            model, preprocess, tokenizer, device = load_model(model_id, requested_device)
        text_vectors, columns, prompts = embed_attribute_prompts(
            model=model,
            tokenizer=tokenizer,
            device=device,
            families=sorted({str(item["family"]) for item in items}),
        )
        prompt_scores = score_attributes(raw, items, text_vectors)
        np.savez_compressed(
            prompt_path,
            raw_scores=prompt_scores.astype(np.float16),
            keys=keys,
            columns=json.dumps(columns, ensure_ascii=False),
            prompts=json.dumps(prompts, ensure_ascii=False),
        )

    residual = residualize_by_family(raw, items)
    prompt_representation = standardize_prompt_scores(prompt_scores, items)
    np.savez_compressed(cache_dir / "B_category_conditioned.npz", matrix=raw.astype(np.float16), keys=keys)
    np.savez_compressed(cache_dir / "C_family_residual.npz", matrix=residual.astype(np.float16), keys=keys)
    np.savez_compressed(
        cache_dir / "D_attribute_scores.npz",
        matrix=prompt_representation.astype(np.float16),
        raw_scores=prompt_scores.astype(np.float16),
        keys=keys,
    )
    metadata = {
        "model_id": model_id,
        "model_revision": model_revision,
        "device": device,
        "dimension": int(raw.shape[1]),
        "attribute_columns": columns,
        "prompts": prompts,
    }
    return {
        "A": raw,
        "B": raw.copy(),
        "C": residual,
        "D": prompt_representation,
    }, metadata
