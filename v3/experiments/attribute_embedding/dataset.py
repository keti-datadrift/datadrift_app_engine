from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

FAMILIES = (
    "tops",
    "outerwear",
    "bottoms",
    "dresses",
    "shoes",
    "bags",
    "sport",
    "headwear",
)

SKIP_TOKENS = (
    "뷰티",
    "디지털",
    "라이프",
    "소품",
    "속옷",
    "홈웨어",
    "언더웨어",
    "이너웨어",
    "주얼리",
    "액세서리",
    "지갑",
    "파우치",
    "스카프",
    "양말",
    "벨트",
    "셋업",
    "푸드",
    "리빙",
    "키친",
    "유아",
)

LABEL_RULES = (
    ("headwear", ("모자", "볼캡", "캡", "비니", "버킷", "페도라", "베레모", "바라클라바", "트루퍼")),
    ("shoes", ("신발", "스니커즈", "부츠", "샌들", "슬리퍼", "플랫", "로퍼", "힐", "구두")),
    ("bags", ("가방", "백팩", "토트", "숄더백", "크로스백", "에코/캔버스백", "보스턴백")),
    ("sport", ("스포츠", "러닝", "요가", "필라테스", "피트니스", "수영", "등산", "하이킹", "테니스")),
    ("dresses", ("원피스", "스커트", "점프수트")),
    ("outerwear", ("아우터",)),
    ("bottoms", ("바지", "하의", "반바지")),
    ("tops", ("상의", "니트웨어", "니트")),
)

CODE_RULES = {
    "001": "tops",
    "002": "outerwear",
    "003": "bottoms",
    "004": "bags",
    "017": "sport",
    "100": "dresses",
    "103": "shoes",
    "120": "headwear",
}

WEAK_LABEL_ALIASES: dict[str, dict[str, tuple[str, ...]]] = {
    "color": {
        "black": ("black", "blk", "블랙", "검정", "흑청"),
        "white": ("white", "wht", "화이트", "아이보리", "ivory", "크림"),
        "gray": ("gray", "grey", "그레이", "회색", "차콜", "charcoal"),
        "navy": ("navy", "네이비"),
        "blue": ("blue", "블루", "파랑", "청색"),
        "red": ("red", "레드", "빨강", "버건디", "burgundy"),
        "green": ("green", "그린", "초록", "카키", "khaki"),
        "brown": ("brown", "브라운", "갈색", "베이지", "beige", "tan"),
        "yellow": ("yellow", "옐로", "노랑"),
        "pink": ("pink", "핑크"),
        "purple": ("purple", "퍼플", "보라"),
        "orange": ("orange", "오렌지"),
    },
    "pattern": {
        "solid": ("solid", "무지", "솔리드"),
        "stripe": ("stripe", "striped", "스트라이프", "단가라"),
        "check": ("check", "checked", "체크", "plaid", "깅엄"),
        "floral": ("floral", "flower", "플라워", "꽃무늬"),
        "dot": ("dot", "도트", "polka"),
        "graphic": ("graphic", "그래픽", "프린트", "print"),
        "logo": ("logo", "로고"),
        "animal": ("leopard", "zebra", "animal", "레오파드", "지브라", "애니멀"),
        "camouflage": ("camo", "camouflage", "카모", "밀리터리"),
    },
    "material": {
        "denim": ("denim", "데님", "진 "),
        "leather": ("leather", "레더", "가죽"),
        "knit": ("knit", "니트"),
        "linen": ("linen", "리넨", "린넨"),
        "wool": ("wool", "울 ", "모직"),
        "suede": ("suede", "스웨이드"),
        "corduroy": ("corduroy", "코듀로이", "골덴"),
        "fleece": ("fleece", "플리스"),
    },
    "shape": {
        "oversized": ("oversized", "oversize", "오버핏", "오버사이즈"),
        "cropped": ("cropped", "crop", "크롭"),
        "wide": ("wide", "와이드"),
        "slim": ("slim", "슬림"),
        "straight": ("straight", "스트레이트", "일자"),
        "bootcut": ("bootcut", "부츠컷"),
        "baggy": ("baggy", "배기", "벌룬"),
        "mini": ("mini", "미니"),
        "maxi": ("maxi", "맥시", "롱 "),
    },
}


def resolve_family(code: Any, label: Any) -> str:
    category_code = str(code or "").strip()
    text = str(label or "").strip()
    if any(token in text for token in SKIP_TOKENS):
        return "skip"
    for family, tokens in LABEL_RULES:
        if any(token in text for token in tokens):
            return family
    if category_code.startswith(("268103", "272103")):
        return "tops"
    if category_code.startswith(("268102", "272102")):
        return "outerwear"
    if category_code.startswith(("268106", "268107", "272104")):
        return "bottoms" if not category_code.startswith("268107") else "dresses"
    if category_code.startswith(("268104", "268115")):
        return "dresses"
    if category_code.startswith(("269", "273")):
        return "bags"
    if category_code.startswith(("270", "274")):
        return "shoes"
    if category_code.startswith(("286",)):
        return "sport"
    if category_code.startswith(("290127", "310", "311")):
        return "headwear"
    return CODE_RULES.get(category_code[:3], "skip")


def weak_labels(product: str) -> dict[str, list[str]]:
    text = f" {product.lower()} "
    result: dict[str, list[str]] = {}
    for label_type, choices in WEAK_LABEL_ALIASES.items():
        matched = [
            label
            for label, aliases in choices.items()
            if any(alias.lower() in text for alias in aliases)
        ]
        if matched:
            result[label_type] = sorted(set(matched))
    return result


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _ranking_items_rows(path: Path, *, source_alias: str) -> Iterable[dict[str, Any]]:
    payload = _read_json(path)
    for row in payload.get("items") or []:
        if not isinstance(row, dict):
            continue
        yield {
            "source": source_alias,
            "product_id": row.get("product_id"),
            "product": row.get("product") or row.get("product_clean"),
            "brand": row.get("brand"),
            "category_code": row.get("category_code"),
            "category_label": row.get("category_label"),
            "image_url": row.get("image_url"),
        }


def _survey_signal_rows(path: Path, *, source_alias: str) -> Iterable[dict[str, Any]]:
    payload = _read_json(path)
    for row in payload if isinstance(payload, list) else []:
        if not isinstance(row, dict) or row.get("entity_type") != "product":
            continue
        meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        yield {
            "source": source_alias,
            "product_id": meta.get("product_id") or meta.get("item_no"),
            "product": meta.get("product") or row.get("entity_name"),
            "brand": row.get("brand") or meta.get("brand"),
            "category_code": row.get("category_code") or meta.get("category_l2_code"),
            "category_label": row.get("category_label") or meta.get("category_l2_name"),
            "image_url": meta.get("image_url"),
        }


SOURCE_READERS = {
    "ranking_items": (
        "active/normalized.json",
        _ranking_items_rows,
    ),
    "survey_signals": (
        "active/survey_signals.json",
        _survey_signal_rows,
    ),
}


def normalize_sources(raw_sources: dict[str, Any] | None) -> dict[str, dict[str, str]]:
    """Return {alias: {disk_name, format}} from config."""

    if not isinstance(raw_sources, dict) or not raw_sources:
        raise ValueError("config.sources must map public aliases to disk channel folders")
    out: dict[str, dict[str, str]] = {}
    for alias, payload in raw_sources.items():
        if not isinstance(payload, dict):
            raise ValueError(f"invalid source entry for {alias!r}")
        disk_name = str(payload.get("disk_name") or "").strip()
        fmt = str(payload.get("format") or "").strip()
        if not alias or not disk_name or fmt not in SOURCE_READERS:
            raise ValueError(
                f"source {alias!r} needs disk_name and format in {sorted(SOURCE_READERS)}"
            )
        if disk_name.startswith("REPLACE_ME"):
            raise ValueError(
                f"source {alias!r} still has placeholder disk_name; copy config.example.yaml "
                "to config.yaml and set local channel folder names"
            )
        out[str(alias)] = {"disk_name": disk_name, "format": fmt}
    return out


def disk_to_alias(sources: dict[str, dict[str, str]]) -> dict[str, str]:
    mapping = {meta["disk_name"]: alias for alias, meta in sources.items()}
    if len(mapping) != len(sources):
        raise ValueError("source disk_name values must be unique")
    return mapping


def load_registry(
    input_root: Path,
    sources: dict[str, dict[str, str]],
) -> dict[tuple[str, str], dict[str, Any]]:
    db_path = input_root / "data" / "retail_images" / "ingest_registry.sqlite"
    uri = f"file:{db_path}?mode=ro"
    alias_by_disk = disk_to_alias(sources)
    registry: dict[tuple[str, str], dict[str, Any]] = {}
    with sqlite3.connect(uri, uri=True) as conn:
        conn.row_factory = sqlite3.Row
        for row in conn.execute(
            """
            SELECT source, product_id, content_sha256, first_seen_run_date, provenance
            FROM ingest_registry WHERE status='embedded' AND content_sha256 IS NOT NULL
            """
        ):
            item = dict(row)
            disk_name = str(item["source"])
            alias = alias_by_disk.get(disk_name)
            if not alias:
                continue
            registry[(alias, str(item["product_id"]))] = {
                **item,
                "source": alias,
                "disk_source": disk_name,
            }
    return registry


def _specificity(row: dict[str, Any]) -> tuple[int, int]:
    family = str(row.get("family") or "skip")
    code = str(row.get("category_code") or "")
    return (1 if family in FAMILIES else 0, len(code))


def _upsert(
    pools: dict[str, dict[tuple[str, str], dict[str, Any]]],
    row: dict[str, Any],
    registry: dict[tuple[str, str], dict[str, Any]],
) -> None:
    source = str(row.get("source") or "")
    product_id = str(row.get("product_id") or "")
    key = (source, product_id)
    if not source or not product_id or key not in registry:
        return
    family = resolve_family(row.get("category_code"), row.get("category_label"))
    if family not in FAMILIES:
        return
    merged = {**row, **registry[key], "family": family}
    current = pools[family].get(key)
    if current is None or _specificity(merged) > _specificity(current):
        pools[family][key] = merged


def _average_hash(path: Path) -> int:
    with Image.open(path) as image:
        pixels = list(image.convert("L").resize((8, 8)).getdata())
    mean = sum(pixels) / len(pixels)
    value = 0
    for pixel in pixels:
        value = (value << 1) | int(pixel >= mean)
    return value


def collect_candidates(
    input_root: Path,
    registry: dict[tuple[str, str], dict[str, Any]],
    *,
    sources: dict[str, dict[str, str]],
    families: tuple[str, ...],
    target_per_family: int,
) -> tuple[dict[str, dict[tuple[str, str], dict[str, Any]]], list[str]]:
    pools: dict[str, dict[tuple[str, str], dict[str, Any]]] = defaultdict(dict)
    survey_root = input_root / "runs" / "survey_days"
    days = sorted((p for p in survey_root.iterdir() if p.is_dir()), reverse=True)
    scanned_days: list[str] = []
    for day in days:
        scanned_days.append(day.name)
        for alias, meta in sources.items():
            rel = SOURCE_READERS[meta["format"]][0]
            reader = SOURCE_READERS[meta["format"]][1]
            path = day / "channels" / meta["disk_name"] / rel
            if not path.is_file():
                continue
            for row in reader(path, source_alias=alias):
                _upsert(pools, row, registry)
        if all(len(pools[family]) >= target_per_family for family in families):
            break
    return pools, scanned_days


def _rank_key(item: dict[str, Any], seed: int) -> str:
    raw = f"{seed}|{item['source']}|{item['product_id']}".encode()
    return hashlib.sha1(raw).hexdigest()


def _image_path(
    input_root: Path,
    item: dict[str, Any],
    *,
    sources: dict[str, dict[str, str]],
) -> Path | None:
    disk_name = str(item.get("disk_source") or sources[str(item["source"])]["disk_name"])
    folder = (
        input_root
        / "data"
        / "retail_images"
        / "webp"
        / disk_name
        / str(item["product_id"])
    )
    digest = str(item.get("content_sha256") or "")
    direct = folder / f"{digest[:16]}.webp"
    if direct.is_file():
        return direct
    candidates = sorted(folder.glob("*.webp")) if folder.is_dir() else []
    return candidates[0] if candidates else None


def sample_items(
    input_root: Path,
    pools: dict[str, dict[tuple[str, str], dict[str, Any]]],
    *,
    sources: dict[str, dict[str, str]],
    families: tuple[str, ...],
    sample_per_family: int,
    seed: int,
    near_duplicate_distance: int = 4,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    duplicate_counts: Counter[str] = Counter()
    missing_counts: Counter[str] = Counter()
    seen_digests: set[str] = set()
    for family in families:
        hashes: list[int] = []
        candidates = sorted(pools[family].values(), key=lambda row: _rank_key(row, seed))
        for item in candidates:
            digest = str(item.get("content_sha256") or "")
            if digest in seen_digests:
                duplicate_counts[family] += 1
                continue
            path = _image_path(input_root, item, sources=sources)
            if path is None:
                missing_counts[family] += 1
                continue
            try:
                phash = _average_hash(path)
            except Exception:
                missing_counts[family] += 1
                continue
            if any((phash ^ other).bit_count() <= near_duplicate_distance for other in hashes):
                duplicate_counts[family] += 1
                continue
            hashes.append(phash)
            seen_digests.add(digest)
            key_hash = int(_rank_key(item, seed)[:8], 16)
            selected.append(
                {
                    **item,
                    "image_path": str(path.resolve()),
                    "weak_labels": weak_labels(str(item.get("product") or "")),
                    "split": "train" if key_hash % 10 < 7 else "test",
                }
            )
            if len(hashes) >= sample_per_family:
                break
    selected.sort(key=lambda row: (str(row["family"]), str(row["source"]), str(row["product_id"])))
    fingerprint = hashlib.sha256(
        "\n".join(
            f"{row['source']}|{row['product_id']}|{row['content_sha256']}" for row in selected
        ).encode()
    ).hexdigest()
    inventory = {
        "selected": len(selected),
        "selected_by_family": dict(Counter(str(row["family"]) for row in selected)),
        "selected_by_source": dict(Counter(str(row["source"]) for row in selected)),
        "candidate_by_family": {family: len(pools[family]) for family in families},
        "near_or_exact_duplicates_skipped": dict(duplicate_counts),
        "missing_or_invalid_images": dict(missing_counts),
        "weak_label_coverage": {
            kind: sum(bool(row["weak_labels"].get(kind)) for row in selected)
            for kind in WEAK_LABEL_ALIASES
        },
        "dataset_fingerprint": fingerprint,
        "source_aliases": sorted(sources),
    }
    return selected, inventory


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
