#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a local Hugging Face DatasetDict from a validated HITL export manifest.

No upload, training, or overwrite behavior is provided.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from PIL import Image


_ALLOWED_SOURCES = {
    "ORIGINAL_OCR_OUTPUT", "HUMAN_LABEL", "AUTO_CORRECTION",
    "LEGACY_LABEL_UNVERIFIED",
}
_REQUIRED_SPLITS = ("train", "val", "test")


def validate_manifest(manifest_path: str | Path) -> tuple[Path, list[dict[str, Any]]]:
    manifest_file = Path(manifest_path).expanduser().resolve(strict=True)
    if not manifest_file.is_file():
        raise FileNotFoundError(manifest_file)
    root = manifest_file.parent
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest.get("source") != "HITL_TRAINING_READY_ONLY":
        raise ValueError("Manifest is not a HITL TRAINING_READY export")
    if manifest.get("format_version") != 1:
        raise ValueError("Unsupported HITL manifest format")
    rows = manifest.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Manifest has no dataset rows")
    if manifest.get("row_count") != len(rows):
        raise ValueError("Manifest row_count does not match rows")

    seen_ids, seen_hashes = set(), set()
    page_splits: dict[str, str] = {}
    validated = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Manifest row must be an object")
        crop_id = str(row.get("crop_id", ""))
        crop_hash = str(row.get("crop_hash", ""))
        text = str(row.get("text", "")).strip()
        page = str(row.get("page", ""))
        split = str(row.get("split", ""))
        source = str(row.get("expected_text_source", ""))
        if not crop_id or crop_id in seen_ids:
            raise ValueError("Missing or duplicate crop_id")
        if not re.fullmatch(r"[0-9a-f]{64}", crop_hash) or crop_hash in seen_hashes:
            raise ValueError("Missing, invalid, or duplicate crop SHA-256")
        if not text or not page or split not in _REQUIRED_SPLITS:
            raise ValueError("Missing corrected text/page or invalid split")
        if source not in _ALLOWED_SOURCES:
            raise ValueError("Invalid expected-text provenance")
        previous_split = page_splits.setdefault(page, split)
        if previous_split != split:
            raise ValueError("Source-page leakage: one page appears in multiple splits")

        relative = Path(str(row.get("image", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Image path must be a safe relative path")
        image_path = (root / relative).resolve(strict=True)
        if root not in image_path.parents:
            raise ValueError("Image path escapes the manifest directory")
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        try:
            with Image.open(image_path) as image:
                image.verify()
        except Exception as exc:
            raise ValueError(f"Invalid image file for crop {crop_id}") from exc
        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        if digest != crop_hash:
            raise ValueError(f"Image hash mismatch for crop {crop_id}")

        seen_ids.add(crop_id)
        seen_hashes.add(crop_hash)
        validated.append({**row, "_absolute_image": str(image_path)})

    if set(row["split"] for row in validated) != set(_REQUIRED_SPLITS):
        raise ValueError("Train, val, and test splits must all be non-empty")
    return root, validated


def build_hf_dataset(manifest_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Validate a manifest, then save a new local DatasetDict directory."""
    root, rows = validate_manifest(manifest_path)
    out = Path(output_dir).expanduser().resolve(strict=False)
    if out == root or root in out.parents or out in root.parents:
        raise ValueError("HF output directory must be separate from the source export")
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {out}")

    try:
        from datasets import Dataset, DatasetDict, Features, Image as HFImage, Value
    except ImportError as exc:
        raise RuntimeError("Install the project requirements (datasets) to build the HF dataset") from exc

    features = Features({
        "image": HFImage(),
        "text": Value("string"),
        "page": Value("string"),
    })
    splits = {}
    for split in _REQUIRED_SPLITS:
        subset = [row for row in rows if row["split"] == split]
        splits[split] = Dataset.from_dict({
            "image": [row["_absolute_image"] for row in subset],
            "text": [str(row["text"]).strip() for row in subset],
            "page": [str(row["page"]) for row in subset],
        }, features=features)

    out.parent.mkdir(parents=True, exist_ok=True)
    DatasetDict(splits).save_to_disk(str(out))
    return {
        "output_dir": str(out),
        "rows": len(rows),
        "split_counts": {split: len(splits[split]) for split in _REQUIRED_SPLITS},
        "source": "HITL_TRAINING_READY_ONLY",
        "uploaded": False,
    }


def main():
    parser = argparse.ArgumentParser(description="تحويل HITL manifest إلى HF Dataset محلي")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True, help="مجلد جديد؛ الموجود يُرفض")
    args = parser.parse_args()
    result = build_hf_dataset(args.manifest, args.out)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
