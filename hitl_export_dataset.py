#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Safe local exporter for explicitly TRAINING_READY HITL corrections.

This exporter never reads the legacy crops.label/verified fields and never
deletes or overwrites an existing output directory. It does not upload data.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from PIL import Image

from crop_correction_store import training_rows


def _safe_file(raw_path: str, roots: tuple[Path, ...]) -> Path:
    path = Path(raw_path).expanduser().resolve(strict=True)
    if not path.is_file():
        raise FileNotFoundError(f"Training image is not a regular file: {path}")
    allowed = tuple(root.expanduser().resolve(strict=True) for root in roots)
    if not any(path == root or root in path.parents for root in allowed):
        raise ValueError(f"Training image is outside allowed crop roots: {path}")
    try:
        with Image.open(path) as image:
            image.verify()
    except Exception as exc:
        raise ValueError(f"Training crop is not a valid image: {path}") from exc
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _target_split_counts(n: int, ratios: tuple[float, float, float]) -> dict[str, int]:
    names = ("train", "val", "test")
    raw = [n * ratio for ratio in ratios]
    counts = [int(value) for value in raw]
    remaining = n - sum(counts)
    order = sorted(range(3), key=lambda i: (raw[i] - counts[i], ratios[i]), reverse=True)
    for i in order[:remaining]:
        counts[i] += 1
    if n >= 3:
        for i in range(3):
            if counts[i] == 0:
                donor = max(range(3), key=lambda j: counts[j])
                if counts[donor] <= 1:
                    raise ValueError("Cannot create non-empty train/val/test page splits")
                counts[donor] -= 1
                counts[i] = 1
    return dict(zip(names, counts))


def _read_prior_page_splits(prior_manifest: str | Path | None) -> tuple[dict[str, str], str | None]:
    if prior_manifest is None:
        return {}, None
    path = Path(prior_manifest).expanduser().resolve(strict=True)
    raw = path.read_bytes()
    manifest = json.loads(raw.decode("utf-8"))
    if manifest.get("source") != "HITL_TRAINING_READY_ONLY":
        raise ValueError("Prior manifest is not a HITL training-ready export")
    prior_rows = manifest.get("rows")
    if not isinstance(prior_rows, list) or not prior_rows:
        raise ValueError("Prior manifest has no page split records")
    mapping: dict[str, str] = {}
    for row in prior_rows:
        page, split = str(row.get("page", "")), str(row.get("split", ""))
        if not page or split not in ("train", "val", "test"):
            raise ValueError("Prior manifest contains invalid page split data")
        if page in mapping and mapping[page] != split:
            raise ValueError("Prior manifest already contains page-level leakage")
        mapping[page] = split
    if set(mapping.values()) != {"train", "val", "test"}:
        raise ValueError("Prior manifest must contain all three page splits")
    return mapping, hashlib.sha256(raw).hexdigest()


def _page_splits(
    page_names: Iterable[str],
    ratios=(0.9, 0.05, 0.05),
    seed=42,
    prior_splits: dict[str, str] | None = None,
):
    ratios = tuple(float(x) for x in ratios)
    if len(ratios) != 3 or any(x <= 0 for x in ratios):
        raise ValueError("Provide three positive split ratios")
    total = sum(ratios)
    if abs(total - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")
    pages = sorted(set(page_names))
    if len(pages) < 3:
        raise ValueError("At least three source pages are required for page-level train/val/test splits")

    prior_splits = prior_splits or {}
    invalid = {value for value in prior_splits.values()} - {"train", "val", "test"}
    if invalid:
        raise ValueError("Prior split map contains an invalid split")
    page_map = {page: prior_splits[page] for page in pages if page in prior_splits}
    if prior_splits and not page_map:
        raise ValueError("No source pages overlap the prior manifest; refusing an implicit split reset")
    new_pages = [page for page in pages if page not in page_map]
    random.Random(seed).shuffle(new_pages)

    targets = _target_split_counts(len(pages), ratios)
    counts = {
        split: sum(1 for value in page_map.values() if value == split)
        for split in ("train", "val", "test")
    }
    for page in new_pages:
        # Fill the largest remaining target deficit; stable tie-break by split order.
        split = max(
            ("train", "val", "test"),
            key=lambda name: (targets[name] - counts[name], -("train", "val", "test").index(name)),
        )
        page_map[page] = split
        counts[split] += 1

    if set(page_map.values()) != {"train", "val", "test"}:
        raise ValueError(
            "Prior manifest does not cover all required splits; add pages or start a new split plan"
        )
    return page_map


def export_training_dataset(
    db_path: str | Path,
    output_dir: str | Path,
    *,
    allowed_crop_roots: Iterable[str | Path],
    ratios=(0.9, 0.05, 0.05),
    seed: int = 42,
    prior_manifest: str | Path | None = None,
) -> dict[str, object]:
    """Export only HITL TRAINING_READY rows into a new, never-overwritten folder."""
    if isinstance(allowed_crop_roots, (str, Path)):
        allowed_crop_roots = (allowed_crop_roots,)
    roots = tuple(Path(root).expanduser().resolve(strict=True) for root in allowed_crop_roots)
    if not roots:
        raise ValueError("At least one allowed crop root is required")
    rows = training_rows(db_path)
    if not rows:
        raise ValueError("No TRAINING_READY HITL corrections are available")
    if any(not str(row.get("corrected_text") or "").strip() for row in rows):
        raise ValueError("A TRAINING_READY row has empty corrected_text")

    prepared = []
    for row in rows:
        path = _safe_file(str(row["crop_image_ref"]), roots)
        actual_hash = _sha256(path)
        if actual_hash != row["crop_hash"]:
            raise ValueError(f"SHA-256 mismatch for crop {row['crop_id']}")
        prepared.append((row, path))

    prior_page_map, prior_fingerprint = _read_prior_page_splits(prior_manifest)
    page_map = _page_splits(
        (str(row["source_image_id"]) for row, _ in prepared),
        ratios, seed, prior_splits=prior_page_map,
    )
    out = Path(output_dir).expanduser().resolve(strict=False)
    if any(out == root or root in out.parents or out in root.parents for root in roots):
        raise ValueError("Output directory must be outside all crop roots")
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive mkdir: even an empty pre-existing directory is never replaced.
    out.mkdir(exist_ok=False)
    images_dir = out / "images"
    images_dir.mkdir()

    manifest_rows = []
    for row, source_path in prepared:
        suffix = source_path.suffix.lower()
        if not suffix or len(suffix) > 10:
            suffix = ".img"
        image_name = f"{row['crop_hash']}{suffix}"
        target = images_dir / image_name
        if not target.exists():
            shutil.copy2(source_path, target)
        manifest_rows.append({
            "crop_id": row["crop_id"],
            "crop_hash": row["crop_hash"],
            "image": f"images/{image_name}",
            "text": row["corrected_text"].strip(),
            "expected_text": row["expected_text"],
            "expected_text_source": row["expected_text_source"],
            "page": row["source_image_id"],
            "split": page_map[str(row["source_image_id"])],
            "crop_config_version": row["crop_config_version"],
            "model_version": row["model_version"],
            "batch_id": row["batch_id"],
        })

    with (out / "dataset.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=[
                "crop_id", "crop_hash", "image", "text", "expected_text",
                "expected_text_source", "page", "split", "crop_config_version",
                "model_version", "batch_id",
            ]
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    counts = {
        split: sum(1 for row in manifest_rows if row["split"] == split)
        for split in ("train", "val", "test")
    }
    manifest = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "HITL_TRAINING_READY_ONLY",
        "seed": seed,
        "ratios": list(ratios),
        "split_source": "REUSED_PRIOR_MANIFEST" if prior_manifest else "SEEDED_PAGE_SPLIT",
        "prior_manifest_sha256": prior_fingerprint,
        "page_count": len(page_map),
        "row_count": len(manifest_rows),
        "split_counts": counts,
        "rows": manifest_rows,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"output_dir": str(out), "row_count": len(manifest_rows), "split_counts": counts}


def main():
    parser = argparse.ArgumentParser(description="تصدير HITL TRAINING_READY فقط إلى Dataset محلي")
    parser.add_argument("--db", required=True, help="مسار قاعدة SQLite الموجودة")
    parser.add_argument("--crop-root", action="append", required=True,
                        help="مجلد مسموح للصور؛ يمكن تكراره")
    parser.add_argument("--out", required=True, help="مجلد إخراج جديد؛ الموجود مسبقاً يُرفض")
    parser.add_argument("--ratios", type=float, nargs=3, default=(0.9, 0.05, 0.05))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prior-manifest", default=None,
                        help="اختياري: إعادة استخدام تقسيم الصفحات السابق للدفعة الثانية")
    args = parser.parse_args()
    db = Path(args.db).expanduser().resolve()
    if not db.is_file():
        parser.error("قاعدة البيانات غير موجودة؛ لن تُنشأ قاعدة جديدة.")
    result = export_training_dataset(
        db, args.out, allowed_crop_roots=args.crop_root,
        ratios=tuple(args.ratios), seed=args.seed,
        prior_manifest=args.prior_manifest,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
