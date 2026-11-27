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

from crop_correction_store import training_rows


def _safe_file(raw_path: str, roots: tuple[Path, ...]) -> Path:
    path = Path(raw_path).expanduser().resolve(strict=True)
    if not path.is_file():
        raise FileNotFoundError(f"Training image is not a regular file: {path}")
    allowed = tuple(root.expanduser().resolve(strict=True) for root in roots)
    if not any(path == root or root in path.parents for root in allowed):
        raise ValueError(f"Training image is outside allowed crop roots: {path}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _page_splits(page_names: Iterable[str], ratios=(0.9, 0.05, 0.05), seed=42):
    ratios = tuple(float(x) for x in ratios)
    if len(ratios) != 3 or any(x <= 0 for x in ratios):
        raise ValueError("Provide three positive split ratios")
    total = sum(ratios)
    if abs(total - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")
    pages = sorted(set(page_names))
    if len(pages) < 3:
        raise ValueError("At least three source pages are required for page-level train/val/test splits")
    random.Random(seed).shuffle(pages)
    n = len(pages)
    n_train = max(1, int(n * ratios[0]))
    n_val = max(1, int(n * ratios[1]))
    if n_train + n_val >= n:
        n_train = n - 2
        n_val = 1
    return {
        **{page: "train" for page in pages[:n_train]},
        **{page: "val" for page in pages[n_train:n_train + n_val]},
        **{page: "test" for page in pages[n_train + n_val:]},
    }


def export_training_dataset(
    db_path: str | Path,
    output_dir: str | Path,
    *,
    allowed_crop_roots: Iterable[str | Path],
    ratios=(0.9, 0.05, 0.05),
    seed: int = 42,
) -> dict[str, object]:
    """Export only HITL TRAINING_READY rows into a new, never-overwritten folder."""
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

    page_map = _page_splits((str(row["source_image_id"]) for row, _ in prepared), ratios, seed)
    out = Path(output_dir).expanduser().resolve(strict=False)
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
            "page": row["source_image_id"],
            "split": page_map[str(row["source_image_id"])],
        })

    with (out / "dataset.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["crop_id", "crop_hash", "image", "text", "page", "split"]
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
    args = parser.parse_args()
    db = Path(args.db).expanduser().resolve()
    if not db.is_file():
        parser.error("قاعدة البيانات غير موجودة؛ لن تُنشأ قاعدة جديدة.")
    result = export_training_dataset(
        db, args.out, allowed_crop_roots=args.crop_root,
        ratios=tuple(args.ratios), seed=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
