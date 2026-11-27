"""Read-only adapter for legacy crops rows.

This adapter never migrates or writes to the legacy database. Legacy labels
are deliberately classified as unverified provenance: neither verified nor
locked proves that a label is the original OCR prediction or that a crop is
geometrically sound.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable


class LegacyAdapterError(RuntimeError):
    """Raised when the legacy database cannot be inspected safely."""


def _readonly_connection(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Legacy database does not exist: {path}")
    uri = path.resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=5.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    return conn


def _path_status(raw_path: str, allowed_roots: tuple[Path, ...]) -> str:
    """Check path existence and containment without opening image contents."""
    if not raw_path:
        return "MISSING_REFERENCE"
    try:
        candidate = Path(raw_path).expanduser().resolve(strict=False)
        if not allowed_roots:
            return "ROOT_NOT_CONFIGURED"
        roots = tuple(root.expanduser().resolve(strict=False) for root in allowed_roots)
        if not any(candidate == root or root in candidate.parents for root in roots):
            return "OUTSIDE_ALLOWED_ROOTS"
        return "PRESENT" if candidate.is_file() else "FILE_NOT_FOUND"
    except (OSError, RuntimeError, ValueError):
        return "INVALID_PATH"


def read_legacy_candidates(
    db_path: str | Path,
    *,
    allowed_crop_roots: Iterable[str | Path] = (),
    limit: int = 500,
) -> list[dict[str, object]]:
    """Return legacy rows as review candidates; never register them in HITL."""
    if not 1 <= limit <= 10000:
        raise ValueError("limit must be between 1 and 10000")
    roots = tuple(Path(root) for root in allowed_crop_roots)
    conn = _readonly_connection(db_path)
    try:
        table = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='crops'"
        ).fetchone()
        if table is None:
            raise LegacyAdapterError("Legacy crops table is missing")
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(crops)")}
        required = {"id", "page_name", "crop_path", "label"}
        missing = sorted(required - columns)
        if missing:
            raise LegacyAdapterError("Legacy crops table lacks required columns: " + ", ".join(missing))

        optional = {name for name in ("ignored", "verified", "confidence", "language",
                                      "x1", "y1", "x2", "y2", "split") if name in columns}
        select_columns = ["id", "page_name", "crop_path", "label"] + sorted(optional)
        has_hitl = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='hitl_crop_corrections'"
        ).fetchone()
        sql = "SELECT " + ", ".join(f'"{name}"' for name in select_columns) + " FROM crops"
        conditions = []
        if "ignored" in columns:
            conditions.append("COALESCE(ignored, 0) = 0")
        if has_hitl:
            conditions.append(
                "NOT EXISTS (SELECT 1 FROM hitl_crop_corrections h "
                "WHERE h.crop_id = 'legacy:' || CAST(crops.id AS TEXT))"
            )
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        sql += " ORDER BY id LIMIT ?"
        rows = conn.execute(sql, (limit,)).fetchall()

        result: list[dict[str, object]] = []
        for row in rows:
            crop_id = f"legacy:{row['id']}"
            raw_path = str(row["crop_path"] or "")
            result.append({
                "crop_id": crop_id,
                "source_image_id": str(row["page_name"] or ""),
                "crop_image_ref": raw_path,
                "path_status": _path_status(raw_path, roots),
                "expected_text": str(row["label"] or ""),
                "expected_text_source": "LEGACY_LABEL_UNVERIFIED",
                "legacy_verified_flag": row["verified"] if "verified" in optional else None,
                "legacy_confidence": row["confidence"] if "confidence" in optional else None,
                "language": row["language"] if "language" in optional else None,
                "split": row["split"] if "split" in optional else None,
                "bounding_box": {
                    key: row[key] for key in ("x1", "y1", "x2", "y2") if key in optional
                },
                "quality_decision_required": True,
                "eligible_for_transcription": False,
                "registration_allowed": False,
                "note": (
                    "Legacy label provenance is unknown; inspect the crop, "
                    "confirm crop quality, and establish OCR-prediction provenance "
                    "before explicit HITL registration."
                ),
            })
        return result
    finally:
        conn.close()
