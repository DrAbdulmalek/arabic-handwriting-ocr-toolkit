"""Human-in-the-loop crop correction store.

This module adds isolated HITL tables to an existing SQLite database.
It never modifies the legacy crops table and never reads image contents.
Call initialize() explicitly; no schema changes happen on import.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional


SCHEMA_VERSION = 1
GOOD_CROP = "GOOD_CROP"
BAD_CROP = "BAD_CROP"
PENDING = "PENDING"
IN_PROGRESS = "IN_PROGRESS"
CORRECTED = "CORRECTED"
SKIPPED = "SKIPPED"
REJECTED = "REJECTED"
VALIDATED = "VALIDATED"
TRAINING_READY = "TRAINING_READY"

_ALLOWED_QUALITY = {GOOD_CROP, BAD_CROP}
_ALLOWED_STATUSES = {PENDING, IN_PROGRESS, CORRECTED, SKIPPED, REJECTED, VALIDATED, TRAINING_READY}


class CorrectionStoreError(Exception):
    """Base error for correction-store operations."""


class DuplicateCropError(CorrectionStoreError):
    """A crop identity or exact image hash is already registered."""


class EligibilityError(CorrectionStoreError):
    """The crop does not satisfy the correction/training gate."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _connect(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def initialize(db_path: str | Path) -> None:
    """Create HITL tables in an existing or new SQLite DB; legacy tables untouched."""
    path = Path(db_path)
    if not path.parent.exists():
        raise FileNotFoundError(f"Database parent does not exist: {path.parent}")
    with _connect(path) as conn:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS hitl_schema_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS hitl_crop_corrections (
                crop_id TEXT PRIMARY KEY,
                crop_hash TEXT NOT NULL UNIQUE,
                crop_image_ref TEXT NOT NULL,
                source_image_id TEXT NOT NULL,
                expected_text TEXT NOT NULL,
                corrected_text TEXT,
                crop_quality TEXT NOT NULL CHECK(crop_quality IN ('GOOD_CROP','BAD_CROP')),
                quality_evidence TEXT NOT NULL,
                correction_status TEXT NOT NULL DEFAULT 'PENDING'
                    CHECK(correction_status IN
                    ('PENDING','IN_PROGRESS','CORRECTED','SKIPPED','REJECTED','VALIDATED','TRAINING_READY')),
                bounding_box_json TEXT,
                crop_config_version TEXT NOT NULL,
                model_version TEXT NOT NULL,
                confidence REAL,
                batch_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                corrected_at TEXT,
                CHECK(confidence IS NULL OR (confidence >= 0.0 AND confidence <= 1.0)),
                CHECK(corrected_text IS NULL OR length(trim(corrected_text)) > 0),
                CHECK(crop_quality != 'BAD_CROP' OR correction_status IN ('REJECTED','SKIPPED'))
            );

            CREATE TABLE IF NOT EXISTS hitl_correction_revisions (
                revision_id INTEGER PRIMARY KEY AUTOINCREMENT,
                crop_id TEXT NOT NULL REFERENCES hitl_crop_corrections(crop_id),
                previous_text TEXT,
                new_text TEXT NOT NULL,
                actor TEXT NOT NULL,
                reason TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS hitl_batch_exports (
                export_id TEXT PRIMARY KEY,
                batch_id TEXT NOT NULL,
                format TEXT NOT NULL,
                manifest_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_hitl_status_quality
                ON hitl_crop_corrections(crop_quality, correction_status);
            CREATE INDEX IF NOT EXISTS idx_hitl_source_image
                ON hitl_crop_corrections(source_image_id);
            CREATE INDEX IF NOT EXISTS idx_hitl_batch
                ON hitl_crop_corrections(batch_id);

            CREATE TRIGGER IF NOT EXISTS hitl_identity_immutable
            BEFORE UPDATE OF crop_id, crop_hash, crop_image_ref, source_image_id,
                             expected_text, bounding_box_json, crop_config_version,
                             model_version, batch_id
            ON hitl_crop_corrections
            BEGIN
                SELECT RAISE(ABORT, 'HITL crop identity and original prediction are immutable');
            END;
            """
        )
        existing = conn.execute(
            "SELECT value FROM hitl_schema_meta WHERE key='schema_version'"
        ).fetchone()
        if existing and int(existing["value"]) > SCHEMA_VERSION:
            raise CorrectionStoreError("Database HITL schema is newer than this module")
        conn.execute(
            "INSERT INTO hitl_schema_meta(key,value) VALUES('schema_version',?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (str(SCHEMA_VERSION),),
        )


def register_crop(
    db_path: str | Path,
    *,
    crop_id: str,
    crop_hash: str,
    crop_image_ref: str,
    source_image_id: str,
    expected_text: str,
    crop_quality: str,
    quality_evidence: Mapping[str, Any],
    crop_config_version: str,
    model_version: str,
    batch_id: str,
    bounding_box: Optional[Mapping[str, Any]] = None,
    confidence: Optional[float] = None,
) -> None:
    """Register one crop. Only GOOD_CROP is eligible for human transcription."""
    if crop_quality not in _ALLOWED_QUALITY:
        raise ValueError(f"Unsupported crop_quality: {crop_quality}")
    required = {
        "crop_id": crop_id, "crop_hash": crop_hash, "crop_image_ref": crop_image_ref,
        "source_image_id": source_image_id, "expected_text": expected_text,
        "crop_config_version": crop_config_version, "model_version": model_version,
        "batch_id": batch_id,
    }
    missing = [key for key, value in required.items() if not isinstance(value, str) or not value.strip()]
    if missing:
        raise ValueError("Missing required fields: " + ", ".join(missing))
    if not isinstance(quality_evidence, Mapping) or not quality_evidence:
        raise ValueError("quality_evidence must document the crop-quality decision")
    if confidence is not None and not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0 and 1")

    now = _utc_now()
    status = PENDING if crop_quality == GOOD_CROP else REJECTED
    try:
        with _connect(db_path) as conn:
            conn.execute(
                """INSERT INTO hitl_crop_corrections
                (crop_id,crop_hash,crop_image_ref,source_image_id,expected_text,
                 crop_quality,quality_evidence,correction_status,bounding_box_json,
                 crop_config_version,model_version,confidence,batch_id,created_at,updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    crop_id, crop_hash, crop_image_ref, source_image_id, expected_text,
                    crop_quality, json.dumps(dict(quality_evidence), ensure_ascii=False, sort_keys=True),
                    status, json.dumps(dict(bounding_box), ensure_ascii=False, sort_keys=True)
                    if bounding_box is not None else None,
                    crop_config_version, model_version, confidence, batch_id, now, now,
                ),
            )
    except sqlite3.IntegrityError as exc:
        if "UNIQUE" in str(exc):
            raise DuplicateCropError("crop_id or exact crop_hash already registered") from exc
        raise


def list_pending(db_path: str | Path, *, batch_id: Optional[str] = None, limit: int = 500) -> list[dict[str, Any]]:
    """Return only quality-approved crops not previously corrected/rejected."""
    if limit < 1 or limit > 10000:
        raise ValueError("limit must be between 1 and 10000")
    sql = """SELECT crop_id,crop_hash,crop_image_ref,source_image_id,expected_text,
                    crop_quality,correction_status,bounding_box_json,crop_config_version,
                    model_version,confidence,batch_id
             FROM hitl_crop_corrections
             WHERE crop_quality='GOOD_CROP' AND correction_status IN ('PENDING','IN_PROGRESS')"""
    params: list[Any] = []
    if batch_id is not None:
        sql += " AND batch_id=?"
        params.append(batch_id)
    sql += " ORDER BY created_at,crop_id LIMIT ?"
    params.append(limit)
    with _connect(db_path) as conn:
        return [dict(row) for row in conn.execute(sql, params).fetchall()]


def save_correction(
    db_path: str | Path, crop_id: str, corrected_text: str, *,
    actor: str = "human", reason: Optional[str] = None,
) -> None:
    """Save a human correction and append an immutable revision record."""
    if not corrected_text or not corrected_text.strip():
        raise ValueError("corrected_text must not be empty")
    now = _utc_now()
    with _connect(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT expected_text,corrected_text,crop_quality,correction_status "
            "FROM hitl_crop_corrections WHERE crop_id=?", (crop_id,)
        ).fetchone()
        if row is None:
            raise KeyError(crop_id)
        if row["crop_quality"] != GOOD_CROP:
            raise EligibilityError("BAD_CROP cannot receive a transcription")
        if row["correction_status"] in (REJECTED, SKIPPED):
            raise EligibilityError("Skipped/rejected crop requires an explicit new review workflow")
        conn.execute(
            """UPDATE hitl_crop_corrections
               SET corrected_text=?, correction_status='CORRECTED',
                   corrected_at=?, updated_at=? WHERE crop_id=?""",
            (corrected_text.strip(), now, now, crop_id),
        )
        conn.execute(
            """INSERT INTO hitl_correction_revisions
               (crop_id,previous_text,new_text,actor,reason,created_at)
               VALUES (?,?,?,?,?,?)""",
            (crop_id, row["corrected_text"], corrected_text.strip(), actor, reason, now),
        )


def set_status(db_path: str | Path, crop_id: str, status: str) -> None:
    if status not in _ALLOWED_STATUSES - {CORRECTED, VALIDATED, TRAINING_READY}:
        raise ValueError("Status is not allowed through this operation")
    now = _utc_now()
    with _connect(db_path) as conn:
        cur = conn.execute(
            """UPDATE hitl_crop_corrections SET correction_status=?,updated_at=?
               WHERE crop_id=? AND crop_quality='GOOD_CROP' AND correction_status IN ('PENDING','IN_PROGRESS')""",
            (status, now, crop_id),
        )
        if cur.rowcount != 1:
            raise EligibilityError("Crop missing, not eligible, or no longer pending")


def validate_correction(db_path: str | Path, crop_id: str) -> None:
    """Explicit second gate before a corrected sample can enter training."""
    now = _utc_now()
    with _connect(db_path) as conn:
        cur = conn.execute(
            """UPDATE hitl_crop_corrections SET correction_status='VALIDATED',updated_at=?
               WHERE crop_id=? AND crop_quality='GOOD_CROP'
                 AND correction_status='CORRECTED' AND corrected_text IS NOT NULL""",
            (now, crop_id),
        )
        if cur.rowcount != 1:
            raise EligibilityError("Only a corrected GOOD_CROP can be validated")


def mark_training_ready(db_path: str | Path, crop_id: str) -> None:
    now = _utc_now()
    with _connect(db_path) as conn:
        cur = conn.execute(
            """UPDATE hitl_crop_corrections SET correction_status='TRAINING_READY',updated_at=?
               WHERE crop_id=? AND crop_quality='GOOD_CROP'
                 AND correction_status='VALIDATED' AND corrected_text IS NOT NULL""",
            (now, crop_id),
        )
        if cur.rowcount != 1:
            raise EligibilityError("Only a validated correction can enter the training-ready set")


def training_rows(db_path: str | Path, *, batch_id: Optional[str] = None) -> list[dict[str, Any]]:
    """Return only validated training-ready human corrections."""
    sql = """SELECT crop_id,crop_hash,crop_image_ref,source_image_id,corrected_text,
                    bounding_box_json,crop_config_version,model_version,batch_id
             FROM hitl_crop_corrections
             WHERE crop_quality='GOOD_CROP' AND correction_status='TRAINING_READY'
               AND corrected_text IS NOT NULL"""
    params: tuple[Any, ...] = ()
    if batch_id is not None:
        sql += " AND batch_id=?"
        params = (batch_id,)
    sql += " ORDER BY source_image_id,crop_id"
    with _connect(db_path) as conn:
        return [dict(row) for row in conn.execute(sql, params).fetchall()]


def correction_history(db_path: str | Path, crop_id: str) -> list[dict[str, Any]]:
    with _connect(db_path) as conn:
        return [dict(row) for row in conn.execute(
            """SELECT revision_id,previous_text,new_text,actor,reason,created_at
               FROM hitl_correction_revisions WHERE crop_id=? ORDER BY revision_id""",
            (crop_id,),
        ).fetchall()]
