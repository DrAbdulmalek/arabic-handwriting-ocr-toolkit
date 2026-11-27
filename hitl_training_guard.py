"""Pure safety helpers for local HITL fine-tuning runs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def verify_hitl_dataset(dataset_dir: str | Path) -> dict[str, Any]:
    """Require intact HITL provenance metadata and return the verified metadata."""
    root = Path(dataset_dir).expanduser().resolve(strict=True)
    metadata_path = root / "hitl_dataset_metadata.json"
    manifest_path = root / "hitl_manifest.json"
    if not metadata_path.is_file() or not manifest_path.is_file():
        raise ValueError("Dataset is missing HITL provenance metadata")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    manifest_bytes = manifest_path.read_bytes()
    actual_hash = hashlib.sha256(manifest_bytes).hexdigest()
    if metadata.get("source") != "HITL_TRAINING_READY_ONLY":
        raise ValueError("Dataset metadata is not HITL-only")
    if metadata.get("source_manifest_sha256") != actual_hash:
        raise ValueError("HITL manifest fingerprint mismatch")
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    if manifest.get("source") != "HITL_TRAINING_READY_ONLY":
        raise ValueError("Source manifest is not HITL-only")
    if manifest.get("row_count") != metadata.get("rows"):
        raise ValueError("Dataset row count differs from the source manifest")
    return {
        "source": "HITL_TRAINING_READY_ONLY",
        "source_manifest_sha256": actual_hash,
        "rows": metadata["rows"],
        "split_counts": metadata.get("split_counts", {}),
    }


def training_metrics_report(
    *,
    dataset_dir: str,
    dataset_manifest_sha256: str | None,
    init_model: str | None,
    baseline: dict[str, Any] | None,
    final: dict[str, Any],
    train_count: int,
    val_count: int,
    test_count: int,
    seed: int,
    epochs: int,
) -> dict[str, Any]:
    """Create an explicit baseline-vs-final report without claiming improvement."""
    baseline_cer = baseline.get("baseline_cer") if baseline else None
    baseline_wer = baseline.get("baseline_wer") if baseline else None
    final_cer = final.get("test_cer")
    final_wer = final.get("test_wer")
    return {
        "report_version": 1,
        "dataset_dir": str(dataset_dir),
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "init_model": init_model,
        "baseline_status": "MEASURED" if baseline else "NOT_MEASURED_NO_INIT_MODEL",
        "baseline_test_cer": baseline_cer,
        "baseline_test_wer": baseline_wer,
        "final_test_cer": final_cer,
        "final_test_wer": final_wer,
        "cer_delta_final_minus_baseline": (
            round(final_cer - baseline_cer, 6)
            if baseline_cer is not None and final_cer is not None else None
        ),
        "wer_delta_final_minus_baseline": (
            round(final_wer - baseline_wer, 6)
            if baseline_wer is not None and final_wer is not None else None
        ),
        "counts": {"train": train_count, "val": val_count, "test": test_count},
        "seed": seed,
        "epochs": epochs,
        "improvement_claimed": False,
    }


def require_new_output_paths(model_dir: str | Path, checkpoints_dir: str | Path) -> None:
    for value in (model_dir, checkpoints_dir):
        path = Path(value).expanduser().resolve(strict=False)
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing training output: {path}")
