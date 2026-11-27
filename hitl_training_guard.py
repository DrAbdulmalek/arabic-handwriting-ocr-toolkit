"""Pure safety helpers for local HITL fine-tuning runs."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
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
    rows = manifest.get("rows")
    if not isinstance(rows, list) or manifest.get("row_count") != metadata.get("rows") or len(rows) != metadata.get("rows"):
        raise ValueError("Dataset row count differs from the source manifest")
    page_splits = {}
    counts = {"train": 0, "val": 0, "test": 0}
    for row in rows:
        page, split = str(row.get("page", "")), str(row.get("split", ""))
        if not page or split not in counts:
            raise ValueError("Manifest contains an invalid page or split")
        if page in page_splits and page_splits[page] != split:
            raise ValueError("Manifest contains page-level train/test leakage")
        page_splits[page] = split
        counts[split] += 1
    if any(value == 0 for value in counts.values()):
        raise ValueError("Manifest must contain non-empty train/val/test splits")
    if counts != metadata.get("split_counts"):
        raise ValueError("Dataset split counts differ from the source manifest")
    return {
        "source": "HITL_TRAINING_READY_ONLY",
        "source_manifest_sha256": actual_hash,
        "rows": metadata["rows"],
        "split_counts": metadata.get("split_counts", {}),
    }


def verify_datasetdict_content(
    dataset_dict: Any,
    dataset_dir: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify every loaded DatasetDict row against the approved HITL manifest.

    ``verify_hitl_dataset`` only proves the manifest *file* is intact. This
    closes the remaining provenance gap: it proves the rows that will actually
    be trained on — texts, raw image bytes, pages, and split membership — match
    the approved manifest row by row, so any later edit of the saved DatasetDict
    (altered correction text, swapped image bytes, a row moved between
    train/val/test while keeping counts equal) is rejected before training.

    Rows are compared as multisets of ``(page, text, image_sha256)`` per split:
    content equality is what matters, not row order. Error messages name the
    split, row index, and failing component only — never text values.

    Raises ValueError on the first mismatch; RuntimeError if ``datasets`` is
    not installed.
    """
    try:
        from datasets import Image as HFImage
    except ImportError as exc:
        raise RuntimeError(
            "Install the project requirements (datasets) to verify DatasetDict content"
        ) from exc

    dataset_dir_path = Path(dataset_dir).expanduser().resolve(strict=True)
    if metadata is None:
        metadata = verify_hitl_dataset(dataset_dir_path)
    manifest = json.loads(
        (dataset_dir_path / "hitl_manifest.json").read_bytes().decode("utf-8")
    )
    required_splits = ("train", "val", "test")
    expected: dict[str, Counter[tuple[str, str, str]]] = {s: Counter() for s in required_splits}
    page_split: dict[str, str] = {}
    for row in manifest.get("rows", []):
        page, split = str(row.get("page", "")), str(row.get("split", ""))
        expected[split][(page, str(row.get("text", "")).strip(), str(row.get("crop_hash", "")))] += 1
        page_split.setdefault(page, split)

    missing = [s for s in required_splits if s not in dataset_dict]
    extra = [s for s in dataset_dict if s not in required_splits]
    if missing or extra:
        raise ValueError(f"DatasetDict splits differ from HITL manifest (missing={missing}, extra={extra})")

    actual: dict[str, Counter[tuple[str, str, str]]] = {}
    for split in required_splits:
        split_ds = dataset_dict[split].cast_column("image", HFImage(decode=False))
        actual[split] = Counter()
        for index, row in enumerate(split_ds):
            page, text = str(row.get("page", "")), str(row.get("text", "")).strip()
            image = row.get("image") or {}
            payload = image.get("bytes")
            if payload is None:
                stored_path = image.get("path")
                payload = Path(stored_path).read_bytes() if stored_path else b""
            image_sha = hashlib.sha256(payload).hexdigest()
            triple = (page, text, image_sha)
            if expected[split][triple] <= actual[split][triple]:
                if page not in page_split:
                    raise ValueError(f"Split '{split}' row {index}: page '{page}' is not in the approved manifest")
                if page_split[page] != split:
                    raise ValueError(
                        f"Split '{split}' row {index}: page '{page}' belongs to split "
                        f"'{page_split[page]}' in the manifest (train/test leakage)"
                    )
                if not any(t == text for (p, t, h) in expected[split] if p == page):
                    raise ValueError(
                        f"Split '{split}' row {index} (page '{page}'): text does not match the "
                        "approved manifest entry"
                    )
                if not any(h == image_sha for (p, t, h) in expected[split] if p == page):
                    raise ValueError(
                        f"Split '{split}' row {index} (page '{page}'): image bytes sha256 "
                        f"{image_sha[:12]}… does not match the approved crop hash"
                    )
                raise ValueError(
                    f"Split '{split}' row {index}: content exceeds the approved manifest rows"
                )
            actual[split][triple] += 1

    for split in required_splits:
        expected_rows = sum(expected[split].values())
        actual_rows = sum(actual[split].values())
        if expected_rows != actual_rows or expected[split] != actual[split]:
            missing_rows = sum((expected[split] - actual[split]).values())
            raise ValueError(
                f"Split '{split}' does not match the approved manifest: "
                f"expected {expected_rows} rows, found {actual_rows}, missing {missing_rows}"
            )
    return {
        "verified": True,
        "manifest_sha256": metadata["source_manifest_sha256"],
        "rows": sum(sum(c.values()) for c in actual.values()),
        "split_counts": {s: sum(actual[s].values()) for s in required_splits},
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


def require_separate_training_output(dataset_dir: str | Path, output_dir: str | Path) -> None:
    dataset = Path(dataset_dir).expanduser().resolve(strict=True)
    output = Path(output_dir).expanduser().resolve(strict=False)
    if output == dataset or dataset in output.parents or output in dataset.parents:
        raise ValueError("Training output must be separate from the source Dataset directory")
