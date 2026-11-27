# HITL Crop Correction — Stage 1

This branch introduces an isolated, local-first correction store for the handwriting OCR toolkit.

## Safety boundary

- Existing `crops` table is not altered.
- No image bytes are read by this module; `crop_image_ref` is a local reference.
- Original text and crop identity are immutable; `expected_text_source` records whether the expected text came from original OCR, human labeling, automatic correction, or an unverified legacy label.
- Only a crop explicitly registered as `GOOD_CROP` can be transcribed.
- Bad crops are rejected from transcription and training.
- A human correction is not training data until it passes both explicit gates:
  `CORRECTED -> VALIDATED -> TRAINING_READY`.
- A lowercase 64-character SHA-256 digest is required for every crop image; exact duplicate image hashes and duplicate crop IDs are rejected.
- Every correction edit is recorded in an append-only revision table.
- Importing this module has no side effects. Call `initialize(db_path)` explicitly.
- Initialization requires an existing database file; it never creates a database on a typo path.
- Connections are closed deterministically, and the module does not change the database journal mode.
- Crop-quality decisions and their evidence are immutable after registration.

## API

```python
from crop_correction_store import (
    initialize, register_crop, list_pending, save_correction,
    validate_correction, mark_training_ready, training_rows,
)

initialize("crops.db")
register_crop(
    "crops.db",
    crop_id="stable-crop-id",
    crop_hash="sha256-of-crop-image",
    crop_image_ref="/local/path/to/crop.png",
    source_image_id="stable-source-page-id",
    expected_text="original OCR prediction",
    expected_text_source="ORIGINAL_OCR_OUTPUT",
    crop_quality="GOOD_CROP",
    quality_evidence={"checks": ["inside_page", "no_clipped_glyphs"]},
    crop_config_version="crop-v1",
    model_version="ocr-v1",
    batch_id="batch-001",
)
# User enters corrected_text only after inspecting the crop.
save_correction("crops.db", "stable-crop-id", "human transcription")
validate_correction("crops.db", "stable-crop-id")
mark_training_ready("crops.db", "stable-crop-id")
rows = training_rows("crops.db")
```

The quality decision is deliberately supplied by the caller and must include evidence.
This stage does not claim that a crop-quality model or the legacy database has been audited.

## Tests

Run locally with:

```bash
python -m unittest discover -s tests -p 'test_crop_correction_store.py' -v
```

This stage does not yet connect the store to the existing review UI, Excel, crop engine, or trainer.
Those integrations are separate stages and must preserve the same eligibility gates.
