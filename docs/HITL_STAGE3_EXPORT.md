# HITL Stage 3 — Safe local dataset export

## Eligibility

The exporter reads only rows returned by `training_rows()`, which requires:
`crop_quality=GOOD_CROP`, `correction_status=TRAINING_READY`, and a non-empty
human `corrected_text`. It never reads legacy `crops.label` or `verified`.

## Safety properties

- Re-resolves every image path and enforces containment in explicit crop roots.
- Recomputes SHA-256 locally and rejects a mismatch with the registered hash.
- Requires at least three distinct source pages and assigns each page to exactly
  one of train/val/test (no crop-level leakage).
- Retains original expected text and its provenance in the audit manifest,
  separately from the human-corrected training text.
- Refuses an existing output directory, including an empty one.
- Refuses to create output inside or above a crop root.
- Never deletes or replaces existing files.
- Performs no network upload.

## Local use

```bash
python hitl_export_dataset.py \
  --db /absolute/path/to/ocr_data/crops.db \
  --crop-root /absolute/path/to/ocr_data/crops \
  --out /absolute/path/to/new-hitl-dataset
```

The output contains `images/`, `dataset.csv`, and `manifest.json`. The
`text` column is the human correction; `expected_text` and
`expected_text_source` are retained for audit only.

## Failure behavior

Validation occurs before creating the output directory. If copying or writing
fails after creation, a partial new output directory may remain for inspection;
the exporter will not silently delete it. Choose a new output path for a retry.

This stage does not train a model or upload any dataset.
