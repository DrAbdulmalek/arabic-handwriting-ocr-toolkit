# HITL Stage 5 — Stable batch-two provenance

## Review versioning

When opening the local review app for a new crop configuration/model, provide
explicit version labels so every correction is traceable:

```bash
python hitl_review_app.py \
  --db /absolute/path/to/ocr_data/crops.db \
  --crop-root /absolute/path/to/ocr_data/crops \
  --reviewer local-reviewer \
  --crop-config-version crop-v2 \
  --model-version ocr-v1 \
  --batch-id batch-two
```

The version strings and batch ID are stored in the HITL correction row; they
are not inferred from the legacy `verified` or `locked` flags.

## Stable page splits

For batch one, export normally and retain its local `manifest.json`. For
batch two, pass that manifest using `--prior-manifest`. Every source page
already present in the prior manifest keeps its exact split. New pages are
assigned to the largest remaining split-ratio deficit, deterministically using
the configured seed.

The exporter fails closed if the prior manifest is malformed, lacks one of
the three splits, or has no page overlap with the current batch. This avoids
silently reshuffling a previously held-out test page into training.

## Evaluation requirement

This stage does not claim model improvement. Before accepting batch two,
record baseline CER/WER on the same held-out test pages, train only on
TRAINING_READY HITL corrections, then evaluate the same test pages again.
Keep test pages out of crop-parameter tuning and training.
