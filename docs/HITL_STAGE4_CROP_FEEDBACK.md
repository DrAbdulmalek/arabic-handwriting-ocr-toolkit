# HITL Stage 4 — Crop geometry feedback

## Purpose

`hitl_crop_feedback.py` reads the local `box_adjustments.jsonl` log and
summarizes how a human changed crop rectangles. It is an analysis aid for a
controlled second-batch crop calibration experiment, not an automatic crop
engine tuner.

## Privacy boundary

The parser uses only `action`, `page`, `id`, and the numeric rectangle
coordinates in `old` / `new`. The output deliberately excludes page names,
crop IDs, paths, labels, and image contents. No source log is uploaded.

## Method

- Multiple edits for the same (page, crop ID) are collapsed to the net change
  from the first old rectangle to the last new rectangle.
- Horizontal and vertical edge shifts, width/height changes, and relative
  percentage changes are summarized by medians.
- Recommendations are emitted only after the configured minimum number of
  distinct crops and a consistent direction threshold.
- Recommendations remain advisory; the script never edits crop settings.

## Local use

```bash
python hitl_crop_feedback.py \
  --log /absolute/path/to/ocr_data/box_adjustments.jsonl \
  --out /absolute/path/to/new-crop-feedback.json \
  --min-samples 20
```

The report is JSON and refuses to overwrite an existing report file.

## Interpretation

A consistent widening signal means reviewers often expanded rectangles. It
does not by itself prove which parameter should change: line segmentation,
word grouping, margins, source scan scale, and manual reviewer behavior can
all contribute. Test one small adjustment on held-out pages, compare crop
quality and OCR CER/WER, and retain the prior configuration for rollback.
