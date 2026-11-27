# HITL Stage 4 — Crop feedback and trainer hand-off

## A. Advisory crop feedback

Run locally against the existing editor log:

```bash
python hitl_crop_feedback.py \
  --log /absolute/path/to/ocr_data/box_adjustments.jsonl \
  --out /absolute/path/to/new-crop-feedback.json \
  --min-samples 20
```

The report contains geometry aggregates only. It never outputs labels, page
names, IDs, paths, or image bytes. It does not modify crop settings. Any
suggested change must be tested on held-out pages and compared using crop
quality plus CER/WER before it is accepted.

## B. Build the Hugging Face Dataset locally

After running `hitl_export_dataset.py`, convert its manifest:

```bash
python hitl_hf_adapter.py \
  --manifest /absolute/path/to/new-hitl-dataset/manifest.json \
  --out /absolute/path/to/new-hitl-hf-dataset
```

The adapter revalidates the manifest, page-level splits, relative image paths,
image readability, and SHA-256 digests. It refuses to overwrite an existing
output directory. The resulting DatasetDict contains `train`, `val`, and
`test`, with the `image`, `text`, and `page` columns expected by
`5_train_trocr_colab.py`.

## C. Training boundary

The adapter only prepares a local DatasetDict; it does not start training.
The existing trainer may download model/tokenizer weights unless they are
already cached. For an offline run, use locally cached model files and set
`HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`. Do not set `--push-repo`
for a local-only run.

No database, crop image, manifest, DatasetDict, or model is uploaded by these
scripts. CI uses synthetic images and temporary databases only.


## D. Fine-tune batch two with a measured baseline

Use the previous locally saved model as `--init-model`. The trainer evaluates
that model on the same held-out test split before fine-tuning, then evaluates
again after training:

```bash
python 5_train_trocr_colab.py \
  --dataset /absolute/path/to/new-hitl-hf-dataset \
  --init-model /absolute/path/to/previous-trained-model \
  --out /absolute/path/to/new-batch-two-model \
  --require-hitl \
  --offline \
  --epochs 20
```

- `--require-hitl` verifies the preserved source manifest fingerprint and
  train/val/test page isolation before training.
- `--init-model` loads the previous trained checkpoint; baseline CER/WER is
  measured on the test split before any new gradient updates.
- `--offline` requires all model/tokenizer/processor files to be cached locally.
- Training output and checkpoint directories must be new and separate from the
  source Dataset directory.
- A HITL-derived model cannot be pushed to Hugging Face Hub by this script.
- The output includes `training_metrics.json` with baseline CER/WER, final
  CER/WER, deltas, split counts, and the source manifest SHA-256.

A negative CER/WER delta is a numerical improvement, but do not claim a
validated improvement until the same held-out test pages, preprocessing,
tokenizer, and evaluation settings are confirmed across both runs.
