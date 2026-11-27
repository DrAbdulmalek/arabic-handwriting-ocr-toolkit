# HITL Stage 2 — Read-only legacy adapter and local review UI

## Legacy adapter contract

The legacy database's `label` provenance is not reliable enough to treat every
label as original OCR output. The review application may overwrite labels, and
the legacy `verified` flag may also be set by automatic dictionary acceptance.

- The adapter opens SQLite using URI `mode=ro` and `query_only`.
- It never creates, migrates, updates, or deletes legacy records.
- It excludes rows explicitly marked ignored.
- It returns stable IDs as `legacy:<row-id>`.
- Every legacy label is tagged `LEGACY_LABEL_UNVERIFIED`.
- The legacy verified flag is informational only.
- It requires an explicit crop-quality decision; it does not infer GOOD_CROP.
- It checks path containment only when allowed roots are supplied.
- It excludes crop IDs already registered in the HITL store before applying
  the query limit.

## Local review UI

`hitl_review_app.py` is a local Gradio interface. It displays the crop image,
legacy expected text (with a provenance warning when unknown), and a separate
human-corrected text field. The reviewer must explicitly choose GOOD_CROP or
BAD_CROP.

The SHA-256 digest is computed locally from the selected crop file. Image bytes
are not sent to GitHub, CI, or a remote inference service.

## Privacy and network boundary

- Gradio `share=False` is hard-coded.
- The default bind address is `127.0.0.1`.
- A crop path must resolve inside one of the explicit `--crop-root` folders.
- The SQLite database must already exist; the app refuses to create a new DB.
- The legacy adapter remains read-only. Only the separate HITL tables are
  written after an explicit human decision.
- No real database or image is included in this repository.

Do not expose the app on a public interface. If changing `--host` from the
loopback default, use only a trusted, isolated network. This minimal stage
does not implement user authentication.

## Local launch (not executed by CI)

After installing the repository requirements, run:

```bash
python hitl_review_app.py \
  --db /absolute/path/to/ocr_data/crops.db \
  --crop-root /absolute/path/to/ocr_data/crops \
  --reviewer local-reviewer \
  --crop-config-version crop-v2 \
  --model-version ocr-v1 \
  --batch-id batch-two
```

Open `http://127.0.0.1:7864` on the same computer.

## Review and training gates

- BAD_CROP is recorded as rejected and cannot receive transcription.
- GOOD_CROP requires a non-empty human transcription.
- A saved correction remains CORRECTED until the independent validation action.
- The separate training-ready action is required after validation.
- Re-editing a correction resets its status to CORRECTED.

## Known limitations

This is the first UI integration, not the complete batch workflow. CSV/XLSX
manifest export/import, tamper/replay protection, explicit skipped-crop
bookkeeping, duplicate-image queue diagnostics, Crop Engine feedback, and a
HITL-only Dataset exporter are still pending. The legacy `4_export_dataset.py`
must not be treated as the HITL export path: it still has destructive output
directory replacement and trusts the legacy `verified` flag. Those issues
must be addressed before using it for this correction dataset.
