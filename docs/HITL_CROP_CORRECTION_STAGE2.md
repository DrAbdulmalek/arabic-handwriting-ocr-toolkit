# HITL Stage 2 — Read-only legacy adapter

## Purpose

The legacy database contains crop records, but its label provenance is not
reliable enough to treat every label as the original OCR output. In particular,
the legacy review application can overwrite label text, and the verified flag
can also be set by automatic dictionary acceptance.

## Adapter contract

- Opens the existing SQLite database with SQLite URI mode=ro and query_only.
- Never creates, migrates, updates, or deletes legacy records.
- Excludes only rows explicitly marked ignored.
- Returns a stable adapter identity in the form legacy:<row-id>.
- Marks every legacy label as LEGACY_LABEL_UNVERIFIED.
- Reports the legacy verified flag as informational metadata only.
- Requires an explicit crop-quality decision; it does not infer GOOD_CROP.
- Does not authorize transcription or HITL registration.
- Checks crop path containment only when allowed_crop_roots is explicitly
  supplied. It checks metadata/path existence but does not open image bytes.
- Omits rows already present in the HITL store under the same adapter ID.

## Important limitation

This adapter does not prove that the local image exists or is the correct crop
until run against the user's local database and filesystem. It also cannot
reconstruct the original OCR prediction where the old review application
overwrote label. Such rows must remain provenance-unknown unless an independent
OCR output record is found.

## Not included in this stage

No UI, bulk registration, Excel/CSV round-trip, image hashing, crop-engine
feedback, database migration, or model training is implemented. No real user
database or crop image is accessed by CI.

## Next integration gates

1. Inspect the adapter output locally in read-only mode against a copy of the
   database, without uploading the database or images.
2. Resolve and verify crop paths under a user-selected allowed root.
3. Add explicit human GOOD_CROP/BAD_CROP review before registration.
4. Add a batch manifest with stable IDs and image hashes calculated locally.
5. Build the correction UI and controlled export/import only after those gates.
