# HITL Stage 6 — Persistent skip and duplicate suppression

The review queue now distinguishes three outcomes:

- `GOOD_CROP`: eligible for transcription after explicit human inspection.
- `BAD_CROP`: explicitly rejected as an unusable crop.
- `USER_SKIPPED`: persisted separately in `hitl_legacy_exclusions`; it is not a crop-quality label and cannot enter training.

The app calculates each candidate's SHA-256 locally. If the hash is already
registered or previously excluded, the candidate is recorded as
`DUPLICATE_HASH` and suppressed. This prevents a second legacy row pointing
to identical image bytes from appearing as a new correction task.

Registered but unfinished GOOD_CROP records can also be skipped; their status
becomes SKIPPED and they are removed from the pending queue. A skip is
persistent across app restarts.

No image, database, hash list, or exclusion log is uploaded. To revisit a
permanently skipped crop, an explicit administrative re-queue operation must
be designed; the current UI intentionally has no one-click undo.
