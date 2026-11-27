"""Unit tests for the isolated HITL correction store."""
import tempfile
import unittest
from pathlib import Path

from crop_correction_store import (
    CORRECTED, GOOD_CROP, BAD_CROP, TRAINING_READY, VALIDATED,
    DuplicateCropError, EligibilityError, initialize, register_crop,
    list_pending, save_correction, validate_correction, mark_training_ready,
    training_rows, correction_history,
)


class CorrectionStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "test.sqlite"
        # Seed a legacy table to prove it is not modified.
        import sqlite3
        with sqlite3.connect(self.db) as conn:
            conn.execute("CREATE TABLE crops (id INTEGER PRIMARY KEY, label TEXT, verified INTEGER)")
            conn.execute("INSERT INTO crops VALUES (7, 'legacy prediction', 0)")
        initialize(self.db)

    def tearDown(self):
        self.tmp.cleanup()

    def register(self, crop_id="c1", crop_hash="a" * 64, quality=GOOD_CROP):
        register_crop(
            self.db, crop_id=crop_id, crop_hash=crop_hash,
            crop_image_ref="local://crops/c1.png", source_image_id="page-1",
            expected_text="توقع OCR", expected_text_source="ORIGINAL_OCR_OUTPUT", crop_quality=quality,
            quality_evidence={"reviewer":"test", "checks":["bounds","nonempty"]},
            crop_config_version="crop-v1", model_version="ocr-v1",
            batch_id="batch-1", bounding_box={"x":1,"y":2,"w":30,"h":10},
            confidence=0.82,
        )

    def test_only_good_crops_are_pending(self):
        self.register()
        self.register("bad1", "b" * 64, BAD_CROP)
        rows = list_pending(self.db)
        self.assertEqual([r["crop_id"] for r in rows], ["c1"])

    def test_prediction_is_preserved_and_revision_recorded(self):
        self.register()
        save_correction(self.db, "c1", "النص الصحيح")
        import sqlite3
        with sqlite3.connect(self.db) as conn:
            row = conn.execute(
                "SELECT expected_text,expected_text_source,corrected_text,correction_status FROM hitl_crop_corrections"
            ).fetchone()
        self.assertEqual(row, ("توقع OCR", "ORIGINAL_OCR_OUTPUT", "النص الصحيح", CORRECTED))
        self.assertEqual(len(correction_history(self.db, "c1")), 1)

    def test_bad_crop_cannot_be_corrected(self):
        self.register("bad1", "b" * 64, BAD_CROP)
        with self.assertRaises(EligibilityError):
            save_correction(self.db, "bad1", "نص")

    def test_exact_hash_duplicate_rejected(self):
        self.register()
        with self.assertRaises(DuplicateCropError):
            self.register("c2", "a" * 64)

    def test_training_requires_two_explicit_gates(self):
        self.register()
        save_correction(self.db, "c1", "النص الصحيح")
        self.assertEqual(training_rows(self.db), [])
        validate_correction(self.db, "c1")
        self.assertEqual(training_rows(self.db), [])
        mark_training_ready(self.db, "c1")
        rows = training_rows(self.db)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["corrected_text"], "النص الصحيح")



    def test_hash_must_be_sha256_hex(self):
        with self.assertRaises(ValueError):
            self.register("invalid-hash", "not-a-sha256")
        self.assertEqual(list_pending(self.db), [])

    def test_expected_text_source_is_required_and_validated(self):
        import sqlite3
        with self.assertRaises(ValueError):
            register_crop(
                self.db, crop_id="bad-source", crop_hash="c" * 64,
                crop_image_ref="local://crops/a.png", source_image_id="page-1",
                expected_text="نص", expected_text_source="GUESSED",
                crop_quality=GOOD_CROP, quality_evidence={"reviewer":"test"},
                crop_config_version="crop-v1", model_version="ocr-v1", batch_id="batch-1",
            )
        self.assertEqual(list_pending(self.db), [])

    def test_recorrection_invalidates_previous_approval(self):
        self.register()
        save_correction(self.db, "c1", "تصحيح أول")
        validate_correction(self.db, "c1")
        mark_training_ready(self.db, "c1")
        save_correction(self.db, "c1", "تصحيح منقح")
        self.assertEqual(training_rows(self.db), [])
        import sqlite3
        with sqlite3.connect(self.db) as conn:
            status = conn.execute(
                "SELECT correction_status FROM hitl_crop_corrections WHERE crop_id='c1'"
            ).fetchone()[0]
        self.assertEqual(status, CORRECTED)

    def test_initialize_is_idempotent(self):
        initialize(self.db)
        initialize(self.db)
        import sqlite3
        with sqlite3.connect(self.db) as conn:
            version = conn.execute(
                "SELECT value FROM hitl_schema_meta WHERE key='schema_version'"
            ).fetchone()[0]
        self.assertEqual(version, "2")

    def test_legacy_crops_table_untouched(self):
        import sqlite3
        with sqlite3.connect(self.db) as conn:
            row = conn.execute("SELECT id,label,verified FROM crops WHERE id=7").fetchone()
        self.assertEqual(row, (7, "legacy prediction", 0))

    def test_initialize_refuses_to_create_a_missing_database(self):
        missing = Path(self.tmp.name) / "typo.sqlite"
        with self.assertRaises(FileNotFoundError):
            initialize(missing)
        self.assertFalse(missing.exists())

    def test_quality_decision_is_immutable(self):
        self.register()
        import sqlite3
        with self.assertRaises(sqlite3.IntegrityError):
            with sqlite3.connect(self.db) as conn:
                conn.execute(
                    "UPDATE hitl_crop_corrections SET crop_quality='BAD_CROP' WHERE crop_id='c1'"
                )

    def test_identity_and_expected_text_are_immutable(self):
        self.register()
        import sqlite3
        with self.assertRaises(sqlite3.IntegrityError):
            with sqlite3.connect(self.db) as conn:
                conn.execute("UPDATE hitl_crop_corrections SET expected_text='changed' WHERE crop_id='c1'")

    def test_repeated_correction_keeps_revision_history(self):
        self.register()
        save_correction(self.db, "c1", "تصحيح أول")
        save_correction(self.db, "c1", "تصحيح نهائي")
        history = correction_history(self.db, "c1")
        self.assertEqual(len(history), 2)
        self.assertEqual(history[-1]["previous_text"], "تصحيح أول")


if __name__ == "__main__":
    unittest.main()
