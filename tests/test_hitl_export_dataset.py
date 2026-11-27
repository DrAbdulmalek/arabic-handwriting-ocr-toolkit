"""Tests for safe HITL-only dataset export."""
import hashlib
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from crop_correction_store import (
    initialize, register_crop, save_correction, validate_correction,
    mark_training_ready,
)
from hitl_export_dataset import export_training_dataset


class HitlExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "crops"
        self.root.mkdir()
        self.db = Path(self.tmp.name) / "crops.sqlite"
        with sqlite3.connect(self.db):
            pass
        initialize(self.db)
        self.files = []
        for i in range(3):
            path = self.root / f"crop-{i}.png"
            payload = f"synthetic-image-{i}".encode()
            path.write_bytes(payload)
            self.files.append(path)
            crop_id = f"crop-{i}"
            register_crop(
                self.db, crop_id=crop_id,
                crop_hash=hashlib.sha256(payload).hexdigest(),
                crop_image_ref=str(path), source_image_id=f"page-{i}",
                expected_text="OCR", expected_text_source="ORIGINAL_OCR_OUTPUT",
                crop_quality="GOOD_CROP",
                quality_evidence={"reviewer":"test","decision":"GOOD_CROP","checks":["visual"]},
                crop_config_version="test", model_version="test", batch_id="batch-test",
            )
            save_correction(self.db, crop_id, f"تصحيح {i}")
            validate_correction(self.db, crop_id)
            mark_training_ready(self.db, crop_id)

    def tearDown(self):
        self.tmp.cleanup()

    def test_exports_only_training_ready_rows_and_page_level_splits(self):
        out = Path(self.tmp.name) / "export"
        result = export_training_dataset(self.db, out, allowed_crop_roots=[self.root])
        self.assertEqual(result["row_count"], 3)
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["source"], "HITL_TRAINING_READY_ONLY")
        self.assertEqual(len(manifest["rows"]), 3)
        self.assertEqual({row["page"] for row in manifest["rows"]}, {"page-0","page-1","page-2"})
        self.assertEqual({row["split"] for row in manifest["rows"]}, {"train","val","test"})
        for row in manifest["rows"]:
            self.assertTrue((out / row["image"]).is_file())
            self.assertTrue(row["text"].startswith("تصحيح"))

    def test_existing_output_is_never_overwritten(self):
        out = Path(self.tmp.name) / "existing"
        out.mkdir()
        sentinel = out / "keep.txt"
        sentinel.write_text("do not delete")
        with self.assertRaises(FileExistsError):
            export_training_dataset(self.db, out, allowed_crop_roots=[self.root])
        self.assertEqual(sentinel.read_text(), "do not delete")

    def test_hash_mismatch_fails_before_output_creation(self):
        self.files[0].write_bytes(b"tampered")
        out = Path(self.tmp.name) / "bad-export"
        with self.assertRaises(ValueError):
            export_training_dataset(self.db, out, allowed_crop_roots=[self.root])
        self.assertFalse(out.exists())

    def test_outside_root_fails_closed(self):
        outside = Path(self.tmp.name) / "outside.png"
        payload = b"outside"
        outside.write_bytes(payload)
        db2 = Path(self.tmp.name) / "outside.sqlite"
        with sqlite3.connect(db2):
            pass
        initialize(db2)
        register_crop(
            db2, crop_id="outside", crop_hash=hashlib.sha256(payload).hexdigest(),
            crop_image_ref=str(outside), source_image_id="page-0",
            expected_text="OCR", expected_text_source="ORIGINAL_OCR_OUTPUT",
            crop_quality="GOOD_CROP",
            quality_evidence={"reviewer":"test","decision":"GOOD_CROP","checks":["visual"]},
            crop_config_version="test", model_version="test", batch_id="batch-test",
        )
        save_correction(db2, "outside", "تصحيح")
        validate_correction(db2, "outside")
        mark_training_ready(db2, "outside")
        out = Path(self.tmp.name) / "outside-export"
        with self.assertRaises(ValueError):
            export_training_dataset(db2, out, allowed_crop_roots=[self.root])
        self.assertFalse(out.exists())

    def test_requires_at_least_three_source_pages(self):
        # Use a second database containing only one approved source page.
        db2 = Path(self.tmp.name) / "one-page.sqlite"
        with sqlite3.connect(db2):
            pass
        initialize(db2)
        payload = b"single"
        path = self.root / "single.png"
        path.write_bytes(payload)
        register_crop(
            db2, crop_id="single", crop_hash=hashlib.sha256(payload).hexdigest(),
            crop_image_ref=str(path), source_image_id="one-page",
            expected_text="OCR", expected_text_source="ORIGINAL_OCR_OUTPUT",
            crop_quality="GOOD_CROP",
            quality_evidence={"reviewer":"test","decision":"GOOD_CROP","checks":["visual"]},
            crop_config_version="test", model_version="test", batch_id="batch-test",
        )
        save_correction(db2, "single", "تصحيح")
        validate_correction(db2, "single")
        mark_training_ready(db2, "single")
        with self.assertRaises(ValueError):
            export_training_dataset(db2, Path(self.tmp.name) / "one-page-export",
                                    allowed_crop_roots=[self.root])


if __name__ == "__main__":
    unittest.main()
