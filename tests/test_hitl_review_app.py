"""Tests for the local-only HITL review UI helpers."""
import sqlite3
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from crop_correction_store import initialize, list_pending, training_rows
from hitl_review_app import ReviewSession, _safe_crop_path, _sha256_file, build_app


class ReviewAppTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "crops"
        self.root.mkdir()
        self.db = Path(self.tmp.name) / "crops.sqlite"
        with sqlite3.connect(self.db) as conn:
            conn.execute("""
                CREATE TABLE crops (
                    id INTEGER PRIMARY KEY, page_name TEXT, crop_path TEXT,
                    label TEXT, ignored INTEGER DEFAULT 0, verified INTEGER DEFAULT 0
                )
            """)
        initialize(self.db)
        self.image = self.root / "one.png"
        Image.new("RGB", (12, 8), "white").save(self.image)
        self.session = ReviewSession(self.db, (self.root,), "test-reviewer")
        self.candidate = {
            "crop_id": "legacy:1",
            "source_image_id": "page-1",
            "crop_image_ref": str(self.image),
            "image_path": str(self.image),
            "expected_text": "نص متوقع",
            "expected_text_source": "LEGACY_LABEL_UNVERIFIED",
            "legacy_confidence": 0.8,
            "bounding_box": {"x1": 1, "y1": 2, "x2": 30, "y2": 15},
            "is_registered": False,
        }

    def tearDown(self):
        self.tmp.cleanup()


    def test_gradio_app_builds_without_network_share(self):
        try:
            import gradio
        except ImportError:
            self.skipTest("Gradio is installed in the dedicated CI smoke-test step")
        app = build_app(self.db, (self.root,), "test-reviewer")
        self.assertIsNotNone(app)

    def test_hash_is_sha256_of_local_bytes(self):
        self.assertEqual(len(_sha256_file(self.image)), 64)
        self.assertEqual(_safe_crop_path(str(self.image), (self.root,)), self.image.resolve())

    def test_crop_path_must_stay_inside_allowed_root(self):
        outside = Path(self.tmp.name) / "outside.png"
        Image.new("RGB", (8, 8), "black").save(outside)
        link = self.root / "escape.png"
        link.symlink_to(outside)
        with self.assertRaises(ValueError):
            _safe_crop_path(str(link), (self.root,))

    def test_good_crop_is_registered_and_corrected_locally(self):
        message = self.session.record(
            self.candidate, "GOOD_CROP", "LEGACY_LABEL_UNVERIFIED", "النص الصحيح"
        )
        self.assertIn("حُفظ التصحيح", message)
        self.assertEqual(list_pending(self.db), [])
        self.assertEqual(training_rows(self.db), [])

    def test_bad_crop_is_excluded_from_transcription_and_training(self):
        message = self.session.record(
            self.candidate, "BAD_CROP", "LEGACY_LABEL_UNVERIFIED", ""
        )
        self.assertIn("غير صالحة", message)
        self.assertEqual(list_pending(self.db), [])
        self.assertEqual(training_rows(self.db), [])

    def test_good_crop_requires_corrected_text_before_registration(self):
        with self.assertRaises(ValueError):
            self.session.record(
                self.candidate, "GOOD_CROP", "LEGACY_LABEL_UNVERIFIED", " "
            )
        with sqlite3.connect(self.db) as conn:
            count = conn.execute("SELECT COUNT(*) FROM hitl_crop_corrections").fetchone()[0]
        self.assertEqual(count, 0)


if __name__ == "__main__":
    unittest.main()
