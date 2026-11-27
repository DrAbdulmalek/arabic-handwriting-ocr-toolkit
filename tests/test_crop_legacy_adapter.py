"""Tests for the read-only legacy crop adapter."""
import sqlite3
import tempfile
import unittest
from pathlib import Path

from crop_legacy_adapter import read_legacy_candidates, LegacyAdapterError


class LegacyAdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "legacy.sqlite"
        with sqlite3.connect(self.db) as conn:
            conn.execute("""
                CREATE TABLE crops (
                    id INTEGER PRIMARY KEY, page_name TEXT, crop_path TEXT,
                    label TEXT, ignored INTEGER DEFAULT 0, verified INTEGER DEFAULT 0,
                    confidence REAL, x1 INTEGER, y1 INTEGER, x2 INTEGER, y2 INTEGER
                )
            """)
            conn.execute(
                "INSERT INTO crops VALUES (1,'page-a','/data/crops/a.png','وسم قديم',0,1,0.99,1,2,30,40)"
            )
            conn.execute(
                "INSERT INTO crops VALUES (2,'page-a','/data/crops/b.png','مستبعد',1,0,NULL,1,2,30,40)"
            )
            conn.commit()

    def tearDown(self):
        self.tmp.cleanup()

    def test_reads_rows_without_treating_verified_as_validation(self):
        rows = read_legacy_candidates(self.db, allowed_crop_roots=["/data/crops"])
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["crop_id"], "legacy:1")
        self.assertEqual(row["expected_text_source"], "LEGACY_LABEL_UNVERIFIED")
        self.assertEqual(row["legacy_verified_flag"], 1)
        self.assertFalse(row["eligible_for_transcription"])
        self.assertFalse(row["registration_allowed"])
        self.assertTrue(row["quality_decision_required"])

    def test_ignored_rows_are_not_returned(self):
        rows = read_legacy_candidates(self.db)
        self.assertEqual([row["crop_id"] for row in rows], ["legacy:1"])

    def test_path_must_be_inside_explicit_allowed_root(self):
        row = read_legacy_candidates(self.db, allowed_crop_roots=["/safe/root"])[0]
        self.assertEqual(row["path_status"], "OUTSIDE_ALLOWED_ROOTS")

    def test_missing_crops_table_fails_closed(self):
        empty = Path(self.tmp.name) / "empty.sqlite"
        sqlite3.connect(empty).close()
        with self.assertRaises(LegacyAdapterError):
            read_legacy_candidates(empty)


if __name__ == "__main__":
    unittest.main()
