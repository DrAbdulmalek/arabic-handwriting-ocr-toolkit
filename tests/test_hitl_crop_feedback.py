"""Tests for geometry-only crop feedback analysis."""
import json
import tempfile
import unittest
from pathlib import Path

from hitl_crop_feedback import analyze_adjustments


class CropFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.log = Path(self.tmp.name) / "box_adjustments.jsonl"

    def tearDown(self):
        self.tmp.cleanup()

    def write_events(self, events):
        self.log.write_text(
            "\n".join(json.dumps(event, ensure_ascii=False) for event in events) + "\n",
            encoding="utf-8",
        )

    def test_consistent_manual_widening_yields_advisory_only(self):
        events = []
        for i in range(5):
            events.append({
                "action": "update", "page": f"page-{i}", "id": i,
                "old": {"x1": 10, "y1": 10, "x2": 30, "y2": 30,
                        "label": "PRIVATE_LABEL"},
                "new": {"x1": 8, "y1": 10, "x2": 32, "y2": 30,
                        "label": "PRIVATE_LABEL"},
            })
        self.write_events(events)
        report = analyze_adjustments(self.log, min_samples=3)
        self.assertEqual(report["distinct_crops"], 5)
        self.assertEqual(report["geometry_medians_px"]["width_delta"], 4.0)
        self.assertTrue(any(
            r.get("axis") == "horizontal"
            and r.get("direction") == "widen"
            and r["status"] == "ADVISORY_CALIBRATION_CANDIDATE"
            for r in report["recommendations"]
        ))
        self.assertFalse(report["auto_applied"])
        serialized = json.dumps(report, ensure_ascii=False)
        self.assertNotIn("PRIVATE_LABEL", serialized)
        self.assertNotIn("page-0", serialized)

    def test_multiple_edits_for_one_crop_are_collapsed_to_net_change(self):
        events = [
            {"action":"update","page":"p","id":1,
             "old":{"x1":10,"y1":10,"x2":30,"y2":30},
             "new":{"x1":9,"y1":10,"x2":31,"y2":30}},
            {"action":"update","page":"p","id":1,
             "old":{"x1":9,"y1":10,"x2":31,"y2":30},
             "new":{"x1":8,"y1":10,"x2":32,"y2":30}},
        ]
        self.write_events(events)
        report = analyze_adjustments(self.log, min_samples=3)
        self.assertEqual(report["distinct_crops"], 1)
        self.assertEqual(report["geometry_medians_px"]["width_delta"], 4.0)

    def test_small_sample_is_insufficient_and_no_data_is_exposed(self):
        self.write_events([{
            "action":"update","page":"secret-page","id":9,
            "old":{"x1":0,"y1":0,"x2":20,"y2":20,"label":"secret"},
            "new":{"x1":1,"y1":0,"x2":21,"y2":20,"label":"secret"},
        }])
        report = analyze_adjustments(self.log, min_samples=20)
        self.assertEqual(report["recommendations"][0]["status"], "INSUFFICIENT_DATA")
        self.assertNotIn("secret-page", json.dumps(report))
        self.assertNotIn("secret", json.dumps(report))

    def test_malformed_and_non_update_events_are_counted(self):
        self.log.write_text(
            "{bad json}\n" + json.dumps({"action":"delete"}) + "\n",
            encoding="utf-8",
        )
        report = analyze_adjustments(self.log)
        self.assertEqual(report["malformed_lines"], 1)
        self.assertEqual(report["ignored_events"], 1)
        self.assertEqual(report["distinct_crops"], 0)

    def test_missing_log_fails_closed(self):
        with self.assertRaises(FileNotFoundError):
            analyze_adjustments(Path(self.tmp.name) / "missing.jsonl")


if __name__ == "__main__":
    unittest.main()
