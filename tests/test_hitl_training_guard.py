"""Tests for local HITL training provenance and baseline report helpers."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from hitl_training_guard import (
    verify_hitl_dataset, training_metrics_report, require_new_output_paths,
)


class TrainingGuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "dataset"
        self.root.mkdir()
        self.manifest = {
            "format_version": 1,
            "source": "HITL_TRAINING_READY_ONLY",
            "row_count": 3,
            "rows": [],
        }
        raw = json.dumps(self.manifest, ensure_ascii=False).encode()
        (self.root / "hitl_manifest.json").write_bytes(raw)
        (self.root / "hitl_dataset_metadata.json").write_text(json.dumps({
            "source": "HITL_TRAINING_READY_ONLY",
            "source_manifest_sha256": hashlib.sha256(raw).hexdigest(),
            "rows": 3,
            "split_counts": {"train":1,"val":1,"test":1},
        }), encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def test_verifies_hf_dataset_manifest_fingerprint(self):
        result = verify_hitl_dataset(self.root)
        self.assertEqual(result["source"], "HITL_TRAINING_READY_ONLY")
        self.assertEqual(result["rows"], 3)
        self.assertEqual(len(result["source_manifest_sha256"]), 64)

    def test_rejects_tampered_manifest(self):
        with (self.root / "hitl_manifest.json").open("a", encoding="utf-8") as stream:
            stream.write(" ")
        with self.assertRaises(ValueError):
            verify_hitl_dataset(self.root)

    def test_rejects_missing_provenance(self):
        (self.root / "hitl_dataset_metadata.json").unlink()
        with self.assertRaises(ValueError):
            verify_hitl_dataset(self.root)

    def test_metrics_report_calculates_delta_without_claiming_improvement(self):
        report = training_metrics_report(
            dataset_dir=str(self.root), dataset_manifest_sha256="a" * 64,
            init_model="previous-model",
            baseline={"baseline_cer":0.30,"baseline_wer":0.50},
            final={"test_cer":0.22,"test_wer":0.40},
            train_count=100,val_count=10,test_count=10,seed=42,epochs=5,
        )
        self.assertAlmostEqual(report["cer_delta_final_minus_baseline"], -0.08)
        self.assertFalse(report["improvement_claimed"])
        self.assertEqual(report["baseline_status"], "MEASURED")

    def test_missing_baseline_is_explicit(self):
        report = training_metrics_report(
            dataset_dir=str(self.root), dataset_manifest_sha256=None,
            init_model=None, baseline=None, final={"test_cer":0.2,"test_wer":0.3},
            train_count=1,val_count=1,test_count=1,seed=1,epochs=1,
        )
        self.assertEqual(report["baseline_status"], "NOT_MEASURED_NO_INIT_MODEL")
        self.assertIsNone(report["cer_delta_final_minus_baseline"])

    def test_training_outputs_must_be_new(self):
        model = Path(self.tmp.name) / "model"
        checkpoints = Path(self.tmp.name) / "checkpoints"
        require_new_output_paths(model, checkpoints)
        model.mkdir()
        with self.assertRaises(FileExistsError):
            require_new_output_paths(model, checkpoints)


if __name__ == "__main__":
    unittest.main()
