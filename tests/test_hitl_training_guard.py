"""Tests for local HITL training provenance and baseline report helpers."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from hitl_training_guard import (
    verify_hitl_dataset, verify_datasetdict_content, training_metrics_report,
    require_new_output_paths, require_separate_training_output,
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
            "rows": [
                {"page":"page-a","split":"train"},
                {"page":"page-b","split":"val"},
                {"page":"page-c","split":"test"},
            ],
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


    def test_rejects_page_level_split_leakage(self):
        manifest = json.loads((self.root / "hitl_manifest.json").read_text(encoding="utf-8"))
        manifest["rows"][1]["page"] = "page-a"
        raw = json.dumps(manifest, ensure_ascii=False).encode()
        (self.root / "hitl_manifest.json").write_bytes(raw)
        metadata = json.loads((self.root / "hitl_dataset_metadata.json").read_text(encoding="utf-8"))
        metadata["source_manifest_sha256"] = hashlib.sha256(raw).hexdigest()
        (self.root / "hitl_dataset_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
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


    def test_training_output_must_be_separate_from_dataset(self):
        with self.assertRaises(ValueError):
            require_separate_training_output(self.root, self.root / "nested-model")
        with self.assertRaises(ValueError):
            require_separate_training_output(self.root, self.root.parent)

    def test_training_outputs_must_be_new(self):
        model = Path(self.tmp.name) / "model"
        checkpoints = Path(self.tmp.name) / "checkpoints"
        require_new_output_paths(model, checkpoints)
        model.mkdir()
        with self.assertRaises(FileExistsError):
            require_new_output_paths(model, checkpoints)


class DatasetDictContentGuardTests(unittest.TestCase):
    """Negative tests: tampering with the loaded DatasetDict must be caught.

    The manifest fingerprint check alone does not prove the rows that will be
    trained on match the approved manifest — these tests prove the row-level
    content guard does (texts, image bytes, pages, split membership).
    """
    def setUp(self):
        try:
            import datasets  # noqa: F401
        except ImportError:
            self.skipTest("datasets is installed in the dedicated CI smoke-test step")
        from datasets import load_from_disk
        from hitl_hf_adapter import build_hf_dataset

        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name) / "export"
        images = root / "images"
        images.mkdir(parents=True)
        rows = []
        for i, split in enumerate(("train", "train", "val", "test")):
            path = images / f"{i}.png"
            Image.new("RGB", (10, 8), (i * 40, 30, 90)).save(path)
            rows.append({
                "crop_id": f"crop-{i}",
                "crop_hash": hashlib.sha256(path.read_bytes()).hexdigest(),
                "image": f"images/{i}.png",
                "text": f"تصحيح {i}",
                "expected_text": "توقع",
                "expected_text_source": "LEGACY_LABEL_UNVERIFIED",
                "page": f"page-{i}",
                "split": split,
                "crop_config_version": "legacy-unknown",
                "model_version": "legacy-unknown",
                "batch_id": "test",
            })
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({
            "format_version": 1,
            "source": "HITL_TRAINING_READY_ONLY",
            "row_count": len(rows),
            "rows": rows,
        }, ensure_ascii=False), encoding="utf-8")
        self.out = Path(self.tmp.name) / "hf-dataset"
        build_hf_dataset(manifest, self.out)
        self.ds = load_from_disk(str(self.out))
        self.load_from_disk = load_from_disk

    def tearDown(self):
        self.tmp.cleanup()

    def test_untouched_datasetdict_matches_manifest(self):
        result = verify_datasetdict_content(self.ds, self.out)
        self.assertTrue(result["verified"])
        self.assertEqual(result["split_counts"], {"train": 2, "val": 1, "test": 1})

    def test_catches_altered_correction_text(self):
        ds = dict(self.ds)
        ds["train"] = self.ds["train"].map(
            lambda ex, idx: {**ex, "text": "نص معدَّل"} if idx == 0 else ex,
            with_indices=True,
        )
        with self.assertRaises(ValueError):
            verify_datasetdict_content(ds, self.out)

    def test_catches_leading_or_trailing_whitespace_mutation(self):
        for changed_text in (" " + self.ds["train"][0]["text"], self.ds["train"][0]["text"] + " "):
            with self.subTest(changed_text=changed_text):
                ds = dict(self.ds)
                ds["train"] = self.ds["train"].map(
                    lambda ex, idx: {**ex, "text": changed_text} if idx == 0 else ex,
                    with_indices=True,
                )
                with self.assertRaises(ValueError):
                    verify_datasetdict_content(ds, self.out)

    def test_catches_swapped_image_bytes(self):
        from datasets import Image as HFImage
        ds = dict(self.ds)
        ds["train"] = (
            self.ds["train"].cast_column("image", HFImage(decode=False)).map(
                lambda ex, idx: {
                    **ex,
                    "image": {"bytes": b"tampered-bytes", "path": ex["image"]["path"]},
                } if idx == 0 else ex,
                with_indices=True,
            )
        )
        with self.assertRaises(ValueError):
            verify_datasetdict_content(ds, self.out)

    def test_catches_row_moved_between_splits(self):
        # Move a train row's page to the test page: counts stay equal, content guard must stop it.
        ds = dict(self.ds)
        ds["train"] = self.ds["train"].map(
            lambda ex, idx: {**ex, "page": "page-3"} if idx == 0 else ex,
            with_indices=True,
        )
        with self.assertRaises(ValueError):
            verify_datasetdict_content(ds, self.out)

    def test_catches_swapped_texts_within_split(self):
        ds = dict(self.ds)
        texts = ["تصحيح 0", "تصحيح 1"]
        ds["train"] = self.ds["train"].map(
            lambda ex, idx: {**ex, "text": texts[1 - idx]},
            with_indices=True,
        )
        with self.assertRaises(ValueError):
            verify_datasetdict_content(ds, self.out)

    def test_catches_rebuilt_dataset_with_missing_row(self):
        from datasets import Dataset
        ds = dict(self.ds)
        ds["val"] = Dataset.from_dict({"image": [], "text": [], "page": []})
        with self.assertRaises(ValueError):
            verify_datasetdict_content(ds, self.out)


if __name__ == "__main__":
    unittest.main()
