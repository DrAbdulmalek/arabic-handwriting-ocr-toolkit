"""Tests for manifest validation and local Hugging Face conversion."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from hitl_hf_adapter import validate_manifest, build_hf_dataset


class HfAdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "export"
        images = self.root / "images"
        images.mkdir(parents=True)
        rows = []
        for i, split in enumerate(("train", "val", "test")):
            path = images / f"{i}.png"
            Image.new("RGB", (10, 8), (i * 50, 20, 90)).save(path)
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
        self.manifest = self.root / "manifest.json"
        self.write_manifest(rows)

    def tearDown(self):
        self.tmp.cleanup()

    def write_manifest(self, rows):
        self.manifest.write_text(json.dumps({
            "format_version": 1,
            "source": "HITL_TRAINING_READY_ONLY",
            "row_count": len(rows),
            "rows": rows,
        }, ensure_ascii=False), encoding="utf-8")

    def test_validates_hashed_images_and_three_splits(self):
        root, rows = validate_manifest(self.manifest)
        self.assertEqual(root, self.root.resolve())
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(Path(row["_absolute_image"]).is_file() for row in rows))

    def test_rejects_hash_mismatch(self):
        manifest = json.loads(self.manifest.read_text(encoding="utf-8"))
        (self.root / manifest["rows"][0]["image"]).write_bytes(b"tampered")
        with self.assertRaises(ValueError):
            validate_manifest(self.manifest)

    def test_rejects_page_leakage(self):
        manifest = json.loads(self.manifest.read_text(encoding="utf-8"))
        manifest["rows"][1]["page"] = manifest["rows"][0]["page"]
        self.write_manifest(manifest["rows"])
        with self.assertRaises(ValueError):
            validate_manifest(self.manifest)

    def test_rejects_path_traversal(self):
        manifest = json.loads(self.manifest.read_text(encoding="utf-8"))
        manifest["rows"][0]["image"] = "../outside.png"
        self.write_manifest(manifest["rows"])
        with self.assertRaises(ValueError):
            validate_manifest(self.manifest)

    def test_builds_hf_dataset_locally_when_dependency_is_available(self):
        try:
            from datasets import load_from_disk
        except ImportError:
            self.skipTest("datasets is installed in the dedicated CI smoke-test step")
        out = Path(self.tmp.name) / "hf-dataset"
        result = build_hf_dataset(self.manifest, out)
        self.assertFalse(result["uploaded"])
        dataset = load_from_disk(str(out))
        self.assertEqual(set(dataset.keys()), {"train", "val", "test"})
        self.assertEqual(sum(len(dataset[s]) for s in dataset), 3)
        self.assertTrue(all(len(dataset[s]) == 1 for s in dataset))

    def test_existing_hf_output_is_not_overwritten(self):
        try:
            from datasets import load_from_disk
        except ImportError:
            self.skipTest("datasets is installed in the dedicated CI smoke-test step")
        out = Path(self.tmp.name) / "existing-hf"
        out.mkdir()
        sentinel = out / "keep.txt"
        sentinel.write_text("keep")
        with self.assertRaises(FileExistsError):
            build_hf_dataset(self.manifest, out)
        self.assertEqual(sentinel.read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
