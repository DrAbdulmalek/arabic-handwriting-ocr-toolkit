#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local-only Gradio UI for human review of legacy handwriting OCR crops.

No image, database, or correction is sent to a remote service. The UI binds
to loopback by default and has no Gradio public-share option.
"""
from __future__ import annotations

import argparse
import hashlib
import uuid
from pathlib import Path
from typing import Any

from PIL import Image

from crop_correction_store import (
    BAD_CROP, GOOD_CROP, CORRECTED, VALIDATED, TRAINING_READY,
    initialize, list_pending, register_crop, save_correction,
    validate_correction, mark_training_ready, training_rows,
    known_crop_hashes, record_legacy_exclusion, DuplicateCropError,
)
from crop_legacy_adapter import read_legacy_candidates


EXPECTED_TEXT_SOURCES = [
    "LEGACY_LABEL_UNVERIFIED",
    "ORIGINAL_OCR_OUTPUT",
    "HUMAN_LABEL",
    "AUTO_CORRECTION",
]


def _safe_crop_path(raw: str, roots: tuple[Path, ...]) -> Path:
    candidate = Path(raw).expanduser().resolve(strict=True)
    if not candidate.is_file():
        raise FileNotFoundError("Crop image is not a regular file")
    allowed = tuple(root.expanduser().resolve(strict=True) for root in roots)
    if not any(candidate == root or root in candidate.parents for root in allowed):
        raise ValueError("Crop path is outside the configured allowed roots")
    try:
        with Image.open(candidate) as image:
            image.verify()
    except Exception as exc:
        raise ValueError("Crop file is not a valid, readable image") from exc
    return candidate


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ReviewSession:
    """Small per-process queue; all persistent decisions live in SQLite."""

    def __init__(
        self, db_path: Path, roots: tuple[Path, ...], reviewer: str,
        crop_config_version: str = "legacy-unknown",
        model_version: str = "legacy-unknown",
        batch_id: str | None = None,
    ):
        self.db_path = db_path
        self.roots = roots
        self.reviewer = reviewer
        self.crop_config_version = crop_config_version
        self.model_version = model_version
        self.batch_id = batch_id or ("legacy-review-" + uuid.uuid4().hex[:12])

    def next_candidate(self) -> dict[str, Any] | None:
        # Resume interrupted GOOD_CROP entries before importing another legacy row.
        for row in list_pending(self.db_path, limit=500):
            try:
                path = _safe_crop_path(row["crop_image_ref"], self.roots)
            except (OSError, ValueError):
                continue
            row = dict(row)
            row["image_path"] = str(path)
            row["is_registered"] = True
            return row

        candidates = read_legacy_candidates(
            self.db_path, allowed_crop_roots=self.roots, limit=10000
        )
        known_hashes = known_crop_hashes(self.db_path)
        for row in candidates:
            if row["path_status"] != "PRESENT" or not row["expected_text"]:
                continue
            if not row["source_image_id"]:
                continue
            try:
                path = _safe_crop_path(str(row["crop_image_ref"]), self.roots)
            except (OSError, ValueError):
                continue
            row = dict(row)
            crop_hash = _sha256_file(path)
            if crop_hash in known_hashes:
                record_legacy_exclusion(
                    self.db_path, crop_id=str(row["crop_id"]), crop_hash=crop_hash,
                    source_image_id=str(row["source_image_id"]),
                    reason="DUPLICATE_HASH", actor=self.reviewer,
                )
                continue
            row["crop_hash"] = crop_hash
            row["image_path"] = str(path)
            row["is_registered"] = False
            return row
        return None

    def record(
        self,
        candidate: dict[str, Any],
        quality: str,
        source: str,
        corrected_text: str,
    ) -> str:
        if quality not in (GOOD_CROP, BAD_CROP):
            raise ValueError("Choose GOOD_CROP or BAD_CROP explicitly")
        if source not in EXPECTED_TEXT_SOURCES:
            raise ValueError("Invalid expected-text provenance")
        if quality == GOOD_CROP and not corrected_text.strip():
            raise ValueError("أدخل النص الصحيح قبل تسجيل القصاصة الجيدة")
        path = _safe_crop_path(candidate["crop_image_ref"], self.roots)
        crop_hash = _sha256_file(path)
        if candidate.get("crop_hash") and candidate["crop_hash"] != crop_hash:
            raise ValueError("Crop image changed after it was loaded; reload before saving")
        crop_id = str(candidate["crop_id"])
        if not candidate.get("is_registered"):
            box = candidate.get("bounding_box") or {}
            try:
                register_crop(
                    self.db_path,
                    crop_id=crop_id,
                    crop_hash=crop_hash,
                    crop_image_ref=str(path),
                    source_image_id=str(candidate["source_image_id"]),
                    expected_text=str(candidate["expected_text"]),
                    expected_text_source=source,
                    crop_quality=quality,
                    quality_evidence={
                        "reviewer": self.reviewer,
                        "decision": quality,
                        "checks": ["human_visual_inspection"],
                    },
                    crop_config_version=self.crop_config_version,
                    model_version=self.model_version,
                    batch_id=self.batch_id,
                    bounding_box=box or None,
                    confidence=candidate.get("legacy_confidence"),
                )
            except DuplicateCropError:
                record_legacy_exclusion(
                    self.db_path, crop_id=crop_id, crop_hash=crop_hash,
                    source_image_id=str(candidate["source_image_id"]),
                    reason="DUPLICATE_HASH", actor=self.reviewer,
                )
                return f"تم استبعاد {crop_id}: صورة القصاصة مكررة ببصمة مطابقة."
        elif quality != GOOD_CROP:
            raise ValueError("A registered GOOD_CROP cannot be changed to BAD_CROP")

        if quality == BAD_CROP:
            return f"تم استبعاد {crop_id} كقصاصة غير صالحة؛ لن تدخل التصحيح أو التدريب."
        save_correction(
            self.db_path, crop_id, corrected_text,
            actor=self.reviewer, reason="human-in-the-loop review",
        )
        return f"حُفظ التصحيح للقصاصة {crop_id}. يلزم تحقق مستقل قبل التدريب."

    def skip(self, candidate: dict[str, Any]) -> str:
        path = _safe_crop_path(candidate["crop_image_ref"], self.roots)
        crop_hash = _sha256_file(path)
        record_legacy_exclusion(
            self.db_path,
            crop_id=str(candidate["crop_id"]),
            crop_hash=crop_hash,
            source_image_id=str(candidate["source_image_id"]),
            reason="USER_SKIPPED",
            actor=self.reviewer,
        )
        return f"تم حفظ تخطي {candidate['crop_id']}؛ لن تظهر مجدداً في الطابور."

    def validate(self, crop_id: str) -> str:
        validate_correction(self.db_path, crop_id)
        return f"تم التحقق من {crop_id}. لم تصبح جاهزة للتدريب بعد."

    def make_training_ready(self, crop_id: str) -> str:
        mark_training_ready(self.db_path, crop_id)
        return f"أصبحت {crop_id} جاهزة للتصدير التدريبي المحلي."

    def summary(self) -> str:
        rows = training_rows(self.db_path)
        return f"العينات المعتمدة للتدريب المحلي: {len(rows)}"


def build_app(db_path: Path, roots: tuple[Path, ...], reviewer: str,
              crop_config_version="legacy-unknown", model_version="legacy-unknown",
              batch_id=None):
    import gradio as gr

    initialize(db_path)
    session = ReviewSession(
        db_path, roots, reviewer, crop_config_version, model_version, batch_id
    )
    first = session.next_candidate()

    def render(candidate):
        if candidate is None:
            return None, "", "", "LEGACY_LABEL_UNVERIFIED", None, "لا توجد قصاصات أخرى مؤهلة للعرض ضمن المسارات المحددة."
        warning = (
            "⚠️ النص المتوقع قديم ومجهول المصدر؛ لا تفترض أنه خرج OCR الأصلي."
            if candidate.get("expected_text_source") == "LEGACY_LABEL_UNVERIFIED"
            else ""
        )
        return (
            candidate["image_path"],
            str(candidate.get("expected_text", "")),
            str(candidate["crop_id"]),
            candidate.get("expected_text_source", "LEGACY_LABEL_UNVERIFIED"),
            candidate,
            warning,
        )

    with gr.Blocks(title="HITL — تصحيح قصاصات OCR") as demo:
        gr.Markdown(
            "## مراجعة القصاصات وتصحيح النص\\n"
            "تعمل هذه الواجهة محلياً. لا ترفع الصور أو قاعدة البيانات أو التصحيحات إلى أي خدمة."
        )
        image = gr.Image(label="صورة القصاصة", type="filepath", interactive=False, height=260)
        with gr.Row():
            crop_id = gr.Textbox(label="معرّف القصاصة", interactive=False)
            source = gr.Dropdown(
                choices=EXPECTED_TEXT_SOURCES,
                value="LEGACY_LABEL_UNVERIFIED",
                label="مصدر النص المتوقع",
            )
        expected = gr.Textbox(label="النص المتوقع (للمقارنة)", interactive=False, rtl=True)
        corrected = gr.Textbox(label="النص الصحيح الذي تدخله", lines=2, rtl=True)
        current = gr.State(first)
        last_saved_id = gr.State(None)
        warning = gr.Markdown("")
        with gr.Row():
            good = gr.Button("قصاصة جيدة — حفظ التصحيح", variant="primary")
            bad = gr.Button("قصاصة سيئة — استبعاد")
            skip_btn = gr.Button("تخطي دائم")
            next_btn = gr.Button("التالي")
        status = gr.Markdown("")
        with gr.Row():
            validate_btn = gr.Button("تحقق مستقل من التصحيح")
            ready_btn = gr.Button("اعتمادها للتدريب المحلي")
        summary = gr.Markdown(session.summary())

        def move(candidate, quality=None, provenance="LEGACY_LABEL_UNVERIFIED", text_value="", last_id=None, skip=False):
            message = ""
            reviewed_id = last_id
            if candidate is not None and skip:
                try:
                    message = session.skip(candidate)
                except Exception as exc:
                    img, exp, cid, src, state, warn = render(candidate)
                    return img, exp, cid, src, state, warn, f"لم يُحفظ التخطي: {exc}", session.summary(), reviewed_id, text_value
            if candidate is not None and quality:
                try:
                    message = session.record(candidate, quality, provenance, text_value)
                    reviewed_id = str(candidate["crop_id"]) if quality == GOOD_CROP else last_id
                except Exception as exc:
                    retry = session.next_candidate()
                    img, exp, cid, src, state, warn = render(retry or candidate)
                    return img, exp, cid, src, state, warn, f"لم يُحفظ القرار: {exc}", session.summary(), reviewed_id, text_value
            nxt = session.next_candidate()
            img, exp, cid, src, state, warn = render(nxt)
            return img, exp, cid, src, state, warn, message, session.summary(), reviewed_id, ""

        good.click(
            lambda c, s, t, last: move(c, GOOD_CROP, s, t, last),
            inputs=[current, source, corrected, last_saved_id],
            outputs=[image, expected, crop_id, source, current, warning, status, summary, last_saved_id, corrected],
        )
        bad.click(
            lambda c, s, last: move(c, BAD_CROP, s, "", last),
            inputs=[current, source, last_saved_id],
            outputs=[image, expected, crop_id, source, current, warning, status, summary, last_saved_id, corrected],
        )
        skip_btn.click(
            lambda c, last: move(c, last_id=last, skip=True),
            inputs=[current, last_saved_id],
            outputs=[image, expected, crop_id, source, current, warning, status, summary, last_saved_id, corrected],
        )
        next_btn.click(
            lambda c, last: move(c, last_id=last),
            inputs=[current, last_saved_id],
            outputs=[image, expected, crop_id, source, current, warning, status, summary, last_saved_id, corrected],
        )
        validate_btn.click(
            lambda cid: (
                session.validate(cid) if cid else "لا توجد قصاصة محفوظة للتحقق.",
                session.summary(),
            ),
            inputs=[last_saved_id], outputs=[status, summary],
        )
        ready_btn.click(
            lambda cid: (
                session.make_training_ready(cid) if cid else "لا توجد قصاصة محفوظة للاعتماد.",
                session.summary(),
            ),
            inputs=[last_saved_id], outputs=[status, summary],
        )
        demo.load(
            lambda: (*render(first), session.summary(), ""),
            outputs=[image, expected, crop_id, source, current, warning, summary, corrected],
        )
    return demo


def main():
    parser = argparse.ArgumentParser(description="واجهة HITL محلية لتصحيح قصاصات OCR")
    parser.add_argument("--db", required=True, help="مسار crops.db الموجود مسبقاً")
    parser.add_argument("--crop-root", action="append", required=True,
                        help="مجلد مسموح لصور القصاصات؛ يمكن تكراره")
    parser.add_argument("--reviewer", required=True, help="معرّف المراجع المحلي")
    parser.add_argument("--crop-config-version", default="legacy-unknown")
    parser.add_argument("--model-version", default="legacy-unknown")
    parser.add_argument("--batch-id", default=None,
                        help="معرّف دفعة ثابت؛ إن تُرك فارغاً يُنشأ معرّف محلي تلقائي")
    parser.add_argument("--host", default="127.0.0.1",
                        help="افتراضياً loopback فقط؛ لا تستخدم 0.0.0.0 على شبكة غير موثوقة")
    parser.add_argument("--port", type=int, default=7864)
    args = parser.parse_args()
    db = Path(args.db).expanduser().resolve()
    if not db.is_file():
        parser.error("قاعدة البيانات غير موجودة؛ لن تُنشأ قاعدة جديدة.")
    roots = tuple(Path(p).expanduser().resolve(strict=True) for p in args.crop_root)
    demo = build_app(
        db, roots, args.reviewer, args.crop_config_version,
        args.model_version, args.batch_id,
    )
    demo.queue(max_size=8).launch(
        server_name=args.host, server_port=args.port,
        share=False, show_error=False, inbrowser=False,
    )


if __name__ == "__main__":
    main()
