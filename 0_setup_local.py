#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الخطوة 0 — تجهيز بيانات المشروع على اللابتوب من المستودع الخاص على Hugging Face.

يجلب ويضع في أماكنها الصحيحة:
  ocr_data/crops.db          قاعدة القصاصات (الوسوم + الإحداثيات + التقسيم)
  ocr_data/crops/            صور القصاصات (من crops.tar.gz — تُفك هنا)
  ocr_data/repo_pages/       لوحات الصفحات الأصلية (كانفس المربعات)
  ocr_data/extracted/.../notes.db   قاعدة الأسطر من مشروع handwriting-notes

الأمر:
  python 0_setup_local.py            # كل شيء (يشمل صور القصاصات — يلزم للتصدير)
  python 0_setup_local.py --no-crops # بدون صور القصاصات (يكفي لتشغيل محرر المربعات)

الرمز: من متغير البيئة HF_TOKEN أو إدخال مخفي عند التشغيل — لا يُكتب في أي ملف.
السكربت آمن للتكرار: يخطي ما هو موجود أصلًا.
"""
import argparse
import shutil
import sys
import tarfile
from pathlib import Path

TOOLKIT = Path(__file__).resolve().parent
DATA = TOOLKIT / "ocr_data"
REPO = "DrAbdulmalek/handwriting-ocr-crops"
NOTES_DST = DATA / ("extracted/handwriting-notes-meta/"
                    "handwriting-notes/data/notes.db")


def have_token():
    import os
    if os.environ.get("HF_TOKEN"):
        return True
    try:
        from getpass import getpass
        tok = getpass("🔑 الصق رمز Hugging Face (Read — لن يظهر): ").strip()
        if tok:
            os.environ["HF_TOKEN"] = tok
            return True
    except (KeyboardInterrupt, EOFError):
        pass
    return False


def main():
    ap = argparse.ArgumentParser(description="تجهيز بيانات المشروع من HF الخاص")
    ap.add_argument("--no-crops", action="store_true",
                    help="تخطي صور القصاصات (يكفي لمحرر المربعات فقط)")
    args = ap.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("❌ ناقص:  pip install -U huggingface_hub")

    if not have_token():
        sys.exit("❌ لا يمكن التنزيل بلا رمز — المستودع خاص. "
                 "أنشئ رمز Read من huggingface.co/settings/tokens")

    DATA.mkdir(parents=True, exist_ok=True)

    patterns = []
    if not (DATA / "crops.db").exists():
        patterns.append("crops.db")
    if not NOTES_DST.exists():
        patterns.append("notes.db")
    if not any((DATA / "repo_pages").glob("*.png")):
        patterns.append("repo_pages/*")
    crops_ok = any((DATA / "crops").glob("*.png")) if (DATA / "crops").exists() else False
    if not args.no_crops and not crops_ok:
        patterns.append("crops.tar.gz")

    if not patterns:
        print("✅ كل البيانات موجودة أصلًا — لا تنزيل")
    else:
        print(f"⏳ تنزيل {len(patterns)} عنصرًا من {REPO} (خاص)…")
        snapshot_download(REPO, repo_type="dataset", local_dir=DATA,
                          allow_patterns=patterns)

    # notes.db إلى موضع القراءة عند الخادم
    src_notes = DATA / "notes.db"
    if src_notes.exists() and not NOTES_DST.exists():
        NOTES_DST.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src_notes), str(NOTES_DST))

    # فك أرشيف القصاصات
    tar = DATA / "crops.tar.gz"
    if tar.exists() and not crops_ok:
        print("⏳ فك أرشيف القصاصات crops.tar.gz…")
        with tarfile.open(tar, "r:gz") as zf:
            zf.extractall(DATA)  # يحوي مجلد crops/ فقط — أعضاء محفوظة الأسماء
        tar.unlink()
    elif tar.exists() and crops_ok:
        tar.unlink()  # القصاصات موجودة — الأرشيف زائد

    # ── ملخص ──
    import sqlite3
    print("\n———— ملخص التجهيز ————")
    try:
        db = sqlite3.connect(DATA / "crops.db")
        n_words = db.execute("SELECT COUNT(*) FROM crops").fetchone()[0]
        n_pages = db.execute("SELECT COUNT(*) FROM pages").fetchone()[0]
        db.close()
        print(f"  قاعدة القصاصات: {n_words} قصاصة على {n_pages} صفحة")
    except Exception as e:
        print(f"  ⚠️ قاعدة القصاصات: {e}")
    n_canvases = sum(1 for _ in (DATA / "repo_pages").glob("*.png"))
    print(f"  لوحات الصفحات:  {n_canvases}")
    if NOTES_DST.exists():
        print("  قاعدة الأسطر:   موجودة ✓")
    n_crops = sum(1 for _ in (DATA / "crops").glob("*.png")) if (DATA / "crops").exists() else 0
    print(f"  صور القصاصات:   {n_crops}")

    print("\n➡️  التالي:")
    print("   python 7_box_editor_server.py --host 0.0.0.0   # محرر المربعات + QR للجوال")
    print("   python 3b_review_app.py                        # مراجعة الوسوم (Gradio)")
    if n_crops:
        print("   python 4_export_dataset.py                     # تحديث مجموعة التدريب بعد المراجعة")
    else:
        print("   (بدون صور القصاصات: المحرر يعمل، والتصدير يحتاج  python 0_setup_local.py  كاملًا)")


if __name__ == "__main__":
    main()
