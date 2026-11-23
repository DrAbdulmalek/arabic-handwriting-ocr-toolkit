#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الخطوة 2 — فحص النسخ الاحتياطية، استخراج الأرشيفات بأمان، وتجهيز صور الصفحات.

ماذا يفعل:
  1. يجمع الأرشيفات (zip/tar/tar.gz) من مجلد التنزيل ويستخرجها بأمان
     (فحص عبور المسارات وحدود الحجم — ولا تنفيذ لأي ملف مستخرج).
  2. يرندر ملفات PDF إلى صور صفحات (إن توفر PyMuPDF).
  3. ينسخ الصور المؤهلة إلى ocr_data/pages بأسماء فريدة مع تصحيح دوران EXIF.
  4. يسجل الصفحات في قاعدة البيانات ويكتب تقرير فحص inventory.json.

ملاحظة: هذه نسخ احتياطية لمستودعات GitHub — قد تحتوي صورًا واجهة/أيقونات
ليست صفحات خط يدوي؛ الأداة في الخطوة 3 تعرضها واحدة واحدة فتجاهل ما لا تريده.
"""
import argparse
import json
import sys
from pathlib import Path

from config import BUCKET_DIR, EXTRACTED_DIR, PAGES_DIR, DB_PATH
import toolkit_utils as tu


def main():
    ap = argparse.ArgumentParser(description="فحص النسخ الاحتياطية وتجهيز صفحات الوسم")
    ap.add_argument("--bucket-dir", default=BUCKET_DIR)
    ap.add_argument("--extracted-dir", default=EXTRACTED_DIR)
    ap.add_argument("--pages-dir", default=PAGES_DIR)
    ap.add_argument("--db", default=DB_PATH)
    ap.add_argument("--min-kb", type=int, default=50, help="أصغر حجم صورة مقبول (KB)")
    ap.add_argument("--min-px", type=int, default=200, help="أصغر بُعد صورة مقبول (بكسل)")
    ap.add_argument("--max-pages", type=int, default=300, help="أقصى عدد صور تُنسخ إلى pages/")
    ap.add_argument("--no-pdf", action="store_true", help="عدم رندر ملفات PDF")
    args = ap.parse_args()

    bucket_dir = Path(args.bucket_dir)
    if not bucket_dir.exists() or not any(bucket_dir.iterdir()):
        sys.exit(f"❌ مجلد التنزيل {bucket_dir} فارغ — شغّل أولًا:  python 1_download_bucket.py")

    # 1) الأرشيفات → استخراج آمن
    archives = [p for p in bucket_dir.rglob("*")
                if p.is_file() and tu.archive_suffix(p) in tu.ARCHIVE_SUFFIXES]
    unsupported = [p for p in bucket_dir.rglob("*")
                   if p.is_file() and p.suffix.lower() in (".7z", ".rar")]

    extracted_files = 0
    extract_report = []
    for arc in sorted(archives):
        try:
            n, subdir = tu.extract_archive(arc, args.extracted_dir)
            extracted_files += n
            extract_report.append({"archive": str(arc), "files": n, "to": str(subdir)})
            print(f"✅ {arc.name}: {n} ملفًا → {subdir}")
        except Exception as e:
            extract_report.append({"archive": str(arc), "error": str(e)})
            print(f"⚠️ تعذر فك {arc.name}: {e}")

    if unsupported:
        print(f"⚠️ {len(unsupported)} أرشيف بصيغة 7z/rar غير مدعومة — فكّها يدويًا ثم ضع الناتج في {args.extracted_dir}")

    # 2) تقرير موجز عن محتوى المستخرجات
    code_files = doc_files = 0
    for root in (bucket_dir, Path(args.extracted_dir)):
        root = Path(root)
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.is_file():
                suf = p.suffix.lower()
                if suf in {".py", ".js", ".ts", ".java", ".c", ".cpp", ".h", ".go", ".rs", ".sh", ".ipynb", ".md"}:
                    code_files += 1
                elif suf in {".pdf", ".docx", ".doc"}:
                    doc_files += 1

    # 3) تجهيز صور الصفحات
    print("\n—— تجهيز صور الصفحات ——")
    stats = tu.stage_images(
        src_roots=[bucket_dir, args.extracted_dir],
        pages_dir=args.pages_dir,
        db_path=args.db,
        min_kb=args.min_kb,
        min_px=args.min_px,
        max_pages=args.max_pages,
        render_pdfs=not args.no_pdf,
    )
    for w in stats["warnings"]:
        print(f"⚠️ {w}")
    if stats.get("pdf_lib_missing"):
        print("⚠️ PyMuPDF غير مثبت — لم تُرندر ملفات PDF. ثبّته:  pip install PyMuPDF")

    # 4) تقرير فحص
    inventory = {
        "archives": extract_report,
        "unsupported_archives": [str(p) for p in unsupported],
        "extracted_files": extracted_files,
        "code_files_seen": code_files,
        "doc_files_seen": doc_files,
        "images_found": stats["images_found"],
        "pdfs_found": stats["pdfs_found"],
        "pdf_pages_rendered": stats["pdf_pages_rendered"],
        "pages_staged": stats["staged"],
        "skipped_small": stats["skipped_small"],
        "skipped_invalid": stats["skipped_invalid"],
        "exif_fixed": stats["exif_fixed"],
        "pages_total": stats["pages_total"],
        "pages_dir": args.pages_dir,
    }
    inv_path = Path(args.pages_dir) / "inventory.json"
    inv_path.parent.mkdir(parents=True, exist_ok=True)
    inv_path.write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"""
———————— تقرير الفحص ————————
أرشيفات استُخرجت          : {len(extract_report)}  ({extracted_files} ملفًا)
صور مؤهلة وُجدت           : {stats['images_found']}   | صفحات PDF مُرندرة: {stats['pdf_pages_rendered']}
صور جاهزة للوسم (staged)  : {stats['staged']}  (صغيرة جدًا: {stats['skipped_small']} | تالفة: {stats['skipped_invalid']} | تصحيح EXIF: {stats['exif_fixed']})
إجمالي صفحات pages/       : {stats['pages_total']}
ملفات برمجية وُجدت (لم تُنفذ): {code_files}
التقرير الكامل            : {inv_path}
""")

    if stats["pages_total"] == 0:
        print("❌ لا توجد صور صفحات جاهزة. تأكد أن النسخ الاحتياطية تحتوي صورًا أو PDF صفحات ممسوحة.")
    else:
        print(f"➡️  الخطوة التالية:  python 3_labeling_app.py   ثم قص وسم الكلمات صفحةً صفحة.")


if __name__ == "__main__":
    main()
