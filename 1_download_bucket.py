#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الخطوة 1 — تنزيل محتويات الـ bucket (المجلد 2026-10-14) إلى مجلد محلي.

يعتمد على تسجيل دخول Hugging Face الخاص بك:
  - في Colab: أضف سرًّا باسم HF_TOKEN (أيقونة المفتاح) وفعّل Notebook access.
  - محليًا:  hf auth login   أو   متغير البيئة HF_TOKEN.
لا تضع أي رمز داخل أي ملف من ملفات هذه الحزمة.
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from config import BUCKET_ID, BUCKET_PREFIX, BUCKET_DIR


def run(cmd, check=True, capture=False):
    print("$ " + " ".join(cmd), flush=True)
    r = subprocess.run(cmd, check=False, capture_output=capture, text=capture)
    if capture and r.stdout.strip():
        print(r.stdout.rstrip())
    if check and r.returncode != 0:
        if capture and r.stderr.strip():
            print(r.stderr.rstrip(), file=sys.stderr)
        sys.exit(r.returncode or 1)
    return r


def main():
    ap = argparse.ArgumentParser(description="تنزيل محتويات HF Bucket إلى مجلد محلي")
    ap.add_argument("--bucket", default=BUCKET_ID, help="معرف الـ bucket (owner/name)")
    ap.add_argument("--prefix", default=BUCKET_PREFIX, help="المجلد الفرعي داخل الـ bucket")
    ap.add_argument("--dest", default=BUCKET_DIR, help="مجلد التنزيل المحلي")
    ap.add_argument("--skip-list", action="store_true", help="تخطي عرض شجرة الملفات قبل التنزيل")
    args = ap.parse_args()

    hf = shutil.which("hf")
    if not hf:
        sys.exit("❌ أمر hf غير موجود. ثبّت أحدث إصدار:  pip install -U huggingface_hub")

    # ميزة Buckets حديثة نسبيًا — نتأكد أن نسخة المكتبة تدعمها
    probe = subprocess.run([hf, "buckets", "--help"], capture_output=True)
    if probe.returncode != 0:
        sys.exit(
            "❌ نسخة huggingface_hub المثبتة لا تدعم Buckets.\n"
            "   حدّثها ثم أعد المحاولة:  pip install -U huggingface_hub"
        )

    # التحقق من تسجيل الدخول (الـ bucket الخاص لا يمكن قراءته بدون رمز)
    who = subprocess.run([hf, "auth", "whoami"], capture_output=True, text=True)
    if who.returncode != 0 and not os.environ.get("HF_TOKEN"):
        sys.exit(
            "❌ لست مسجل الدخول إلى Hugging Face.\n"
            "   في Colab: أضف سرًّا باسم HF_TOKEN من أيقونة المفتاح وفعّل Notebook access،\n"
            "   أو شغّل:  hf auth login\n"
            "   ملاحظة أمنية: أنشئ رمزًا جديدًا (Read) من huggingface.co/settings/tokens\n"
            "   ولا تلصق الرمز في أي خلية أو ملف."
        )
    run([hf, "auth", "whoami"])

    src_prefix = f"{args.bucket}/{args.prefix.strip('/')}".strip("/")
    if not args.skip_list:
        print("\n—— شجرة الملفات المتاحة للتنزيل ——")
        listed = run([hf, "buckets", "list", src_prefix, "-R"], check=False, capture=True)
        if listed.returncode != 0:
            print("⚠️ تعذر عرض القائمة بالمسار الفرعي؛ محاولة عرض جذر الـ bucket…")
            run([hf, "buckets", "list", args.bucket, "-R"], check=False, capture=True)

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    print(f"\n—— التنزيل إلى {dest} ——")
    run([hf, "buckets", "cp", f"hf://buckets/{args.bucket}/{args.prefix.strip('/')}/", f"{dest}/"])

    # ملخص ما نُزّل
    files = [p for p in dest.rglob("*") if p.is_file()]
    total_mb = sum(p.stat().st_size for p in files) / 1024 ** 2
    print(f"\n✅ تم تنزيل {len(files)} ملفًا ({total_mb:.1f} MB) إلى {dest}")
    print("➡️  الخطوة التالية:  python 2_inspect_backups.py")


if __name__ == "__main__":
    main()
