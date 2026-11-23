#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
اختبار ذاتي للحزمة — لا يحتاج إنترنت ولا GPU ولا نماذج:
  python test_toolkit.py

يفحص: القص والحشو، الاستخراج الآمن (رفض عبور المسارات)، قاعدة البيانات،
توزيع التقسيم على مستوى الصفحة، والتصدير الكامل (إن كانت مكتبة datasets مثبتة).
"""
import py_compile
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import zipfile
import tarfile
import io
from pathlib import Path

import PIL.Image as Image

import toolkit_utils as tu

PKG = Path(__file__).resolve().parent
RESULTS = []


def check(name, fn):
    try:
        fn()
        RESULTS.append((name, "PASS", ""))
        print(f"✅ {name}")
    except Exception as e:
        RESULTS.append((name, "FAIL", str(e)))
        print(f"❌ {name}: {e}")


def t_crop():
    page = Image.new("RGB", (800, 600), "white")
    for x in range(100, 300):
        for y in range(50, 120):
            page.putpixel((x, y), (0, 0, 0))
    c0 = tu.crop_word_image(page, 100, 50, 300, 120, pad=0)
    assert c0.size == (200, 70), c0.size
    c1 = tu.crop_word_image(page, 100, 50, 300, 120, pad=6)
    assert c1.size == (212, 82), c1.size
    # إحداثيات خارج الحدود تُقص تلقائيًا ولا تُنهار
    c2 = tu.crop_word_image(page, 780, 560, 900, 700, pad=0)
    assert c2.size == (20, 40), c2.size
    try:
        tu.crop_word_image(page, 10, 10, 11, 11, pad=0)
        raise AssertionError("كان يجب رفض الصندوق الأصغر من 3 بكسل")
    except ValueError:
        pass


def t_pad_square():
    im = Image.new("RGB", (100, 50), "red")   # العرض هو البُعد الأكبر → حشوة عمودية
    s = tu.pad_to_square(im)
    assert s.size == (100, 100)
    assert s.getpixel((95, 50)) == (255, 0, 0)      # داخل الصورة بعد اللصق
    assert s.getpixel((50, 5)) == (255, 255, 255)   # حشوة بيضاء أعلى الصورة
    s2 = tu.pad_to_square(Image.new("RGB", (50, 100), "blue"))
    assert s2.getpixel((5, 50)) == (255, 255, 255)  # حشوة بيضاء يمين الصورة العمودية


def t_safe_zip():
    tmp = Path(tempfile.mkdtemp())
    try:
        evil = tmp / "evil.zip"
        with zipfile.ZipFile(evil, "w") as zf:
            zf.writestr("../escape.txt", "pwn")
            zf.writestr("ok/data.txt", "fine" * 100)
        try:
            tu.safe_extract_zip(evil, tmp / "out_evil")
            raise AssertionError("كان يجب رفض مسار ../ داخل الأرشيف")
        except ValueError:
            pass
        assert not (tmp / "escape.txt").exists()

        good = tmp / "good.zip"
        with zipfile.ZipFile(good, "w") as zf:
            zf.writestr("docs/page1.txt", "abc")
            zf.writestr("docs/sub/page2.txt", "def")
        n = tu.safe_extract_zip(good, tmp / "out_good")
        assert n == 2 and (tmp / "out_good/docs/sub/page2.txt").read_text() == "def"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def t_safe_tar():
    tmp = Path(tempfile.mkdtemp())
    try:
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tf:
            data = b"hello"
            info = tarfile.TarInfo("../evil.txt"); info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
        buf.seek(0)
        evil = tmp / "evil.tar.gz"
        evil.write_bytes(buf.getvalue())
        try:
            tu.safe_extract_tar(evil, tmp / "out_evil")
            raise AssertionError("كان يجب رفض مسار ../ داخل الأرشيف")
        except ValueError:
            pass

        buf2 = io.BytesIO()
        with tarfile.open(fileobj=buf2, mode="w:gz") as tf:
            info = tarfile.TarInfo("repo/img.png"); info.size = 4
            tf.addfile(info, io.BytesIO(b"PNG!"))
        buf2.seek(0)
        good = tmp / "good.tar.gz"; good.write_bytes(buf2.getvalue())
        n = tu.safe_extract_tar(good, tmp / "out_good")
        assert n == 1 and (tmp / "out_good/repo/img.png").exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def t_db_and_splits():
    tmp = Path(tempfile.mkdtemp())
    try:
        db = tmp / "t.db"
        conn = tu.connect_db(db)
        pages = [f"page_{i:03d}.jpg" for i in range(12)]
        for p in pages:
            tu.add_page_if_missing(conn, p)
            for w in range(5):
                conn.execute(
                    "INSERT INTO crops(page_name,crop_path,label,x1,y1,x2,y2) VALUES (?,?,?,?,?,?,?)",
                    (p, f"crops/{p[:-4]}_w{w:03d}.png", f"كلمة{w}", 1, 2, 3, 4))
        conn.commit()
        n = conn.execute("SELECT COUNT(*) FROM crops").fetchone()[0]
        assert n == 60, n
        splits = tu.assign_splits(pages, (0.9, 0.05, 0.05), 42)
        assert set(splits.values()) <= {"train", "val", "test"}
        assert len(splits) == 12
        # نفس الصفحة لا تنقسم بين تقسيمين (لا تسريب)
        q = conn.execute("""
            SELECT page_name, COUNT(DISTINCT split) AS c FROM crops
            WHERE split != '' GROUP BY page_name HAVING c > 1""").fetchall()
        assert not q, q
        assert tu.charset_of(["سلام", "عليك"]) == "".join(sorted(set("سلامعليك")))
        conn.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def t_export_e2e():
    try:
        import datasets  # noqa: F401
        import pandas  # noqa: F401
    except ImportError:
        print("   (تخطي اختبار التصدير الكامل — مكتبة datasets غير مثبتة محليًا؛")
        print("    تعمل تلقائيًا داخل Colab بعد pip install -r requirements.txt)")
        return
    tmp = Path(tempfile.mkdtemp())
    try:
        db = tmp / "t.db"
        conn = tu.connect_db(db)
        for pi in range(8):
            page = f"page_{pi:03d}.jpg"
            tu.add_page_if_missing(conn, page)
            for w in range(4):
                cp = tmp / f"{page[:-4]}_w{w:03d}.png"
                Image.new("RGB", (60, 30), "white").save(cp)
                conn.execute(
                    "INSERT INTO crops(page_name,crop_path,label,x1,y1,x2,y2) VALUES (?,?,?,?,?,?,?)",
                    (page, str(cp), f"كلمة{w}", 0, 0, 60, 30))
        conn.commit(); conn.close()
        r = subprocess.run(
            [sys.executable, "4_export_dataset.py", "--db", str(db),
             "--out", str(tmp / "ds"), "--audit", str(tmp / "audit"), "--seed", "7"],
            cwd=PKG, capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        from datasets import load_from_disk
        ds = load_from_disk(str(tmp / "ds"))
        assert "train" in ds and "text" in ds["train"].column_names
        assert len(ds["train"]) + len(ds.get("val", [])) + len(ds.get("test", [])) == 32
        assert (tmp / "audit/labels_train.csv").exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def t_compile():
    for f in sorted(PKG.glob("*.py")):
        if f.name != Path(__file__).name:
            py_compile.compile(str(f), doraise=True)


if __name__ == "__main__":
    check("تجميع كل ملفات الحزمة py_compile", t_compile)
    check("قص الكلمات + الحشو + حدود الصورة", t_crop)
    check("حشو الصورة إلى مربع", t_pad_square)
    check("استخراج zip آمن (رفض ../)", t_safe_zip)
    check("استخراج tar آمن (رفض ../)", t_safe_tar)
    check("قاعدة البيانات + التقسيم بلا تسريب", t_db_and_splits)
    check("تصدير مجموعة البيانات كاملًا", t_export_e2e)

    fails = [r for r in RESULTS if r[1] == "FAIL"]
    print(f"\nالنتيجة: {len(RESULTS) - len(fails)}/{len(RESULTS)} نجاح")
    if fails:
        sys.exit(1)
