# -*- coding: utf-8 -*-
"""
دوال مشتركة بين السكربتات: استخراج آمن للأرشيفات، تجهيز صور الصفحات،
قص الكلمات، وقاعدة بيانات SQLite.

الأمان: الأرشيفات القادمة من الـ bucket تُعامل كبيانات غير موثوقة —
الاستخراج يفحص مسارات العبور (../) ولا ينفّذ أي ملف مستخرج.
"""
import re
import shutil
import sqlite3
import zipfile
import tarfile
import hashlib
from pathlib import Path

from PIL import Image, ImageOps

IMAGE_EXTS   = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz", ".tbz2", ".tar.bz2")
PDF_EXT = ".pdf"

MAX_ARCHIVE_ENTRIES = 50000
MAX_ARCHIVE_BYTES   = 10 * 1024 ** 3     # 10 GB فك ضغط كحد أقصى لكل أرشيف
MAX_MEMBER_BYTES    = 1 * 1024 ** 3      # 1 GB لكل ملف مفرد

# ────────────────────────────── قاعدة البيانات ──────────────────────────────

SCHEMA = """
CREATE TABLE IF NOT EXISTS crops (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    page_name TEXT NOT NULL,
    crop_path TEXT NOT NULL,
    label     TEXT NOT NULL,
    x1 INTEGER, y1 INTEGER, x2 INTEGER, y2 INTEGER,
    split     TEXT NOT NULL DEFAULT '',
    ignored   INTEGER NOT NULL DEFAULT 0,
    verified  INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS pages (
    name TEXT PRIMARY KEY,
    done INTEGER NOT NULL DEFAULT 0,
    imported_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_crops_page  ON crops(page_name);
CREATE INDEX IF NOT EXISTS idx_crops_split ON crops(split);
"""


def connect_db(db_path):
    """فتح/إنشاء قاعدة البيانات مع المخطط الكامل."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.executescript(SCHEMA)
    return conn


def add_page_if_missing(conn, name):
    conn.execute("INSERT OR IGNORE INTO pages(name) VALUES (?)", (name,))
    conn.commit()


def page_images(pages_dir):
    """قائمة صور الصفحات مرتبةً تصاعديًا بالاسم."""
    d = Path(pages_dir)
    if not d.exists():
        return []
    return sorted(str(p) for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


# ────────────────────────────── الاستخراج الآمن ─────────────────────────────

def _safe_target(dest: Path, member_name: str) -> Path:
    """التحقق من عدم وجود عبور مسار (path traversal) أو مسار مطلق."""
    name = member_name.replace("\\", "/")
    if name.startswith("/") or re.match(r"^[A-Za-z]:", name):
        raise ValueError(f"مسار مطلق غير آمن داخل الأرشيف: {member_name!r}")
    if ".." in Path(name).parts:
        raise ValueError(f"مسار عبور غير آمن داخل الأرشيف: {member_name!r}")
    return dest / name


def safe_extract_zip(zip_path, dest, max_entries=MAX_ARCHIVE_ENTRIES,
                     max_total=MAX_ARCHIVE_BYTES, max_member=MAX_MEMBER_BYTES):
    """استخراج zip مع حماية من Zip Bomb وعبور المسارات."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    count, total = 0, 0
    with zipfile.ZipFile(zip_path) as zf:
        infos = zf.infolist()
        if len(infos) > max_entries:
            raise ValueError(f"عدد ملفات الأرشيف ({len(infos)}) يتجاوز الحد المسموح {max_entries}")
        for info in infos:
            if info.file_size > max_member:
                raise ValueError(f"ملف أكبر من الحد المسموح داخل الأرشيف: {info.filename}")
            total += info.file_size
            if total > max_total:
                raise ValueError("الحجم الكلي للأرشيف يتجاوز الحد المسموح (احتمال Zip Bomb)")
            target = _safe_target(dest, info.filename)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)
            count += 1
    return count


def safe_extract_tar(tar_path, dest, max_entries=MAX_ARCHIVE_ENTRIES):
    """استخراج tar/tar.gz مع فحص كل الأعضاء قبل الفك (مسارات، أنواع، روابط)."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:*") as tf:
        members = tf.getmembers()
        if len(members) > max_entries:
            raise ValueError(f"عدد ملفات الأرشيف ({len(members)}) يتجاوز الحد المسموح {max_entries}")
        for m in members:
            _safe_target(dest, m.name)          # فحص المسار
            if not (m.isreg() or m.isdir()):
                raise ValueError(f"عضو غير مدعوم داخل الأرشيف (رابط/جهاز): {m.name!r}")
        try:
            tf.extractall(dest, filter="data")   # Python 3.12+
        except TypeError:
            tf.extractall(dest)
    return sum(1 for m in members if m.isreg())


def archive_suffix(path):
    """إرجاع اللاحقة الكاملة (تدعم tar.gz/tar.bz2)."""
    low = str(path).lower()
    for suf in ARCHIVE_SUFFIXES:
        if low.endswith(suf):
            return suf
    p = Path(path)
    return p.suffix.lower()


def extract_archive(path, dest_root):
    """فك ضغط أرشيف واحد داخل dest_root/<اسم_الأرشيف>/ وإرجاع (عدد الملفات، المجلد)."""
    path, dest_root = Path(path), Path(dest_root)
    suf = archive_suffix(path)
    subdir = dest_root / path.name[: -len(suf) if suf else None].replace("/", "_")
    if suf == ".zip":
        n = safe_extract_zip(path, subdir)
    elif suf in ARCHIVE_SUFFIXES:
        n = safe_extract_tar(path, subdir)
    else:
        raise ValueError(f"صيغة أرشيف غير مدعومة: {path}")
    return n, subdir


# ────────────────────────────── تجهيز صور الصفحات ───────────────────────────

def _unique_page_name(rel_path, taken):
    """اسم صفحة فريد ومستقر من مسار المصدر (hash + الاسم المبدل)."""
    key = hashlib.sha1(str(rel_path).encode("utf-8")).hexdigest()[:8]
    stem = Path(str(rel_path)).stem
    stem = re.sub(r"[^\w.\-]+", "_", stem, flags=re.UNICODE)[:60] or "page"
    ext = Path(str(rel_path)).suffix.lower()
    name = f"{key}_{stem}{ext}"
    i = 1
    while name in taken:
        name = f"{key}_{stem}_{i}{ext}"
        i += 1
    return name


def stage_images(src_roots, pages_dir, db_path=None, min_kb=50, min_px=200,
                 max_pages=300, render_pdfs=True, pdf_max_pages=50):
    """
    نسخ صور الصفحات من مجلدات المصدر إلى pages_dir بأسماء فريدة ومستقرة،
    مع تصحيح دوران EXIF، ورندر ملفات PDF إلى صور (إن توفر PyMuPDF).

    الإرجاع: dict بإحصاءات العملية.
    """
    pages_dir = Path(pages_dir)
    pages_dir.mkdir(parents=True, exist_ok=True)
    taken = {p.name for p in pages_dir.iterdir()}
    conn = connect_db(db_path) if db_path else None

    images, pdfs = [], []
    for root in src_roots:
        root = Path(root)
        if not root.exists():
            continue
        for p in sorted(root.rglob("*")):
            if not p.is_file():
                continue
            suf = archive_suffix(p) if p.name.lower().endswith(ARCHIVE_SUFFIXES) else p.suffix.lower()
            if suf in IMAGE_EXTS:
                images.append(p)
            elif suf == PDF_EXT:
                pdfs.append(p)

    stats = {"images_found": len(images), "pdfs_found": len(pdfs),
             "staged": 0, "pdf_pages_rendered": 0, "skipped_small": 0,
             "skipped_invalid": 0, "exif_fixed": 0, "pdf_lib_missing": False, "warnings": []}

    # PDFs → صور
    if pdfs and render_pdfs:
        try:
            import fitz  # PyMuPDF
            for pdf in pdfs:
                key = hashlib.sha1(str(pdf).encode("utf-8")).hexdigest()[:8]
                stem = re.sub(r"[^\w.\-]+", "_", pdf.stem)[:40] or "doc"
                doc = fitz.open(str(pdf))
                for i in range(min(len(doc), pdf_max_pages)):
                    pix = doc[i].get_pixmap(dpi=250)
                    name = f"{key}_{stem}_p{i:03d}.png"
                    if name not in taken:
                        pix.save(str(pages_dir / name))
                        taken.add(name)
                    stats["pdf_pages_rendered"] += 1
                doc.close()
                if len(doc) > pdf_max_pages:
                    stats["warnings"].append(f"PDF أطول من {pdf_max_pages} صفحة، تم تجاهل الباقي: {pdf.name}")
        except ImportError:
            stats["pdf_lib_missing"] = True

    # الصور
    staged_names = []
    for img_path in images:
        if stats["staged"] >= max_pages:
            stats["warnings"].append(f"تم الوصول للحد الأقصى {max_pages} صورة؛ الباقي لم يُنسخ (ارفع max_pages)")
            break
        if img_path.stat().st_size < min_kb * 1024:
            stats["skipped_small"] += 1
            continue
        rel = None
        for root in src_roots:
            root = Path(root)
            try:
                rel = img_path.relative_to(root)
                break
            except ValueError:
                continue
        rel = rel if rel is not None else Path(img_path.name)
        name = _unique_page_name(rel, taken)
        target = pages_dir / name
        try:
            with Image.open(img_path) as im:
                im.load()
                w, h = im.size
                if w < min_px or h < min_px:
                    stats["skipped_small"] += 1
                    continue
                fixed = ImageOps.exif_transpose(im)
                if fixed is not im:
                    fixed.convert("RGB").save(target)   # إعادة حفظ بعد تصحيح الدوران
                    stats["exif_fixed"] += 1
                else:
                    shutil.copy2(img_path, target)
        except Exception:
            stats["skipped_invalid"] += 1
            continue
        taken.add(name)
        staged_names.append(name)
        stats["staged"] += 1

    if conn:
        for name in sorted(taken):
            add_page_if_missing(conn, name)
        conn.close()
    stats["pages_total"] = len(taken)
    return stats


# ────────────────────────────── قص الكلمات ──────────────────────────────────

def crop_word_image(page_img, x1, y1, x2, y2, pad=6, bg=(255, 255, 255)):
    """قص كلمة من صفحة PIL مع هوامش بيضاء؛ الإحداثيات مقصوصة داخل حدود الصورة."""
    W, H = page_img.size
    x1, x2 = sorted((max(0, min(int(x1), W)), max(0, min(int(x2), W))))
    y1, y2 = sorted((max(0, min(int(y1), H)), max(0, min(int(y2), H))))
    if x2 - x1 < 3 or y2 - y1 < 3:
        raise ValueError("صندوق القص أصغر من 3 بكسل — حدد الكلمة بدقة أكبر")
    crop = page_img.crop((x1, y1, x2, y2)).convert("RGB")
    if pad > 0:
        canvas = Image.new("RGB", (crop.width + 2 * pad, crop.height + 2 * pad), bg)
        canvas.paste(crop, (pad, pad))
        crop = canvas
    return crop


def pad_to_square(img, fill=(255, 255, 255)):
    """تمديد الصورة إلى مربع بحشوة بيضاء (لإدخال ViT دون تشويه النسبة)."""
    img = img.convert("RGB")
    w, h = img.size
    m = max(w, h)
    canvas = Image.new("RGB", (m, m), fill)
    canvas.paste(img, ((m - w) // 2, (m - h) // 2))
    return canvas


# ────────────────────────────── تقسيم البيانات ──────────────────────────────

def assign_splits(page_names, ratios=(0.9, 0.05, 0.05), seed=42):
    """
    توزيع الصفحات على train/val/test على مستوى الصفحة (كل قصاصات الصفحة
    في نفس التقسيم) مع إرجاع dict: اسم الصفحة → التقسيم.
    """
    pages = sorted(set(page_names))
    rng = __import__("random").Random(seed)
    rng.shuffle(pages)
    n = len(pages)
    n_train = int(round(n * ratios[0]))
    n_val = int(round(n * ratios[1])) if n >= 3 else 0
    splits = {}
    for i, p in enumerate(pages):
        if i < n_train:
            splits[p] = "train"
        elif i < n_train + n_val:
            splits[p] = "val"
        else:
            splits[p] = "test"
    return splits


def charset_of(labels):
    """مجموعة المحارف المستخدمة في الوسوم (للمراجعة قبل التدريب)."""
    chars = set()
    for t in labels:
        chars.update(str(t).strip())
    return "".join(sorted(chars))
