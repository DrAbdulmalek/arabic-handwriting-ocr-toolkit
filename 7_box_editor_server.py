#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الخطوة 7 — خادم مراجعة وتعديل مربعات الكلمات على الصفحات.

يعرض كل صفحة كاملة أو سطرًا أو جزءًا من سطر مع مربعات الكلمات، ويسمح بسحب حواف
كل مربع لتوسيعه أو تضييقه (لأن التقسيم الآلي قد يقطع الكلمة أو يضمّ أكثر من كلمة)،
مع تقسيم مربع، دمج مربعين، إضافة مربع جديد، وتصحيح الوسم.

التشغيل:
    python 7_box_editor_server.py            # ثم افتح http://127.0.0.1:7863
    python 7_box_editor_server.py --port 9000

في Colab (خلية مجهزة في OCR_Pipeline_Colab.ipynb تشغّله وتطبع رابط الوكيل).

نقاط النهاية:
  GET  /api/pages    قائمة الصفحات + عدّادات المراجعة
  GET  /api/page     تفاصيل صفحة: الأسطر + الكلمات + الإحداثيات
  GET  /api/auto_box مربع ضيق حول الكلمة المُنقورة فقط (بروفيل أعمدة الحبر)
  GET  /api/analyze  تحليل الصفحة كله: مربعات مبدئية لكل الكلمات + فواصل الأعمدة
  GET  /api/export_page تصدير نص الصفحة (txt/docx) بترتيب القراءة — الأعلى أولًا
  GET  /api/log      آخر أحداث سجل النشاط app_log.jsonl
  GET  /api/consent  حالة إذن إرسال بيانات الاستخدام المجهولة للمطوّر
  GET  /img/page     صورة الصفحة كاملة
  GET  /img/crop     قص أي منطقة من الصفحة (عرض السطر/الجزء المكبّر)
  POST /api/update   تحديث مربعات/وسوم/حالات دفعة واحدة
  POST /api/add      إضافة مربع جديد
  POST /api/add_many إضافة دفعة مربعات (نتيجة التحليل الآلي)
  POST /api/tighten تقليص المربعات إلى حدود الحبر تمامًا (نمط ABBYY) وإعادة قصها
  POST /api/delete   تجاهل مربعات (تبقى في القاعدة معلّمة ignored=1)
  POST /api/merge    دمج مربعين في واحد
  POST /api/consent  تسجيل إذن/رفض المستخدم للإرسال
  POST /api/telemetry استقبال أحداث الواجهة (يُسجَّل فقط بعد الإذن)
كل تغيير يُسجَّل في ocr_data/box_adjustments.jsonل للتدقيق، وسجل النشاط العام
في ocr_data/app_log.jsonl، وصورة القصاصة تُعاد توليدها تلقائيًا عند تعديل المربع.
"""
import argparse
import io
import json
import sqlite3
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

try:
    from PIL import Image
except ImportError:
    sys.exit("❌ مكتبة Pillow مطلوبة:  pip install pillow")

TOOLKIT = Path(__file__).resolve().parent
DB_PATH = TOOLKIT / "ocr_data/crops.db"
NOTES_DB = TOOLKIT / ("ocr_data/extracted/handwriting-notes-meta/"
                      "handwriting-notes/data/notes.db")
PAGES_DIR = TOOLKIT / "ocr_data/repo_pages"   # لوحات processed/<الصفحة>/page.png
CROPS_DIR = TOOLKIT / "ocr_data/crops"
HTML_PATH = TOOLKIT / "box_editor.html"
PANEL_PATH = TOOLKIT / "control_panel.html"
ADJ_LOG = TOOLKIT / "ocr_data/box_adjustments.jsonl"
APP_LOG = TOOLKIT / "ocr_data/app_log.jsonl"      # سجل النشاط العام
TELEMETRY = TOOLKIT / "ocr_data/telemetry.json"   # إذن إرسال البيانات للمطوّر
APP_VERSION = "1.5.0"

_db_lock = threading.RLock()  # قابل لإعادة الدخول: log_adj يُستدعى داخل الحاملين للقفل
_img_cache = {}
_img_cache_lock = threading.Lock()

PAGE_MAP = {}   # page_name -> Path لملف صورة الصفحة
PAGE_SIZE = {}  # page_name -> (width, height)


# ─────────────────────────── تجهيز خرائط الصفحات ───────────────────────────
def build_page_maps():
    if not DB_PATH.exists():
        sys.exit(f"❌ قاعدة البيانات {DB_PATH} غير موجودة")
    if not NOTES_DB.exists():
        sys.exit(f"❌ قاعدة الأسطر {NOTES_DB} غير موجودة (مطلوبة لعرض الأسطر)")
    nconn = sqlite3.connect(f"file:{NOTES_DB}?mode=ro", uri=True)
    src = {}
    for sname, w, h in nconn.execute(
            "SELECT source_name, width, height FROM pages"):
        src[sname] = (w, h)
    nconn.close()

    files = {f.name[:-4]: f for f in PAGES_DIR.glob("*.png")}  # <page_name>.png

    unmapped = []
    for name in src:
        f = files.get(name)
        if f:
            PAGE_MAP[name] = f
            PAGE_SIZE[name] = src[name]
        else:
            unmapped.append(name)
    if unmapped:
        print(f"⚠️ {len(unmapped)} صفحة بلا صورة (ستُتجاهل): {unmapped[:3]}")
    print(f"✓ خرائط الصفحات: {len(PAGE_MAP)} صفحة بصور من أصل {len(src)}")


def page_image(name):
    with _img_cache_lock:
        if name in _img_cache:
            return _img_cache[name]
    img = Image.open(PAGE_MAP[name]).convert("RGB")
    with _img_cache_lock:
        if len(_img_cache) > 7:
            _img_cache.pop(next(iter(_img_cache)))
        _img_cache[name] = img
    return img


def log_adj(action, page, rec_id, old, new):
    rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "action": action, "page": page, "id": rec_id,
           "old": old, "new": new}
    with _db_lock:
        with open(ADJ_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def app_log(event, **kw):
    """سجل النشاط العام: كل ما يجري في البرنامج (بلا صور أو نصوص القصاصات).

    يُكتب محليًا دومًا في ocr_data/app_log.jsonl لأي تطوير لاحق؛ وإرساله
    إلى المطوّر مشروط بإذن المستخدم المسجَّل في telemetry.json.
    """
    rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "event": event, "v": APP_VERSION}
    rec.update(kw)
    try:
        with _db_lock:
            APP_LOG.parent.mkdir(parents=True, exist_ok=True)
            with open(APP_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass
    return rec


def consent_state():
    """إذن المستخدم: True/False بعد الإجابة، None إن لم يُسأل بعد."""
    try:
        return json.loads(TELEMETRY.read_text(encoding="utf-8"))["consent"]
    except Exception:
        return None


def set_consent(v):
    TELEMETRY.parent.mkdir(parents=True, exist_ok=True)
    TELEMETRY.write_text(json.dumps(
        {"consent": bool(v),
         "ts": datetime.now(timezone.utc).isoformat(timespec="seconds")}),
        encoding="utf-8")
    app_log("consent", value=bool(v))


def api_log(tail=200):
    if not APP_LOG.exists():
        return {"events": []}
    with _db_lock:
        lines = APP_LOG.read_text(encoding="utf-8").splitlines()[-max(1, tail):]
    out = []
    for ln in lines:
        try:
            out.append(json.loads(ln))
        except Exception:
            pass
    return {"events": out}


def regen_crop_png(page_name, crop_path, box):
    """إعادة قص صورة القصاصة من صفحة الصفحة بعد تعديل المربع."""
    x1, y1, x2, y2 = [int(v) for v in box]
    if x2 <= x1:
        x2 = x1 + 1
    if y2 <= y1:
        y2 = y1 + 1
    img = page_image(page_name).crop((x1, y1, x2, y2))
    local = CROPS_DIR / Path(crop_path).name
    local.parent.mkdir(parents=True, exist_ok=True)
    img.save(local, "PNG")
    return local


def crop_png_bytes(name, x1, y1, x2, y2, scale=1):
    img = page_image(name).crop((max(0, x1), max(0, y1), x2, y2))
    if scale and scale != 1:
        img = img.resize((int(img.width * scale), int(img.height * scale)),
                         Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


# ─────────────────────── اكتشاف حدود كلمة من نقطة نقر ───────────────────────
_gray_cache = {}


def page_gray(name):
    with _img_cache_lock:
        if name in _gray_cache:
            return _gray_cache[name]
    g = page_image(name).convert("L")
    with _img_cache_lock:
        if len(_gray_cache) > 7:
            _gray_cache.pop(next(iter(_gray_cache)))
        _gray_cache[name] = g
    return g


def _otsu(hist):
    """عتبة Otsu من مدرج تكراري 256 خانة — فصل الحبر عن الخلفية تلقائيًا."""
    total = sum(hist)
    if not total:
        return 128
    sum_all = sum(i * hist[i] for i in range(256))
    sum_b = w_b = 0
    best_t, best_var = 128, -1.0
    for t in range(256):
        w_b += hist[t]
        if not w_b:
            continue
        w_f = total - w_b
        if not w_f:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_all - sum_b) / w_f
        var = w_b * w_f * (m_b - m_f) * (m_b - m_f)
        if var > best_var:
            best_var, best_t = var, t
    return best_t


def _strip_ink(img_l, box):
    """قناع الحبر لمقطع من الصفحة الرمادية بعتبة Otsu الخاصة بالمقطع."""
    win = img_l.crop(box)
    wlen, wht = win.size
    px = list(win.getdata())
    hist = [0] * 256
    for v in px:
        hist[v] += 1
    thr = _otsu(hist)
    if thr < 1:
        # صورة ثنائية القيمة الصارمة (0/255): Otsu يعيد 0 — الحبر هو القيمة الداكنة
        thr = 1
    return wlen, wht, [v < thr for v in px]


def _col_segments(ink, wlen, wht, y1, y2, empty_max=1, min_gap=5):
    """شرائح الكلمات أفقياً داخل نطاق الصفوف: أعمدة تحوي حبرًا تفصلها
    فجوات عمودية ≥ min_gap — الفجوة بين الكلمات تفصل، والضربات المتصلة
    والنقط تبقى في الشريحة نفسها."""
    y1 = max(0, y1)
    y2 = min(wht, y2)
    cols = [0] * wlen
    for j in range(y1, y2):
        base = j * wlen
        for i in range(wlen):
            if ink[base + i]:
                cols[i] += 1
    segs, i, n = [], 0, wlen
    while i < n and cols[i] <= empty_max:
        i += 1
    while i < n:
        start = last = i
        g = 0
        i += 1
        while i < n:
            if cols[i] > empty_max:
                last = i
                g = 0
            else:
                g += 1
                if g >= min_gap:
                    break
            i += 1
        segs.append((start, last + 1))
        while i < n and cols[i] <= empty_max:
            i += 1
    return segs


def _y_extent(ink, wlen, wht, x1, x2, y1, y2):
    """مدى الصفوف الحبرية ضمن شريحة أعمدة [x1, x2) وبند صفوف [y1, y2)."""
    ry1 = ry2 = None
    for j in range(max(0, y1), min(wht, y2)):
        base = j * wlen
        for i in range(x1, x2):
            if ink[base + i]:
                if ry1 is None:
                    ry1 = j
                ry2 = j
                break
    return (ry1, ry2 + 1) if ry1 is not None else None


def _band_rows(ink, wlen, wht, near_y, reach=18):
    """شريط الصفوف حول near_y مع امتصاص التشغيلات المجاورة ضمن reach —
    فتلتحم ضربات الكلمة بنقطها وشُدّتها حتى لو انقطع الحبر بينها صفوفًا."""
    rows = [0] * wht
    for j in range(wht):
        base = j * wlen
        c = 0
        for i in range(wlen):
            if ink[base + i]:
                c += 1
        rows[j] = c
    runs, j = [], 0
    while j < wht:
        if rows[j] > 0:
            start = last = j
            g = 0
            j += 1
            while j < wht:
                if rows[j] > 0:
                    last = j
                    g = 0
                else:
                    g += 1
                    if g >= 8:
                        break
                j += 1
            runs.append((start, last + 1))
        else:
            j += 1
    if not runs:
        return None
    k = min(range(len(runs)),
            key=lambda i: abs((runs[i][0] + runs[i][1]) / 2 - near_y))
    lo, hi = runs[k]
    changed = True
    while changed:                     # امتصاص سلسلي للجيران القريبين
        changed = False
        for a, b in runs:
            if a >= lo and b <= hi:
                continue
            if max(lo - b, a - hi) <= reach:
                lo, hi = min(lo, a), max(hi, b)
                changed = True
    return (lo, hi)


def auto_word_box(name, x, y, band=None, gap=5, vreach=18):
    """مربع ضيق حول الكلمة المُنقورة فقط — ملاصق لحبرها بلا أي هامش أبيض
    (نمط حفظ الـpatterns في ABBYY FineReader) — بلا جيرانها.

    يقسّم شريط السطر حول النقرة إلى كلمات بفجوات الأعمدة الحبرية
    (البروفيل العمودي)، ويختار الشريحة التي تحتوي عمود النقرة (أو الأقرب
    إليها إن نُقر في فجوة) — فتُلتقط الكلمة بنقطها وشُدّتها دون امتداد
    للجار. إن لم توجد فجوات فاصلة (كلمات ملتصقة) لجأ إلى مكوّنات الحبر
    المتصلة بفجوة صغيرة (يتحمل النقر داخل ثقب حرف مثل ـحـ).
    band=(y1, y2) يحدّ سطر الصفحة إن مُمرِّر، وإلا يُشتق من الصفوف الحبرية.
    """
    gap = max(1, int(gap))
    vreach = int(vreach)
    img = page_gray(name)
    W, H = img.size
    x = max(0, min(int(x), W - 1))
    y = max(0, min(int(y), H - 1))
    hw, hh = 220, 170
    wx1, wy1 = max(0, x - hw), max(0, y - hh)
    wx2, wy2 = min(W, x + hw), min(H, y + hh)
    if wx2 - wx1 < 10 or wy2 - wy1 < 10:
        return {"found": False}
    wlen, wht, ink = _strip_ink(img, (wx1, wy1, wx2, wy2))
    lx, ly = x - wx1, y - wy1

    # شريط صفوف الكلمة: band الممرَّر، أو السطر المشتق حول النقرة
    if band is not None:
        ry1 = max(0, int(band[0]) - wy1 - vreach)
        ry2 = min(wht, int(band[1]) - wy1 + vreach)
        if ry2 - ry1 < 6:
            return {"found": False}
    else:
        br = _band_rows(ink, wlen, wht, ly)
        if br is None:
            return {"found": False}
        ry1 = max(0, br[0] - 6)
        ry2 = min(wht, br[1] + 6)

    segs = _col_segments(ink, wlen, wht, ry1, ry2, empty_max=1, min_gap=gap)
    seg = next((s for s in segs if s[0] <= lx < s[1]), None)
    if seg is None and segs:                     # النقرة في فجوة: أقرب شريحة قريبة
        cand = min(segs, key=lambda s: min(abs(s[0] - lx), abs(s[1] - lx)))
        if min(abs(cand[0] - lx), abs(cand[1] - lx)) <= 50:
            seg = cand
    if seg:
        ye = _y_extent(ink, wlen, wht, seg[0], seg[1], ry1, ry2)
        # النقرة قريبة عموديًا من الشريحة؟ (يمنع التقاط كلمة بعيدة أسفل/أعلى)
        if ye and ye[0] - 40 <= ly <= ye[1] + 40:
            # pad=0: الحدود على الحبر تمامًا (seg/ye نهايات حصرية لأقصى عمود/صف حبري)
            bx1 = max(0, wx1 + seg[0])
            by1 = max(0, wy1 + ye[0])
            bx2 = min(W, wx1 + seg[1])
            by2 = min(H, wy1 + ye[1])
            if bx2 - bx1 >= 5 and by2 - by1 >= 5:
                return {"found": True, "x1": int(bx1), "y1": int(by1),
                        "x2": int(bx2), "y2": int(by2)}
    # احتياط: كلمات ملتصقة بلا فجوة عمودية — مكوّنات متصلة بفجوة صغيرة
    return _cc_word_box(img, (wx1, wy1, wx2, wy2), (lx, ly), band, vreach)


def _cc_word_box(img, win, seed_pt, band, vreach):
    """الاحتياط: بدءًا من مكوّن الحبر عند البذرة يدمج القريب أفقيًا (≤6px)."""
    wx1, wy1, wx2, wy2 = win
    W, H = img.size
    wlen, wht, ink = _strip_ink(img, win)
    lx, ly = seed_pt
    seed = None
    if 0 <= lx < wlen and 0 <= ly < wht and ink[ly * wlen + lx]:
        seed = (lx, ly)
    else:
        for r in range(1, 41):
            hit = None
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if max(abs(dx), abs(dy)) != r:
                        continue
                    i, j = lx + dx, ly + dy
                    if 0 <= i < wlen and 0 <= j < wht and ink[j * wlen + i]:
                        hit = (i, j)
                        break
                if hit:
                    break
            if hit:
                seed = hit
                break
    if not seed:
        return {"found": False}
    label = [[0] * wlen for _ in range(wht)]
    boxes = [None]
    nlab = 0
    stack = []
    for j0 in range(wht):
        row0 = label[j0]
        for i0 in range(wlen):
            if row0[i0] or not ink[j0 * wlen + i0]:
                continue
            nlab += 1
            b = [i0, j0, i0, j0]
            boxes.append(b)
            stack.append((i0, j0))
            row0[i0] = nlab
            while stack:
                ci, cj = stack.pop()
                if ci < b[0]: b[0] = ci
                elif ci > b[2]: b[2] = ci
                if cj < b[1]: b[1] = cj
                elif cj > b[3]: b[3] = cj
                for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ni, nj = ci + di, cj + dj
                    if 0 <= ni < wlen and 0 <= nj < wht:
                        rl = label[nj]
                        if not rl[ni] and ink[nj * wlen + ni]:
                            rl[ni] = nlab
                            stack.append((ni, nj))
    sl = label[seed[1]][seed[0]]
    if not sl:
        return {"found": False}
    cur = list(boxes[sl])
    changed = True
    gap = 6
    while changed:
        changed = False
        for k in range(1, nlab + 1):
            if k == sl:
                continue
            b = boxes[k]
            if b is None:
                continue
            if max(cur[0] - b[2], b[0] - cur[2], 0) > gap:
                continue
            if max(cur[1] - b[3], b[1] - cur[3], 0) > vreach:
                continue
            if band is not None:
                bc = (b[1] + b[3]) / 2
                if bc < band[0] - wy1 - vreach or bc > band[1] - wy1 + vreach:
                    continue
            cur = [min(cur[0], b[0]), min(cur[1], b[1]),
                   max(cur[2], b[2]), max(cur[3], b[3])]
            boxes[k] = None
            changed = True
    # pad=0: المربع ملاصق للحبر (cur صندوق شامل الحواف → النهاية الحصرية +1)
    x1 = max(0, wx1 + cur[0])
    y1 = max(0, wy1 + cur[1])
    x2 = min(W, wx1 + cur[2] + 1)
    y2 = min(H, wy1 + cur[3] + 1)
    if x2 - x1 < 5 or y2 - y1 < 5:
        return {"found": False}
    return {"found": True, "x1": int(x1), "y1": int(y1),
            "x2": int(x2), "y2": int(y2)}


# ─────────────────────────── عمليات البيانات ───────────────────────────
def _page_lines(name):
    """أسطر الصفحة من قاعدة الأسطر (level='line')."""
    nconn = sqlite3.connect(f"file:{NOTES_DB}?mode=ro", uri=True)
    pid = nconn.execute("SELECT id FROM pages WHERE source_name=?",
                        (name,)).fetchone()
    lines = []
    if pid:
        for lid, bb in nconn.execute(
                "SELECT id, bbox FROM snippets WHERE page_id=? AND level='line'",
                (pid[0],)):
            x, y, w, h = json.loads(bb)
            lines.append({"id": lid, "x1": x, "y1": y, "x2": x + w, "y2": y + h})
    nconn.close()
    lines.sort(key=lambda l: (l["y1"], l["x1"]))
    return lines


def api_pages():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT page_name, COUNT(*), SUM(verified=1), SUM(ignored=1),
               MIN(id)
        FROM crops GROUP BY page_name ORDER BY MIN(id)""").fetchall()
    conn.close()
    pages = [{"name": r[0], "words": r[1], "verified": r[2] or 0,
              "ignored": r[3] or 0} for r in rows]
    return {"pages": pages,
            "total_active": sum(p["words"] - p["ignored"] for p in pages),
            "total_verified": sum(p["verified"] for p in pages)}


def api_page(name):
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT id, x1, y1, x2, y2, label, language, confidence,
               verified, ignored, split
        FROM crops WHERE page_name=? ORDER BY id""", (name,)).fetchall()
    conn.close()
    lines = _page_lines(name)
    words = [{"id": r[0], "x1": r[1], "y1": r[2], "x2": r[3], "y2": r[4],
              "label": r[5] or "", "language": r[6] or "arabic",
              "confidence": r[7], "verified": bool(r[8]),
              "ignored": bool(r[9]), "split": r[10] or "train"}
             for r in rows]
    words.sort(key=lambda w: (w["y1"], w["x1"]))
    w, h = PAGE_SIZE.get(name, (0, 0))
    if not w:
        img = page_image(name)
        w, h = img.size
    return {"name": name, "width": w, "height": h,
            "image": f"/img/page?name={name}",
            "lines": lines, "words": words}


def api_update(items):
    out = []
    with _db_lock:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        for it in items:
            row = conn.execute("SELECT * FROM crops WHERE id=?",
                               (it["id"],)).fetchone()
            if not row:
                continue
            old = {"x1": row["x1"], "y1": row["y1"], "x2": row["x2"],
                   "y2": row["y2"], "label": row["label"],
                   "verified": row["verified"], "ignored": row["ignored"],
                   "language": row["language"]}
            sets, vals = [], []
            for k in ("x1", "y1", "x2", "y2"):
                if k in it:
                    sets.append(f"{k}=?"); vals.append(int(it[k]))
            for k in ("label", "language"):
                if k in it:
                    sets.append(f"{k}=?"); vals.append(str(it[k]))
            for k in ("verified", "ignored"):
                if k in it:
                    sets.append(f"{k}=?"); vals.append(1 if it[k] else 0)
            if sets:
                conn.execute(f"UPDATE crops SET {', '.join(sets)} WHERE id=?",
                             vals + [it["id"]])
            conn.commit()
            new = conn.execute("SELECT * FROM crops WHERE id=?",
                               (it["id"],)).fetchone()
            bbox_changed = any(new[k] != old[k] for k in ("x1", "y1", "x2", "y2"))
            if bbox_changed:
                regen_crop_png(row["page_name"], row["crop_path"],
                               (new["x1"], new["y1"], new["x2"], new["y2"]))
            log_adj("update", row["page_name"], it["id"], old,
                    {k: new[k] for k in old})
            out.append({"id": new["id"], "x1": new["x1"], "y1": new["y1"],
                        "x2": new["x2"], "y2": new["y2"],
                        "label": new["label"], "language": new["language"],
                        "verified": bool(new["verified"]),
                        "ignored": bool(new["ignored"]),
                        "split": new["split"]})
        conn.close()
    return {"words": out}


def api_add(p):
    name = p["page_name"]
    x1, y1, x2, y2 = (int(p[k]) for k in ("x1", "y1", "x2", "y2"))
    with _db_lock:
        conn = sqlite3.connect(DB_PATH)
        sp = conn.execute("""SELECT split FROM crops WHERE page_name=?
                             GROUP BY split ORDER BY COUNT(*) DESC LIMIT 1""",
                          (name,)).fetchone()
        split = sp[0] if sp else "train"
        n = conn.execute("SELECT COUNT(*) FROM crops WHERE page_name=?",
                         (name,)).fetchone()[0]
        fname = f"e{uuid.uuid4().hex[:8]}_w{n:04d}.png"
        crop_path = str(CROPS_DIR / fname)
        cur = conn.execute("""
            INSERT INTO crops (page_name, crop_path, label, x1, y1, x2, y2,
                               split, created_at, source, language,
                               src_id, ignored, verified)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (name, crop_path, p.get("label", ""), x1, y1, x2, y2, split,
             datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
             "box_editor", p.get("language", "arabic"), None, 0, 0))
        new_id = cur.lastrowid
        conn.commit()
        conn.close()
    regen_crop_png(name, crop_path, (x1, y1, x2, y2))
    log_adj("add", name, new_id, None,
            {"x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return {"word": {"id": new_id, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                     "label": p.get("label", ""),
                     "language": p.get("language", "arabic"),
                     "confidence": None, "verified": False,
                     "ignored": False, "split": split}}


def api_delete(ids):
    with _db_lock:
        conn = sqlite3.connect(DB_PATH)
        for i in ids:
            r = conn.execute("SELECT page_name FROM crops WHERE id=?",
                             (i,)).fetchone()
            if r:
                conn.execute("UPDATE crops SET ignored=1 WHERE id=?", (i,))
                log_adj("delete", r[0], i, None, {"ignored": 1})
        conn.commit()
        conn.close()
    return {"ok": True}


def api_merge(keep_id, drop_id):
    with _db_lock:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        k = conn.execute("SELECT * FROM crops WHERE id=?", (keep_id,)).fetchone()
        d = conn.execute("SELECT * FROM crops WHERE id=?", (drop_id,)).fetchone()
        if not k or not d or k["page_name"] != d["page_name"]:
            conn.close()
            return {"error": "المربعان غير صالحين للدمج"}
        box = (min(k["x1"], d["x1"]), min(k["y1"], d["y1"]),
               max(k["x2"], d["x2"]), max(k["y2"], d["y2"]))
        label = k["label"] or d["label"]
        conn.execute("UPDATE crops SET x1=?,y1=?,x2=?,y2=?,label=? WHERE id=?",
                     (*box, label, keep_id))
        conn.execute("UPDATE crops SET ignored=1 WHERE id=?", (drop_id,))
        conn.commit()
        regen_crop_png(k["page_name"], k["crop_path"], box)
        log_adj("merge", k["page_name"], keep_id,
                {"keep": dict(k), "drop": dict(d)},
                {"x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3],
                 "label": label})
        conn.close()
    return {"word": {"id": keep_id, "x1": box[0], "y1": box[1],
                     "x2": box[2], "y2": box[3], "label": label,
                     "verified": bool(k["verified"])}}


def api_add_many(p):
    """إضافة دفعة مربعات (نتيجة زر التحليل) باتصال قاعدة واحدة."""
    name = p["page_name"]
    items = p.get("items", [])
    out = []
    with _db_lock:
        conn = sqlite3.connect(DB_PATH)
        sp = conn.execute("""SELECT split FROM crops WHERE page_name=?
                             GROUP BY split ORDER BY COUNT(*) DESC LIMIT 1""",
                          (name,)).fetchone()
        split = sp[0] if sp else "train"
        n = conn.execute("SELECT COUNT(*) FROM crops WHERE page_name=?",
                         (name,)).fetchone()[0]
        for k, it in enumerate(items):
            try:
                x1, y1, x2, y2 = (int(it[kk]) for kk in ("x1", "y1", "x2", "y2"))
            except Exception:
                continue
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            fname = f"e{uuid.uuid4().hex[:8]}_w{n + k:04d}.png"
            crop_path = str(CROPS_DIR / fname)
            cur = conn.execute("""
                INSERT INTO crops (page_name, crop_path, label, x1, y1, x2, y2,
                                   split, created_at, source, language,
                                   src_id, ignored, verified)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (name, crop_path, it.get("label", ""), x1, y1, x2, y2, split,
                 datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                 "box_editor", it.get("language", "arabic"), None, 0, 0))
            out.append({"id": cur.lastrowid, "x1": x1, "y1": y1,
                        "x2": x2, "y2": y2, "label": it.get("label", ""),
                        "language": it.get("language", "arabic"),
                        "confidence": None, "verified": False,
                        "ignored": False, "split": split,
                        "crop_path": crop_path})
        conn.commit()
        conn.close()
    for wd in out:
        regen_crop_png(name, wd.pop("crop_path"),
                       (wd["x1"], wd["y1"], wd["x2"], wd["y2"]))
    app_log("add_many", page=name, n=len(out))
    return {"words": out}


# ─────────────── القصّ إلى حدود الحبر تمامًا (نمط ABBYY) ───────────────
def _ink_bbox_in(img_l, box):
    """صندوق الحبر الدقيق داخل منطقة من الصفحة الرمادية — بإحداثيات الصفحة،
    أو None إن كانت المنطقة خوية من الحبر."""
    x1 = max(0, int(box[0]))
    y1 = max(0, int(box[1]))
    x2 = min(img_l.size[0], int(box[2]))
    y2 = min(img_l.size[1], int(box[3]))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    wlen, wht, ink = _strip_ink(img_l, (x1, y1, x2, y2))
    ix1 = ix2 = iy1 = iy2 = None
    for j in range(wht):
        base = j * wlen
        for i in range(wlen):
            if ink[base + i]:
                if ix1 is None or i < ix1:
                    ix1 = i
                if ix2 is None or i > ix2:
                    ix2 = i
                if iy1 is None:
                    iy1 = j
                iy2 = j
    if ix1 is None:
        return None
    return (x1 + ix1, y1 + iy1, x1 + ix2 + 1, y1 + iy2 + 1)


def api_tighten(p):
    """تقليص المربعات إلى حدود الحبر تمامًا — بلا أي فراغ أبيض — كحفظ
    الـpatterns في ABBYY FineReader، مع إعادة قص صور القصاصات فورًا حتى
    لا تختلف أبعاد البيانات عند التدريب. ids=None يقلّص كل مربعات الصفحة
    غير المتجاهلة؛ التقليص لا يوسّع أبدًا ولا يلمس ما هو ملاصق أصلًا."""
    name = p["page_name"]
    ids = p.get("ids")
    conn = sqlite3.connect(DB_PATH)
    if ids:
        ph = ",".join("?" for _ in ids)
        rows = conn.execute(
            f"""SELECT id, crop_path, x1, y1, x2, y2 FROM crops
                WHERE page_name=? AND ignored=0 AND id IN ({ph})""",
            (name, *[int(i) for i in ids])).fetchall()
    else:
        rows = conn.execute(
            """SELECT id, crop_path, x1, y1, x2, y2 FROM crops
               WHERE page_name=? AND ignored=0 ORDER BY id""",
            (name,)).fetchall()
    img = page_gray(name)
    updated, regen = [], []
    unchanged = empty = 0
    for wid, crop_path, x1, y1, x2, y2 in rows:
        nb = _ink_bbox_in(img, (x1, y1, x2, y2))
        if nb is None:
            empty += 1
            continue
        if nb == (x1, y1, x2, y2):
            unchanged += 1
            continue
        conn.execute("UPDATE crops SET x1=?, y1=?, x2=?, y2=? WHERE id=?",
                     (*nb, wid))
        log_adj("tighten", name, wid, [x1, y1, x2, y2], list(nb))
        updated.append({"id": wid, "x1": nb[0], "y1": nb[1],
                        "x2": nb[2], "y2": nb[3]})
        regen.append((crop_path, nb))
    conn.commit()
    conn.close()
    for crop_path, nb in regen:
        regen_crop_png(name, crop_path, nb)
    app_log("tighten", page=name, n=len(updated),
            all=ids is None, empty=empty)
    return {"updated": updated, "unchanged": unchanged, "empty": empty}


# ─────────────────── تحليل الصفحة كله وفواصل الأعمدة ───────────────────
def find_columns(name, min_gap=24, min_run=0.4, flank=0.18):
    """فواصل الأعمدة في صفحات العمودين.

    أولًا من أسطر القاعدة: تجميع الأسطر المتقاطعة أفقيًا يكوّن عمودًا،
    والفجوة الأفقية بين عنقودين من الأسطر هي الفاصل — أدق طريقة لأنها
    مبنية على بنية الصفحة نفسها. وإن لم تتوفر أسطر، فالممر العمودي شبه
    الفارغ الممتد ≥ 40% من الارتفاع ومحاط بأعمدة كثيفة الحبر من الجانبين.
    """
    lines = _page_lines(name)
    if lines:
        spans = sorted((l["x1"], l["x2"]) for l in lines)
        clusters = []
        for a, b in spans:
            if clusters and a <= clusters[-1][1] + min_gap:
                clusters[-1][1] = max(clusters[-1][1], b)
            else:
                clusters.append([a, b])
        return [{"x1": c1[1], "x2": c2[0]}
                for c1, c2 in zip(clusters, clusters[1:])
                if c2[0] - c1[1] >= min_gap]
    img = page_gray(name)
    W, H = img.size
    f = 4
    sw, sh = max(1, W // f), max(1, H // f)
    _, sht, sink = _strip_ink(img.resize((sw, sh)), (0, 0, sw, sh))
    longest = [0] * sw      # أطول تشغيلة صفوف فارغة في العمود
    col_ink = [0] * sw      # عدد صفوف الحبر في العمود
    for i in range(sw):
        run = best = 0
        for j in range(sht):
            if sink[j * sw + i]:
                col_ink[i] += 1
                best = max(best, run)
                run = 0
            else:
                run += 1
        longest[i] = max(best, run)
    thr = max(2, int(sht * min_run))
    zones, i = [], 0
    while i < sw:
        if longest[i] >= thr:
            start = i
            while i < sw and longest[i] >= thr:
                i += 1
            if (i - start) >= max(2, min_gap // f) and 0 < start and i < sw:
                lf = max(col_ink[max(0, start - 12):start] or [0])
                rf = max(col_ink[i:i + 12] or [0])
                if lf >= flank * sht and rf >= flank * sht:
                    zones.append({"x1": start * f, "x2": min(W, i * f)})
        else:
            i += 1
    return zones


def _merge_overlaps(boxes, ratio=0.5):
    """دمج المربعات المتداخلة بشدة (شريطان متجاوران يلتقطان الكلمة نفسها)."""
    out = []
    for b in sorted(boxes, key=lambda r: (r["y1"], r["x1"])):
        merged = False
        for o in out:
            ix = min(o["x2"], b["x2"]) - max(o["x1"], b["x1"])
            iy = min(o["y2"], b["y2"]) - max(o["y1"], b["y1"])
            if ix > 0 and iy > 0:
                a_b = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
                a_o = (o["x2"] - o["x1"]) * (o["y2"] - o["y1"])
                if ix * iy / max(1, min(a_b, a_o)) > ratio:
                    o["x1"] = min(o["x1"], b["x1"])
                    o["y1"] = min(o["y1"], b["y1"])
                    o["x2"] = max(o["x2"], b["x2"])
                    o["y2"] = max(o["y2"], b["y2"])
                    merged = True
                    break
        if not merged:
            out.append(dict(b))
    return out


def analyze_page(name, min_gap=5, vreach=16, min_size=6):
    """مربعات مبدئية ضيقة لكل كلمات الصفحة + فواصل الأعمدة.

    يعتمد أسطر قاعدة الأسطر كأشرطة، ثم يضيف أي كتل حبر خارجها (مجموعات
    زوايا الصفحة وأعلاها) عبر بروفيل صفوف مصغّر، ويقسّم كل شريط إلى كلمات
    بفجوات الأعمدة الحبرية كما في auto_word_box.
    """
    img = page_gray(name)
    W, H = img.size
    bands = [(l["y1"] - vreach, l["y2"] + vreach) for l in _page_lines(name)]
    # كتل حبر خارج الأسطر: زوايا الصفحة ومجموعات أعلى/أسفل
    f = 4
    sw, sh = max(1, W // f), max(1, H // f)
    _, sht, sink = _strip_ink(img.resize((sw, sh)), (0, 0, sw, sh))
    rowc = [0] * sht
    for j in range(sht):
        base = j * sw
        for i in range(sw):
            if sink[base + i]:
                rowc[j] += 1
    covered = [(max(0, int(b[0]) // f), min(sht, int(b[1]) // f)) for b in bands]

    def in_covered(j):
        return any(a - 1 <= j <= b + 1 for a, b in covered)

    j = 0
    while j < sht:
        if rowc[j] >= 2 and not in_covered(j):
            start = last = j
            while j < sht and rowc[j] >= 1 and not in_covered(j):
                if rowc[j] >= 2:
                    last = j
                j += 1
            if last - start >= 2:
                bands.append((start * f, min(H, (last + 1) * f)))
        else:
            j += 1
    words = []
    for by1, by2 in bands:
        if by2 - by1 < 8 or by1 >= H:
            continue
        oy = max(0, by1)
        wlen, wht, ink = _strip_ink(img, (0, oy, W, min(H, by2)))
        for sx1, sx2 in _col_segments(ink, wlen, wht, 0, wht,
                                      empty_max=2, min_gap=min_gap):
            ye = _y_extent(ink, wlen, wht, sx1, sx2, 0, wht)
            if not ye:
                continue
            x1, x2 = sx1, sx2
            y1, y2 = oy + ye[0], oy + ye[1]
            if (x2 - x1) < min_size or (y2 - y1) < min_size:
                continue
            words.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
    words = _merge_overlaps(words)
    columns = find_columns(name)
    app_log("analyze", page=name, words=len(words), columns=len(columns))
    return {"found": bool(words), "words": words, "columns": columns}


# ─────────────── ترتيب القراءة وتصدير الصفحة (txt / docx) ───────────────
def _cluster_lines(ws):
    """تجميع كلمات منطقة إلى أسطر بمركز y (فجوة > 0.7×الطول الوسيط = سطر جديد)."""
    if not ws:
        return []
    hs = sorted(w["y2"] - w["y1"] for w in ws)
    med_h = max(8, hs[len(hs) // 2])
    ws = sorted(ws, key=lambda w: (w["y1"] + w["y2"]) / 2)
    lines, cur, base = [], [], None
    for w in ws:
        cy = (w["y1"] + w["y2"]) / 2
        if base is None or cy - base <= 0.7 * med_h:
            cur.append(w)
            base = cy if base is None else base
        else:
            lines.append(cur)
            cur = [w]
            base = cy
    if cur:
        lines.append(cur)
    return lines


def _cluster_blocks(lines, med_h):
    """تجميع الأسطر إلى كتل: فجوة عمودية كبيرة (مجموعة زاوية) = كتلة جديدة."""
    blocks, cur = [], []
    prev_bottom = None
    for ln in sorted(lines, key=lambda l: min(w["y1"] for w in l)):
        top = min(w["y1"] for w in ln)
        if prev_bottom is not None and top - prev_bottom > 1.4 * med_h:
            blocks.append(cur)
            cur = []
        cur.append(ln)
        prev_bottom = max(w["y2"] for w in ln)
    if cur:
        blocks.append(cur)
    return blocks


def _reading_order(words, columns, mode="rows"):
    """ترتيب القراءة للتصدير.

    rows (الافتراضي): الكتل من الأعلى للأسفل — المجموعة الأولى في الأعلى ثم
    الثانية ثم الثالثة… وعند تعادل عمودي يُقدَّم العمود الأيمن (عربي).
    columns: إتمام العمود الأيمن كاملًا ثم الأيسر (ترتيب أعمدة).
    داخل السطر: الكلمات من اليمين لليسار للأسطر العربية، والعكس للاتينية.
    """
    regions = []
    if columns:
        edges = sorted((c["x1"], c["x2"]) for c in columns)
        prev = 0
        for gx1, gx2 in edges:
            regions.append((prev, gx1))
            prev = gx2
        regions.append((prev, 10 ** 9))
    else:
        regions = [(0, 10 ** 9)]

    def region_of(w):
        cx = (w["x1"] + w["x2"]) / 2
        for k, (a, b) in enumerate(regions):
            if a <= cx < b:
                return k
        return len(regions) - 1

    all_hs = sorted(w["y2"] - w["y1"] for w in words) or [30]
    med_h = max(8, all_hs[len(all_hs) // 2])
    by_region = {}
    for w in words:
        by_region.setdefault(region_of(w), []).append(w)

    blocks = []                     # (y1, region_k, block) — k الأكبر = أقصى اليمين
    for rk in sorted(by_region):
        ws = by_region[rk]
        lines = _cluster_lines(ws)
        for blk in _cluster_blocks(lines, med_h):
            blocks.append((min(w["y1"] for ln in blk for w in ln), rk, blk))
    if mode == "columns":
        # العمود الأيمن (أكبر k) كاملًا أولًا ثم ما يليه لليسار
        blocks.sort(key=lambda b: (-b[1], b[0]))
    else:
        # الكتل من الأعلى للأسفل؛ عند تعادل العمودي يُقدَّم العمود الأيمن
        blocks.sort(key=lambda b: (b[0], -b[1]))

    ordered = []
    for _, _, blk in blocks:
        for ln in blk:
            arab = sum(1 for w in ln if (w.get("language") or "arabic")
                       .startswith("ar"))
            ln.sort(key=lambda w: -w["x1"] if arab * 2 >= len(ln) else w["x1"])
        ordered.append(blk)
    return ordered


def export_page(name, fmt="txt", mode="rows"):
    """تصدير نص الصفحة بترتيب القراءة — كل الكلمات الموثّقة غير المتجاهلة."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""SELECT x1, y1, x2, y2, label, language
                           FROM crops WHERE page_name=? AND ignored=0
                           AND label!='' ORDER BY id""", (name,)).fetchall()
    conn.close()
    words = [{"x1": r[0], "y1": r[1], "x2": r[2], "y2": r[3],
              "label": r[4], "language": r[5] or "arabic"} for r in rows]
    ordered = _reading_order(words, find_columns(name), mode=mode)
    if fmt == "docx":
        try:
            from docx import Document
            from docx.enum.text import WD_ALIGN_PARAGRAPH
            from docx.oxml import OxmlElement
            from docx.oxml.ns import qn
        except ImportError:
            raise RuntimeError(
                "مكتبة python-docx غير مثبتة — ثبّتها أولًا:  pip install python-docx")
        doc = Document()
        for blk in ordered:
            for ln in blk:
                para = doc.add_paragraph(" ".join(w["label"] for w in ln))
                para.alignment = WD_ALIGN_PARAGRAPH.RIGHT
                ppr = para._p.get_or_add_pPr()
                bidi = OxmlElement("w:bidi")
                ppr.append(bidi)
            doc.add_paragraph("")
        buf = io.BytesIO()
        doc.save(buf)
        app_log("export", page=name, fmt="docx", blocks=len(ordered))
        return buf.getvalue(), ("application/vnd.openxmlformats-officedocument"
                                ".wordprocessingml.document")
    txt = "\n\n".join("\n".join(" ".join(w["label"] for w in ln) for ln in blk)
                      for blk in ordered)
    app_log("export", page=name, fmt="txt", blocks=len(ordered))
    return txt.encode("utf-8"), "text/plain; charset=utf-8"


# ─────────────────────────── خادم HTTP ───────────────────────────
class Handler(BaseHTTPRequestHandler):

    def log_message(self, *a):  # إسكات سجل الطلبات
        pass

    def _cors(self):  # السماح للوحة التحكم بقراءة الحالة فقط (GET) — POST بلا CORS
        if self.command == "GET":
            self.send_header("Access-Control-Allow-Origin", "*")

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self._cors()
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path, ctype):
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self._cors()
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path == "/":
                return self._file(HTML_PATH, "text/html; charset=utf-8")
            if u.path == "/panel":
                return self._file(PANEL_PATH, "text/html; charset=utf-8")
            if u.path == "/api/health":
                return self._json({"ok": True, "pages": len(PAGE_MAP),
                                   "version": APP_VERSION,
                                   "consent": consent_state()})
            if u.path == "/api/pages":
                return self._json(api_pages())
            if u.path == "/api/page":
                name = q["name"][0]
                app_log("page_open", page=name)
                return self._json(api_page(name))
            if u.path == "/api/analyze":
                name = q["name"][0]
                if name not in PAGE_MAP:
                    return self._json({"error": "صفحة غير موجودة"}, 404)
                return self._json(analyze_page(name))
            if u.path == "/api/log":
                return self._json(api_log(int(q.get("tail", ["200"])[0])))
            if u.path == "/api/consent":
                return self._json({"consent": consent_state()})
            if u.path == "/api/auto_box":
                name = q["name"][0]
                if name not in PAGE_MAP:
                    return self._json({"error": "صفحة غير موجودة"}, 404)
                bx, by = float(q["x"][0]), float(q["y"][0])
                band = None
                if "ly1" in q and "ly2" in q:
                    band = (float(q["ly1"][0]), float(q["ly2"][0]))
                gap = float(q.get("gap", ["5"])[0])
                vreach = float(q.get("vreach", ["18"])[0])
                r = auto_word_box(name, bx, by, band=band, gap=gap, vreach=vreach)
                app_log("auto_box", page=name, x=int(bx), y=int(by),
                        found=bool(r.get("found")))
                return self._json(r)
            if u.path == "/api/export_page":
                name = q["name"][0]
                fmt = q.get("fmt", ["txt"])[0].lower()
                if fmt not in ("txt", "docx"):
                    return self._json({"error": "صيغة غير مدعومة"}, 400)
                mode = q.get("order", ["rows"])[0].lower()
                body, ctype = export_page(name, fmt, mode)
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Disposition",
                                 f'attachment; filename="{name}.{fmt}"')
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                return self.wfile.write(body)
            if u.path == "/img/page":
                name = q["name"][0]
                if name not in PAGE_MAP:
                    return self._json({"error": "صفحة غير موجودة"}, 404)
                return self._file(PAGE_MAP[name], "image/png")
            if u.path == "/img/crop":
                name = q["name"][0]
                x1, y1, x2, y2 = (int(q[k][0]) for k in ("x1", "y1", "x2", "y2"))
                scale = float(q.get("scale", ["1"])[0])
                body = crop_png_bytes(name, x1, y1, x2, y2, scale)
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                return self.wfile.write(body)
            return self._json({"error": "مسار غير معروف"}, 404)
        except Exception as e:
            app_log("error", path=u.path, method="GET", error=str(e))
            return self._json({"error": str(e)}, 500)

    def _body(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n) or b"{}")

    def do_POST(self):
        u = urlparse(self.path)
        try:
            if u.path == "/api/update":
                return self._json(api_update(self._body().get("items", [])))
            if u.path == "/api/add":
                return self._json(api_add(self._body()))
            if u.path == "/api/add_many":
                return self._json(api_add_many(self._body()))
            if u.path == "/api/tighten":
                return self._json(api_tighten(self._body()))
            if u.path == "/api/delete":
                return self._json(api_delete(self._body().get("ids", [])))
            if u.path == "/api/merge":
                b = self._body()
                return self._json(api_merge(b["keep_id"], b["drop_id"]))
            if u.path == "/api/consent":
                b = self._body()
                v = bool(b.get("consent"))
                set_consent(v)
                return self._json({"ok": True, "consent": v})
            if u.path == "/api/telemetry":
                b = self._body()
                if consent_state() is not True:
                    return self._json({"recorded": False,
                                       "reason": "لا إذن من المستخدم"})
                n = 0
                for ev in b.get("events", []):
                    if isinstance(ev, dict) and ev.get("event"):
                        app_log("client_" + str(ev["event"]),
                                **{k: v for k, v in ev.items()
                                   if k not in ("event",)})
                        n += 1
                return self._json({"recorded": True, "count": n})
            return self._json({"error": "مسار غير معروف"}, 404)
        except Exception as e:
            app_log("error", path=u.path, method="POST", error=str(e))
            return self._json({"error": str(e)}, 500)


def lan_urls(port):
    """عناوين الشبكة المحلية للجهاز (لفتح الواجهة من الجوال على نفس Wi-Fi)."""
    import socket
    ips = set()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))  # لا يرسل بيانات — يجد فقط عنوان الشبكة
        ips.add(s.getsockname()[0])
        s.close()
    except Exception:
        pass
    try:
        for ip in socket.gethostbyname_ex(socket.gethostname())[2]:
            if not ip.startswith("127."):
                ips.add(ip)
    except Exception:
        pass
    return [f"http://{ip}:{port}" for ip in sorted(ips)]


def main():
    ap = argparse.ArgumentParser(description="خادم مراجعة مربعات الكلمات")
    ap.add_argument("--port", type=int, default=7863)
    ap.add_argument("--host", default="127.0.0.1",
                    help="استخدم 0.0.0.0 لفتح الواجهة من الجوال على نفس Wi-Fi")
    ap.add_argument("--consent", choices=["yes", "no"], default=None,
                    help="تسجيل إذن إرسال بيانات الاستخدام المجهولة للمطوّر "
                         "بدون سؤال (تُحفظ في ocr_data/telemetry.json)")
    ap.add_argument("--no-prompt", action="store_true",
                    help="لا تسأل عن الإذن في الطرفية عند البدء")
    args = ap.parse_args()

    build_page_maps()

    # إذن إرسال بيانات الاستخدام — يُسأل عنه مرة واحدة عند البدء
    c = consent_state()
    if c is None and args.consent:
        set_consent(args.consent == "yes")
        c = args.consent == "yes"
    if c is None and not args.no_prompt and sys.stdin.isatty():
        print("📊 هل توافق على إرسال بيانات استخدام مجهولة (أنواع العمليات "
              "وأعدادها ونسخة البرنامج — بلا صور ولا نصوص) إلى المطوّر؟")
        print("   تُسجَّل محليًا دومًا في ocr_data/app_log.jsonl. [y/N] ",
              end="", flush=True)
        try:
            ans = input().strip().lower()
        except EOFError:
            ans = ""
        set_consent(ans in ("y", "yes"))
        c = ans in ("y", "yes")
    app_log("server_start", port=args.port, host=args.host, consent=c)

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    srv.daemon_threads = True

    print(f"✅ واجهة المراجعة جاهزة:  http://127.0.0.1:{args.port}/panel  (لوحة التحكم)")
    print(f"   المحرر مباشرة:            http://127.0.0.1:{args.port}/")
    if args.host == "0.0.0.0":
        urls = lan_urls(args.port)
        for u in urls:
            print(f"   📱 من الجوال (نفس Wi-Fi):  {u}")
        if urls:
            try:
                import qrcode
                qr = qrcode.QRCode(border=1)
                qr.add_data(urls[0])
                qr.make()
                print("   ── امسح هذا الرمز بكاميرا الجوال ──")
                qr.print_ascii(invert=True)
            except ImportError:
                print("   💡 لطباعة رمز QR يمسحه الجوال:  pip install qrcode")
    else:
        print("   💡 للفتح من الجوال على نفس Wi-Fi:  python 7_box_editor_server.py --host 0.0.0.0")
    print("   (في Colab: الخلية 6ج تطبع رابط الوكيل — افتحه في تبويب جديد)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("⛔ أُغلق الخادم")


if __name__ == "__main__":
    main()
