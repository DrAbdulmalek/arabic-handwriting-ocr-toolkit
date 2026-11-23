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
  GET  /img/page     صورة الصفحة كاملة
  GET  /img/crop     قص أي منطقة من الصفحة (عرض السطر/الجزء المكبّر)
  POST /api/update   تحديث مربعات/وسوم/حالات دفعة واحدة
  POST /api/add      إضافة مربع جديد
  POST /api/delete   تجاهل مربعات (تبقى في القاعدة معلّمة ignored=1)
  POST /api/merge    دمج مربعين في واحد
كل تغيير يُسجَّل في ocr_data/box_adjustments.jsonl للتدقيق، وصورة القصاصة
تُعاد توليدها تلقائيًا من صفحة الصفحة الأصلية عند تعديل المربع.
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


# ─────────────────────────── عمليات البيانات ───────────────────────────
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
                return self._json({"ok": True, "pages": len(PAGE_MAP)})
            if u.path == "/api/pages":
                return self._json(api_pages())
            if u.path == "/api/page":
                return self._json(api_page(q["name"][0]))
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
            if u.path == "/api/delete":
                return self._json(api_delete(self._body().get("ids", [])))
            if u.path == "/api/merge":
                b = self._body()
                return self._json(api_merge(b["keep_id"], b["drop_id"]))
            return self._json({"error": "مسار غير معروف"}, 404)
        except Exception as e:
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
    args = ap.parse_args()

    build_page_maps()
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
