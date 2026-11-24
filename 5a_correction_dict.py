#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الخطوة 5أ — قاموس التصحيح المستمر والتعلم النشط (الإصدار 1.7).

يحوّل تصحيحاتك المتكررة إلى أصل معرفي محلي يتسعّ مع كل مراجعة:
  • يبني قاموسًا (خطأ → صواب) من سجل التعديلات `box_adjustments.jsonl`
    الذي يسجّله محرر المربعات وتطبيق المراجعة تلقائيًا.
  • يطابق نصوص OCR بخمس طبقات: تام → مطبَّع (بلا تشكيل وألف التنوين
    موحّدة) → بصري (خريطة الالتباس في الخط اليدوي العربي: بتثن، جحخ،
    دذ، رز، سش، صض، طظ، عغ، فق) → تقريبي (Levenshtein ≤ 2) → طبي.
  • يوجّه المراجعة: قبول تلقائي واثق (طبقة تامة/مطبَّعة بتكرار ≥ 2)
    مع تدقيق عشوائي 5%، واقتراح بنقرة واحدة لما هو أقل ثقة، وتسجيل
    كل قرار في `corrections.db`.

محلي بالكامل — لا شبكة ولا نماذج خارجية، وقاعدة القاموس منفصلة عن
قاعدة القصاصات (`crops.db`) فلا تلمس بيانات التدريب.

الاستخدام:
  python 5a_correction_dict.py build                       # بناء/تحديث القاموس
  python 5a_correction_dict.py stats                       # إحصاءات القاموس
  python 5a_correction_dict.py search "بحت"                # اقتراحات لنص
  python 5a_correction_dict.py import-medical medical_terms.txt
  python 5a_correction_dict.py decisions 20                # آخر القرارات
"""
import argparse
import json
import random
import re
import sqlite3
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

VERSION = "1.7.0"

# ───────────────────────── التطبيع والمفاتيح البصرية ─────────────────────────

# حركات الإعراب والشد + التدويل + ألف فوقية
_TASHKEEL_RE = re.compile(r"[\u064B-\u065F\u0670\u0640]")

_NORM_MAP = str.maketrans({
    "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",
    "ى": "ي", "ئ": "ي", "ؤ": "و",
    "ة": "ه",
    "ک": "ك", "ی": "ي", "ے": "ي",
    "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
    "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
})

# مجموعات الالتباس البصري في الخط اليدوي العربي: حروف يخطئ فيها النموذج
# بسبب تشابه الرسم، فتُطبع كلها إلى قائد المجموعة عند بناء المفتاح البصري.
_VISUAL_GROUPS = (("بتثن", "ب"), ("جحخ", "ح"), ("دذ", "د"), ("رز", "ر"),
                  ("سش", "س"), ("صض", "ص"), ("طظ", "ط"), ("عغ", "ع"),
                  ("فق", "ف"))
_VISUAL_MAP = str.maketrans({c: lead for grp, lead in _VISUAL_GROUPS
                             for c in grp})


def normalize_ar(text):
    """تطبيع عربي: بلا تشكيل/تدويل، ألف جامدة، ة→ه، ى→ي، أرقام لاتينية."""
    if not text:
        return ""
    t = _TASHKEEL_RE.sub("", str(text)).translate(_NORM_MAP)
    return " ".join(t.split())


def visual_key(text):
    """مفتاح بصري: النص المطبَّع بعد طبع كل حرف إلى قائد مجموعته."""
    return normalize_ar(text).translate(_VISUAL_MAP)


def levenshtein(a, b, max_dist=2):
    """مسافة التحرير مع قطع مبكر عند تجاوز الحد المسموح."""
    if a == b:
        return 0
    if abs(len(a) - len(b)) > max_dist:
        return max_dist + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        best = i
        for j, cb in enumerate(b, 1):
            v = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
            cur.append(v)
            if v < best:
                best = v
        if best > max_dist:
            return max_dist + 1
        prev = cur
    return prev[-1] if prev[-1] <= max_dist else max_dist + 1


# ────────────────────────────── قاعدة القاموس ────────────────────────────────

def open_corr_db(path):
    """يفتح/ينشئ قاعدة القاموس: corrections + medical_terms + decisions."""
    conn = sqlite3.connect(str(path))
    conn.execute("""CREATE TABLE IF NOT EXISTS corrections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        wrong TEXT NOT NULL, wrong_norm TEXT NOT NULL, wrong_vis TEXT NOT NULL,
        right TEXT NOT NULL, right_norm TEXT NOT NULL,
        count INTEGER NOT NULL DEFAULT 1,
        pages TEXT NOT NULL DEFAULT '',
        last_seen TEXT NOT NULL DEFAULT '')""")
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS u_wrong_right "
                 "ON corrections(wrong, right)")
    conn.execute("CREATE INDEX IF NOT EXISTS i_wnorm ON corrections(wrong_norm)")
    conn.execute("CREATE INDEX IF NOT EXISTS i_wvis ON corrections(wrong_vis)")
    conn.execute("""CREATE TABLE IF NOT EXISTS medical_terms (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        term TEXT NOT NULL, term_norm TEXT NOT NULL UNIQUE,
        abbr TEXT, category TEXT)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS decisions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL, crop_id INTEGER, page_name TEXT,
        given TEXT, suggestion TEXT, score REAL, tier TEXT,
        action TEXT NOT NULL, final TEXT)""")
    conn.execute("CREATE TABLE IF NOT EXISTS meta "
                 "(k TEXT PRIMARY KEY, v TEXT)")
    conn.commit()
    return conn


# ─────────────────────── البناء من سجل التعديلات ─────────────────────────────

def _jsonl_signature(jsonl_path):
    p = Path(jsonl_path)
    if not p.exists():
        return ""
    st = p.stat()
    return f"{st.st_mtime_ns}:{st.st_size}"


def ensure_fresh(corr, jsonl_path):
    """يعيد البناء فقط إن تغيّر سجل التعديلات منذ آخر بناء (رخيص لكل تشغيل)."""
    sig = _jsonl_signature(jsonl_path)
    row = corr.execute("SELECT v FROM meta WHERE k='adj_sig'").fetchone()
    if row and row[0] == sig:
        return None
    out = build_from_jsonl(corr, jsonl_path)
    corr.execute("INSERT INTO meta(k, v) VALUES('adj_sig', ?) "
                 "ON CONFLICT(k) DO UPDATE SET v=excluded.v", (sig,))
    corr.commit()
    return out


def build_from_jsonl(corr, jsonl_path):
    """يستخرج أزواج (خطأ → صواب) من أحداث تعديل الوسوم في السجل."""
    p = Path(jsonl_path)
    n_ev = n_pairs = n_cosmetic = 0
    if not p.exists():
        return {"events": 0, "pairs": 0, "cosmetic": 0}
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        act = rec.get("action")
        if act not in ("update", "review_edit"):
            continue
        old_l = (rec.get("old") or {}).get("label")
        new_l = (rec.get("new") or {}).get("label")
        if not old_l or not new_l:
            continue
        old_l, new_l = old_l.strip(), new_l.strip()
        if not old_l or not new_l or old_l == new_l:
            continue
        n_ev += 1
        if normalize_ar(old_l) == normalize_ar(new_l):
            n_cosmetic += 1        # تشكيل/تدويل فقط — ليس تصحيحًا معجميًا
            continue
        _upsert(corr, old_l, new_l, str(rec.get("page") or ""))
        n_pairs += 1
    corr.commit()
    return {"events": n_ev, "pairs": n_pairs, "cosmetic": n_cosmetic}


def _upsert(corr, wrong, right, page):
    wn, wv, rn = normalize_ar(wrong), visual_key(wrong), normalize_ar(right)
    row = corr.execute("SELECT id, count, pages FROM corrections "
                       "WHERE wrong=? AND right=?", (wrong, right)).fetchone()
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    if row:
        pages = set(filter(None, row[2].split(","))) | ({page} if page else set())
        corr.execute("UPDATE corrections SET count=?, pages=?, last_seen=? "
                     "WHERE id=?", (row[1] + 1, ",".join(sorted(pages)), ts, row[0]))
    else:
        corr.execute("""INSERT INTO corrections (wrong, wrong_norm, wrong_vis,
                        right, right_norm, count, pages, last_seen)
                        VALUES (?,?,?,?,?,1,?,?)""",
                     (wrong, wn, wv, right, rn, page or "", ts))


def import_medical(corr, path):
    """استيراد مصطلحات طبية من ملف: مصطلح|اختصار|فئة لكل سطر."""
    p = Path(path)
    if not p.exists():
        return 0
    n = 0
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [x.strip() for x in line.split("|")]
        term = parts[0]
        if not term:
            continue
        abbr = parts[1] if len(parts) > 1 and parts[1] else None
        cat = parts[2] if len(parts) > 2 and parts[2] else None
        corr.execute("""INSERT INTO medical_terms (term, term_norm, abbr, category)
                        VALUES (?,?,?,?) ON CONFLICT(term_norm) DO UPDATE
                        SET abbr=excluded.abbr, category=excluded.category""",
                     (term, normalize_ar(term), abbr, cat))
        n += 1
    corr.commit()
    return n


# ──────────────────────────── المطابقة والاقتراح ─────────────────────────────

@dataclass
class Suggestion:
    text: str        # النص المصحَّح المقترح
    score: float     # درجة الثقة (طبقة المطابقة)
    tier: str        # exact / norm / visual / fuzzy / medical
    count: int       # عدد مرات رصد هذا التصحيح في السجل
    wrong: str       # النص الخاطئ المطابق


MIN_SCORE = 0.80
AUTO_TIERS = ("exact", "norm", "medical")
MIN_AUTO_COUNT = 2


def is_confident(s):
    """قبول تلقائي؟ طبقة حاسمة (تامة/مطبَّعة/طبية) بتكرار كافٍ."""
    return s.tier in AUTO_TIERS and s.count >= MIN_AUTO_COUNT


def lookup(conn, text, top_k=3, min_score=MIN_SCORE, use_medical=True):
    """اقتراحات متعددة الطبقات لنص واحد، مرتبة بالثقة ثم التكرار."""
    if not (text or "").strip():
        return []
    norm = normalize_ar(text)
    vis = visual_key(text)
    results = {}                       # keyed by نص الاقتراح

    def merge(right, score, tier, count, wrong):
        cur = results.get(right)
        if cur is None or (score, count) > (cur.score, cur.count):
            results[right] = Suggestion(right, score, tier, count, wrong)

    # 1) مطابقة تامة على النص الخام
    for r in conn.execute("SELECT right, count FROM corrections "
                          "WHERE wrong=? LIMIT 5", (text.strip(),)):
        merge(r[0], 1.0, "exact", r[1], text.strip())
    # 2) مطابقة على النص المطبَّع
    if len(results) < top_k:
        for r in conn.execute("SELECT wrong, right, count FROM corrections "
                              "WHERE wrong_norm=? LIMIT 5", (norm,)):
            merge(r[1], 0.98, "norm", r[2], r[0])
    # 3) مطابقة بصرية (خريطة الالتباس)
    if len(results) < top_k and vis:
        for r in conn.execute("SELECT wrong, right, count FROM corrections "
                              "WHERE wrong_vis=? LIMIT 5", (vis,)):
            merge(r[1], 0.92, "visual", r[2], r[0])
    # 4) تشابه تقريبي (مرشّحون بالسابقة نفسها لتخفيف الفحص)
    if len(results) < top_k and len(norm) >= 2:
        prefix = norm[:2]
        for r in conn.execute("""SELECT wrong, right, count, wrong_norm FROM
                corrections WHERE wrong_norm LIKE ? ORDER BY count DESC
                LIMIT 50""", (prefix + "%",)):
            d = levenshtein(norm, r[3], max_dist=2)
            if d <= 2:
                merge(r[1], 0.90 - 0.05 * d, "fuzzy", r[2], r[0])
    # 5) المصطلحات الطبية: مطابقة تامة = موثوق، شبه مطابقة = اقتراح
    if use_medical:
        for r in conn.execute("SELECT term, term_norm, abbr FROM medical_terms"):
            tn = r[1]
            if tn == norm:
                merge(r[0], 0.95, "medical", 999, text.strip())
            elif levenshtein(norm, tn, max_dist=1) <= 1:
                merge(r[0], 0.85, "medical_near", 999, text.strip())
    out = [s for s in results.values() if s.score >= min_score]
    out.sort(key=lambda s: (s.score, s.count), reverse=True)
    return out[:top_k]


def best(conn, text):
    """أفضل اقتراح واحد أو None."""
    s = lookup(conn, text)
    return s[0] if s else None


# ─────────────────────────── توجيه المراجعة ──────────────────────────────────

def route(rows, conn, rng=None, audit_rate=0.05):
    """تصنيف قصاصات معلّقة: auto (قبول تلقائي) / audit (تدقيق عشوائي)
    / suggest (اقتراح معروض) / plain (مراجعة عادية).

    rows: قائمة صفوف تشبه صفوف crops (id, path, label, conf, lang, page).
    القرار التلقائي هنا لا يعدّل شيئًا — المتصل يطبّقه ويسجلّه."""
    rng = rng or random.Random()
    out = []
    for row in rows:
        suggs = lookup(conn, row[2] or "")
        top = suggs[0] if suggs else None
        if top and is_confident(top):
            if rng.random() < audit_rate:
                out.append({"row": row, "action": "audit", "suggestion": top})
            else:
                out.append({"row": row, "action": "auto", "suggestion": top})
        elif top:
            out.append({"row": row, "action": "suggest", "suggestion": top})
        else:
            out.append({"row": row, "action": "plain", "suggestion": None})
    return out


def gray_zone(confidences, low_pct=25, high_pct=75):
    """النطاق الرمادي إحصائيًا (أربيعات ثقة العينات المعلّقة) أو None."""
    data = [c for c in confidences if c is not None]
    if len(data) < 4:
        return None
    lo, hi = statistics.quantiles(data, n=4)[0], statistics.quantiles(data, n=4)[2]
    return (lo, hi)


def log_decision(corr, crop_id, page, given, sugg, action, final=None):
    """تسجيل قرار في corrections.db — مسار التدقيق الكامل للقاموس."""
    corr.execute("""INSERT INTO decisions (ts, crop_id, page_name, given,
                    suggestion, score, tier, action, final)
                    VALUES (?,?,?,?,?,?,?,?,?)""",
                 (time.strftime("%Y-%m-%d %H:%M:%S"), crop_id, page, given,
                  sugg.text if sugg else None,
                  sugg.score if sugg else None,
                  sugg.tier if sugg else None, action, final))
    corr.commit()


def stats_summary(corr):
    n_pairs = corr.execute("SELECT COUNT(*), COALESCE(SUM(count),0) "
                           "FROM corrections").fetchone()
    n_med = corr.execute("SELECT COUNT(*) FROM medical_terms").fetchone()[0]
    dec = dict(corr.execute("SELECT action, COUNT(*) FROM decisions "
                            "GROUP BY action").fetchall())
    return {"pairs": n_pairs[0], "occurrences": n_pairs[1], "medical": n_med,
            "decisions": dec}


# ─────────────────────────────── واجهة الأوامر ───────────────────────────────

def main(argv=None):
    try:
        from config import DB_PATH as CROPS_DB
        default_root = Path(CROPS_DB).parent
    except Exception:
        default_root = Path("ocr_data")
    ap = argparse.ArgumentParser(description="قاموس التصحيح والتعلم النشط")
    ap.add_argument("cmd", choices=["build", "stats", "search", "import-medical",
                                    "decisions"])
    ap.add_argument("text", nargs="?", help="نص البحث في الأمر search")
    ap.add_argument("--db", default=str(default_root / "corrections.db"))
    ap.add_argument("--jsonl", default=str(default_root / "box_adjustments.jsonl"))
    ap.add_argument("--medical", default=str(Path(__file__).parent /
                                             "medical_terms.txt"))
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args(argv)

    conn = open_corr_db(args.db)
    if args.cmd == "build":
        seeded = 0
        if Path(args.medical).exists():
            if conn.execute("SELECT COUNT(*) FROM medical_terms").fetchone()[0] == 0:
                seeded = import_medical(conn, args.medical)
        out = ensure_fresh(conn, args.jsonl) or build_from_jsonl(conn, args.jsonl)
        print(f"✅ القاموس: {conn.execute('SELECT COUNT(*) FROM corrections').fetchone()[0]} زوج "
              f"(أحداث معدَّلة: {out['events']}، تجميلية متخطاة: {out['cosmetic']}) "
              f"— مصطلحات طبية مُستوردة الآن: {seeded}")
    elif args.cmd == "stats":
        s = stats_summary(conn)
        print(f"أزواج التصحيح: {s['pairs']} — ورود إجمالي: {s['occurrences']} — "
              f"مصطلحات طبية: {s['medical']}")
        for k, v in sorted(s["decisions"].items()):
            print(f"  قرار {k}: {v}")
    elif args.cmd == "search":
        if not args.text:
            raise SystemExit("❌ اكتب النص: search \"النص\"")
        suggs = lookup(conn, args.text)
        if not suggs:
            print("لا اقتراحات.")
        for s in suggs:
            print(f"  {s.text!r}  [طبقة: {s.tier} | ثقة: {s.score:.2f} | "
                  f"تكرار: {s.count}] ← من {s.wrong!r}")
    elif args.cmd == "import-medical":
        n = import_medical(conn, args.medical)
        print(f"✅ استُوردت {n} سطرًا مصطلحات طبية.")
    elif args.cmd == "decisions":
        for r in conn.execute("SELECT ts, crop_id, given, suggestion, tier, "
                              "action, final FROM decisions ORDER BY id DESC "
                              "LIMIT ?", (args.limit,)):
            print(" | ".join(str(x) if x is not None else "—" for x in r))
    conn.close()


if __name__ == "__main__":
    main()
