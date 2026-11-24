#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الخطوة 3ب — مراجعة الوسوم الآلية (Gradio + SQLite).

الوسوم المستوردة من قاعدة بيانات مشروع handwriting-notes حالة «auto»:
نصوص OCR أولية بثقة متفاوتة تحتاج تأكيدًا بشريًا قبل التدريب.

طريقة العمل:
  1. تعرض الأداة القصاصات ذات الثقة الأدنى أولًا (الأكثر احتمالًا للخطأ) مع
     الوسم الآلي في حقل نصي جاهز للتعديل.
  2. ✅ «قبول كما هو» → verified=1 (يمر للتصدير دومًا حتى لو كان قصيرًا).
     💾 «حفظ التصحيح» → تصحيح النص + verified=1.
     🚫 «تجاهل» → ignored=1 (يُستبعد من التصدير — قصاصة غير قابلة للقراءة).
     ⏭️ «تخطي» → ينقلها لآخر الدورة دون قرار.
  3. «تراجع» يعكس آخر قرار محفوظ.
  4. قاموس التصحيح (الإصدار 1.7): عند الإقلاع يبني القاموس من سجل
     `box_adjustments.jsonl` ويقبل تلقائيًا التصحيحات الحاسمة المتكررة
     (طبقة تامة/مطبَّعة بتكرار ≥ 2) مع تدقيق عشوائي 5% يبقيها للمراجعة،
     ويُظهر اقتراحًا بنقرة واحدة لبقية الطبقات (بصرية/تقريبية/طبية)،
     ويُسجّل كل قرار في `corrections.db` — وكل تصحيح تكتبه هنا يعود
     إلى القاموس في التشغيل التالي. «--no-auto» يوقف القبول التلقائي
     ويكتفي بالاقتراح.
"""
import argparse
import importlib.util
import json
import os
import random
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import toolkit_utils as tu
from config import DB_PATH


# ────────────────────────────── قاعدة البيانات ──────────────────────────────

def open_db(db_path):
    conn = tu.connect_db(db_path)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(crops)")]
    for col, typ in (("ignored", "INTEGER NOT NULL DEFAULT 0"),
                     ("verified", "INTEGER NOT NULL DEFAULT 0"),
                     ("confidence", "REAL"), ("language", "TEXT NOT NULL DEFAULT ''")):
        if col not in cols:
            conn.execute(f"ALTER TABLE crops ADD COLUMN {col} {typ}")
    conn.commit()
    return conn


def load_dict_module():
    """يحمّل 5a_correction_dict.py (أرقام بادئة الملفات تمنع الاستيراد العادي)."""
    p = Path(__file__).parent / "5a_correction_dict.py"
    spec = importlib.util.spec_from_file_location("correction_dict", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def stats(conn, lang):
    where = "WHERE ignored=0" + (" AND language=?" if lang and lang != "الكل" else "")
    params = () if lang in (None, "الكل") else (lang,)
    left = conn.execute(f"SELECT COUNT(*) FROM crops {where} AND verified=0", params).fetchone()[0]
    verified = conn.execute("SELECT COUNT(*) FROM crops WHERE verified=1").fetchone()[0]
    ignored = conn.execute("SELECT COUNT(*) FROM crops WHERE ignored=1").fetchone()[0]
    langs = [r[0] or "unknown" for r in conn.execute(
        "SELECT DISTINCT language FROM crops ORDER BY 1")]
    return left, verified, ignored, langs


def fetch_next(conn, lang, newest_first, skip_ids):
    """أول قصاصة غير مُراجَعة وغير مُتخطاة حسب الفلتر والترتيب."""
    order = "id DESC" if newest_first else "confidence IS NULL, confidence ASC, id ASC"
    where = "ignored=0 AND verified=0" + (" AND language=?" if lang and lang != "الكل" else "")
    params = () if lang in (None, "الكل") else (lang,)
    rows = conn.execute(
        f"SELECT id, crop_path, label, confidence, language, page_name "
        f"FROM crops WHERE {where} ORDER BY {order} LIMIT 300", params).fetchall()
    for r in rows:
        if r[0] not in skip_ids:
            return r
    return None


# ────────────────────────────── منطق المراجعة ───────────────────────────────

def resolve_crop(p, db_path):
    """مسارات القصاصات المستوردة قد تعود لجهاز آخر — إن لم يوجد الملف
    فجرّب مجلد crops بجوار القاعدة (نفس قاعدة الخطوة 4)."""
    q = Path(p)
    if q.exists():
        return q
    alt = Path(db_path).parent / "crops" / q.name
    return alt if alt.exists() else q


def render_row(row, db_path):
    if row is None:
        return None, "", "", "🎉 لا مزيد من القصاصات تحت هذا الفلتر — وسّع الفلتر أو انتقل للخطوة 4 (التصدير)."
    cid, path, label, conf, lang, page = row
    badge = f"#{cid} | اللغة: {lang or 'unknown'} | الثقة: {conf if conf is not None else '—'} | الصفحة: {page}"
    return str(resolve_crop(path, db_path)), label or "", badge, ""


def review_action(conn, action, cid, new_label, adj_path=None):
    """ينفّذ القرار ويعيد لقطة للتراجع؛ وتصحيح النص يُسجَّل في سجل
    التعديلات ليتعلم منه قاموس التصحيح في التشغيل التالي."""
    row = conn.execute(
        "SELECT label, verified, ignored, page_name FROM crops WHERE id=?",
        (cid,)).fetchone()
    if row is None:
        return None
    label0, verified0, ignored0, page0 = row
    if action == "accept":
        conn.execute("UPDATE crops SET verified=1 WHERE id=?", (cid,))
    elif action == "edit":
        new_l = (new_label or "").strip()
        conn.execute(
            "UPDATE crops SET label=?, verified=1 WHERE id=?",
            (new_l, cid))
        if adj_path and new_l and new_l != (label0 or "").strip():
            rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   "action": "review_edit", "page": page0, "id": cid,
                   "old": {"label": label0}, "new": {"label": new_l}}
            Path(adj_path).parent.mkdir(parents=True, exist_ok=True)
            with open(adj_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    elif action == "ignore":
        conn.execute("UPDATE crops SET ignored=1 WHERE id=?", (cid,))
    conn.commit()
    return (cid, label0, verified0, ignored0)


def undo_action(conn, snapshot):
    if not snapshot:
        return "⚠️ لا قرار محفوظ للتراجع."
    cid, label0, verified0, ignored0 = snapshot
    conn.execute(
        "UPDATE crops SET label=?, verified=?, ignored=? WHERE id=?",
        (label0, verified0, ignored0, cid))
    conn.commit()
    return f"↩️ رُجع القرار عن القصاصة #{cid}."


# ────────────────────────────── الواجهة ─────────────────────────────────────

def suggestion_md(s):
    if not s:
        return ""
    return (f"✨ **اقتراح القاموس: {s.text}** — طبقة: {s.tier} | "
            f"ثقة: {s.score:.2f} | تكرار: {s.count}")


def dict_prog(corr):
    if corr is None:
        return ""
    dec = dict(corr.execute(
        "SELECT action, COUNT(*) FROM decisions GROUP BY action").fetchall())
    return (f" | 🤖 تلقائي: {dec.get('auto', 0)} | تدقيق: {dec.get('audit', 0)}"
            f" | اقتراح مُقبل: {dec.get('accepted', 0)}")


def build_demo(db_path, corr_db=None, adj_log=None, no_auto=False):
    import gradio as gr
    conn = open_db(db_path)
    left0, ver0, ign0, langs0 = stats(conn, None)
    skip_ids0, last0 = [], None

    # ── قاموس التصحيح والقبول التلقائي (الإصدار 1.7) ──
    cd_m, corr = None, None
    try:
        cd_m = load_dict_module()
        corr = cd_m.open_corr_db(
            corr_db or (Path(db_path).parent / "corrections.db"))
        adj_log = adj_log or (Path(db_path).parent / "box_adjustments.jsonl")
        cd_m.ensure_fresh(corr, adj_log)
        med = Path(__file__).parent / "medical_terms.txt"
        if med.exists() and corr.execute(
                "SELECT COUNT(*) FROM medical_terms").fetchone()[0] == 0:
            cd_m.import_medical(corr, med)
        if not no_auto:
            pending = conn.execute(
                """SELECT id, crop_path, label, confidence, language, page_name
                   FROM crops WHERE ignored=0 AND verified=0""").fetchall()
            for r in cd_m.route(pending, corr, random.Random()):
                if r["action"] == "auto":
                    row = r["row"]
                    conn.execute(
                        "UPDATE crops SET label=?, verified=1 WHERE id=?",
                        (r["suggestion"].text, row[0]))
                    cd_m.log_decision(corr, row[0], row[5], row[2] or "",
                                      r["suggestion"], "auto",
                                      r["suggestion"].text)
        conn.commit()
        dec = dict(corr.execute(
            "SELECT action, COUNT(*) FROM decisions GROUP BY action").fetchall())
        auto_msg = (f"🤖 قبول تلقائي من القاموس: **{dec.get('auto', 0)}** قصاصة "
                    f"(تدقيق عشوائي 5% يبقى للمراجعة) | أزواج القاموس: "
                    f"**{corr.execute('SELECT COUNT(*) FROM corrections').fetchone()[0]}** | "
                    f"مصطلحات طبية: **{corr.execute('SELECT COUNT(*) FROM medical_terms').fetchone()[0]}**")
    except Exception as e:
        cd_m = corr = None
        auto_msg = f"⚠️ قاموس التصحيح معطل: {e}"

    first = fetch_next(conn, "الكل", False, skip_ids0)
    img0, label0, badge0, msg0 = render_row(first, db_path)
    sugg0 = None
    if corr is not None and first:
        b0 = cd_m.lookup(corr, first[2] or "")
        sugg0 = b0[0] if b0 else None

    with gr.Blocks(title="مراجعة الوسوم الآلية — OCR") as demo:
        gr.Markdown(
            "## 🔎 مراجعة وسوم OCR الآلية (عربي / إنجليزي / ألماني)\n"
            "تُعرض القصاصات **الأقل ثقة أولًا**. صحّح النص أو أكّده أو تجاهل القصاصة — "
            "كل قرار يُحفظ فورًا في `crops.db` ويؤثر مباشرة على مجموعة التصدير (الخطوة 4)."
        )
        skip_s   = gr.State(skip_ids0)
        last_s   = gr.State(last0)
        cur_id_s = gr.State(first[0] if first else None)
        sugg_s   = gr.State(sugg0)

        lang_dd = gr.Dropdown(choices=["الكل"] + langs0, value="الكل",
                              label="فلتر اللغة", interactive=True)
        order_r = gr.Radio(["الأقل ثقة أولًا", "الأحدث أولًا"], value="الأقل ثقة أولًا",
                           label="الترتيب", interactive=True)
        progress = gr.Markdown(f"متبقي للمراجعة: **{left0}** | مُراجَع: **{ver0}** | متجاهَل: **{ign0}**"
                               + dict_prog(corr))

        img = gr.Image(value=img0, label="القصاصة", interactive=False, height=220)
        badge = gr.Markdown(badge0)
        txt = gr.Textbox(label="نص الكلمة (عدّله إن كان خاطئًا)", value=label0, rtl=True)
        chip = gr.Markdown(suggestion_md(sugg0), visible=sugg0 is not None)

        with gr.Row():
            btn_accept = gr.Button("✅ قبول كما هو", variant="primary")
            btn_sugg   = gr.Button("✨ تطبيق الاقتراح", visible=sugg0 is not None)
            btn_edit   = gr.Button("💾 حفظ التصحيح")
            btn_ignore = gr.Button("🚫 تجاهل القصاصة")
            btn_skip   = gr.Button("⏭️ تخطي")
            btn_undo   = gr.Button("↩️ تراجع")

        status = gr.Markdown(msg0 + " " + auto_msg)

        def advance(lang, newest, skip, last, action, cid, new_label, add_skip,
                    sugg=None):
            """قرار واحد (اختياري) + تخطٍ (اختياري) ثم جلب التالي."""
            msg = ""
            if action and cid is not None:
                last = review_action(conn, action, cid, new_label, adj_log)
                msg = {"accept": "✅ أُكّد الوسم.", "edit": "💾 حُفظ التصحيح.",
                       "ignore": "🚫 أُهملت القصاصة (لن تُصدَّر)."}[action]
                if corr is not None and sugg is not None:
                    page = conn.execute("SELECT page_name FROM crops WHERE id=?",
                                        (cid,)).fetchone()
                    dec = ("accepted" if action == "edit" and
                           (new_label or "").strip() == sugg.text else "rejected")
                    cd_m.log_decision(corr, cid, page[0] if page else None,
                                      sugg.wrong, sugg, dec, new_label)
            elif add_skip and cid is not None:            # تخطي دون قرار
                skip = skip + [cid]
            nxt = fetch_next(conn, lang, newest, skip)
            if nxt is None and skip:                      # كل المتبقي متخطى؟ دورة جديدة
                skip = []
                nxt = fetch_next(conn, lang, newest, skip)
            left, ver, ign, _ = stats(conn, lang)
            img_, label_, badge_, msg2_ = render_row(nxt, db_path)
            sugg_ = None
            if corr is not None and nxt:
                b = cd_m.lookup(corr, nxt[2] or "")
                sugg_ = b[0] if b else None
            msg_all = (msg + " " if msg else "") + (msg2_ or "")
            return (skip, last, nxt[0] if nxt else None, img_, label_, badge_,
                    msg_all,
                    f"متبقي للمراجعة: **{left}** | مُراجَع: **{ver}** | متجاهَل: **{ign}**"
                    + dict_prog(corr),
                    suggestion_md(sugg_), sugg_)

        def on_undo_click(lang, newest, skip, last, cid):
            umsg = undo_action(conn, last) if last else "⚠️ لا قرار محفوظ للتراجع."
            _, last2, cur2, img_, label_, badge_, msg_, prog_, chip_, sugg_ = advance(
                lang, newest, skip, last, None, None, None, False, None)
            return skip, last2, cur2, img_, label_, badge_, f"{umsg}\n\n{msg_ or ''}".strip(), prog_, chip_, sugg_

        for btn, act, add_skip in ((btn_accept, "accept", False), (btn_edit, "edit", False),
                                   (btn_ignore, "ignore", False), (btn_skip, None, True)):
            btn.click(
                lambda lang, newest, skip, last, cid, nl, sg, a=act, sk=add_skip:
                    advance(lang, newest, skip, last, a, cid, nl, sk, sg),
                inputs=[lang_dd, order_r, skip_s, last_s, cur_id_s, txt, sugg_s],
                outputs=[skip_s, last_s, cur_id_s, img, txt, badge, status,
                         progress, chip, sugg_s],
            )

        btn_sugg.click(
            lambda lang, newest, skip, last, cid, sg:
                advance(lang, newest, skip, last,
                        "edit" if sg else None, cid,
                        sg.text if sg else "", False, sg),
            inputs=[lang_dd, order_r, skip_s, last_s, cur_id_s, sugg_s],
            outputs=[skip_s, last_s, cur_id_s, img, txt, badge, status,
                     progress, chip, sugg_s],
        )

        btn_undo.click(
            on_undo_click,
            inputs=[lang_dd, order_r, skip_s, last_s, cur_id_s],
            outputs=[skip_s, last_s, cur_id_s, img, txt, badge, status,
                     progress, chip, sugg_s],
        )

    return demo


def main():
    ap = argparse.ArgumentParser(description="مراجعة الوسوم الآلية (Gradio + SQLite)")
    ap.add_argument("--db", default=DB_PATH)
    ap.add_argument("--port", type=int, default=7861)
    ap.add_argument("--share", action="store_true", help="رابط عام (يُفعَّل تلقائيًا في Colab)")
    ap.add_argument("--dict", default=None, help="مسار قاعدة قاموس التصحيح")
    ap.add_argument("--no-auto", action="store_true",
                    help="اقتراح فقط — بلا قبول تلقائي للتصحيحات المتكررة")
    args = ap.parse_args()

    conn = open_db(args.db)
    left, ver, ign, _ = stats(conn, None)
    conn.close()
    print(f"📊 للمراجعة: {left} | مُراجَع: {ver} | متجاهَل: {ign}")
    if ver + ign + left == 0:
        print("⚠️ قاعدة البيانات فارغة — شغّل سكربت البناء أو الخطوة 3 أولًا.")

    demo = build_demo(args.db, corr_db=args.dict, no_auto=args.no_auto)
    in_colab = bool(os.environ.get("COLAB_RELEASE_TAG"))
    demo.queue(max_size=8).launch(
        server_name="0.0.0.0", server_port=args.port,
        share=args.share or in_colab, show_error=True, inbrowser=False,
    )
    if in_colab:
        print("\n🛑 عند الانتهاء من المراجعة: أوقف هذه الخلية ثم أعد تشغيل الخطوة 4 (التصدير).")


if __name__ == "__main__":
    main()
