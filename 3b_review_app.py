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
"""
import argparse
import os
import sqlite3
from pathlib import Path

import gradio as gr

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


def review_action(conn, action, cid, new_label):
    """ينفّذ القرار ويعيد لقطة للتراجع."""
    row = conn.execute(
        "SELECT label, verified, ignored FROM crops WHERE id=?", (cid,)).fetchone()
    if row is None:
        return None
    label0, verified0, ignored0 = row
    if action == "accept":
        conn.execute("UPDATE crops SET verified=1 WHERE id=?", (cid,))
    elif action == "edit":
        conn.execute(
            "UPDATE crops SET label=?, verified=1 WHERE id=?",
            ((new_label or "").strip(), cid))
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

def build_demo(db_path):
    conn = open_db(db_path)
    left0, ver0, ign0, langs0 = stats(conn, None)
    skip_ids0, last0 = [], None
    first = fetch_next(conn, "الكل", False, skip_ids0)
    img0, label0, badge0, msg0 = render_row(first, db_path)

    with gr.Blocks(title="مراجعة الوسوم الآلية — OCR") as demo:
        gr.Markdown(
            "## 🔎 مراجعة وسوم OCR الآلية (عربي / إنجليزي / ألماني)\n"
            "تُعرض القصاصات **الأقل ثقة أولًا**. صحّح النص أو أكّده أو تجاهل القصاصة — "
            "كل قرار يُحفظ فورًا في `crops.db` ويؤثر مباشرة على مجموعة التصدير (الخطوة 4)."
        )
        skip_s   = gr.State(skip_ids0)
        last_s   = gr.State(last0)
        cur_id_s = gr.State(first[0] if first else None)

        lang_dd = gr.Dropdown(choices=["الكل"] + langs0, value="الكل",
                              label="فلتر اللغة", interactive=True)
        order_r = gr.Radio(["الأقل ثقة أولًا", "الأحدث أولًا"], value="الأقل ثقة أولًا",
                           label="الترتيب", interactive=True)
        progress = gr.Markdown(f"متبقي للمراجعة: **{left0}** | مُراجَع: **{ver0}** | متجاهَل: **{ign0}**")

        img = gr.Image(value=img0, label="القصاصة", interactive=False, height=220)
        badge = gr.Markdown(badge0)
        txt = gr.Textbox(label="نص الكلمة (عدّله إن كان خاطئًا)", value=label0, rtl=True)

        with gr.Row():
            btn_accept = gr.Button("✅ قبول كما هو", variant="primary")
            btn_edit   = gr.Button("💾 حفظ التصحيح")
            btn_ignore = gr.Button("🚫 تجاهل القصاصة")
            btn_skip   = gr.Button("⏭️ تخطي")
            btn_undo   = gr.Button("↩️ تراجع")

        status = gr.Markdown(msg0)

        def advance(lang, newest, skip, last, action, cid, new_label, add_skip):
            """قرار واحد (اختياري) + تخطٍ (اختياري) ثم جلب التالي."""
            msg = ""
            if action and cid is not None:
                last = review_action(conn, action, cid, new_label)
                msg = {"accept": "✅ أُكّد الوسم.", "edit": "💾 حُفظ التصحيح.",
                       "ignore": "🚫 أُهملت القصاصة (لن تُصدَّر)."}[action]
            elif add_skip and cid is not None:            # تخطي دون قرار
                skip = skip + [cid]
            nxt = fetch_next(conn, lang, newest, skip)
            if nxt is None and skip:                      # كل المتبقي متخطى؟ دورة جديدة
                skip = []
                nxt = fetch_next(conn, lang, newest, skip)
            left, ver, ign, _ = stats(conn, lang)
            img_, label_, badge_, msg2_ = render_row(nxt, db_path)
            msg_all = (msg + " " if msg else "") + (msg2_ or "")
            return (skip, last, nxt[0] if nxt else None, img_, label_, badge_, msg_all,
                    f"متبقي للمراجعة: **{left}** | مُراجَع: **{ver}** | متجاهَل: **{ign}**")

        def on_undo_click(lang, newest, skip, last, cid):
            umsg = undo_action(conn, last) if last else "⚠️ لا قرار محفوظ للتراجع."
            _, last2, cur2, img_, label_, badge_, msg_, prog_ = advance(
                lang, newest, skip, last, None, None, None, False)
            return skip, last2, cur2, img_, label_, badge_, f"{umsg}\n\n{msg_ or ''}".strip(), prog_

        for btn, act, add_skip in ((btn_accept, "accept", False), (btn_edit, "edit", False),
                                   (btn_ignore, "ignore", False), (btn_skip, None, True)):
            btn.click(
                lambda lang, newest, skip, last, cid, nl, a=act, sk=add_skip:
                    advance(lang, newest, skip, last, a, cid, nl, sk),
                inputs=[lang_dd, order_r, skip_s, last_s, cur_id_s, txt],
                outputs=[skip_s, last_s, cur_id_s, img, txt, badge, status, progress],
            )

        btn_undo.click(
            on_undo_click,
            inputs=[lang_dd, order_r, skip_s, last_s, cur_id_s],
            outputs=[skip_s, last_s, cur_id_s, img, txt, badge, status, progress],
        )

    return demo


def main():
    ap = argparse.ArgumentParser(description="مراجعة الوسوم الآلية (Gradio + SQLite)")
    ap.add_argument("--db", default=DB_PATH)
    ap.add_argument("--port", type=int, default=7861)
    ap.add_argument("--share", action="store_true", help="رابط عام (يُفعَّل تلقائيًا في Colab)")
    args = ap.parse_args()

    conn = open_db(args.db)
    left, ver, ign, _ = stats(conn, None)
    conn.close()
    print(f"📊 للمراجعة: {left} | مُراجَع: {ver} | متجاهَل: {ign}")
    if ver + ign + left == 0:
        print("⚠️ قاعدة البيانات فارغة — شغّل سكربت البناء أو الخطوة 3 أولًا.")

    demo = build_demo(args.db)
    in_colab = bool(os.environ.get("COLAB_RELEASE_TAG"))
    demo.queue(max_size=8).launch(
        server_name="0.0.0.0", server_port=args.port,
        share=args.share or in_colab, show_error=True, inbrowser=False,
    )
    if in_colab:
        print("\n🛑 عند الانتهاء من المراجعة: أوقف هذه الخلية ثم أعد تشغيل الخطوة 4 (التصدير).")


if __name__ == "__main__":
    main()
