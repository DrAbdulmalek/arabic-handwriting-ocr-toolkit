#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الخطوة 3 — أداة قص الكلمات ووسمها (Gradio + SQLite).

طريقة العمل لكل صفحة:
  1. انقر على زاوية الكلمة العلوية (يسار أو يمين) ثم على الزاوية السفلية المقابلة
     — يُرسم صندوق أحمر.
  2. اكتب نص الكلمة كما هي بخط اليد في حقل "نص الكلمة".
  3. اضغط "حفظ القصاصة" — تُقص الكلمة من الصورة الأصلية وتُحفظ في ocr_data/crops
     ويُسجل صف في قاعدة بيانات SQLite.
  4. عند إنهاء الصفحة اضغط "إنهاء الصفحة" للانتقال التلقائي لما تبقى.

إحداثيات النقر تُحوَّل تلقائيًا من صورة العرض إلى الصورة الأصلية، وكل صندوق
يُحفظ بإحداثياته الأصلية في قاعدة البيانات لإعادة القص مستقبلًا دون إعادة وسم.
"""
import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw

import gradio as gr

import toolkit_utils as tu
from config import PAGES_DIR, CROPS_DIR, DB_PATH

MAX_DISPLAY_W = 800   # عرض صورة العرض؛ الإحداثيات تُعاد إلى المسار الأصلي تلقائيًا


# ────────────────────────────── دوال العرض والرسم ───────────────────────────

def load_page_display(path):
    """الصورة الأصلية + نسخة معروضة مصغرة + معامل التحويل (قيمة→أصل = 1/scale)."""
    orig = Image.open(path).convert("RGB")
    scale = min(1.0, MAX_DISPLAY_W / orig.width)
    disp = orig.resize((max(1, int(orig.width * scale)), max(1, int(orig.height * scale)))) if scale < 1 else orig.copy()
    return orig, disp, scale


def draw_overlay(disp, points, saved_boxes_val):
    im = disp.copy()
    d = ImageDraw.Draw(im)
    for (x1, y1, x2, y2) in saved_boxes_val:
        d.rectangle([x1, y1, x2, y2], outline=(16, 150, 72), width=3)
    if len(points) == 1:
        x, y = points[0]
        d.ellipse([x - 7, y - 7, x + 7, y + 7], outline=(220, 60, 60), width=3)
    elif len(points) >= 2:
        (xa, ya), (xb, yb) = points[0], points[1]
        d.rectangle([min(xa, xb), min(ya, yb), max(xa, xb), max(ya, yb)], outline=(220, 60, 60), width=3)
    return im


def saved_boxes_val(conn, page_name, scale):
    """صناديق الكلمات المحفوظة للصفحة محوَّلة إلى إحداثيات صورة العرض."""
    rows = conn.execute(
        "SELECT x1,y1,x2,y2 FROM crops WHERE page_name=? ORDER BY id", (page_name,)
    ).fetchall()
    return [tuple(round(v * scale) for v in r) for r in rows]


def page_table(conn, page_name):
    rows = conn.execute(
        "SELECT id,label,crop_path FROM crops WHERE page_name=? ORDER BY id", (page_name,)
    ).fetchall()
    return [[i + 1, lbl, Path(p).name] for i, (_, lbl, p) in enumerate(rows)]


def next_undone(files, conn, after=None):
    done = {r[0] for r in conn.execute("SELECT name FROM pages WHERE done=1")}
    order = list(range(len(files)))
    if after is not None:
        order = order[after + 1:] + order[:after + 1]   # البحث بدءًا من التالي ثم دورة كاملة
    for i in order:
        if Path(files[i]).name not in done:
            return i
    return None


# ────────────────────────────── أحداث الواجهة ───────────────────────────────

def on_click(evt: gr.SelectData, files, idx, points, scale, disp, db_path):
    x, y = evt.index[0], evt.index[1]
    W, H = disp.size
    x, y = max(0, min(int(x), W - 1)), max(0, min(int(y), H - 1))
    pts = points + [(x, y)]
    if len(pts) > 2:
        pts = [(x, y)]
    conn = tu.connect_db(db_path)
    boxes = saved_boxes_val(conn, Path(files[idx]).name, scale)
    conn.close()
    hint = "✓ الزاوية الأولى — انقر الآن الزاوية المقابلة." if len(pts) == 1 else \
           "✅ اكتب نص الكلمة ثم اضغط «حفظ القصاصة»."
    return draw_overlay(disp, pts, boxes), pts, hint


def on_save(label, files, idx, points, scale, disp, db_path, crops_dir):
    label = (label or "").strip()
    if not label:
        return draw_overlay(disp, points, []), points, "❌ اكتب نص الكلمة أولًا.", gr.update(), gr.update(), ""
    if len(points) != 2:
        return draw_overlay(disp, points, []), points, "❌ حدد الكلمة بالنقر على زاويتيها أولًا.", gr.update(), gr.update(), ""

    page_path = Path(files[idx])
    page_name = page_path.name
    conn = tu.connect_db(db_path)
    try:
        (x1v, y1v), (x2v, y2v) = points
        ox1 = round(min(x1v, x2v) / scale); oy1 = round(min(y1v, y2v) / scale)
        ox2 = round(max(x1v, x2v) / scale); oy2 = round(max(y1v, y2v) / scale)

        orig = Image.open(page_path).convert("RGB")
        crop = tu.crop_word_image(orig, ox1, oy1, ox2, oy2, pad=6)

        crops_dir = Path(crops_dir); crops_dir.mkdir(parents=True, exist_ok=True)
        n = conn.execute("SELECT COUNT(*) FROM crops WHERE page_name=?", (page_name,)).fetchone()[0]
        crop_path = crops_dir / f"{page_path.stem}_w{n:03d}.png"
        while crop_path.exists():
            n += 1
            crop_path = crops_dir / f"{page_path.stem}_w{n:03d}.png"
        crop.save(crop_path)

        conn.execute(
            "INSERT INTO crops(page_name,crop_path,label,x1,y1,x2,y2) VALUES (?,?,?,?,?,?,?)",
            (page_name, str(crop_path), label, ox1, oy1, ox2, oy2),
        )
        conn.commit()
        boxes = saved_boxes_val(conn, page_name, scale)
        table = page_table(conn, page_name)
        total = conn.execute("SELECT COUNT(*) FROM crops").fetchone()[0]
        msg = f"✅ قصاصة {crop_path.name} محفوظة (النص: {label}) — إجمالي القصاصات: {total}"
    finally:
        conn.close()

    return draw_overlay(disp, [], boxes), [], msg, table, gr.update(), ""


def on_manual_add(x1, y1, x2, y2, label, files, idx, disp, scale, db_path, crops_dir):
    """إضافة صندوق بإحداثيات يدوية (بكسل، على الصورة الأصلية) — بديل احتياطي للنقر."""
    label = (label or "").strip()
    if not label:
        return draw_overlay(disp, [], []), "❌ اكتب نص الكلمة أولًا.", None, ""
    if None in (x1, y1, x2, y2) or x1 == "" or y1 == "" or x2 == "" or y2 == "":
        return draw_overlay(disp, [], []), "❌ أدخل الإحداثيات الأربعة.", None, ""
    page_path = Path(files[idx]); page_name = page_path.name
    conn = tu.connect_db(db_path)
    try:
        ox1, ox2 = sorted((int(x1), int(x2))); oy1, oy2 = sorted((int(y1), int(y2)))
        orig = Image.open(page_path).convert("RGB")
        crop = tu.crop_word_image(orig, ox1, oy1, ox2, oy2, pad=6)
        crops_dir = Path(crops_dir); crops_dir.mkdir(parents=True, exist_ok=True)
        n = conn.execute("SELECT COUNT(*) FROM crops WHERE page_name=?", (page_name,)).fetchone()[0]
        crop_path = crops_dir / f"{page_path.stem}_w{n:03d}.png"
        while crop_path.exists():
            n += 1
            crop_path = crops_dir / f"{page_path.stem}_w{n:03d}.png"
        crop.save(crop_path)
        conn.execute(
            "INSERT INTO crops(page_name,crop_path,label,x1,y1,x2,y2) VALUES (?,?,?,?,?,?,?)",
            (page_name, str(crop_path), label, ox1, oy1, ox2, oy2),
        )
        conn.commit()
        boxes = saved_boxes_val(conn, page_name, scale)
        table = page_table(conn, page_name)
        msg = f"✅ قصاصة {crop_path.name} محفوظة يدويًا"
    finally:
        conn.close()
    return draw_overlay(disp, [], boxes), msg, table, ""


def on_undo(files, idx, disp, scale, db_path, crops_dir):
    page_name = Path(files[idx]).name
    conn = tu.connect_db(db_path)
    try:
        row = conn.execute(
            "SELECT id,crop_path FROM crops WHERE page_name=? ORDER BY id DESC LIMIT 1",
            (page_name,),
        ).fetchone()
        if not row:
            return draw_overlay(disp, [], saved_boxes_val(conn, page_name, scale)), \
                page_table(conn, page_name), "⚠️ لا قصاصات سابقة لهذه الصفحة."
        cid, cpath = row
        conn.execute("DELETE FROM crops WHERE id=?", (cid,))
        conn.commit()
        try:
            p = Path(cpath)
            if p.exists() and p.resolve().parent == Path(crops_dir).resolve():
                p.unlink()
        except OSError:
            pass
        msg = f"🗑️ حُذفت القصاصة الأخيرة ({Path(cpath).name})"
        boxes = saved_boxes_val(conn, page_name, scale)
        table = page_table(conn, page_name)
    finally:
        conn.close()
    return draw_overlay(disp, [], boxes), table, msg


def on_clear_points(disp, scale, db_path, files, idx):
    conn = tu.connect_db(db_path)
    boxes = saved_boxes_val(conn, Path(files[idx]).name, scale)
    conn.close()
    return draw_overlay(disp, [], boxes), [], "تم تجاهل التحديد الحالي."


def on_done(files, idx, db_path):
    """وسم الصفحة كمكتملة والانتقال إلى التالية غير المكتملة."""
    conn = tu.connect_db(db_path)
    try:
        conn.execute("UPDATE pages SET done=1 WHERE name=?", (Path(files[idx]).name,))
        conn.commit()
        nxt = next_undone(files, conn, after=idx)
        remaining = conn.execute("SELECT COUNT(*) FROM pages WHERE done=0").fetchone()[0]
    finally:
        conn.close()
    if nxt is None:
        return idx, "🎉 كل الصفحات موسومة! انتقل إلى الخطوة 4 (التصدير).", gr.update(choices=[Path(f).name for f in files], value=Path(files[idx]).name)
    return nxt, f"✅ اكتملت الصفحة — بقي {remaining} صفحة. الصفحة الحالية: {Path(files[nxt]).name}", \
        gr.update(choices=[Path(f).name for f in files], value=Path(files[nxt]).name)


def open_page(files, idx, db_path):
    """فتح صفحة بالفهرس: عرض + صناديق محفوظة + جدول."""
    conn = tu.connect_db(db_path)
    try:
        orig, disp, scale = load_page_display(files[idx])
        page_name = Path(files[idx]).name
        boxes = saved_boxes_val(conn, page_name, scale)
        table = page_table(conn, page_name)
    finally:
        conn.close()
    return draw_overlay(disp, [], boxes), disp, scale, [], table, f"📄 الصفحة: {page_name} — انقر زاويتي كل كلمة ثم اكتب نصها."


def refresh_pages(pages_dir, db_path, files, idx):
    """إعادة فحص مجلد الصفحات وتسجيل الجديد منها والانتقال لأول غير مكتملة."""
    files_new = tu.page_images(pages_dir)
    conn = tu.connect_db(db_path)
    try:
        for f in files_new:
            tu.add_page_if_missing(conn, Path(f).name)
        nxt = next_undone(files_new, conn)
        remaining = conn.execute("SELECT COUNT(*) FROM pages WHERE done=0").fetchone()[0]
    finally:
        conn.close()
    if not files_new:
        return [], 0, None, 1.0, None, [], [], "❌ لا توجد صور في مجلد الصفحات — شغّل الخطوة 2 أولًا.", gr.update(choices=[], value=None)
    nxt = nxt if nxt is not None else 0
    out = open_page(files_new, nxt, db_path)   # (img, disp, scale, points, table, status)
    return files_new, nxt, out[0], out[2], out[1], out[4], out[3], out[5], \
        gr.update(choices=[Path(f).name for f in files_new], value=Path(files_new[nxt]).name)


def on_dropdown_select(page_name, files, db_path):
    idx = next((i for i, f in enumerate(files) if Path(f).name == page_name), None)
    if idx is None:
        return None, None, 1.0, [], [], "❌ الصفحة غير موجودة."
    out = open_page(files, idx, db_path)   # (img, disp, scale, points, table, status)
    return idx, out[0], out[2], out[4], out[5], out[1]


# ────────────────────────────── بناء الواجهة ────────────────────────────────

def build_demo(pages_dir, db_path, crops_dir):
    files = tu.page_images(pages_dir)
    conn = tu.connect_db(db_path)
    try:
        for f in files:
            tu.add_page_if_missing(conn, Path(f).name)
        idx0 = next_undone(files, conn)
        remaining = conn.execute("SELECT COUNT(*) FROM pages WHERE done=0").fetchone()[0]
        total_crops = conn.execute("SELECT COUNT(*) FROM crops").fetchone()[0]
    finally:
        conn.close()
    idx0 = idx0 if idx0 is not None else (0 if files else None)

    if files and idx0 is not None:
        orig, disp0, scale0 = load_page_display(files[idx0])
        c = tu.connect_db(db_path)
        boxes0 = saved_boxes_val(c, Path(files[idx0]).name, scale0)
        table0 = page_table(c, Path(files[idx0]).name)
        c.close()
        first_img = draw_overlay(disp0, [], boxes0)
        first_status = f"📄 الصفحة: {Path(files[idx0]).name} — انقر زاويتي كل كلمة ثم اكتب نصها."
    else:
        disp0, scale0, first_img, table0, first_status = None, 1.0, None, [], \
            f"❌ لا توجد صور في {pages_dir} — شغّل الخطوة 2 أولًا (python 2_inspect_backups.py)."

    with gr.Blocks(title="وسم الكلمات بخط اليد — OCR") as demo:
        gr.Markdown(
            "## ✍️ قص ووسم الكلمات العربية بخط اليد\n"
            "انقر على **زاويتي الكلمة** (أعلى ثم أسفل) ← اكتب نصها ← «حفظ القصاصة».\n"
            "كل قصاصة تُحفظ في SQLite مع إحداثياتها الأصلية، وتُستخدم لاحقًا لتدريب TrOCR."
        )
        files_s  = gr.State(files)
        idx_s    = gr.State(idx0 if idx0 is not None else 0)
        points_s = gr.State([])
        scale_s  = gr.State(scale0)
        disp_s   = gr.State(disp0)

        with gr.Row():
            btn_prev = gr.Button("◀ السابقة", scale=1)
            dd_pages = gr.Dropdown(choices=[Path(f).name for f in files],
                                   value=Path(files[idx0]).name if files else None,
                                   label="الصفحات", interactive=True, scale=4)
            btn_next = gr.Button("التالية ▶", scale=1)
            btn_refresh = gr.Button("🔄 تحديث القائمة", scale=1)

        img = gr.Image(value=first_img, label="الصفحة — انقر زاويتي الكلمة", interactive=False, height=560)

        txt_label = gr.Textbox(label="نص الكلمة", placeholder="اكتب الكلمة كما هي مكتوبة بخط اليد", rtl=True)
        with gr.Row():
            btn_save   = gr.Button("💾 حفظ القصاصة", variant="primary")
            btn_clear  = gr.Button("↩️ تجاهل التحديد")
            btn_undo   = gr.Button("🗑️ تراجع عن آخر قصاصة")
            btn_done   = gr.Button("✅ إنهاء الصفحة والانتقال")

        status = gr.Textbox(value=first_status, label="الحالة", interactive=False)
        table  = gr.Dataframe(value=table0, headers=["#", "النص", "القصاصة"],
                              label="قصاصات الصفحة الحالية", interactive=False, wrap=True)

        with gr.Accordion("⚙️ إضافة صندوق بإحداثيات يدوية (بديل احتياطي)", open=False):
            gr.Markdown("أدخل إحداثيات الصورة **الأصلية** بالبكسل: x1,y1 أعلى، وx2,y2 أسفل.")
            with gr.Row():
                mx1 = gr.Number(label="x1", precision=0)
                my1 = gr.Number(label="y1", precision=0)
                mx2 = gr.Number(label="x2", precision=0)
                my2 = gr.Number(label="y2", precision=0)
            btn_manual = gr.Button("إضافة صندوق يدوي")

        # ── الأحداث ──
        img.select(on_click,
                   inputs=[files_s, idx_s, points_s, scale_s, disp_s, gr.State(db_path)],
                   outputs=[img, points_s, status])

        btn_save.click(on_save,
                       inputs=[txt_label, files_s, idx_s, points_s, scale_s, disp_s, gr.State(db_path), gr.State(crops_dir)],
                       outputs=[img, points_s, status, table, idx_s, txt_label])

        btn_manual.click(on_manual_add,
                         inputs=[mx1, my1, mx2, my2, txt_label, files_s, idx_s, disp_s, scale_s, gr.State(db_path), gr.State(crops_dir)],
                         outputs=[img, status, table, txt_label])

        btn_undo.click(on_undo,
                       inputs=[files_s, idx_s, disp_s, scale_s, gr.State(db_path), gr.State(crops_dir)],
                       outputs=[img, table, status])

        btn_clear.click(on_clear_points,
                        inputs=[disp_s, scale_s, gr.State(db_path), files_s, idx_s],
                        outputs=[img, points_s, status])

        done_evt = btn_done.click(on_done, inputs=[files_s, idx_s, gr.State(db_path)],
                                  outputs=[idx_s, status, dd_pages])
        done_evt.then(lambda files, idx, db: open_page(files, int(idx), db)[0:6],
                      inputs=[files_s, idx_s, gr.State(db_path)],
                      outputs=[img, disp_s, scale_s, points_s, table, status])

        def go(delta, files, idx, db):
            n = len(files)
            if n == 0:
                return 0, None, None, 1.0, [], [], "❌ لا صفحات."
            idx2 = (int(idx) + delta) % n
            img_, disp_, scale_, pts_, table_, status_ = open_page(files, idx2, db)
            return idx2, img_, disp_, scale_, pts_, table_, status_

        prev_evt = btn_prev.click(lambda *a: go(-1, *a),
                                  inputs=[files_s, idx_s, gr.State(db_path)],
                                  outputs=[idx_s, img, disp_s, scale_s, points_s, table, status])
        next_evt = btn_next.click(lambda *a: go(1, *a),
                                  inputs=[files_s, idx_s, gr.State(db_path)],
                                  outputs=[idx_s, img, disp_s, scale_s, points_s, table, status])
        for evt in (prev_evt, next_evt):
            evt.then(lambda files, idx: gr.update(value=Path(files[int(idx)]).name),
                     inputs=[files_s, idx_s], outputs=[dd_pages])

        dd_pages.select(on_dropdown_select,
                        inputs=[dd_pages, files_s, gr.State(db_path)],
                        outputs=[idx_s, img, scale_s, table, status, disp_s])

        btn_refresh.click(refresh_pages,
                          inputs=[gr.State(pages_dir), gr.State(db_path), files_s, idx_s],
                          outputs=[files_s, idx_s, img, scale_s, disp_s, table, points_s, status, dd_pages])

    return demo, remaining, total_crops


def main():
    ap = argparse.ArgumentParser(description="أداة قص ووسم الكلمات (Gradio + SQLite)")
    ap.add_argument("--pages", default=PAGES_DIR)
    ap.add_argument("--db", default=DB_PATH)
    ap.add_argument("--crops", default=CROPS_DIR)
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="إنشاء رابط عام (يُفعَّل تلقائيًا داخل Colab)")
    args = ap.parse_args()

    demo, remaining, total_crops = build_demo(args.pages, args.db, args.crops)
    in_colab = bool(os.environ.get("COLAB_RELEASE_TAG"))
    share = args.share or in_colab

    print(f"📊 قصاصات محفوظة حتى الآن: {total_crops} | صفحات غير مكتملة: {remaining}")
    if not tu.page_images(args.pages):
        print(f"⚠️ مجلد الصفحات {args.pages} فارغ — شغّل الخطوة 2 أولًا.")

    demo.queue(max_size=8).launch(
        server_name="0.0.0.0", server_port=args.port, share=share,
        show_error=True, inbrowser=False,
    )
    if in_colab:
        print("\n🛑 عند الانتهاء من الوسم: أوقف هذه الخلية (زر Stop) ثم انتقل للخطوة 4.")


if __name__ == "__main__":
    main()
