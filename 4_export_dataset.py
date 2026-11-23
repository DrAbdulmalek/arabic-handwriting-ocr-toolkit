#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الخطوة 4 — تحويل قصاصات SQLite إلى مجموعة بيانات Hugging Face جاهزة للتدريب.

ماذا يفعل:
  - يقرأ القصاصات من قاعدة البيانات ويحذف الوسوم الفارغة/الملفات المفقودة (مع تقرير).
  - يوزع الصفحات على train/val/test على مستوى الصفحة (كل قصاصات الصفحة معًا)
    لتجنب تسريب نمط الخط والحبر بين التقسيمات.
  - يحفظ مجموعة DatasetDict في ocr_data/hf_dataset (أعمدة: image, text, page).
  - يكتب نسخة تدقيق: CSV لكل تقسيم + نسخ من القصاصات.
"""
import argparse
import shutil
import sqlite3
import sys
from pathlib import Path

import pandas as pd

import toolkit_utils as tu
from config import DB_PATH, DATASET_DIR, AUDIT_DIR, SPLIT_RATIOS, SEED


def main():
    ap = argparse.ArgumentParser(description="تصدير القصاصات إلى HF Dataset")
    ap.add_argument("--db", default=DB_PATH)
    ap.add_argument("--out", default=DATASET_DIR, help="مجلد حفظ مجموعة البيانات")
    ap.add_argument("--audit", default=AUDIT_DIR, help="مجلد نسخة التدقيق")
    ap.add_argument("--ratios", type=float, nargs=3, default=SPLIT_RATIOS,
                    metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--strict", action="store_true", help="التوقف عند أي ملف مفقود بدل حذفه")
    ap.add_argument("--no-copies", action="store_true",
                    help="توفير المساحة: نسخ CSV فقط دون نسخ صور التدقيق")
    ap.add_argument("--no-clean", action="store_true",
                    help="تعطيل تنظيف الوسوم الآلية (إزالة الشوائب والشظايا غير المقروءة)")
    ap.add_argument("--min-confidence", type=float, default=0.0,
                    help="استيراد الوسوم بثقة OCR أعلى من هذه القيمة فقط (0 = الكل)")
    args = ap.parse_args()

    if not Path(args.db).exists():
        sys.exit(f"❌ قاعدة البيانات {args.db} غير موجودة — شغّل الخطوة 3 ووسم صفحات أولًا.")

    conn = sqlite3.connect(args.db)
    try:
        df = pd.read_sql_query(
            "SELECT id, page_name, crop_path, label, split, ignored, verified FROM crops ORDER BY id",
            conn)
    except Exception:
        df = pd.read_sql_query(
            "SELECT id, page_name, crop_path, label, split, 0 AS ignored, 0 AS verified FROM crops ORDER BY id",
            conn)
    conn.close()

    if df.empty:
        sys.exit("❌ لا قصاصات في قاعدة البيانات — وسم صفحات أولًا (الخطوة 3).")

    n_ignored = int((df["ignored"] == 1).sum())
    if n_ignored:
        df = df[df["ignored"] != 1].copy()
        print(f"ℹ️ استُبعدت {n_ignored} قصاصة معلّمة كمتجاهلة من المراجعة.")

    if "verified" not in df:
        df["verified"] = 0

    # ── فلترة بالثقة (اختياري) ──
    try:
        conn = sqlite3.connect(args.db)
        conf = pd.read_sql_query("SELECT id, confidence, source, language FROM crops", conn)
        conn.close()
        df = df.merge(conf, on="id", how="left")
        if "language" not in df:
            df["language"] = ""
    except Exception:
        df["confidence"] = None
        df["source"] = ""
        df["language"] = ""

    # ── تنظيف الوسوم الآلية ──
    if not args.no_clean:
        import re as _re
        import unicodedata

        def _clean(s):
            s = unicodedata.normalize("NFC", s)                             # أوماوت مركّبة: u + ̈ ← ü
            s = _re.sub("[\u200e\u200f\u202a-\u202e\u2066-\u2069]", "", s)  # علامات الاتجاه المخفية
            s = s.replace("\u0640", "")                                      # التطويل
            return _re.sub(r"\s+", " ", s).strip()

        def _readable(s):
            """وسم قابل للتدريب: كلمتان عربيتان، أو لاتيني/ألماني (ö/ü/ä/ß) بثلاثة
            أحرف، أو رقمان — الشظايا والعلامات تُحذف."""
            if not (2 <= len(s) <= 30):
                return False
            ar = len(_re.findall(r"[\u0621-\u064A\u066E-\u06D3]", s))
            la = len(_re.findall(r"[A-Za-zÄÖÜäöüß]", s))
            dg = len(_re.findall(r"[0-9\u0660-\u0669]", s))
            return ar >= 2 or la >= 3 or dg >= 2

        df["label"] = df["label"].astype(str).map(_clean)
        df = df[df["label"] != ""].copy()
        mask_ok = df["label"].map(_readable)
        mask_reviewed = df["verified"] == 1   # ما راجعه البشر يمرّ حتى لو قصر الوسم
        before = len(df)
        df = df[mask_ok | mask_reviewed].copy()
        print(f"🧹 تنظيف الوسوم: حُذف {before - len(df)} وسمًا غير مقروء — باقي {len(df)} "
              f"(الوسوم المُراجَعة بشريًا تمرّ دومًا ولو قصرت: ja, im, zu…)")

    # ── تنظيف ──
    df["label"] = df["label"].astype(str).str.strip()
    n_empty = int((df["label"] == "").sum())
    if n_empty:
        df = df[df["label"] != ""]
        print(f"⚠️ حُذفت {n_empty} قصاصة بوسم فارغ.")

    # ── حلّ المسارات القديمة المطلقة: إن لم توجد، جرّب مجلد القصاصات بجوار القاعدة ──
    def _resolve(p):
        q = Path(p)
        if q.exists():
            return q
        alt = Path(args.db).parent / "crops" / q.name
        return alt if alt.exists() else q

    df["crop_path"] = df["crop_path"].astype(str).map(_resolve)

    missing = [p for p in df["crop_path"] if not Path(p).exists()]
    if missing:
        print(f"⚠️ {len(missing)} ملف قصاصة مفقود. أمثلة: {missing[:3]}")
        if args.strict:
            sys.exit("❌ --strict مفعّل: أعد توليد القصاصات المفقودة أو شغّل دون --strict للحذف.")
        df = df[df["crop_path"].map(Path.exists)]
    if df.empty:
        sys.exit("❌ بعد التنظيف لا توجد قصاصات صالحة.")

    n_dup = int(df.duplicated(subset=["crop_path"]).sum())
    if n_dup:
        df = df.drop_duplicates(subset=["crop_path"], keep="first")
        print(f"⚠️ حُذفت {n_dup} قصاصة مكررة (نفس المسار).")

    # ── التقسيم على مستوى الصفحة ──
    preset = df[df["split"].isin(["train", "val", "test"])]
    to_assign = df[~df["split"].isin(["train", "val", "test"])]
    page_map = tu.assign_splits(to_assign["page_name"].unique(), tuple(args.ratios), args.seed)
    df.loc[to_assign.index, "split"] = to_assign["page_name"].map(page_map)
    print(f"ℹ️ تقسيم تلقائي على مستوى الصفحة (seed={args.seed}): "
          f"{df['split'].value_counts().to_dict()}")

    if preset.empty and df["page_name"].nunique() < 3:
        print("⚠️ عدد الصفحات أقل من 3: val/test قد تكون فارغة، وسيأخذ التدريب "
              "10% من train للتحقق (تسريب جزئي — وسم صفحات أكثر يفصلها).")
        tr = df[df["split"] == "train"]
        n_val = max(1, int(0.1 * len(tr)))
        df.loc[tr.sample(n=n_val, random_state=args.seed).index, "split"] = "val"

    # ── نسخة التدقيق ──
    audit = Path(args.audit)
    for split in ("train", "val", "test"):
        sub = df[df["split"] == split]
        if sub.empty:
            continue
        (audit / split).mkdir(parents=True, exist_ok=True)
        if not args.no_copies:
            for _, r in sub.iterrows():
                shutil.copy2(r["crop_path"], audit / split / Path(r["crop_path"]).name)
        sub[["crop_path", "label", "page_name", "language"]].to_csv(
            audit / f"labels_{split}.csv", index=False, encoding="utf-8-sig")

    # ── بناء مجموعة Hugging Face ──
    try:
        from datasets import Dataset, DatasetDict, Features, Image as HFImage, Value
    except ImportError:
        sys.exit("❌ مكتبة datasets غير مثبتة:  pip install datasets")

    feats = Features({
        "image": HFImage(),
        "text": Value("string"),
        "page": Value("string"),
        "language": Value("string"),
    })
    splits_ds = {}
    for split in ("train", "val", "test"):
        sub = df[df["split"] == split]
        if sub.empty:
            continue
        if "language" not in sub.columns:
            sub = sub.assign(language="unknown")
        splits_ds[split] = Dataset.from_dict({
            "image": sub["crop_path"].tolist(),
            "text": sub["label"].tolist(),
            "page": sub["page_name"].tolist(),
            "language": sub["language"].fillna("unknown").tolist(),
        }, features=feats)

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    DatasetDict(splits_ds).save_to_disk(str(out))

    # ── إحصاءات ──
    lens = df["label"].str.len()
    n_verified = int((df["verified"] == 1).sum()) if "verified" in df else 0
    lang_counts = df["language"].fillna("unknown").value_counts().to_dict() if "language" in df else {}
    lang_examples = {}
    if "language" in df:
        for lang, grp in df.fillna({"language": "unknown"}).groupby("language"):
            lang_examples[lang] = grp["label"].head(3).tolist()
    print(f"""
———————— ملخص مجموعة البيانات ————————
train: {len(splits_ds.get('train', []))}  |  val: {len(splits_ds.get('val', []))}  |  test: {len(splits_ds.get('test', []))}
صفحات: {df['page_name'].nunique()}  |  كلمات فريدة: {df['label'].nunique()}
طول الوسم: متوسط {lens.mean():.1f} حرفًا، أقصى {lens.max()}
وسوم مُراجَعة بشريًا: {n_verified} من {len(df)} (مراجعة أكتر = دقة أعلى)
حسب اللغة: {lang_counts}
أمثلة لكل لغة: {lang_examples}
مجموعة المحارف: {tu.charset_of(df['label'])}
مجموعة البيانات: {out}   |   نسخة التدقيق: {audit}
➡️  الخطوة التالية:  python 5_train_trocr_colab.py --epochs 20
""")
    if len(splits_ds.get("train", [])) < 200:
        print("⚠️ أقل من 200 قصاصة تدريب — كافية لبداية تجربة، لكن توقع نتائج متواضعة؛ "
              "وسم 500–2000 كلمة يرفع الدقة بوضوح.")


if __name__ == "__main__":
    main()
