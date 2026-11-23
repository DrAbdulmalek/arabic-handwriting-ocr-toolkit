#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الخطوة 5 — ضبط دقيق لنموذج OCR بخط اليد على قصاصاتك (يُشغَّل على Colab T4).

المعمارية (المسار الافتراضي): بنية TrOCR — ترميز ViT + فك BERT متعدد اللغات
(mbert) مع طبقات cross-attention، كما في ورقة TrOCR الأصلية. نقاط microsoft/trocr-*
الرسمية غير صالحة للعربية (مفرداتها لاتينية فقط)، ونقاط AraBERT لا تحتوي ö/ü/ä/ß
الألمانية؛ لذا الفك الافتراضي bert-base-multilingual-cased: يغطي عربي + إنجليزي +
ألماني معًا (ملاحظاتك الثلاثية اللغات).

لتخصيص الفك:  --decoder aubmindlab/bert-base-arabertv2   (عربي بحت)
المسار اللاتيني (--latin): ضبط مباشر لـ microsoft/trocr-small-handwritten.

نصائح T4:
  - ابدأ:  python 5_train_trocr_colab.py --epochs 20
  - عند OOM: قلّل الدفعة --batch-size 4 أو الصورة --image-size 224.
  - الرمز إن رفعت للـ Hub يُقرأ من متغير البيئة HF_TOKEN (Colab Secrets) فقط.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import jiwer

import toolkit_utils as tu
from config import (
    DATASET_DIR, TRAIN_OUTPUT_DIR, ENCODER_ID, DECODER_ID, ARABIC_DECODER_ID,
    LATIN_FALLBACK_MODEL, IMAGE_SIZE, MAX_LABEL_TOKENS, SEED,
)


def build_model_and_processors(latin: bool, image_size: int, decoder_id: str):
    from transformers import (
        VisionEncoderDecoderModel, AutoTokenizer, ViTImageProcessor,
    )
    if latin:
        model = VisionEncoderDecoderModel.from_pretrained(LATIN_FALLBACK_MODEL)
        tokenizer = AutoTokenizer.from_pretrained(LATIN_FALLBACK_MODEL)
        processor = ViTImageProcessor.from_pretrained(LATIN_FALLBACK_MODEL)
        if hasattr(processor, "size") and isinstance(processor.size, dict):
            processor.size = {"height": image_size, "width": image_size}
        return model, tokenizer, processor

    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        ENCODER_ID, decoder_id,
    )
    tokenizer = AutoTokenizer.from_pretrained(decoder_id)
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.sep_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.max_length = MAX_LABEL_TOKENS
    model.config.num_beams = 4
    gc = model.generation_config
    gc.max_length = MAX_LABEL_TOKENS
    gc.num_beams = 4
    processor = ViTImageProcessor.from_pretrained(
        ENCODER_ID, size={"height": image_size, "width": image_size},
    )
    return model, tokenizer, processor


def make_collator(tokenizer, processor, max_label_tokens):
    """يحوّل الدفعة إلى pixel_values + labels مع إخفاء الحشو بـ -100."""
    pad_id = tokenizer.pad_token_id

    def collate(batch):
        images = [tu.pad_to_square(b["image"].convert("RGB")) for b in batch]
        pixel_values = processor(images=images, return_tensors="pt").pixel_values
        labels = tokenizer(
            [b["text"] for b in batch],
            padding="max_length",
            max_length=max_label_tokens,
            truncation=True,
            return_tensors="pt",
        ).input_ids
        labels[labels == pad_id] = -100
        return {"pixel_values": pixel_values, "labels": labels}

    return collate


def make_compute_metrics(tokenizer):
    pad_id = tokenizer.pad_token_id

    def compute_metrics(eval_pred):
        preds = eval_pred.predictions
        if isinstance(preds, tuple):
            preds = preds[0]
        labels = np.asarray(eval_pred.label_ids)
        labels[labels == -100] = pad_id
        pred_str = tokenizer.batch_decode(preds, skip_special_tokens=True)
        ref_str = tokenizer.batch_decode(labels, skip_special_tokens=True)
        pairs = [(r, h) for r, h in zip(ref_str, pred_str) if r.strip()]
        if not pairs:
            return {"cer": 1.0, "wer": 1.0}
        refs, hyps = zip(*pairs)
        return {"cer": float(jiwer.cer(list(refs), list(hyps))),
                "wer": float(jiwer.wer(list(refs), list(hyps)))}

    return compute_metrics


def main():
    ap = argparse.ArgumentParser(description="تدريب TrOCR على قصاصاتك")
    ap.add_argument("--dataset", default=DATASET_DIR)
    ap.add_argument("--out", default=TRAIN_OUTPUT_DIR)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=4e-5)
    ap.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    ap.add_argument("--max-label-tokens", type=int, default=MAX_LABEL_TOKENS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--latin", action="store_true",
                    help="ضبط microsoft/trocr-small-handwritten (إنجليزي بحت)")
    ap.add_argument("--decoder", default=DECODER_ID,
                    help=f"فك النموذج للإخراج (الافتراضي {DECODER_ID} يغطي ar/en/de؛ "
                         f"لعربي بحت: {ARABIC_DECODER_ID})")
    ap.add_argument("--push-repo", default="", help="مثال: DrAbdulmalek/trocr-multilingual-handwritten")
    args = ap.parse_args()

    from datasets import load_from_disk, Image as HFImage
    from transformers import (
        Seq2SeqTrainer, Seq2SeqTrainingArguments, EarlyStoppingCallback,
    )

    if not Path(args.dataset).exists():
        sys.exit(f"❌ مجموعة البيانات {args.dataset} غير موجودة — شغّل الخطوة 4 أولًا.")

    print(f"⏳ تحميل النموذج ({'مسار لاتيني' if args.latin else 'ViT + فك ' + args.decoder})…")
    model, tokenizer, processor = build_model_and_processors(args.latin, args.image_size, args.decoder)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    if device == "cpu":
        print("⚠️ لا توجد GPU — التدريب سيكون بطيئًا جدًا (استخدم Colab مع T4).")

    ds = load_from_disk(args.dataset)
    train_ds = ds["train"].select_columns(["image", "text"])
    eval_ds = ds["val"].select_columns(["image", "text"]) if "val" in ds and len(ds["val"]) else None
    if eval_ds is None:
        split = train_ds.train_test_split(test_size=0.1, seed=args.seed)
        train_ds, eval_ds = split["train"], split["test"]
        print("⚠️ لا يوجد تقسيم val — استُخدم 10% من train للتحقق.")
    test_ds = ds["test"].select_columns(["image", "text"]) if "test" in ds and len(ds["test"]) else eval_ds
    print(f"📊 train={len(train_ds)} | val={len(eval_ds)} | test={len(test_ds)}")

    collate = make_collator(tokenizer, processor, args.max_label_tokens)
    metrics = make_compute_metrics(tokenizer)

    targs = Seq2SeqTrainingArguments(
        output_dir=args.out + "_checkpoints",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.05,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        save_total_limit=2,
        logging_steps=25,
        fp16=torch.cuda.is_available(),
        seed=args.seed,
        predict_with_generate=True,
        remove_unused_columns=False,
        report_to="none",
        dataloader_pin_memory=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collate,
        processing_class=tokenizer,
        compute_metrics=metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=4)],
    )

    print("🚀 بدء التدريب…")
    trainer.train()
    trainer.save_model(args.out)
    processor.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)

    final = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="test")
    print(f"🏆 نتائج الاختبار النهائية — CER: {final.get('test_cer', -1):.4f} | WER: {final.get('test_wer', -1):.4f}")

    # عينات توضيحية من الاختبار
    model.eval()
    n_show = min(8, len(test_ds))
    for i in range(n_show):
        item = test_ds[i]
        pv = processor(images=tu.pad_to_square(item["image"].convert("RGB")),
                       return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            gen = model.generate(pv, max_length=args.max_label_tokens, num_beams=4)
        pred = tokenizer.batch_decode(gen, skip_special_tokens=True)[0].strip()
        print(f"  [{i}] المتوقع: {item['text']!r}  →  النموذج: {pred!r}")

    print(f"✅ النموذج محفوظ في {args.out}")

    # رفع اختياري إلى Hugging Face Hub — الرمز من البيئة فقط
    if args.push_repo:
        token = os.environ.get("HF_TOKEN")
        if not token:
            print("⚠️ لم يُرفع: HF_TOKEN غير موجود في البيئة (Colab Secrets). أنشئ رمز Write مؤقتًا ثم ألغه بعد الرفع.")
        else:
            from huggingface_hub import HfApi
            readme = f"""---
license: apache-2.0
tags: [ocr, handwriting, multilingual, arabic, german, english, trocr, vision-encoder-decoder]
language: [ar, en, de]
base_model: {ENCODER_ID if not args.latin else LATIN_FALLBACK_MODEL}
---
# TrOCR للخط اليدوي (عربي + إنجليزي + ألماني) — ضبط مخصص

نموذج OCR لكلمات الخط اليدوي الثلاثية اللغات (العربية والإنجليزية والألمانية)، مبني بمعمارية
TrOCR (ViT + فك {'bert-base-multilingual-cased' if not args.latin else 'RoBERTa'}) وضُبط على قصاصات مكتوبة بخط اليد.
**تنويه:** السلوك والقدرات مشروطة ببيانات التدريب الخاصة؛ اختبره قبل الاستخدام الفعلي.
"""
            try:
                with open(os.path.join(args.out, "README.md"), "w", encoding="utf-8") as f:
                    f.write(readme)
                api = HfApi(token=token)
                api.create_repo(args.push_repo, repo_type="model", exist_ok=True)
                api.upload_folder(folder_path=args.out, repo_id=args.push_repo, repo_type="model")
                print(f"✅ تم الرفع: https://huggingface.co/{args.push_repo}")
                print("🔴 ألغِ رمز الكتابة الآن من huggingface.co/settings/tokens")
            except Exception as e:
                print(f"⚠️ فشل الرفع: {e}")


if __name__ == "__main__":
    main()
