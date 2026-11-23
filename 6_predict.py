#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الخطوة 6 — تجربة النموذج المدرَّب على صور صفحات أو قصاصات.

أمثلة:
  python 6_predict.py --model trocr-arabic-handwritten --image ocr_data/crops
  python 6_predict.py --model trocr-arabic-handwritten --image ocr_data/pages/page_001.jpg
"""
import argparse
from pathlib import Path

import torch
from PIL import Image

import toolkit_utils as tu
from config import MAX_LABEL_TOKENS


def load_model(model_dir, device):
    from transformers import AutoTokenizer, ViTImageProcessor, VisionEncoderDecoderModel
    model = VisionEncoderDecoderModel.from_pretrained(model_dir).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    processor = ViTImageProcessor.from_pretrained(model_dir)
    return model, tokenizer, processor


def predict_image(model, tokenizer, processor, img, device, num_beams=4):
    pv = processor(images=tu.pad_to_square(img.convert("RGB")),
                   return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        gen = model.generate(pv, max_length=MAX_LABEL_TOKENS, num_beams=num_beams)
    return tokenizer.batch_decode(gen, skip_special_tokens=True)[0].strip()


def main():
    ap = argparse.ArgumentParser(description="تنبؤ OCR على صور")
    ap.add_argument("--model", default="trocr-arabic-handwritten", help="مجلد النموذج المدرَّب")
    ap.add_argument("--image", required=True, help="صورة واحدة أو مجلد صور")
    ap.add_argument("--beams", type=int, default=4)
    ap.add_argument("--max", type=int, default=20, help="أقصى عدد صور في المجلد")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, processor = load_model(args.model, device)
    print(f"✅ النموذج جاهز على {device} ({args.model})")

    src = Path(args.image)
    if src.is_dir():
        paths = [p for p in sorted(src.iterdir()) if p.suffix.lower() in tu.IMAGE_EXTS][: args.max]
    elif src.is_file():
        paths = [src]
    else:
        raise SystemExit(f"❌ المسار غير موجود: {src}")

    if not paths:
        raise SystemExit("❌ لا توجد صور للتنبؤ.")

    for p in paths:
        text = predict_image(model, tokenizer, processor, Image.open(p), device, args.beams)
        print(f"🖼️ {p.name}  →  {text!r}")


if __name__ == "__main__":
    main()
