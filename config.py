# -*- coding: utf-8 -*-
"""
الإعدادات الافتراضية لحزمة OCR الخط اليدوي العربي.
عدّل القيم هنا مباشرة، أو تجاوزها من سطر الأوامر في كل سكربت (--help).
"""

# ── مصدر البيانات: مستودع النسخ الاحتياطية على Hugging Face Buckets ──
BUCKET_ID = "DrAbdulmalek/github-repo-backups"
BUCKET_PREFIX = "2026-10-14"            # المجلد الفرعي المطلوب داخل الـ bucket

# ── مجلدات العمل (تُنشأ تلقائيًا نسبةً إلى مجلد التشغيل) ──
DATA_DIR      = "ocr_data"
BUCKET_DIR    = "ocr_data/bucket_downloads"   # ما يُنزَّل من الـ bucket
EXTRACTED_DIR = "ocr_data/extracted"          # النسخ المستخرجة من الأرشيفات
PAGES_DIR     = "ocr_data/pages"              # صور الصفحات الجاهزة للقص والوسم
CROPS_DIR     = "ocr_data/crops"              # قصاصات الكلمات الموسومة
DB_PATH       = "ocr_data/crops.db"           # قاعدة بيانات SQLite للقصاصات
DATASET_DIR   = "ocr_data/hf_dataset"         # مجموعة التدريب النهائية (HF Dataset)
AUDIT_DIR     = "ocr_data/dataset_audit"      # نسخة تدقيق (CSV + صور منسوخة)

# ── النموذج ──
# ملاحظة مهمة: نقاط TrOCR الرسمية (microsoft/trocr-*) مبنية على مفردات لاتينية
# ولا تستطيع توليد حروف عربية أصلًا. لذلك نبني بنية TrOCR نفسها (ترميز ViT +
# فك BERT مع cross-attention) كما في ورقة TrOCR الأصلية.
# ملاحظاتك تحتوي ثلاث لغات (عربي/إنجليزي/ألماني)، لذا الفك الافتراضي mBERT
# (bert-base-multilingual-cased): مفرداته تغطي العربية كاملة + ö/ü/ä/ß الألمانية
# + اللاتينية. مفردات AraBERT لا تحتوي علامات الترميم الألمانية، وتسقط إلى [UNK].
ENCODER_ID = "google/vit-base-patch16-224-in21k"    # ViT مهيأ مسبقًا للصور
DECODER_ID = "bert-base-multilingual-cased"         # فك ثلاثي اللغات (ar/en/de)
ARABIC_DECODER_ID = "aubmindlab/bert-base-arabertv2"  # بديل عربي بحت (--decoder)
LATIN_FALLBACK_MODEL = "microsoft/trocr-small-handwritten"  # يُستخدم مع --latin فقط

IMAGE_SIZE       = 384    # أبعاد إدخال الترميز (قلّلها إلى 224 عند نفاد ذاكرة T4)
MAX_LABEL_TOKENS = 32     # أقصى طول لمفردات الكلمة الواحدة بعد التقطيع

# ── تقسيم البيانات (على مستوى الصفحة لتجنب تسريب النمط بين القصاصات) ──
SPLIT_RATIOS = (0.9, 0.05, 0.05)   # train / val / test
SEED = 42

# ── مخرجات التدريب ──
TRAIN_OUTPUT_DIR = "trocr-multilingual-handwritten"   # عربي + إنجليزي + ألماني
