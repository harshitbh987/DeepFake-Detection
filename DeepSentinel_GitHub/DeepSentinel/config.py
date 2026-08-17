import os

# ── Base directory (auto-detected — works on Windows, Mac, Linux) ─────────
ROOT_DIR         = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR   = os.path.join(ROOT_DIR, "checkpoints")
TEMP_UPLOAD_DIR  = os.path.join(ROOT_DIR, "temp_uploads")
LOG_DIR          = os.path.join(ROOT_DIR, "logs")
SAMPLE_OUT_DIR   = os.path.join(ROOT_DIR, "sample_outputs")
DATASET_ROOT     = os.path.join(ROOT_DIR, "dataset")

PRIMARY_WEIGHTS  = os.path.join(CHECKPOINT_DIR, "sentinel_b0_best.pt")

# ── Face detection ────────────────────────────────────────────────────────
FACE_CROP_SIZE       = 224
IMAGE_MEAN           = [0.485, 0.456, 0.406]
IMAGE_STD            = [0.229, 0.224, 0.225]
FRAMES_PER_VIDEO     = 12
MIN_VALID_FRAMES     = 3
FACE_MARGIN_RATIO    = 0.30
MIN_FACE_SIZE        = 40
FACE_CONF_THRESH     = 0.92

# ── Model ─────────────────────────────────────────────────────────────────
BACKBONE             = "efficientnet_b0"
NUM_CLASSES          = 1
DROPOUT_PROB         = 0.35

# ── Training ──────────────────────────────────────────────────────────────
BATCH_SIZE           = 16
NUM_WORKERS          = 0          # 0 = safe on Windows
WARMUP_LR            = 3e-4
FINETUNE_LR          = 5e-6
WARMUP_EPOCHS        = 5
FINETUNE_EPOCHS      = 8
WEIGHT_DECAY         = 1e-4
LABEL_SMOOTHING      = 0.05
GRAD_CLIP            = 1.0

# ── Dataset ───────────────────────────────────────────────────────────────
TOTAL_VIDEOS             = 8000
REAL_RATIO               = 0.50
VAL_SPLIT                = 0.15
FRAMES_PER_VIDEO_TRAIN   = 12

# ── Inference ─────────────────────────────────────────────────────────────
FAKE_DECISION_BOUNDARY   = 0.50
CONFIDENCE_HIGH          = 0.80
CONFIDENCE_MEDIUM        = 0.60

# ── GradCAM ───────────────────────────────────────────────────────────────
GRADCAM_LAYER    = "features.8"
GRADCAM_ALPHA    = 0.45

# ── Upload ────────────────────────────────────────────────────────────────
MAX_UPLOAD_MB        = 200
ALLOWED_VIDEO_EXT    = ["mp4", "avi", "mov", "mkv"]

# ── Create directories automatically ─────────────────────────────────────
for _d in [CHECKPOINT_DIR, TEMP_UPLOAD_DIR, LOG_DIR, SAMPLE_OUT_DIR, DATASET_ROOT]:
    os.makedirs(_d, exist_ok=True)
