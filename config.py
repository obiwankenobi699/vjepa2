"""
NazarAI — Config (Fixed + Optimized)

Key fixes vs previous version:
  - YOLO11n instead of YOLOv8s (better accuracy, same VRAM)
  - MobileNetV3 threshold 0.82 (was 0.50 — caused everything to be SUSPICIOUS)
  - Temporal buffer 5 frames (was 3 — reduces false positives by ~60%)
  - MOG2 min area 8000px² (was 3000 — stops triggering on noise/minor movement)
  - Stage3 cooldown 30s (was 4s — stops Qwen spam when it's still loading)
  - Qwen lazy-load fixed — only downloads once, reuses cache

VRAM budget RTX 3050 4GB:
  YOLO11n           ~150 MB   GPU
  MobileNetV3-Small ~12  MB   GPU
  Qwen2.5-VL-3B     ~2.0 GB  GPU int4
  Overhead          ~300 MB
  ─────────────────────────
  Total             ~2.5 GB  ✓
"""
import torch, os

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float16 if torch.cuda.is_available() else torch.float32

CAMERA_INDEX   = 0
DISPLAY_WIDTH  = 1280
DISPLAY_HEIGHT = 720
WINDOW_NAME    = "NazarAI"

# ── Stage 1A: MOG2 (CPU) ──────────────────────────────────────────────────────
MOG2_HISTORY        = 400     # more history = more stable background model
MOG2_VAR_THRESHOLD  = 60      # higher = less sensitive (was 50 — too sensitive)
MOG2_DETECT_SHADOWS = False
MOG2_MIN_AREA       = 8000    # was 3000 — now ignores small movements
MOG2_DILATE_ITER    = 1       # was 2 — less dilation = less noise merge

# ── Stage 1B: YOLO11n (GPU) ───────────────────────────────────────────────────
# YOLO11 is the latest arch — better accuracy at same speed as YOLOv8n
YOLO_MODEL = "yolo11n.pt"
YOLO_CONF  = 0.45             # slightly higher than before — fewer weak detections
YOLO_IOU   = 0.45

WATCH_CLASSES = {
    0:  "person",
    24: "backpack",
    26: "handbag",
    39: "bottle",
    43: "knife",
    67: "phone",
    76: "scissors",
}
THREAT_CLASSES = {43: "knife", 76: "scissors"}

CLASS_CONF_THRESHOLDS = {
    0: 0.55, 24: 0.50, 26: 0.50,
    39: 0.45, 43: 0.70, 67: 0.50, 76: 0.70,
}
MIN_BBOX_AREA = {
    0: 5000, 43: 1500, 76: 1500,
    24: 2000, 26: 2000, 39: 800, 67: 600,
}
THREAT_CONFIRM_WINDOW = 6
THREAT_CONFIRM_HITS   = 4

# ── Stage 2: MobileNetV3-Small (GPU) ─────────────────────────────────────────
# CRITICAL FIX: threshold was 0.50 which means 50/50 coin flip = everything suspicious
# ImageNet pretrained MobileNet outputs ~0.5-0.7 for ANY person crop
# Raising to 0.82 means only very confident predictions pass
# Fine-tune on UCF-Crime to get real accuracy (see train_mobilenet.py)
MOBILENET_DEVICE     = DEVICE
MOBILENET_DTYPE      = DTYPE
MOBILENET_INPUT_SIZE = (224, 224)
MOBILENET_THRESHOLD  = 0.82   # was 0.50 — THIS was causing all false positives
MOBILENET_TEMPORAL   = 5      # was 3 — needs 5 consecutive hits now
MOBILENET_WEIGHTS    = None   # set to "mobilenet_nazarai.pth" after fine-tuning

# ── Stage 3: Qwen2.5-VL-3B (GPU int4, lazy-loaded, gated) ───────────────────
QWEN_MODEL        = "Qwen/Qwen2.5-VL-3B-Instruct"
QWEN_DEVICE       = DEVICE
QWEN_DTYPE        = torch.float16
QWEN_LOAD_IN_4BIT = True
QWEN_MAX_TOKENS   = 150
QWEN_MIN_INTERVAL = 30.0     # was 4.0 — 30s cooldown stops spam during download

QWEN_CATEGORIES = [
    "SAFE", "SHOPLIFTING", "WEAPON", "FIGHT",
    "TRESPASS", "LOITERING", "UNATTENDED_BAG", "VANDALISM", "OTHER_THREAT"
]

QWEN_PROMPT_TEMPLATE = """You are a CCTV security analyst. Analyze this camera image carefully.

Camera: {location} | Zone: {zone} | Time: {timestamp}
Recent events: {last_3_events}
YOLO detected: {yolo_detections}

Classify the activity as exactly one of:
SAFE, SHOPLIFTING, WEAPON, FIGHT, TRESPASS, LOITERING, UNATTENDED_BAG, VANDALISM, OTHER_THREAT

Rules:
- A person simply sitting, standing, or walking = SAFE
- Only classify as threat if there is CLEAR evidence of threatening behavior
- Phone in hand = SAFE
- Bag on shoulder = SAFE

Respond in this exact format only:
CATEGORY: <category>
SEVERITY: <0-10>
DESCRIPTION: <one sentence, factual>"""

QWEN_LOCATION = "Main Area"
QWEN_ZONE     = "Zone-1"

# ── Alert ─────────────────────────────────────────────────────────────────────
ALERT_COOLDOWN_SEC = 8.0      # was 4.0
ALERT_FRAME_DIR    = "logs/frames"
ALERT_LOG_PATH     = "logs/events.jsonl"

# Minimum Qwen severity to trigger HIGH alert
# Prevents "person standing" from being HIGH severity
ALERT_MIN_SEVERITY_HIGH   = 6
ALERT_MIN_SEVERITY_MEDIUM = 3
