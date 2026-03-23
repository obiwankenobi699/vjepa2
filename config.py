"""HawkWatch — Central Configuration (Dual Model)"""
import torch

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float16 if torch.cuda.is_available() else torch.float32

CAMERA_INDEX   = 0
DISPLAY_WIDTH  = 1280
DISPLAY_HEIGHT = 720

# ── YOLO ──────────────────────────────────────
YOLO_MODEL = "yolov8s.pt"
YOLO_CONF  = 0.40
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
    0:  0.50, 24: 0.50, 26: 0.50,
    39: 0.45, 43: 0.72, 67: 0.50, 76: 0.72,
}
MIN_BBOX_AREA = {
    0: 4000, 43: 1500, 76: 1500,
    24: 2000, 26: 2000, 39: 800, 67: 600,
}
THREAT_CONFIRM_WINDOW = 6
THREAT_CONFIRM_HITS   = 4

# ── V-JEPA 2 — motion/temporal analysis ───────
VJEPA_MODEL = "facebook/vjepa2-vitl-fpc16-256-ssv2"
NUM_FRAMES  = 16
FRAME_STEP  = 8
CLIP_SIZE   = (256, 256)

# SSv2 label substrings that indicate suspicious motion
VJEPA_SUSPICIOUS_KEYWORDS = [
    "throw", "hit", "push", "pull", "grab", "pick", "carry",
    "hide", "run", "fall", "kick", "punch", "attack", "stab",
    "strike", "point", "aim", "snatch", "drag", "struggle"
]
VJEPA_SUSPICIOUS_CONF = 0.32

# ── Moondream2 — scene/spatial understanding ──
VLM_MODEL        = "vikhyatk/moondream2"
VLM_REVISION     = "2025-01-09"
VLM_SAMPLE_EVERY = 45          # every ~1.5s at 30fps
VLM_MAX_TOKENS   = 80

VLM_SCENE_QUESTION = (
    "What is the person doing? "
    "Are there any weapons, threats, or suspicious actions? "
    "Answer in one short sentence."
)

VLM_THREAT_KEYWORDS = [
    "weapon", "knife", "gun", "threat", "attack", "aggress",
    "fight", "punch", "hit", "stab", "choke", "grab", "steal",
    "suspicious", "danger", "violent", "threatening", "assault",
    "intrud", "break", "smash", "harm", "hostil"
]

# ── Alert Engine ──────────────────────────────
ALERT_COOLDOWN_SEC = 5.0
ALERT_FRAME_DIR    = "logs/frames"
ALERT_LOG_PATH     = "logs/events.jsonl"

# Alert tiers:
#   CRITICAL  — YOLO confirmed weapon
#   HIGH      — VJEPA suspicious + Moondream confirms
#   MEDIUM    — either VJEPA or Moondream alone (not both)
ALERT_ON_SINGLE_SIGNAL = True   # set False to only alert on dual confirmation

# ── Display ───────────────────────────────────
WINDOW_NAME = "HawkWatch"
