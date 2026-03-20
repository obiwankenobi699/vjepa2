"""HawkWatch — Central Configuration"""
import torch

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float16 if torch.cuda.is_available() else torch.float32

CAMERA_INDEX   = 0
DISPLAY_WIDTH  = 1280
DISPLAY_HEIGHT = 720

# YOLO
YOLO_MODEL = "yolov8n.pt"
YOLO_CONF  = 0.45
YOLO_IOU   = 0.45

# COCO class IDs to monitor
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

# V-JEPA 2
VJEPA_MODEL = "facebook/vjepa2-vitl-fpc16-256-ssv2"
NUM_FRAMES  = 16
FRAME_STEP  = 8
CLIP_SIZE   = (256, 256)

SUSPICIOUS_KEYWORDS = [
    "throw", "hit", "push", "pull", "grab", "pick", "carry",
    "hide", "run", "fall", "kick", "punch", "attack", "stab",
    "strike", "point", "aim", "snatch", "drag", "struggle"
]

# Alert engine
ALERT_COOLDOWN_SEC        = 4.0
ALERT_FRAME_DIR           = "logs/frames"
ALERT_LOG_PATH            = "logs/events.jsonl"
VJEPA_SUSPICIOUS_CONF     = 0.30
PERSON_ALERT_WITH_ACTION  = True

WINDOW_NAME = "HawkWatch"
