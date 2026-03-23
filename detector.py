"""
HawkWatch — Real-Time YOLO Object Detector
YOLOv8s with per-class confidence, bbox area, and temporal confirmation.
"""
import cv2
import numpy as np
from collections import deque, defaultdict
from ultralytics import YOLO
import config

CLASS_COLORS = {
    0:  (0,   200,  80),   # person   green
    24: (0,   165, 255),   # backpack orange
    26: (0,   165, 255),   # handbag  orange
    39: (0,   210, 210),   # bottle   cyan
    43: (0,    30, 255),   # knife    red
    67: (180,   0, 255),   # phone    purple
    76: (0,    30, 255),   # scissors red
}


class TemporalConfirmer:
    """
    Tracks per-class detection history over a rolling window.
    A class is 'confirmed' only when it appears in THREAT_CONFIRM_HITS
    out of the last THREAT_CONFIRM_WINDOW frames.
    This eliminates single-frame false positives (fan flicker, glare, etc.)
    """
    def __init__(self):
        self._history: dict[int, deque] = defaultdict(
            lambda: deque(maxlen=config.THREAT_CONFIRM_WINDOW)
        )

    def update(self, detected_class_ids: list[int]):
        """Call once per frame with all detected class IDs."""
        # Push 1 for detected, 0 for not detected — for each threat class
        for cid in config.THREAT_CLASSES:
            self._history[cid].append(1 if cid in detected_class_ids else 0)

    def is_confirmed(self, class_id: int) -> bool:
        """Returns True only if class has enough consecutive hits."""
        history = self._history[class_id]
        if len(history) < config.THREAT_CONFIRM_WINDOW:
            return False   # not enough frames seen yet
        return sum(history) >= config.THREAT_CONFIRM_HITS


def _passes_filters(d: dict) -> bool:
    """
    Apply per-class confidence and bbox area filters.
    Returns False if the detection should be rejected.
    """
    cid  = d["class_id"]
    conf = d["confidence"]
    x1, y1, x2, y2 = d["bbox"]

    # Per-class confidence threshold
    min_conf = config.CLASS_CONF_THRESHOLDS.get(cid, config.YOLO_CONF)
    if conf < min_conf:
        return False

    # Minimum bbox area
    area = (x2 - x1) * (y2 - y1)
    min_area = config.MIN_BBOX_AREA.get(cid, 0)
    if area < min_area:
        return False

    # Scissors-specific: aspect ratio guard
    # A fan blade assembly is nearly square (circular projection)
    # Real scissors are elongated along one axis
    if cid == 76:  # scissors
        w = x2 - x1
        h = y2 - y1
        if w == 0 or h == 0:
            return False
        ratio = max(w, h) / min(w, h)
        if ratio < 1.3:
            # Too square — likely a circular object (fan, plate, wheel)
            return False

    return True


class ObjectDetector:
    def __init__(self):
        print(f"[YOLO] Loading {config.YOLO_MODEL} (small — better accuracy)...")
        self.model     = YOLO(config.YOLO_MODEL)
        self.model.to(config.DEVICE)
        self.confirmer = TemporalConfirmer()

        # Warm-up
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        self.model(dummy, verbose=False)
        print(f"[YOLO] Warmed up on {config.DEVICE}")

    def detect(self, frame: np.ndarray) -> list:
        """
        Returns filtered, temporally-confirmed detections.
        Threat classes only appear in output once confirmed across frames.
        """
        results = self.model(
            frame,
            conf=config.YOLO_CONF,   # permissive global conf — filtered below
            iou=config.YOLO_IOU,
            classes=list(config.WATCH_CLASSES.keys()),
            verbose=False,
        )

        raw = []
        for box in results[0].boxes:
            cid  = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            raw.append({
                "class_id":   cid,
                "label":      config.WATCH_CLASSES.get(cid, str(cid)),
                "confidence": conf,
                "bbox":       (x1, y1, x2, y2),
                "is_threat":  cid in config.THREAT_CLASSES,
            })

        # Apply per-class filters
        filtered = [d for d in raw if _passes_filters(d)]

        # Update temporal confirmer with all detected threat IDs
        threat_ids_this_frame = [
            d["class_id"] for d in filtered if d["is_threat"]
        ]
        self.confirmer.update(threat_ids_this_frame)

        # For threats: only pass through if temporally confirmed
        # Non-threats (person, bag, etc.) pass immediately
        final = []
        for d in filtered:
            if d["is_threat"]:
                if self.confirmer.is_confirmed(d["class_id"]):
                    final.append(d)
                # else: detected but not yet confirmed — silently accumulate
            else:
                final.append(d)

        return final

    def draw(self, frame: np.ndarray, detections: list) -> np.ndarray:
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            color = CLASS_COLORS.get(d["class_id"], (200, 200, 200))
            thick = 3 if d["is_threat"] else 2

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)

            label_txt = f"{d['label']} {d['confidence']:.2f}"
            (tw, th), _ = cv2.getTextSize(
                label_txt, cv2.FONT_HERSHEY_DUPLEX, 0.52, 1
            )
            cv2.rectangle(frame, (x1, y1-th-8), (x1+tw+6, y1), color, -1)
            cv2.putText(frame, label_txt, (x1+3, y1-4),
                        cv2.FONT_HERSHEY_DUPLEX, 0.52, (0, 0, 0), 1)

            if d["is_threat"]:
                L = 20
                for px, py in [(x1,y1),(x2,y1),(x1,y2),(x2,y2)]:
                    sx = 1 if px == x1 else -1
                    sy = 1 if py == y1 else -1
                    cv2.line(frame, (px,py), (px+sx*L,py), (0,0,255), 3)
                    cv2.line(frame, (px,py), (px,py+sy*L), (0,0,255), 3)

        return frame
