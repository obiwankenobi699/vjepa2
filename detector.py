"""
NazarAI — Stage 1: MOG2 + YOLO11n

Changes:
  - Uses YOLO11n (ultralytics auto-downloads if not present)
  - MOG2 min area 8000px² (stricter — ignores minor movement)
  - Returns is_threat_class flag per detection for Stage2
"""
import cv2
import numpy as np
from collections import deque, defaultdict
from ultralytics import YOLO
import config

CLASS_COLORS = {
    0:  (0,   200,  80),
    24: (0,   165, 255),
    26: (0,   165, 255),
    39: (0,   210, 210),
    43: (0,    30, 255),
    67: (180,   0, 255),
    76: (0,    30, 255),
}


class MOG2Filter:
    def __init__(self):
        self.mog2 = cv2.createBackgroundSubtractorMOG2(
            history       = config.MOG2_HISTORY,
            varThreshold  = config.MOG2_VAR_THRESHOLD,
            detectShadows = config.MOG2_DETECT_SHADOWS,
        )
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self._passed  = 0
        self._dropped = 0

    def has_motion(self, frame: np.ndarray) -> tuple:
        mask = self.mog2.apply(frame)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.dilate(mask, self.kernel, iterations=config.MOG2_DILATE_ITER)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = max((cv2.contourArea(c) for c in contours), default=0.0)
        motion = max_area > config.MOG2_MIN_AREA
        if motion: self._passed += 1
        else:      self._dropped += 1
        return motion, mask, float(max_area)

    @property
    def pass_rate(self) -> float:
        total = self._passed + self._dropped
        return self._passed / total if total else 0.0


class TemporalConfirmer:
    def __init__(self):
        self._h: dict = defaultdict(
            lambda: deque(maxlen=config.THREAT_CONFIRM_WINDOW)
        )

    def update(self, ids: list):
        for cid in config.THREAT_CLASSES:
            self._h[cid].append(1 if cid in ids else 0)

    def confirmed(self, cid: int) -> bool:
        h = self._h[cid]
        return (len(h) == config.THREAT_CONFIRM_WINDOW
                and sum(h) >= config.THREAT_CONFIRM_HITS)


def _passes_filters(d: dict) -> bool:
    cid = d["class_id"]
    x1, y1, x2, y2 = d["bbox"]
    if d["confidence"] < config.CLASS_CONF_THRESHOLDS.get(cid, config.YOLO_CONF):
        return False
    if (x2 - x1) * (y2 - y1) < config.MIN_BBOX_AREA.get(cid, 0):
        return False
    if cid == 76:
        w, h = x2-x1, y2-y1
        if not w or not h or max(w,h)/min(w,h) < 1.3:
            return False
    return True


class ObjectDetector:
    def __init__(self):
        self.mog2 = MOG2Filter()
        print(f"[YOLO] Loading {config.YOLO_MODEL} (YOLO11 — latest arch)...")
        self.model = YOLO(config.YOLO_MODEL)
        self.model.to(config.DEVICE)
        self.confirmer = TemporalConfirmer()
        self.model(np.zeros((480,640,3), dtype=np.uint8), verbose=False)
        print(f"[YOLO] YOLO11n ready on {config.DEVICE}")
        self._yolo_run = 0
        self._yolo_skip = 0

    def detect(self, frame: np.ndarray) -> tuple:
        has_motion, mask, area = self.mog2.has_motion(frame)
        if not has_motion:
            self._yolo_skip += 1
            return [], False, mask

        self._yolo_run += 1
        res = self.model(
            frame,
            conf=config.YOLO_CONF, iou=config.YOLO_IOU,
            classes=list(config.WATCH_CLASSES.keys()),
            verbose=False,
        )
        raw = []
        for box in res[0].boxes:
            cid  = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
            raw.append({
                "class_id":        cid,
                "label":           config.WATCH_CLASSES.get(cid, str(cid)),
                "confidence":      conf,
                "bbox":            (x1,y1,x2,y2),
                "is_threat":       cid in config.THREAT_CLASSES,
                "is_threat_class": cid in config.THREAT_CLASSES,
            })

        filtered   = [d for d in raw if _passes_filters(d)]
        threat_ids = [d["class_id"] for d in filtered if d["is_threat"]]
        self.confirmer.update(threat_ids)

        final = []
        for d in filtered:
            if d["is_threat"]:
                if self.confirmer.confirmed(d["class_id"]):
                    final.append(d)
            else:
                final.append(d)
        return final, True, mask

    def extract_crops(self, frame: np.ndarray, detections: list) -> list:
        H, W = frame.shape[:2]
        crops = []
        for d in detections:
            x1,y1,x2,y2 = d["bbox"]
            pad = 20
            x1,y1 = max(0,x1-pad),max(0,y1-pad)
            x2,y2 = min(W,x2+pad),min(H,y2+pad)
            if x2>x1 and y2>y1:
                crops.append((frame[y1:y2,x1:x2].copy(), d.get("is_threat_class",False)))
        return crops

    def draw(self, frame: np.ndarray, detections: list) -> np.ndarray:
        for d in detections:
            x1,y1,x2,y2 = d["bbox"]
            c = CLASS_COLORS.get(d["class_id"],(200,200,200))
            cv2.rectangle(frame,(x1,y1),(x2,y2),c,3 if d["is_threat"] else 2)
            txt = f"{d['label']} {d['confidence']:.2f}"
            (tw,th),_ = cv2.getTextSize(txt,cv2.FONT_HERSHEY_DUPLEX,0.52,1)
            cv2.rectangle(frame,(x1,y1-th-8),(x1+tw+6,y1),c,-1)
            cv2.putText(frame,txt,(x1+3,y1-4),cv2.FONT_HERSHEY_DUPLEX,0.52,(0,0,0),1)
            if d["is_threat"]:
                L=20
                for px,py in [(x1,y1),(x2,y1),(x1,y2),(x2,y2)]:
                    sx=1 if px==x1 else -1; sy=1 if py==y1 else -1
                    cv2.line(frame,(px,py),(px+sx*L,py),(0,0,255),3)
                    cv2.line(frame,(px,py),(px,py+sy*L),(0,0,255),3)
        return frame
