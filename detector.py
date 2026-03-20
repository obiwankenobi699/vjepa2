"""HawkWatch — Real-Time YOLO Object Detector (every frame, <10ms)"""
import cv2
import numpy as np
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

class ObjectDetector:
    def __init__(self):
        print(f"[YOLO] Loading {config.YOLO_MODEL} ...")
        self.model = YOLO(config.YOLO_MODEL)
        self.model.to(config.DEVICE)
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        self.model(dummy, verbose=False)
        print(f"[YOLO] Warmed up on {config.DEVICE}")

    def detect(self, frame: np.ndarray) -> list:
        results = self.model(
            frame,
            conf=config.YOLO_CONF,
            iou=config.YOLO_IOU,
            classes=list(config.WATCH_CLASSES.keys()),
            verbose=False,
        )
        out = []
        for box in results[0].boxes:
            cid  = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            out.append({
                "class_id":   cid,
                "label":      config.WATCH_CLASSES.get(cid, str(cid)),
                "confidence": conf,
                "bbox":       (x1, y1, x2, y2),
                "is_threat":  cid in config.THREAT_CLASSES,
            })
        return out

    def draw(self, frame: np.ndarray, detections: list) -> np.ndarray:
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            color = CLASS_COLORS.get(d["class_id"], (200, 200, 200))
            thick = 3 if d["is_threat"] else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
            label_txt = f"{d['label']} {d['confidence']:.2f}"
            (tw, th), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_DUPLEX, 0.52, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv2.putText(frame, label_txt, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_DUPLEX, 0.52, (0, 0, 0), 1)
            if d["is_threat"]:
                L = 20
                for px, py in [(x1,y1),(x2,y1),(x1,y2),(x2,y2)]:
                    sx = 1 if px == x1 else -1
                    sy = 1 if py == y1 else -1
                    cv2.line(frame,(px,py),(px+sx*L,py),(0,0,255),3)
                    cv2.line(frame,(px,py),(px,py+sy*L),(0,0,255),3)
        return frame
