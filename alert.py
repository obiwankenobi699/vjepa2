"""HawkWatch — Alert Engine. Saves ONLY flagged frames."""
import os, cv2, json, time, datetime
import numpy as np
import config

class AlertEngine:
    def __init__(self):
        os.makedirs(config.ALERT_FRAME_DIR, exist_ok=True)
        self._last_alert   = 0.0
        self.incident_count = 0
        self.active        = False
        self.active_reason = ""

    def evaluate(self, frame: np.ndarray, detections: list, vjepa_state: dict):
        now            = time.time()
        threat_objects = [d for d in detections if d["is_threat"]]
        persons        = [d for d in detections if d["class_id"] == 0]
        action_threat  = (
            bool(persons) and
            vjepa_state["suspicious"] and
            config.PERSON_ALERT_WITH_ACTION
        )
        should_alert = bool(threat_objects) or action_threat
        if not should_alert:
            self.active = False
            return None
        if (now - self._last_alert) < config.ALERT_COOLDOWN_SEC:
            self.active = True
            return None

        self._last_alert = now
        self.incident_count += 1
        reasons = []
        if threat_objects:
            reasons.append("weapon: " + ", ".join(d["label"] for d in threat_objects))
        if action_threat:
            reasons.append(f"action: {vjepa_state['label']} ({vjepa_state['confidence']:.2f})")
        reason = " | ".join(reasons)
        self.active        = True
        self.active_reason = reason

        ts    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        fname = f"{ts}_incident_{self.incident_count:04d}.jpg"
        fpath = os.path.join(config.ALERT_FRAME_DIR, fname)
        cv2.imwrite(fpath, frame, [cv2.IMWRITE_JPEG_QUALITY, 92])

        event = {
            "timestamp":       datetime.datetime.now().isoformat(),
            "incident_id":     self.incident_count,
            "reason":          reason,
            "frame_path":      fpath,
            "yolo_detections": [
                {"label": d["label"], "confidence": round(d["confidence"],3),
                 "bbox": d["bbox"], "is_threat": d["is_threat"]}
                for d in detections
            ],
            "vjepa": {
                "label":      vjepa_state["label"],
                "confidence": round(vjepa_state["confidence"],3),
                "suspicious": vjepa_state["suspicious"],
            },
        }
        with open(config.ALERT_LOG_PATH, "a") as f:
            f.write(json.dumps(event) + "\n")

        print(f"[ALERT] Incident #{self.incident_count:04d} — {reason}")
        print(f"        Saved: {fpath}")
        return event
