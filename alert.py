"""NazarAI — Alert Engine (fixed severity thresholds)"""
import os, cv2, json, time, datetime
import numpy as np
import config


class AlertEngine:
    def __init__(self):
        os.makedirs(config.ALERT_FRAME_DIR, exist_ok=True)
        self._last         = 0.0
        self.incident_count = 0
        self.active        = False
        self.active_tier   = "CLEAR"
        self.active_reason = ""

    def evaluate(self, frame: np.ndarray, detections: list,
                 s2_state: dict, s3_state: dict):
        now     = time.time()
        threats = [d for d in detections if d["is_threat"]]
        persons = [d for d in detections if d["class_id"] == 0]

        s2_confirmed = s2_state.get("confirmed", False)
        s2_score     = s2_state.get("score", 0.0)
        s3_susp      = s3_state.get("suspicious", False)
        s3_cat       = s3_state.get("category", "SAFE")
        s3_sev       = int(s3_state.get("severity", 0))

        # Tier logic — much stricter than before
        if threats:
            tier = "CRITICAL"
        elif s3_susp and s3_sev >= config.ALERT_MIN_SEVERITY_HIGH:
            tier = "HIGH"
        elif (s3_susp and s3_sev >= config.ALERT_MIN_SEVERITY_MEDIUM) or s2_confirmed:
            tier = "MEDIUM"
        else:
            # CLEAR — reset and return
            self.active      = False
            self.active_tier = "CLEAR"
            return None

        # Don't alert for MEDIUM unless person actually present
        if tier == "MEDIUM" and not persons and not threats:
            self.active      = False
            self.active_tier = "CLEAR"
            return None

        if (now - self._last) < config.ALERT_COOLDOWN_SEC:
            self.active      = True
            self.active_tier = tier
            return None

        self._last = now
        self.incident_count += 1

        parts = []
        if threats:
            parts.append("WEAPON: " + ", ".join(d["label"] for d in threats))
        if s3_susp:
            parts.append(f"Qwen: [{s3_cat}] sev={s3_sev} — {s3_state.get('description','')[:50]}")
        elif s2_confirmed:
            parts.append(f"MobileNet confirmed (score={s2_score:.2f})")

        reason = " | ".join(parts) or tier
        self.active        = True
        self.active_tier   = tier
        self.active_reason = reason

        ts    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        fname = f"{ts}_{tier.lower()}_#{self.incident_count:04d}.jpg"
        fpath = os.path.join(config.ALERT_FRAME_DIR, fname)
        cv2.imwrite(fpath, frame, [cv2.IMWRITE_JPEG_QUALITY, 92])

        event = {
            "timestamp":   datetime.datetime.now().isoformat(),
            "incident_id": self.incident_count,
            "tier":        tier,
            "reason":      reason,
            "frame_path":  fpath,
            "yolo":  [{"label":d["label"],"conf":round(d["confidence"],3),
                       "bbox":d["bbox"],"threat":d["is_threat"]} for d in detections],
            "stage2": {"score":round(s2_score,3),"confirmed":s2_confirmed},
            "stage3": {"category":s3_cat,"severity":s3_sev,
                       "description":s3_state.get("description",""),
                       "calls":s3_state.get("calls",0)},
        }
        with open(config.ALERT_LOG_PATH, "a") as f:
            f.write(json.dumps(event) + "\n")

        print(f"\n{'█'*56}")
        print(f"  [ALERT] [{tier}] #{self.incident_count:04d}")
        print(f"  {reason}")
        print(f"  Saved: {fpath}")
        print(f"{'█'*56}\n")
        return event
