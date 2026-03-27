"""HawkWatch — FastAPI Bridge for React Native"""

import os
os.environ["QT_QPA_PLATFORM"]  = "xcb"
os.environ["QT_LOGGING_RULES"] = "*.debug=false"

import cv2
import numpy as np
import base64
import time
import requests
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

import config
from detector   import ObjectDetector
from classifier import BehaviorClassifier
from alert      import AlertEngine

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Models loaded once at startup ─────────────────────────────────
detector   = ObjectDetector()
classifier = BehaviorClassifier()
alerter    = AlertEngine()

# ── Frame save directory ───────────────────────────────────────────
FRAME_LOG_DIR = Path("logs/appframe")
FRAME_LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── Moondream vLLM endpoint ────────────────────────────────────────
# Adjust port if your vllm serve uses a different one
MOONDREAM_URL = "http://localhost:8000/v1/chat/completions"
MOONDREAM_MODEL = "vikhyatk/moondream2"  # match whatever you passed to vllm serve

frame_counter = 0  # global counter for saved frames


# ── Moondream helper ───────────────────────────────────────────────
def ask_moondream(frame_bgr: np.ndarray) -> str:
    """
    Send a frame to local Moondream vLLM and get a plain-English scene description.
    Returns empty string on failure so the rest of the pipeline still works.
    """
    try:
        # Encode frame to JPEG base64
        _, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 60])
        b64 = base64.b64encode(buf).decode("utf-8")

        payload = {
            "model": MOONDREAM_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                        },
                        {
                            "type": "text",
                            "text": (
                                "You are a security camera AI. "
                                "Describe what is happening in this scene in 1-2 sentences. "
                                "Focus on: people, actions, objects, anything suspicious. "
                                "Be concise and factual."
                            )
                        }
                    ]
                }
            ],
            "max_tokens": 120,
            "temperature": 0.2,
        }

        resp = requests.post(MOONDREAM_URL, json=payload, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()

    except requests.exceptions.ConnectionError:
        print("⚠️  Moondream vLLM not reachable (is it running?)")
        return ""
    except Exception as e:
        print(f"⚠️  Moondream error: {e}")
        return ""


# ── Frame saver ────────────────────────────────────────────────────
def save_frame(frame_bgr: np.ndarray) -> str:
    """Save frame to logs/appframe and return the filename."""
    global frame_counter
    frame_counter += 1
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"frame_{ts}_{frame_counter:05d}.jpg"
    path = FRAME_LOG_DIR / filename
    cv2.imwrite(str(path), frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return filename


# ── /health ────────────────────────────────────────────────────────
@app.get("/health")
def health():
    # Quick check if Moondream is reachable
    moondream_status = "unknown"
    try:
        r = requests.get("http://localhost:8000/health", timeout=2)
        moondream_status = "online" if r.status_code == 200 else "error"
    except Exception:
        moondream_status = "offline"

    return {
        "status": "ok",
        "model": "HawkWatch Dual",
        "python": "3.11",
        "moondream": moondream_status,
        "frames_saved": frame_counter,
    }


# ── /frame  (React Native real-time mode) ─────────────────────────
@app.post("/frame")
async def analyze_frame(payload: dict):
    try:
        t_start = time.time()

        # 1. Decode image
        img_data = base64.b64decode(payload["frame"])
        np_arr   = np.frombuffer(img_data, np.uint8)
        frame    = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return {"error": "Could not decode frame"}

        # 2. Save frame to disk
        saved_filename = save_frame(frame)

        # 3. YOLO detection
        detections = detector.detect(frame)

        # 4. VJEPA behavior classification
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        classifier.push_frame(frame_rgb)
        state = classifier.get_state()

        # 5. Alert engine
        alerter.evaluate(frame, detections, state)

        # 6. Moondream scene description (only every 3rd frame to save GPU)
        scene_description = ""
        run_moondream = (frame_counter % 3 == 0)
        if run_moondream:
            scene_description = ask_moondream(frame)

        # 7. Build human-readable summary
        detection_labels = [d.label for d in detections if hasattr(d, "label")]
        label_counts: dict = {}
        for lbl in detection_labels:
            label_counts[lbl] = label_counts.get(lbl, 0) + 1

        summary_parts = []
        if label_counts:
            objects = ", ".join(f"{v}x {k}" for k, v in label_counts.items())
            summary_parts.append(f"Detected: {objects}")
        if state:
            tier = state.get("tier", "")
            behavior = state.get("behavior", "")
            if tier:
                summary_parts.append(f"Threat: {tier}")
            if behavior:
                summary_parts.append(f"Behavior: {behavior}")
        if alerter.incident_count:
            summary_parts.append(f"Incidents: {alerter.incident_count}")

        summary = " | ".join(summary_parts) if summary_parts else "Scene clear"

        t_end = time.time()
        latency_ms = round((t_end - t_start) * 1000)

        # 8. Console log so you can see it in terminal
        print(
            f"[FRAME #{frame_counter}] {saved_filename} | "
            f"{len(detections)} detections | "
            f"tier={state.get('tier', '?') if state else '?'} | "
            f"{latency_ms}ms"
        )
        if scene_description:
            print(f"[MOONDREAM] {scene_description}")

        return {
            # Core ML output
            "detections":        _serialize(detections),
            "state":             state,
            "incidents":         alerter.incident_count,

            # Human-readable
            "summary":           summary,
            "scene":             scene_description,   # Moondream description

            # Meta
            "saved_frame":       saved_filename,
            "latency_ms":        latency_ms,
            "frame_number":      frame_counter,

            # Alert
            "alert":             alerter.incident_count > 0,
            "message":           f"⚠️ {summary}" if alerter.incident_count > 0 else "",
        }

    except Exception as e:
        print(f"[ERROR] /frame: {e}")
        return {"error": str(e), "fallback": True}


# ── /analyze  (React Native video upload mode) ────────────────────
@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        tmp_path = "/tmp/hawkwatch_upload.mp4"

        with open(tmp_path, "wb") as f:
            f.write(contents)

        cap     = cv2.VideoCapture(tmp_path)
        results = []
        count   = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            count += 1
            if count % 5 != 0:
                continue

            detections = detector.detect(frame)
            frame_rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            classifier.push_frame(frame_rgb)
            state = classifier.get_state()
            alerter.evaluate(frame, detections, state)

            results.append({
                "frame":      count,
                "detections": _serialize(detections),
                "state":      state,
            })

        cap.release()

        return {
            "frames_analyzed": len(results),
            "incidents":       alerter.incident_count,
            "results":         results,
        }
    except Exception as e:
        return {"error": str(e)}


# ── /status  (new — React Native can poll this for live insights) ──
@app.get("/status")
def get_status():
    """
    Lightweight GET endpoint the React Native app can poll
    to show live status without sending a frame.
    """
    state = classifier.get_state() if classifier else {}
    return {
        "incidents":     alerter.incident_count,
        "state":         state,
        "frames_saved":  frame_counter,
        "tier":          state.get("tier", "idle") if state else "idle",
        "behavior":      state.get("behavior", "") if state else "",
        "alert":         alerter.incident_count > 0,
    }


# ── helper ─────────────────────────────────────────────────────────
def _serialize(detections) -> list:
    out = []
    for d in detections:
        try:
            out.append({
                "label":      d.label       if hasattr(d, "label")      else str(d),
                "confidence": float(d.conf) if hasattr(d, "conf")       else 0.0,
                "bbox":       list(d.bbox)  if hasattr(d, "bbox")       else [],
            })
        except Exception:
            pass
    return out
