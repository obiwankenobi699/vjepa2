"""HawkWatch — Main Entry Point"""
import os
os.environ["QT_QPA_PLATFORM"]  = "xcb"
os.environ["QT_LOGGING_RULES"] = "*.debug=false"

import sys
import cv2

import config
from detector   import ObjectDetector
from classifier import BehaviorClassifier
from alert      import AlertEngine
from hud        import draw_hud


def main(camera_index: int = 0):
    print("\n" + "="*52)
    print("  HawkWatch — Initializing")
    print("="*52)

    detector   = ObjectDetector()
    classifier = BehaviorClassifier()
    alerter    = AlertEngine()

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {camera_index}")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # minimize capture buffer lag

    cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(config.WINDOW_NAME, config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT)

    print(f"\n[INFO] Live on camera {camera_index}")
    print(f"[INFO] q = quit   l = open log\n")

    fps   = 0.0
    count = 0
    tick  = cv2.getTickCount()

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        count += 1

        # ── YOLO: every frame, real-time ─────
        detections = detector.detect(frame)
        frame      = detector.draw(frame, detections)

        # ── Push frame to async VJEPA buffer ─
        import cv2 as _cv2, numpy as _np
        resized = _cv2.resize(
            _cv2.cvtColor(frame, _cv2.COLOR_BGR2RGB),
            config.CLIP_SIZE
        )
        classifier.push_frame(resized)
        vjepa_state = classifier.get_state()

        # ── Alert engine ─────────────────────
        alerter.evaluate(frame, detections, vjepa_state)

        # ── FPS ──────────────────────────────
        if count % 30 == 0:
            elapsed = (cv2.getTickCount() - tick) / cv2.getTickFrequency()
            fps     = 30 / elapsed
            tick    = cv2.getTickCount()

        # ── Render ───────────────────────────
        display = draw_hud(frame, fps, vjepa_state, alerter, detections, count)
        cv2.imshow(config.WINDOW_NAME, display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("l"):
            os.system("xdg-open logs/events.jsonl &")
        if count > 60 and cv2.getWindowProperty(
            config.WINDOW_NAME, cv2.WND_PROP_VISIBLE
        ) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[INFO] Session ended")
    print(f"[INFO] {alerter.incident_count} incidents logged")
    print(f"[INFO] Frames : logs/frames/")
    print(f"[INFO] Events : logs/events.jsonl")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
