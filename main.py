"""HawkWatch — Main Entry Point (Dual Model)"""
import os

import cv2

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
    print("  HawkWatch — Dual Model Init")
    print("="*52)

    detector   = ObjectDetector()
    classifier = BehaviorClassifier()
    alerter    = AlertEngine()

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {camera_index}")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(config.WINDOW_NAME, config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT)

    print(f"\n[INFO] Camera {camera_index} live")
    print(f"[INFO] V-JEPA 2  : motion analysis  (async)")
    print(f"[INFO] Moondream2: scene description (every {config.VLM_SAMPLE_EVERY} frames)")
    print(f"[INFO] q = quit   l = open log\n")

    fps   = 0.0
    count = 0
    tick  = cv2.getTickCount()

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        count += 1

        # YOLO — every frame
        detections = detector.detect(frame)
        frame      = detector.draw(frame, detections)

        # Both models — single push_frame call handles routing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        classifier.push_frame(frame_rgb)
        state = classifier.get_state()

        # Alert engine
        alerter.evaluate(frame, detections, state)

        # FPS
        if count % 30 == 0:
            elapsed = (cv2.getTickCount() - tick) / cv2.getTickFrequency()
            fps     = 30 / elapsed
            tick    = cv2.getTickCount()

        display = draw_hud(frame, fps, state, alerter, detections, count)
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
    print(f"\n[INFO] Session ended — {alerter.incident_count} incidents logged")
    print(f"[INFO] Frames : logs/frames/  |  Events : logs/events.jsonl")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
