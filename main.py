"""NazarAI — Main (fixed pipeline)"""
import os, sys, cv2
os.environ["QT_QPA_PLATFORM"]  = "xcb"
os.environ["QT_LOGGING_RULES"] = "*.debug=false"

def _preflight():
    errors = []; warns = []
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"[INFO] GPU: {name} ({vram:.1f}GB)")
            if vram < 3.5:
                warns.append(f"GPU VRAM {vram:.1f}GB — Qwen int4 needs ~2.5GB, might be tight")
        else:
            warns.append("CUDA not available — running on CPU (very slow)")
    except ImportError:
        errors.append("torch not installed")

    for pkg, inst in [
        ("cv2",         "opencv-python"),
        ("ultralytics", "ultralytics"),
        ("torchvision", "torchvision"),
    ]:
        try: __import__(pkg)
        except ImportError: errors.append(f"{pkg} missing — pip install {inst}")

    # Qwen is optional at startup (lazy-loaded)
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration
    except ImportError:
        warns.append("transformers<4.51 — Qwen won't load. pip install 'transformers>=4.51'")
    try:
        import bitsandbytes
    except ImportError:
        warns.append("bitsandbytes missing — Qwen will use fp16 (needs 6GB VRAM). pip install bitsandbytes")

    for w in warns: print(f"[WARN] {w}")
    if errors:
        print("\n[FATAL] Fix these before running:")
        for e in errors: print(f"  ✗ {e}")
        sys.exit(1)

import config
from detector import ObjectDetector
from stage2   import Stage2Classifier
from stage3   import Stage3VLM
from alert    import AlertEngine
from hud      import draw_hud


def main(camera_index: int = 0):
    _preflight()
    print("\n" + "="*56)
    print("  NazarAI — 3-Stage Cascade Pipeline")
    print("="*56)
    print(f"  S1: MOG2 (CPU) + {config.YOLO_MODEL} (GPU)")
    print(f"  S2: MobileNetV3-Small | threshold={config.MOBILENET_THRESHOLD}")
    print(f"      temporal buffer={config.MOBILENET_TEMPORAL} frames")
    print(f"  S3: Qwen2.5-VL-3B int4 — lazy-loaded on first threat")
    print(f"      cooldown={config.QWEN_MIN_INTERVAL}s between calls")
    print("="*56)
    print("\n  [NOTE] Normal person sitting still will NOT trigger alerts.")
    print("  [NOTE] Weapon or aggressive motion needed for HIGH/CRITICAL.\n")

    detector = ObjectDetector()
    stage2   = Stage2Classifier()
    stage3   = Stage3VLM()
    alerter  = AlertEngine()

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[FATAL] Cannot open camera {camera_index}"); sys.exit(1)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(config.WINDOW_NAME, config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT)
    print(f"[INFO] Camera {camera_index} live | q=quit  l=log\n")

    fps=0.0; count=0; tick=cv2.getTickCount()
    s2_state = {"score":0.0,"label":"NORMAL","confirmed":False}
    s3_state = {"category":"SAFE","severity":0,"description":"standby",
                "suspicious":False,"busy":False,"calls":0,"loaded":False}

    while True:
        ret, frame = cap.read()
        if not ret: continue
        count += 1

        # S1: MOG2 → YOLO11n
        detections, motion, mask = detector.detect(frame)
        frame = detector.draw(frame, detections)

        weapon_dets = [d for d in detections if d["is_threat"]]

        # S2: MobileNetV3 on YOLO crops (only when motion)
        if motion and detections:
            crops = detector.extract_crops(frame, detections)
            if crops:
                # Use biggest crop
                biggest_crop, is_threat_cls = max(
                    crops, key=lambda x: x[0].shape[0]*x[0].shape[1]
                )
                s2_res   = stage2.infer_crop(
                    biggest_crop, "cam0", is_threat_class=is_threat_cls
                )
                s2_state = s2_res

                # S3: Qwen — only on 5-frame confirmed suspicious OR weapon
                if s2_res["confirmed"]:
                    stage3.push_crop(biggest_crop, detections, "cam0")

        elif not motion:
            stage2.reset_buffer("cam0")

        # Weapon → force Stage3 immediately (bypass temporal buffer)
        if weapon_dets and not stage3.busy:
            x1,y1,x2,y2 = weapon_dets[0]["bbox"]
            H,W2 = frame.shape[:2]
            wc = frame[max(0,y1-30):min(H,y2+30),
                       max(0,x1-30):min(W2,x2+30)]
            if wc.size > 0:
                stage3.push_crop(wc, detections, "cam0")

        s3_state = stage3.get_state()
        alerter.evaluate(frame, detections, s2_state, s3_state)

        if count % 30 == 0:
            e   = (cv2.getTickCount()-tick)/cv2.getTickFrequency()
            fps = 30/max(e,1e-6); tick=cv2.getTickCount()

        display = draw_hud(frame,fps,s2_state,s3_state,
                           alerter,detections,count,detector.mog2.pass_rate)
        cv2.imshow(config.WINDOW_NAME, display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"): break
        if key == ord("l"): os.system("xdg-open logs/events.jsonl &")
        if count>60 and cv2.getWindowProperty(config.WINDOW_NAME,cv2.WND_PROP_VISIBLE)<1:
            break

    cap.release(); cv2.destroyAllWindows()
    s3 = stage3.get_state()
    print(f"\n[INFO] {alerter.incident_count} incidents | {count} frames")
    print(f"[INFO] MOG2 pass rate : {detector.mog2.pass_rate*100:.1f}%")
    print(f"[INFO] YOLO ran on   : {detector._yolo_run} frames")
    print(f"[INFO] Qwen calls    : {s3.get('calls',0)}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
