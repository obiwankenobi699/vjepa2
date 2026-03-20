import os
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["QT_LOGGING_RULES"] = "*.debug=false"

import sys
import cv2
import torch
import numpy as np
import threading
from collections import deque
from transformers import AutoVideoProcessor, AutoModelForVideoClassification

MODEL_ID      = "facebook/vjepa2-vitl-fpc16-256-ssv2"
NUM_FRAMES    = 16
FRAME_STEP    = 8
INPUT_SIZE    = (256, 256)
THREAT_THRESH = 0.4
DEVICE        = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE         = torch.float16 if torch.cuda.is_available() else torch.float32
WINDOW        = "V-JEPA 2 — Real-Time Detection"

def load_model():
    print(f"[INFO] Model  : {MODEL_ID}")
    print(f"[INFO] Device : {DEVICE} | Dtype: {DTYPE}")
    processor = AutoVideoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForVideoClassification.from_pretrained(
        MODEL_ID,
        dtype=DTYPE,
        device_map=DEVICE,
        attn_implementation="sdpa",
    )
    model.eval()
    print(f"[INFO] Loaded : {model.config.num_labels} classes")
    return processor, model

class InferenceWorker:
    def __init__(self, processor, model):
        self.processor = processor
        self.model     = model
        self.label     = "warming up..."
        self.prob      = 0.0
        self.busy      = False
        self._lock     = threading.Lock()

    def submit(self, frames):
        if self.busy:
            return
        self.busy = True
        threading.Thread(target=self._run, args=(frames,), daemon=True).start()

    def _run(self, frames):
        try:
            clip      = np.stack(frames)               # T x H x W x C
            idx       = np.arange(0, len(clip), FRAME_STEP)
            sampled   = clip[idx]

            inputs = self.processor(list(sampled), return_tensors="pt")
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            with torch.no_grad():
                probs = self.model(**inputs).logits.softmax(dim=-1)[0]

            top     = probs.argmax().item()
            label   = self.model.config.id2label.get(top, str(top))
            prob    = probs[top].item()

            with self._lock:
                self.label = label
                self.prob  = prob

            del inputs
            if DEVICE != "cpu":
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"[WARN] Inference error: {e}")
        finally:
            self.busy = False

    def result(self):
        with self._lock:
            return self.label, self.prob

def draw_overlay(frame, label, prob, fps, busy, is_threat):
    color  = (0, 0, 255) if is_threat else (0, 210, 0)
    status = "inferring..." if busy else f"{label[:38]} ({prob:.2f})"
    prefix = "THREAT" if is_threat else "CLASS "
    cv2.rectangle(frame, (0, 0), (580, 70), (10, 10, 10), -1)
    cv2.putText(frame, f"{prefix}: {status}",
                (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.60, color, 2)
    cv2.putText(frame, f"FPS: {fps:.1f}   device: {DEVICE}",
                (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160, 160, 160), 1)
    return frame

def main(camera_index=0):
    processor, model = load_model()
    worker = InferenceWorker(processor, model)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {camera_index}")
        sys.exit(1)

    # Create window explicitly before loop — prevents premature WND_PROP_VISIBLE=0
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 960, 540)

    print("[INFO] Running. Press 'q' or close window to quit.")

    buffer      = deque(maxlen=NUM_FRAMES)
    fps         = 0.0
    count       = 0
    tick        = cv2.getTickCount()

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        resized = cv2.resize(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), INPUT_SIZE
        )
        buffer.append(resized)

        if len(buffer) == NUM_FRAMES:
            worker.submit(list(buffer))
            buffer.clear()

        label, prob = worker.result()

        count += 1
        if count % 20 == 0:
            elapsed = (cv2.getTickCount() - tick) / cv2.getTickFrequency()
            fps     = 20 / elapsed
            tick    = cv2.getTickCount()

        display = draw_overlay(
            frame.copy(), label, prob, fps, worker.busy, prob > THREAT_THRESH
        )
        cv2.imshow(WINDOW, display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        # Only check window close after 30 frames to allow render time
        if count > 30 and cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Session ended.")

if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
