"""HawkWatch — Async V-JEPA 2 Behavior Classifier (background thread)"""
import threading
import numpy as np
import torch
from collections import deque
from transformers import AutoVideoProcessor, AutoModelForVideoClassification
import config

def _is_suspicious(label: str) -> bool:
    l = label.lower()
    return any(kw in l for kw in config.SUSPICIOUS_KEYWORDS)

class BehaviorClassifier:
    def __init__(self):
        print(f"[VJEPA] Loading {config.VJEPA_MODEL} ...")
        self.processor = AutoVideoProcessor.from_pretrained(config.VJEPA_MODEL)
        self.model = AutoModelForVideoClassification.from_pretrained(
            config.VJEPA_MODEL,
            dtype=config.DTYPE,
            device_map=config.DEVICE,
            attn_implementation="sdpa",
        )
        self.model.eval()
        print(f"[VJEPA] Ready — {self.model.config.num_labels} action classes")
        self.label      = "observing..."
        self.confidence = 0.0
        self.suspicious = False
        self.busy       = False
        self._lock      = threading.Lock()
        self._buffer    = deque(maxlen=config.NUM_FRAMES)
        self._buf_lock  = threading.Lock()

    def push_frame(self, frame_rgb: np.ndarray):
        with self._buf_lock:
            self._buffer.append(frame_rgb)
        if len(self._buffer) == config.NUM_FRAMES and not self.busy:
            with self._buf_lock:
                frames = list(self._buffer)
                self._buffer.clear()
            self.busy = True
            threading.Thread(target=self._run, args=(frames,), daemon=True).start()

    def _run(self, frames: list):
        try:
            clip    = np.stack(frames)
            idx     = np.arange(0, len(clip), config.FRAME_STEP)
            sampled = clip[idx]
            inputs  = self.processor(list(sampled), return_tensors="pt")
            inputs  = {k: v.to(config.DEVICE) for k, v in inputs.items()}
            with torch.no_grad():
                probs = self.model(**inputs).logits.softmax(dim=-1)[0]
            top_idx  = probs.argmax().item()
            top_conf = probs[top_idx].item()
            top_lbl  = self.model.config.id2label.get(top_idx, str(top_idx))
            with self._lock:
                self.label      = top_lbl
                self.confidence = top_conf
                self.suspicious = (
                    _is_suspicious(top_lbl) and
                    top_conf >= config.VJEPA_SUSPICIOUS_CONF
                )
            del inputs
            if config.DEVICE != "cpu":
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"[VJEPA] Error: {e}")
        finally:
            self.busy = False

    def get_state(self) -> dict:
        with self._lock:
            return {
                "label":      self.label,
                "confidence": self.confidence,
                "suspicious": self.suspicious,
                "busy":       self.busy,
            }
