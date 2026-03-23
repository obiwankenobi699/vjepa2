"""
HawkWatch — Dual Model Behavior Analyser
  Thread 1: V-JEPA 2  — motion patterns (16 frames)
  Thread 2: BLIP-2    — natural language scene description
  
  BLIP-2 replaces Moondream2 — no custom code, no pyvips, pure transformers
  VRAM: VJEPA ~850MB + BLIP-2 ~900MB + YOLO ~200MB = ~1.95GB total
"""
import threading
import numpy as np
import torch
from collections import deque
from PIL import Image
from transformers import (
    AutoVideoProcessor,
    AutoModelForVideoClassification,
    BlipProcessor,
    BlipForConditionalGeneration,
)
import config


def _vjepa_suspicious(label: str) -> bool:
    return any(kw in label.lower() for kw in config.VJEPA_SUSPICIOUS_KEYWORDS)

def _vlm_suspicious(text: str) -> bool:
    return any(kw in text.lower() for kw in config.VLM_THREAT_KEYWORDS)


# ══════════════════════════════════════════════
#  V-JEPA 2 — motion/temporal worker
# ══════════════════════════════════════════════
class VJEPAWorker:
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

    def push_frame(self, frame_rgb_small: np.ndarray):
        with self._buf_lock:
            self._buffer.append(frame_rgb_small)
        if len(self._buffer) == config.NUM_FRAMES and not self.busy:
            with self._buf_lock:
                frames = list(self._buffer)
                self._buffer.clear()
            self.busy = True
            threading.Thread(target=self._run, args=(frames,), daemon=True).start()

    def _run(self, frames):
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
                    _vjepa_suspicious(top_lbl) and
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


# ══════════════════════════════════════════════
#  BLIP-2 — scene/spatial understanding worker
# ══════════════════════════════════════════════
class VLMWorker:
    def __init__(self):
        model_id = "Salesforce/blip-image-captioning-large"
        print(f"[VLM] Loading BLIP captioning (~900MB)...")
        self.processor = BlipProcessor.from_pretrained(model_id)
        self.model = BlipForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=config.DTYPE,
        ).to(config.DEVICE)
        self.model.eval()
        print(f"[VLM] BLIP ready on {config.DEVICE}")

        self.description  = "scene loading..."
        self.suspicious   = False
        self.busy         = False
        self._lock        = threading.Lock()
        self._frame_count = 0

    def push_frame(self, frame_rgb_full: np.ndarray):
        self._frame_count += 1
        if self._frame_count % config.VLM_SAMPLE_EVERY != 0:
            return
        if self.busy:
            return
        self.busy = True
        threading.Thread(
            target=self._run,
            args=(frame_rgb_full.copy(),),
            daemon=True
        ).start()

    def _run(self, frame_rgb):
        try:
            image  = Image.fromarray(frame_rgb)
            # Conditional captioning — steers BLIP toward security context
            prompt = "a security camera captures"
            inputs = self.processor(image, prompt, return_tensors="pt").to(
                config.DEVICE, config.DTYPE
            )
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    num_beams=3,
                )
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            # Strip the prompt prefix from output
            caption = caption.replace("a security camera captures", "").strip()
            susp    = _vlm_suspicious(caption)

            with self._lock:
                self.description = caption if caption else "scene unclear"
                self.suspicious  = susp
            print(f"[VLM] {caption[:80]}")

        except Exception as e:
            print(f"[VLM] Error: {e}")
            with self._lock:
                self.description = "analysis error"
        finally:
            self.busy = False

    def get_state(self) -> dict:
        with self._lock:
            return {
                "description": self.description,
                "suspicious":  self.suspicious,
                "busy":        self.busy,
            }


# ══════════════════════════════════════════════
#  Unified facade
# ══════════════════════════════════════════════
class BehaviorClassifier:
    def __init__(self):
        self.vjepa = VJEPAWorker()
        self.vlm   = VLMWorker()

    def push_frame(self, frame_rgb_full: np.ndarray):
        import cv2
        small = cv2.resize(frame_rgb_full, config.CLIP_SIZE)
        self.vjepa.push_frame(small)
        self.vlm.push_frame(frame_rgb_full)

    def get_state(self) -> dict:
        v = self.vjepa.get_state()
        m = self.vlm.get_state()

        both      = v["suspicious"] and m["suspicious"]
        either    = v["suspicious"] or m["suspicious"]
        combined  = both or (config.ALERT_ON_SINGLE_SIGNAL and either)
        tier      = "HIGH" if both else ("MEDIUM" if either else "CLEAR")

        return {
            "label":          v["label"],
            "confidence":     v["confidence"],
            "suspicious":     combined,
            "busy":           v["busy"] or m["busy"],
            "description":    m["description"],
            "vlm_suspicious": m["suspicious"],
            "tier":           tier,
            "vjepa_susp":     v["suspicious"],
            "vlm_susp":       m["suspicious"],
        }
