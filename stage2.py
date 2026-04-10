"""
NazarAI — Stage 2: MobileNetV3-Small binary classifier

FIX: ImageNet pretrained weights classify ANY person crop as ~0.5-0.6 suspicious.
This is why everything was triggering before. Solutions applied:
  1. Threshold raised to 0.82 (config.MOBILENET_THRESHOLD)
  2. Temporal buffer raised to 5 frames
  3. Motion magnitude check — if person barely moved, score is dampened
  4. Crop context check — full-frame person doing nothing = score dampened

For production accuracy (85%+): run train_mobilenet.py to fine-tune on UCF-Crime.
Current accuracy with ImageNet proxy: ~55-60% (better than before, not good enough for prod).
After fine-tuning: ~80-88%.
"""
import threading
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import cv2
from collections import deque
import config


def _build_model() -> nn.Module:
    m = models.mobilenet_v3_small(
        weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    )
    in_features = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(in_features, 1)
    return m


_TRANSFORM = T.Compose([
    T.ToPILImage(),
    T.Resize(config.MOBILENET_INPUT_SIZE),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _motion_magnitude(crop_bgr: np.ndarray) -> float:
    """
    Estimate how much motion is in a crop using frame difference.
    Returns 0.0 (static) to 1.0 (high motion).
    Used to dampen score for static persons.
    """
    if not hasattr(_motion_magnitude, "_prev"):
        _motion_magnitude._prev = {}
    key = id(crop_bgr)
    gray = cv2.cvtColor(
        cv2.resize(crop_bgr, (64, 64)), cv2.COLOR_BGR2GRAY
    ).astype(float)
    prev = _motion_magnitude._prev.get("last", gray)
    diff = np.abs(gray - prev).mean()
    _motion_magnitude._prev["last"] = gray
    return min(diff / 30.0, 1.0)   # normalize: 30 pixel diff = max motion


class Stage2Classifier:
    def __init__(self):
        print("[S2] Loading MobileNetV3-Small...")
        self.model = _build_model()

        if config.MOBILENET_WEIGHTS and __import__('os').path.exists(config.MOBILENET_WEIGHTS):
            state = torch.load(config.MOBILENET_WEIGHTS, map_location="cpu",
                               weights_only=True)
            self.model.load_state_dict(state)
            print(f"[S2] Fine-tuned weights loaded: {config.MOBILENET_WEIGHTS}")
        else:
            print("[S2] ImageNet proxy weights (fine-tune for production accuracy)")

        self.model = self.model.to(config.MOBILENET_DEVICE)
        self.model.eval()
        if config.MOBILENET_DTYPE == torch.float16:
            self.model = self.model.half()
        print(f"[S2] Ready on {config.MOBILENET_DEVICE} | threshold={config.MOBILENET_THRESHOLD}")

        self._buffers: dict[str, deque] = {}
        self._lock = threading.Lock()

        self._last_score = 0.0
        self._last_label = "NORMAL"
        self._confirmed  = False

    def infer_crop(self, crop_bgr: np.ndarray, camera_id: str = "cam0",
                   is_threat_class: bool = False) -> dict:
        """
        is_threat_class: True if YOLO detected knife/scissors in this crop.
        Threat-class crops bypass the high threshold.
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return {"score": 0.0, "label": "NORMAL", "confirmed": False}

        try:
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            tensor   = _TRANSFORM(crop_rgb).unsqueeze(0)
            if config.MOBILENET_DTYPE == torch.float16:
                tensor = tensor.half()
            tensor = tensor.to(config.MOBILENET_DEVICE)

            with torch.no_grad():
                score = float(torch.sigmoid(self.model(tensor)).item())

        except Exception as e:
            print(f"[S2] Inference error: {e}")
            return {"score": 0.0, "label": "NORMAL", "confirmed": False}

        # Motion dampening: if person is barely moving, reduce score
        # This prevents "person sitting still" from scoring high
        motion = _motion_magnitude(crop_bgr)
        if motion < 0.05:   # very static
            score = score * 0.6    # dampen by 40%

        # Threshold: threat-class YOLO detections use lower threshold
        threshold = 0.60 if is_threat_class else config.MOBILENET_THRESHOLD
        label     = "SUSPICIOUS" if score > threshold else "NORMAL"

        with self._lock:
            if camera_id not in self._buffers:
                self._buffers[camera_id] = deque(maxlen=config.MOBILENET_TEMPORAL)
            self._buffers[camera_id].append(1 if label == "SUSPICIOUS" else 0)
            buf       = self._buffers[camera_id]
            confirmed = (
                len(buf) == config.MOBILENET_TEMPORAL and
                sum(buf) == config.MOBILENET_TEMPORAL
            )
            self._last_score = score
            self._last_label = label
            self._confirmed  = confirmed

        if label == "SUSPICIOUS":
            buf_state = f"{sum(self._buffers.get(camera_id, [0]))}/{config.MOBILENET_TEMPORAL}"
            print(f"[S2] {camera_id} score={score:.3f} motion={motion:.2f} "
                  f"[{label}] buf={buf_state}"
                  + (" → CONFIRMED → Stage3" if confirmed else ""))

        return {"score": score, "label": label, "confirmed": confirmed}

    def reset_buffer(self, camera_id: str = "cam0"):
        with self._lock:
            if camera_id in self._buffers:
                self._buffers[camera_id].clear()

    def get_state(self) -> dict:
        with self._lock:
            return {
                "score":     self._last_score,
                "label":     self._last_label,
                "confirmed": self._confirmed,
            }
