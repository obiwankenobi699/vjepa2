"""
NazarAI — Stage 3: Qwen2.5-VL-3B-Instruct (fixed)

Fixes vs previous version:
  - Model download status shown clearly
  - Busy-skip logging reduced (only log every 10 skips)
  - Cooldown 30s prevents spam
  - SAFE responses never trigger alerts
  - Fallback: if Qwen not yet loaded, mark as pending (not threat)
"""
import threading
import time
import datetime
import numpy as np
import torch
from PIL import Image
from collections import deque
import config

_qwen_model     = None
_qwen_processor = None
_load_lock      = threading.Lock()
_load_attempted = False
_load_done      = False
_skip_count     = 0


def _load_qwen():
    global _qwen_model, _qwen_processor, _load_attempted, _load_done
    with _load_lock:
        if _load_done:
            return
        if _load_attempted:
            return
        _load_attempted = True

        print("\n[S3] ═══════════════════════════════════════")
        print("[S3] Loading Qwen2.5-VL-3B-Instruct...")
        print("[S3] First run: downloads ~7.5GB (one time only)")
        print("[S3] Subsequent runs: loads from cache in ~30s")
        print("[S3] ═══════════════════════════════════════\n")

        try:
            from transformers import (
                Qwen2_5_VLForConditionalGeneration,
                AutoProcessor,
                BitsAndBytesConfig,
            )
        except ImportError as e:
            print(f"[S3] FATAL: {e}")
            print("[S3] Run: pip install 'transformers>=4.51' accelerate bitsandbytes")
            return

        kwargs = {"device_map": config.QWEN_DEVICE}

        if config.QWEN_LOAD_IN_4BIT:
            try:
                import bitsandbytes
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                print("[S3] int4 quantisation active (~2GB VRAM)")
            except ImportError:
                kwargs["torch_dtype"] = config.QWEN_DTYPE
                print("[S3] bitsandbytes missing — using fp16 (~6GB VRAM)")
        else:
            kwargs["torch_dtype"] = config.QWEN_DTYPE

        try:
            _qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                config.QWEN_MODEL, **kwargs
            )
            _qwen_model.eval()
            _qwen_processor = AutoProcessor.from_pretrained(
                config.QWEN_MODEL, trust_remote_code=True
            )
            _load_done = True
            print(f"\n[S3] ✓ Qwen2.5-VL-3B ready on {config.QWEN_DEVICE}\n")
        except Exception as e:
            print(f"[S3] Load failed: {e}")


def _parse_response(text: str) -> dict:
    result = {
        "category":    "OTHER_THREAT",
        "severity":    5,
        "description": text[:120].strip(),
        "raw":         text,
    }
    for line in text.split("\n"):
        line = line.strip()
        upper = line.upper()
        if upper.startswith("CATEGORY:"):
            cat = line.split(":", 1)[1].strip().upper()
            if cat in config.QWEN_CATEGORIES:
                result["category"] = cat
        elif upper.startswith("SEVERITY:"):
            try:
                result["severity"] = int(line.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif upper.startswith("DESCRIPTION:"):
            result["description"] = line.split(":", 1)[1].strip()
    return result


class Stage3VLM:
    def __init__(self):
        self.category    = "SAFE"
        self.severity    = 0
        self.description = "Qwen loading on first suspicious event"
        self.raw         = ""
        self.suspicious  = False
        self.busy        = False
        self.loaded      = False
        self._calls      = 0
        self._last_call  = 0.0
        self._skip_log   = 0
        self._cam_history: dict[str, deque] = {}
        self._lock = threading.Lock()

    def push_crop(self, crop_bgr: np.ndarray, detections: list,
                  camera_id: str = "cam0"):
        global _skip_count
        now = time.time()

        if self.busy:
            _skip_count += 1
            if _skip_count % 10 == 0:  # only log every 10 skips
                print(f"[S3] Busy — skipped {_skip_count} crops so far")
            return

        if now - self._last_call < config.QWEN_MIN_INTERVAL:
            remaining = config.QWEN_MIN_INTERVAL - (now - self._last_call)
            if int(remaining) % 5 == 0:  # log every 5s
                print(f"[S3] Cooldown — {remaining:.0f}s remaining")
            return

        self._last_call = now
        self.busy       = True
        _skip_count     = 0
        threading.Thread(
            target=self._run,
            args=(crop_bgr.copy(), list(detections), camera_id),
            daemon=True,
        ).start()

    def _run(self, crop_bgr: np.ndarray, detections: list, camera_id: str):
        global _load_done
        try:
            # Load model if not done yet
            if not _load_done:
                _load_qwen()
            if not _load_done:
                print("[S3] Model not ready yet — skipping inference")
                with self._lock:
                    self.description = "Loading model... (first suspicious event)"
                return

            self.loaded = True
            ts     = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            hist   = self._cam_history.setdefault(camera_id, deque(maxlen=3))
            last_3 = "; ".join(hist) or "none"
            det_str = ", ".join(
                f"{d['label']}({d['confidence']:.2f})" for d in detections
            ) if detections else "none"

            prompt_text = config.QWEN_PROMPT_TEMPLATE.format(
                location        = config.QWEN_LOCATION,
                zone            = config.QWEN_ZONE,
                timestamp       = ts,
                last_3_events   = last_3,
                yolo_detections = det_str,
            )

            import cv2 as _cv2
            crop_rgb = _cv2.cvtColor(crop_bgr, _cv2.COLOR_BGR2RGB)
            pil_img  = Image.fromarray(crop_rgb)

            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text",  "text":  prompt_text},
                ],
            }]

            text_input = _qwen_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            try:
                from qwen_vl_utils import process_vision_info
                image_inputs, video_inputs = process_vision_info(messages)
            except ImportError:
                image_inputs, video_inputs = [pil_img], None

            inputs = _qwen_processor(
                text=[text_input],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(config.QWEN_DEVICE) for k, v in inputs.items()}

            with torch.no_grad():
                out = _qwen_model.generate(
                    **inputs,
                    max_new_tokens=config.QWEN_MAX_TOKENS,
                    do_sample=False,
                )
            generated = out[:, inputs["input_ids"].shape[1]:]
            response  = _qwen_processor.batch_decode(
                generated, skip_special_tokens=True
            )[0].strip()

            parsed = _parse_response(response)
            susp   = parsed["category"] != "SAFE" and parsed["severity"] >= config.ALERT_MIN_SEVERITY_MEDIUM
            self._calls += 1
            hist.append(f"{parsed['category']} sev={parsed['severity']}")

            with self._lock:
                self.category    = parsed["category"]
                self.severity    = parsed["severity"]
                self.description = parsed["description"]
                self.raw         = response
                self.suspicious  = susp

            icon = "🚨" if susp else "✓"
            print(f"[S3] #{self._calls} {icon} [{parsed['category']}] "
                  f"sev={parsed['severity']} | {parsed['description'][:70]}")

            del inputs
            if config.DEVICE != "cpu":
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"[S3] ERROR: {e}")
            import traceback; traceback.print_exc()
            with self._lock:
                self.category    = "OTHER_THREAT"
                self.description = f"Error: {str(e)[:60]}"
                self.suspicious  = True
        finally:
            self.busy = False

    def get_state(self) -> dict:
        with self._lock:
            return {
                "category":    self.category,
                "severity":    self.severity,
                "description": self.description,
                "suspicious":  self.suspicious,
                "busy":        self.busy,
                "calls":       self._calls,
                "loaded":      self.loaded,
            }
