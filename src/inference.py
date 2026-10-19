"""Task 4.1: model loaded ONCE; same preprocessing as training (src.common)."""
import json, os
from pathlib import Path
import numpy as np

class Classifier:
    def __init__(self, model_path, threshold=None):
        import keras
        self.model = keras.models.load_model(model_path)  # once, at start-up
        meta_file = Path(model_path).with_name("meta.json")
        meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
        env_thr = os.getenv("THRESHOLD")
        self.threshold = float(threshold if threshold is not None else (env_thr or meta.get("threshold", 0.5)))
        self.size = int(meta.get("img_size", self.model.input_shape[1]))
        self.name = meta.get("model_name", "visionguard")

    def predict(self, raw: bytes) -> dict:
        from .common import decode_resize
        x = decode_resize(raw, self.size).numpy()[None]
        p = float(self.model.predict(x, verbose=0)[0][0])
        is_ai = p >= self.threshold
        return {"label": "AI-generated" if is_ai else "Real", "confidence": p if is_ai else 1 - p,
                "p_ai": p, "model_name": self.name, "threshold": self.threshold}
