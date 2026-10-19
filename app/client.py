"""Local model or remote API (Task 7/8), chosen by MODEL_API_URL."""
import os
import requests

_local = None

def load_local(cfg):
    global _local
    if os.path.exists(cfg.MODEL_PATH):
        from src.inference import Classifier
        _local = Classifier(cfg.MODEL_PATH, float(cfg.THRESHOLD) if cfg.THRESHOLD else None)
    return _local

def model_ready(cfg):
    return bool(cfg.MODEL_API_URL) or _local is not None

def predict(raw, filename, cfg):
    if cfg.MODEL_API_URL:
        r = requests.post(cfg.MODEL_API_URL.rstrip("/") + "/predict", files={"file": (filename, raw)},
                          headers={"X-API-Key": cfg.MODEL_API_KEY}, timeout=60)
        r.raise_for_status()
        return r.json()
    if _local is None:
        raise RuntimeError("model not loaded")
    return _local.predict(raw)
