"""Lightweight VisionGuard model API (TFLite via LiteRT, no TensorFlow): fits in ~512 MB of RAM.
POST /predict  header X-API-Key, multipart field 'file'."""
import hmac, json, os
from pathlib import Path
import numpy as np
from flask import Flask, jsonify, request
from ai_edge_litert.interpreter import Interpreter
from preprocess import decode_resize

HERE = Path(__file__).parent / "model"
meta = json.loads((HERE / "meta.json").read_text())
interp = Interpreter(model_path=str(HERE / "model.tflite"))  # loaded once at start-up
interp.allocate_tensors()
IN, OUT = interp.get_input_details()[0]["index"], interp.get_output_details()[0]["index"]
SIZE = int(meta.get("img_size", 224))
THR = float(os.getenv("THRESHOLD") or meta.get("threshold", 0.5))
NAME = meta.get("model_name", "visionguard")
KEY = os.getenv("MODEL_API_KEY", "")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

@app.get("/")
def root():
    return jsonify(service="VisionGuard API", model=NAME)

@app.get("/health")
def health():
    return jsonify(status="ok", model=NAME)

@app.post("/predict")
def predict():
    if not KEY or not hmac.compare_digest(request.headers.get("X-API-Key", ""), KEY):
        return jsonify(error="unauthorized"), 401
    f = request.files.get("file")
    if not f:
        return jsonify(error="no file"), 400
    try:
        x = decode_resize(f.read(), SIZE)[None]
        interp.set_tensor(IN, x.astype(np.float32))
        interp.invoke()
        p = float(interp.get_tensor(OUT)[0][0])
    except Exception:
        return jsonify(error="could not process image"), 422
    is_ai = p >= THR
    return jsonify(label="AI-generated" if is_ai else "Real", confidence=p if is_ai else 1 - p,
                   p_ai=p, model_name=NAME, threshold=THR)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "10000")), debug=False)
