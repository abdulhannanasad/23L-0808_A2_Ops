"""Model API deployed to the cloud. POST /predict with header X-API-Key and multipart field 'file'."""
import hmac, os
from flask import Flask, request, jsonify
from src.inference import Classifier

app = Flask(__name__)
KEY = os.getenv("MODEL_API_KEY", "")
clf = Classifier(os.getenv("MODEL_PATH", "models/best.keras"))

@app.get("/health")
def health():
    return jsonify(status="ok", model=clf.name)

@app.post("/predict")
def predict():
    if not KEY or not hmac.compare_digest(request.headers.get("X-API-Key", ""), KEY):
        return jsonify(error="unauthorized"), 401
    f = request.files.get("file")
    if not f:
        return jsonify(error="no file"), 400
    try:
        return jsonify(clf.predict(f.read()))
    except Exception:
        return jsonify(error="could not process image"), 422

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")), debug=False)
