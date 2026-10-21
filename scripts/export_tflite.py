"""Convert models/best.keras -> lite_api/model/model.tflite and verify it against the Keras model.
python scripts/export_tflite.py [models/best.keras] [--check-images splits/test.csv]"""
import json, shutil, sys
from pathlib import Path
import numpy as np, pandas as pd, tensorflow as tf, keras
sys.path[:0] = [".", "lite_api"]
from preprocess import decode_resize as np_decode
from src.common import decode_resize as tf_decode

src = Path(sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "models/best.keras")
out = Path("lite_api/model"); out.mkdir(parents=True, exist_ok=True)
model = keras.models.load_model(src)
meta = json.loads(src.with_name("meta.json").read_text())
size = int(meta.get("img_size", 224))

# Keras 3 models convert via an exported SavedModel (from_keras_model is unsupported in TF 2.16 + Keras 3)
model.export("out/saved_model")
conv = tf.lite.TFLiteConverter.from_saved_model("out/saved_model")
(out / "model.tflite").write_bytes(conv.convert())
shutil.copy(src.with_name("meta.json"), out / "meta.json")
print("model.tflite size (MB):", round((out / "model.tflite").stat().st_size / 1e6, 1))

csv = sys.argv[sys.argv.index("--check-images") + 1] if "--check-images" in sys.argv else "splits/test.csv"
paths = pd.read_csv(csv).sample(20, random_state=0).path if Path(csv).exists() else []
interp = tf.lite.Interpreter(model_path=str(out / "model.tflite")); interp.allocate_tensors()
i, o = interp.get_input_details()[0]["index"], interp.get_output_details()[0]["index"]
pre_diff, prob_diff, flips = [], [], 0
for p in paths:
    raw = open(p, "rb").read()
    a = tf_decode(raw, size).numpy(); b = np_decode(raw, size)
    pre_diff.append(float(np.abs(a - b).max()))
    pk = float(model.predict(a[None], verbose=0)[0][0])
    interp.set_tensor(i, b[None].astype(np.float32)); interp.invoke()
    pt = float(interp.get_tensor(o)[0][0])
    prob_diff.append(abs(pk - pt)); flips += (pk >= meta["threshold"]) != (pt >= meta["threshold"])
if len(paths):
    print(f"preprocess max |tf-numpy| pixel diff: {max(pre_diff):.3f} (0-255 scale)")
    print(f"max |p_keras - p_tflite|: {max(prob_diff):.5f}; label flips: {flips}/{len(paths)}")
