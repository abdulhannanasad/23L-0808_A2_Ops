"""Bonus: Grad-CAM for 4 correct + 4 wrong test predictions.
python -m src.gradcam out/<run_name>/keras_model/best.keras [--mlflow]"""
import argparse, json, os
from pathlib import Path
import numpy as np, pandas as pd, tensorflow as tf, keras, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .common import decode_resize

def cam(model, x):
    """Walk the (sequential) layers, keep the LAST 4-D feature map, differentiate the output w.r.t. it."""
    with tf.GradientTape() as tape:
        h, conv = x, None
        for layer in model.layers[1:]:  # skip InputLayer
            h = layer(h, training=False)
            if len(h.shape) == 4:
                conv = h; tape.watch(conv)  # noqa: E702
        score = h[:, 0]  # sigmoid output = P(AI)
    g = tape.gradient(score, conv)
    w = tf.reduce_mean(g, axis=(1, 2), keepdims=True)
    c = tf.nn.relu(tf.reduce_sum(w * conv, axis=-1))[0].numpy()
    return c / (c.max() + 1e-8), float(score[0])

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("--mlflow", action="store_true")
    a = ap.parse_args()
    model = keras.models.load_model(a.model)
    meta = json.loads((Path(a.model).with_name("meta.json")).read_text())
    thr, size = meta["threshold"], meta["img_size"]
    df = pd.read_csv("splits/test.csv"); df = df.sample(min(400, len(df)), random_state=0).reset_index(drop=True)
    right, wrong = [], []
    for path, y in zip(df.path, df.label):
        if len(right) >= 4 and len(wrong) >= 4:
            break
        x = decode_resize(tf.io.read_file(path), size)[None]
        heat, p = cam(model, x)
        item = (x[0].numpy() / 255.0, heat, p, int(y))
        bucket = right if (p >= thr) == bool(y) else wrong
        if len(bucket) < 4:
            bucket.append(item)
    fig, ax = plt.subplots(2, 8, figsize=(24, 6.5))
    for col, (img, heat, p, y) in enumerate(right + wrong):
        hm = tf.image.resize(heat[..., None], (size, size)).numpy()[..., 0]
        tag = "correct" if col < len(right) else "WRONG"
        ax[0, col].imshow(img); ax[0, col].set_title(f"{tag} | true={'ai' if y else 'real'} p_ai={p:.2f}", fontsize=9)
        ax[1, col].imshow(img); ax[1, col].imshow(hm, cmap="jet", alpha=0.45)
        ax[0, col].axis("off"); ax[1, col].axis("off")
    Path("reports").mkdir(exist_ok=True)
    fig.savefig("reports/gradcam.png", dpi=110, bbox_inches="tight")
    print("saved reports/gradcam.png", f"(correct={len(right)}, wrong={len(wrong)})")
    if a.mlflow:
        import mlflow
        from dotenv import load_dotenv; load_dotenv()
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"]); mlflow.set_experiment("23L-0808_A2_Ops")
        with mlflow.start_run(run_name="gradcam"):
            mlflow.log_artifact("reports/gradcam.png")

if __name__ == "__main__":
    main()
