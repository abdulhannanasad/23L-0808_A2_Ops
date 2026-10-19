"""Task 3.1/3.2 toy: python scripts/toy_mlflow.py --lr 0.001   (local if MLFLOW_TRACKING_URI unset, else DagsHub)"""
import argparse, os
import keras, mlflow, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv
load_dotenv()
ap = argparse.ArgumentParser(); ap.add_argument("--lr", type=float, default=1e-3); a = ap.parse_args()
if os.getenv("MLFLOW_TRACKING_URI"):
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment("toy-mnist")
(x, y), _ = keras.datasets.mnist.load_data(); x = x[:10000] / 255.0; y = y[:10000]
m = keras.Sequential([keras.layers.Flatten(input_shape=(28, 28)), keras.layers.Dense(64, activation="relu"), keras.layers.Dense(10, activation="softmax")])
m.compile(keras.optimizers.Adam(a.lr), "sparse_categorical_crossentropy", metrics=["accuracy"])
with mlflow.start_run(run_name=f"toy_lr{a.lr}"):
    mlflow.log_param("lr", a.lr); mlflow.set_tag("purpose", "toy")
    h = m.fit(x, y, epochs=5, validation_split=0.2, verbose=0)
    for i in range(5):
        mlflow.log_metrics({"loss": h.history["loss"][i], "val_loss": h.history["val_loss"][i], "val_accuracy": h.history["val_accuracy"][i]}, step=i)
    plt.plot(h.history["loss"], label="train"); plt.plot(h.history["val_loss"], label="val"); plt.legend(); plt.savefig("toy_loss.png")
    mlflow.log_artifact("toy_loss.png"); m.save("toy.keras"); mlflow.log_artifact("toy.keras")
