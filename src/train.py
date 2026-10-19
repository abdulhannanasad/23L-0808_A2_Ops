"""Tasks 2 + 3: train, evaluate once on test, log everything to MLflow/DagsHub.
Example: python -m src.train --model mobilenet --run-name r1_mnv2_lr1e-3 --lr 1e-3 --logging manual"""
import argparse, json, os, random, re, subprocess, sys
from pathlib import Path
import numpy as np, tensorflow as tf, keras, mlflow
from dotenv import load_dotenv
from .data import make_ds, SEED
from .models import build_cnn, build_transfer, unfreeze_last
from .evaluate import predict, pick_threshold, evaluate_and_plot

def sh(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, text=True).strip()
    except Exception:
        return "unknown"

def dvc_version():
    f = Path("data/raw.dvc")
    m = re.search(r"md5:\s*(\S+)", f.read_text()) if f.exists() else None
    return m.group(1) if m else "untracked"

class EpochLogger(keras.callbacks.Callback):  # manual instrumentation
    def __init__(self, offset=0): super().__init__(); self.offset = offset
    def on_epoch_end(self, epoch, logs=None):
        mlflow.log_metrics({k: float(v) for k, v in (logs or {}).items()}, step=self.offset + epoch)

def fit(model, train, val, lr, epochs, ckpt, offset, manual):
    model.compile(keras.optimizers.Adam(lr), "binary_crossentropy",
                  metrics=["accuracy", keras.metrics.AUC(name="auc")])
    cbs = [keras.callbacks.EarlyStopping("val_loss", patience=4, restore_best_weights=True),
           keras.callbacks.ModelCheckpoint(ckpt, "val_loss", save_best_only=True),
           keras.callbacks.ReduceLROnPlateau("val_loss", factor=0.3, patience=2)]
    if manual:
        cbs.append(EpochLogger(offset))
    h = model.fit(train, validation_data=val, epochs=epochs, callbacks=cbs, verbose=2)
    best = int(np.argmin(h.history["val_loss"]))
    return len(h.history["loss"]), best + 1, float(min(h.history["val_loss"]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["cnn", "mobilenet", "efficientnet"], default="mobilenet")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--experiment", default="23L-0808_A2_Ops")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ft-lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--ft-epochs", type=int, default=10)
    ap.add_argument("--no-finetune", action="store_true")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--aug", choices=["on", "off"], default="on")
    ap.add_argument("--logging", choices=["manual", "autolog"], default="manual")
    a = ap.parse_args()

    load_dotenv()
    random.seed(SEED); np.random.seed(SEED); keras.utils.set_random_seed(SEED)
    if os.getenv("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(a.experiment)
    manual = a.logging == "manual"
    if not manual:
        mlflow.tensorflow.autolog(log_models=False)

    out = Path("out") / a.run_name; out.mkdir(parents=True, exist_ok=True)
    train = make_ds("splits/train.csv", a.img_size, a.batch, True, a.aug == "on")
    val = make_ds("splits/val.csv", a.img_size, a.batch)
    test = make_ds("splits/test.csv", a.img_size, a.batch)

    with mlflow.start_run(run_name=a.run_name):
        mlflow.set_tags({"git_commit": sh("git rev-parse HEAD"), "dvc_data_version": dvc_version(),
                         "seed": SEED, "logging_mode": a.logging})
        mlflow.log_params(vars(a))
        if a.model == "cnn":
            model = build_cnn(a.img_size, a.dropout)
        else:
            model = build_transfer(a.img_size, {"mobilenet": "mobilenetv2", "efficientnet": "efficientnetb0"}[a.model], a.dropout)
        model.summary()
        (out / "trainable_stage1.txt").write_text("\n".join(l.name for l in model.layers if l.trainable))
        ckpts = []
        ep, best_ep, vl = fit(model, train, val, a.lr, a.epochs, str(out / "stage1.keras"), 0, manual)
        ckpts.append((vl, out / "stage1.keras")); mlflow.log_metric("best_epoch_stage1", best_ep)
        if a.model != "cnn" and not a.no_finetune:
            trainable = unfreeze_last(model)
            (out / "trainable_stage2.txt").write_text("\n".join(trainable))
            ep2, best2, vl2 = fit(model, train, val, a.ft_lr, a.ft_epochs, str(out / "stage2.keras"), ep, manual)
            ckpts.append((vl2, out / "stage2.keras")); mlflow.log_metric("best_epoch_stage2", best2)
        best_path = min(ckpts)[1]
        model = keras.models.load_model(best_path)

        yv, pv = predict(model, val)
        thr = pick_threshold(yv, pv)
        yt, pt = predict(model, test)  # test set used ONCE
        m = evaluate_and_plot(yt, pt, thr, "splits/test.csv", str(out))
        mlflow.log_metrics({f"test_{k}": v for k, v in m.items()})
        print(json.dumps(m, indent=2))

        pkg = out / "keras_model"; pkg.mkdir(exist_ok=True)
        model.save(pkg / "best.keras")
        (pkg / "meta.json").write_text(json.dumps({"threshold": thr, "img_size": a.img_size, "label_1": "ai", "label_0": "real"}))
        (out / "requirements_frozen.txt").write_text(sh(f"{sys.executable} -m pip freeze"))
        for f in ["confusion_matrix.png", "roc_curve.png", "misclassified.png", "requirements_frozen.txt",
                  "trainable_stage1.txt", "trainable_stage2.txt"]:
            if (out / f).exists():
                mlflow.log_artifact(str(out / f))
        for s in ["train", "val", "test"]:
            mlflow.log_artifact(f"splits/{s}.csv", "splits")
        mlflow.log_artifacts(str(pkg), "keras_model")

if __name__ == "__main__":
    main()
