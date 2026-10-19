"""Task 2.5 / 2.6 evaluation, threshold choice, error analysis."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, roc_curve, ConfusionMatrixDisplay)

def predict(model, ds):
    y = np.concatenate([b[1].numpy() for b in ds])
    p = model.predict(ds, verbose=0).ravel()
    return y, p

def pick_threshold(y, p):
    """Choose on VALIDATION data only: threshold maximising F1 (balanced view of both error types).
    Swap for a recall-constrained rule if you decide one error type is worse (justify in report)."""
    grid = np.linspace(0.05, 0.95, 91)
    return float(grid[int(np.argmax([f1_score(y, p >= t) for t in grid]))])

def evaluate_and_plot(y, p, thr, test_csv, outdir):
    pred = (p >= thr).astype(int)
    m = dict(accuracy=accuracy_score(y, pred), precision=precision_score(y, pred),
             recall=recall_score(y, pred), f1=f1_score(y, pred), roc_auc=roc_auc_score(y, p), threshold=thr)
    cm = confusion_matrix(y, pred)
    ConfusionMatrixDisplay(cm, display_labels=["real", "ai"]).plot()
    plt.savefig(f"{outdir}/confusion_matrix.png", dpi=120); plt.close()
    fpr, tpr, _ = roc_curve(y, p)
    plt.plot(fpr, tpr); plt.plot([0, 1], [0, 1], "--"); plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title(f"ROC (AUC={m['roc_auc']:.3f})"); plt.savefig(f"{outdir}/roc_curve.png", dpi=120); plt.close()
    df = pd.read_csv(test_csv)
    wrong = np.where(pred != y)[0][:10]
    fig, ax = plt.subplots(2, 5, figsize=(15, 6))
    for a, i in zip(ax.ravel(), wrong):
        a.imshow(Image.open(df.path[i]).convert("RGB")); a.axis("off")
        a.set_title(f"true={'ai' if y[i] else 'real'} p_ai={p[i]:.2f}", fontsize=9)
    fig.savefig(f"{outdir}/misclassified.png", dpi=120); plt.close(fig)
    return {k: float(v) for k, v in m.items()}
