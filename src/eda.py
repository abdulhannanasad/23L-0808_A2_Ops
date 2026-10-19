"""Task 1.3 visual EDA: python -m src.eda"""
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

df = pd.read_csv("reports/audit.csv")
df = df[df.ok]
for cls, g in df.groupby("cls"):
    s = g.sample(12, random_state=0)
    fig, ax = plt.subplots(3, 4, figsize=(10, 8))
    for a, p in zip(ax.ravel(), s.path):
        a.imshow(Image.open(p).convert("RGB")); a.axis("off")
    fig.suptitle(f"Samples: {cls}"); fig.savefig(f"reports/grid_{cls}.png", dpi=120); plt.close(fig)
for col in ["w", "h", "size"]:
    fig, ax = plt.subplots()
    for cls, g in df.groupby("cls"):
        ax.hist(g[col], bins=40, alpha=0.5, label=cls)
    ax.set_xlabel(col); ax.legend(); fig.savefig(f"reports/hist_{col}.png", dpi=120); plt.close(fig)
print("saved to reports/")
