"""Build stratified train/val/test CSVs (70/15/15) from the audit. Drops corrupt files and duplicates."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42

def main():
    df = pd.read_csv("reports/audit.csv")
    df = df[df.ok & ~df.dup].copy()
    df["label"] = df["cls"].str.lower().str.startswith("ai").astype(int)  # 1 = AI-generated
    tr, rest = train_test_split(df, test_size=0.30, stratify=df.label, random_state=SEED)
    va, te = train_test_split(rest, test_size=0.50, stratify=rest.label, random_state=SEED)
    Path("splits").mkdir(exist_ok=True)
    for name, d in [("train", tr), ("val", va), ("test", te)]:
        d[["path", "label"]].to_csv(f"splits/{name}.csv", index=False)
        print(name, len(d), d.label.mean().round(3))

if __name__ == "__main__":
    main()
