"""Task 1.2 dataset audit: python -m src.audit data/raw"""
import hashlib, sys
from pathlib import Path
import pandas as pd
from PIL import Image

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()

def main(root):
    rows = []
    for p in sorted(Path(root).rglob("*")):
        if not p.is_file():
            continue
        r = dict(path=p.as_posix(), cls=p.parent.name, size=p.stat().st_size,
                 sha256=sha256(p), ok=True, w=None, h=None, mode=None, fmt=None)
        try:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:
                r.update(w=im.width, h=im.height, mode=im.mode, fmt=im.format)
        except Exception:
            r["ok"] = False
        rows.append(r)
    df = pd.DataFrame(rows)
    df["dup"] = df.duplicated("sha256", keep="first")
    Path("reports").mkdir(exist_ok=True)
    df.to_csv("reports/audit.csv", index=False)
    good = df[df.ok]
    out = ["Counts per class:", str(df.groupby("cls").size()),
           f"\nCorrupt files: {(~df.ok).sum()}", f"Exact duplicates: {df.dup.sum()}",
           "\nmin/median/max:", str(good[["w", "h", "size"]].agg(["min", "median", "max"])),
           "\nModes:", str(good["mode"].value_counts()), "\nFormats:", str(good["fmt"].value_counts())]
    txt = "\n".join(out)
    print(txt)
    Path("reports/audit_summary.txt").write_text(txt)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/raw")
