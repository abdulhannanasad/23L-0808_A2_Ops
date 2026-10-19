"""Task 5: every prediction is stored (what, when, which model, how confident)."""
import time
from datetime import datetime, timezone
from sqlalchemy import create_engine, text

_engine = None

def init_db(url, retries=10):
    global _engine
    _engine = create_engine(url, pool_pre_ping=True)
    for i in range(retries):
        try:
            with _engine.begin() as c:
                pk = "INTEGER PRIMARY KEY AUTOINCREMENT" if url.startswith("sqlite") else "SERIAL PRIMARY KEY"
                c.execute(text(f"""CREATE TABLE IF NOT EXISTS predictions (
                    id {pk}, created_at TEXT, filename TEXT, label TEXT, confidence REAL,
                    p_ai REAL, model_name TEXT, threshold REAL)"""))
            return
        except Exception:
            if i == retries - 1:
                raise
            time.sleep(2)

def add_record(filename, r):
    with _engine.begin() as c:
        c.execute(text("""INSERT INTO predictions (created_at, filename, label, confidence, p_ai, model_name, threshold)
                          VALUES (:t, :f, :l, :c, :p, :m, :th)"""),
                  dict(t=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"), f=filename, l=r["label"],
                       c=r["confidence"], p=r["p_ai"], m=r["model_name"], th=r["threshold"]))

def recent(limit=100):
    with _engine.connect() as c:
        return [dict(r._mapping) for r in c.execute(text("SELECT * FROM predictions ORDER BY id DESC LIMIT :n"), dict(n=limit))]

def ping():
    try:
        with _engine.connect() as c:
            c.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
