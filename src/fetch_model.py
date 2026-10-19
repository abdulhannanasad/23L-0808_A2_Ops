"""Download the champion ONCE to ./models (the app never calls DagsHub per request).
python -m src.fetch_model"""
import json, os, shutil
from pathlib import Path
import mlflow
from mlflow import MlflowClient
from dotenv import load_dotenv

load_dotenv()
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mv = MlflowClient().get_model_version_by_alias("visionguard", "champion")
local = mlflow.artifacts.download_artifacts(artifact_uri=mv.source, dst_path="out/champion")
Path("models").mkdir(exist_ok=True)
shutil.copy(Path(local) / "best.keras", "models/best.keras")
meta = json.loads((Path(local) / "meta.json").read_text()); meta["model_name"] = f"visionguard v{mv.version} (champion)"
Path("models/meta.json").write_text(json.dumps(meta))
print("saved models/best.keras", meta)
