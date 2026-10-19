"""Task 3.6: python -m src.register <run_id> [--alias champion]"""
import argparse, os
import mlflow
from mlflow import MlflowClient
from dotenv import load_dotenv

NAME = "visionguard"
load_dotenv()
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
ap = argparse.ArgumentParser(); ap.add_argument("run_id"); ap.add_argument("--alias", default=None)
a = ap.parse_args()
mv = mlflow.register_model(f"runs:/{a.run_id}/keras_model", NAME)
print("registered version", mv.version)
if a.alias:
    MlflowClient().set_registered_model_alias(NAME, a.alias, mv.version)
    print(f"alias {a.alias} -> v{mv.version}")
