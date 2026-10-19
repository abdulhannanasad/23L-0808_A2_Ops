"""Task 3.6 separate script: load the model by alias and classify one image.
python -m src.load_champion path/to/image.jpg"""
import sys
from src.fetch_model import *  # noqa: F401,F403  (downloads champion to ./models)
from src.inference import Classifier

clf = Classifier("models/best.keras")
print(clf.predict(open(sys.argv[1], "rb").read()))
