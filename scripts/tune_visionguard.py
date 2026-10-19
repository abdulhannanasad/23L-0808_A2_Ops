"""Task 2.7 apply: >=8 trials over lr, dropout, #unfrozen layers"""
import keras, keras_tuner as kt
from src.data import make_ds
from src.models import build_transfer, unfreeze_last

train = make_ds("splits/train.csv", 224, 32, True); val = make_ds("splits/val.csv", 224, 32)
def build(hp):
    m = build_transfer(224, "mobilenetv2", hp.Float("dropout", 0.2, 0.6, step=0.1))
    n = hp.Choice("unfrozen_layers", [0, 10, 20, 30])
    if n:
        unfreeze_last(m, n)
    m.compile(keras.optimizers.Adam(hp.Float("lr", 1e-5, 1e-3, sampling="log")), "binary_crossentropy", metrics=["accuracy"])
    return m
t = kt.RandomSearch(build, "val_accuracy", max_trials=8, directory="out/tuner", project_name="visionguard", seed=42)
t.search(train, validation_data=val, epochs=4)
t.results_summary()
