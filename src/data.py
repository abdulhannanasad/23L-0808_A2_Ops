"""Task 2.1 tf.data pipeline."""
import pandas as pd, tensorflow as tf, keras
from .common import decode_resize

SEED = 42
AUTOTUNE = tf.data.AUTOTUNE

def augmenter():
    return keras.Sequential([
        keras.layers.RandomFlip("horizontal", seed=SEED),
        keras.layers.RandomRotation(0.08, seed=SEED),
        keras.layers.RandomZoom(0.1, seed=SEED),
        keras.layers.RandomContrast(0.1, seed=SEED),
    ])

def make_ds(csv, img_size=224, batch=32, train=False, augment=True):
    df = pd.read_csv(csv)
    ds = tf.data.Dataset.from_tensor_slices((df.path.values, df.label.values.astype("float32")))
    if train:
        ds = ds.shuffle(len(df), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(lambda p, y: (decode_resize(tf.io.read_file(p), img_size), y), num_parallel_calls=AUTOTUNE)
    if train and augment:
        aug = augmenter()
        ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=AUTOTUNE)
    return ds.batch(batch).prefetch(AUTOTUNE)
