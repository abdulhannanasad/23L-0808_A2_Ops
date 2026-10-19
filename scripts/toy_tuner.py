"""Task 2.7 toy: Keras Tuner on Fashion-MNIST"""
import keras, keras_tuner as kt
(x, y), _ = keras.datasets.fashion_mnist.load_data(); x = x[:10000] / 255.0; y = y[:10000]
def build(hp):
    m = keras.Sequential([keras.layers.Flatten(input_shape=(28, 28)),
                          keras.layers.Dense(hp.Int("units", 32, 256, step=32), activation="relu"),
                          keras.layers.Dense(10, activation="softmax")])
    m.compile(keras.optimizers.Adam(hp.Float("lr", 1e-4, 1e-2, sampling="log")), "sparse_categorical_crossentropy", metrics=["accuracy"])
    return m
t = kt.RandomSearch(build, "val_accuracy", max_trials=8, directory="out/tuner_toy", project_name="fmnist", seed=42)
t.search(x, y, epochs=3, validation_split=0.2)
t.results_summary()
print(t.get_best_hyperparameters(1)[0].values)
