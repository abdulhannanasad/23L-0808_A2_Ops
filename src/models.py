"""Task 2.2 / 2.3 models. Preprocessing lives INSIDE the model, so serving sends raw 0-255 pixels."""
import keras
from keras import layers

def build_cnn(img_size=224, dropout=0.4):
    inp = keras.Input((img_size, img_size, 3))
    x = layers.Rescaling(1 / 255.0)(inp)
    for f in (32, 64, 128):
        x = layers.Conv2D(f, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inp, out, name="baseline_cnn")

LAST_BLOCK = {"mobilenetv2": "block_16_expand", "efficientnetb0": "block7a_expand_conv"}

def build_transfer(img_size=224, backbone="mobilenetv2", dropout=0.3):
    inp = keras.Input((img_size, img_size, 3))
    if backbone == "mobilenetv2":
        base = keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_shape=(img_size, img_size, 3))
        x = layers.Rescaling(1 / 127.5, offset=-1)(inp)  # MobileNetV2 expects [-1, 1]
    elif backbone == "efficientnetb0":
        base = keras.applications.EfficientNetB0(include_top=False, weights="imagenet", input_shape=(img_size, img_size, 3))
        x = inp  # EfficientNet has its own rescaling inside and expects [0, 255]
    else:
        raise ValueError(backbone)
    base.trainable = False
    x = base(x, training=False)  # keeps BatchNorm in inference mode, even when unfrozen later
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    m = keras.Model(inp, out, name=f"transfer_{backbone}")
    m.backbone_name = backbone
    return m

def unfreeze_last(model, n=None):
    """Unfreeze the last block (default) or the last n layers of the backbone. BatchNorm stays frozen."""
    base = model.get_layer(index=[i for i, l in enumerate(model.layers) if isinstance(l, keras.Model)][0])
    base.trainable = True
    names = [l.name for l in base.layers]
    start = names.index(LAST_BLOCK[model.backbone_name]) if n is None else max(0, len(names) - n)
    for i, l in enumerate(base.layers):
        l.trainable = i >= start and not isinstance(l, layers.BatchNormalization)
    return [l.name for l in base.layers if l.trainable]
