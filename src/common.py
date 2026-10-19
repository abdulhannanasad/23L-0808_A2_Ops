"""Preprocessing shared by training AND serving so they cannot drift apart."""
IMG_SIZE = 224

def decode_resize(raw, size=IMG_SIZE):
    import tensorflow as tf
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    # pad (not stretch) so aspect ratio is preserved; output float32 in [0,255]
    return tf.image.resize_with_pad(img, size, size)
