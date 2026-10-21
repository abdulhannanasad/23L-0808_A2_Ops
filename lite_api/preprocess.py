"""NumPy re-implementation of src.common.decode_resize (tf.image.resize_with_pad, bilinear, no antialias),
so the cloud API needs no TensorFlow. scripts/export_tflite.py checks it against TensorFlow."""
import io
import numpy as np
from PIL import Image

def _axis(n_out, n_in):
    scale = np.float32(n_in) / np.float32(n_out)
    x = (np.arange(n_out, dtype=np.float32) + np.float32(0.5)) * scale - np.float32(0.5)
    f = np.floor(x)
    lo = np.maximum(f, 0).astype(np.int64)
    hi = np.minimum(np.ceil(x), n_in - 1).astype(np.int64)
    return lo, hi, (x - f).astype(np.float32)

def resize_bilinear(img, oh, ow):
    h, w, _ = img.shape
    ylo, yhi, ly = _axis(oh, h)
    xlo, xhi, lx = _axis(ow, w)
    g = lambda yi, xi: img[yi][:, xi].astype(np.float32)  # gather first, convert only the small result
    tl, tr, bl, br = g(ylo, xlo), g(ylo, xhi), g(yhi, xlo), g(yhi, xhi)
    lx = lx[None, :, None]; ly = ly[:, None, None]
    top = tl + (tr - tl) * lx
    bot = bl + (br - bl) * lx
    return top + (bot - top) * ly

def decode_resize(raw: bytes, size: int) -> np.ndarray:
    img = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
    h, w, _ = img.shape
    ratio = max(w / size, h / size)
    rh, rw = int(np.floor(np.float32(h) / np.float32(ratio))), int(np.floor(np.float32(w) / np.float32(ratio)))
    rh, rw = max(rh, 1), max(rw, 1)
    out = np.zeros((size, size, 3), np.float32)
    ph = max(0, int(np.floor((size - h / ratio) / 2)))
    pw = max(0, int(np.floor((size - w / ratio) / 2)))
    out[ph:ph + rh, pw:pw + rw] = resize_bilinear(img, rh, rw)
    return out
