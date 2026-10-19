import io, os
os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ["MODEL_PATH"] = "does-not-exist.keras"
from PIL import Image
from app.app import create_app

def client():
    return create_app().test_client()

def test_health():
    r = client().get("/health")
    assert r.status_code == 200 and r.json["status"] == "ok"

def test_rejects_wrong_type():
    r = client().post("/predict", data={"file": (io.BytesIO(b"hi"), "a.txt")}, content_type="multipart/form-data")
    assert r.status_code == 400

def test_rejects_fake_image():
    r = client().post("/predict", data={"file": (io.BytesIO(b"not an image"), "a.png")}, content_type="multipart/form-data")
    assert r.status_code == 400

def test_valid_image_without_model_gives_503():
    buf = io.BytesIO(); Image.new("RGB", (8, 8)).save(buf, "PNG"); buf.seek(0)
    r = client().post("/predict", data={"file": (buf, "ok.png")}, content_type="multipart/form-data")
    assert r.status_code == 503
