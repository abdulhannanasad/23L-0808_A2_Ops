import io
from flask import Flask, render_template, request, jsonify
from PIL import Image
from werkzeug.utils import secure_filename
from .config import Config
from . import db, client

ALLOWED = {"jpg", "jpeg", "png", "webp"}

def create_app(cfg=Config):
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = cfg.MAX_UPLOAD_MB * 1024 * 1024
    db.init_db(cfg.DATABASE_URL)
    client.load_local(cfg)

    def fail(msg, code=400):
        return render_template("error.html", message=msg, code=code), code

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.post("/predict")
    def predict():
        f = request.files.get("file")
        if not f or not f.filename:
            return fail("Please choose an image file.")
        name = secure_filename(f.filename)
        if "." not in name or name.rsplit(".", 1)[1].lower() not in ALLOWED:
            return fail("Unsupported file type. Use JPG, PNG or WEBP.")
        raw = f.read()  # kept in memory only, never written to disk
        try:
            with Image.open(io.BytesIO(raw)) as im:
                im.verify()
        except Exception:
            return fail("That file looks like an image but could not be decoded.")
        try:
            r = client.predict(raw, name, cfg)
        except Exception as e:
            app.logger.exception("inference failed")
            return fail("The model is unavailable right now. Please try again later.", 503)
        db.add_record(name, r)
        return render_template("result.html", r=r, filename=name)

    @app.get("/history")
    def history():
        return render_template("history.html", rows=db.recent())

    @app.get("/health")
    def health():
        return jsonify(status="ok", db=db.ping(), model_ready=client.model_ready(cfg),
                       mode="remote" if cfg.MODEL_API_URL else "local")

    @app.errorhandler(413)
    def too_big(_):
        return fail(f"File too large (max {cfg.MAX_UPLOAD_MB} MB).", 413)

    @app.errorhandler(404)
    def nf(_):
        return fail("Page not found.", 404)

    @app.errorhandler(500)
    def err(_):
        return fail("Something went wrong on our side.", 500)

    return app

if __name__ == "__main__":
    # Flask server, debug OFF: debug mode exposes an interactive console (remote code execution) and auto-reloads.
    create_app().run(host=Config.HOST, port=Config.PORT, debug=False)
