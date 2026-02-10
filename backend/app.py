import os

from flask import Flask, jsonify, request, send_file, send_from_directory
from flask_cors import CORS

from utils import (
    DEFAULT_HEATMAP_PATH,
    DEFAULT_MODEL_PATH,
    ensure_dirs,
    get_model,
    predict_fracture,
)


def create_app():
    # Serve frontend from dist directory
    frontend_dist = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'dist')
    app = Flask(__name__, static_folder=frontend_dist, static_url_path='/')
    CORS(app)

    ensure_dirs()
    model = get_model(DEFAULT_MODEL_PATH)

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "model_loaded": model is not None})

    @app.post("/analyze")
    def analyze():
        if "image" not in request.files:
            return jsonify({"error": "Missing 'image' file"}), 400

        uploaded = request.files["image"]
        if not uploaded.filename:
            return jsonify({"error": "Empty filename"}), 400

        uploads_dir = os.path.join(os.path.dirname(__file__), "uploads")
        img_path = os.path.join(uploads_dir, uploaded.filename)
        uploaded.save(img_path)

        # Avoid serving a stale heatmap if Grad-CAM fails for this request.
        try:
            if os.path.exists(DEFAULT_HEATMAP_PATH):
                os.remove(DEFAULT_HEATMAP_PATH)
        except Exception:
            pass

        result = predict_fracture(model=model, img_path=img_path)
        return jsonify(result)

    @app.get("/get_heatmap")
    def get_heatmap():
        if not os.path.exists(DEFAULT_HEATMAP_PATH):
            return jsonify({"error": "No heatmap available yet. POST /analyze first."}), 404
        resp = send_file(DEFAULT_HEATMAP_PATH, mimetype="image/jpeg")
        resp.headers["Cache-Control"] = "no-store, max-age=0"
        return resp

    @app.route('/', defaults={'path': ''})
    @app.route('/<path:path>')
    def serve_frontend(path):
        # Serve static files
        if path and os.path.exists(os.path.join(app.static_folder, path)):
            return send_from_directory(app.static_folder, path)
        # Serve index.html for SPA routing
        return send_from_directory(app.static_folder, 'index.html')

    return app



if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)