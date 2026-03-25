import os
import json

from flask import Flask, jsonify, request, send_file, send_from_directory
from flask_cors import CORS

from database import (
    check_db_health,
    create_analysis,
    get_user_by_email,
    init_db,
    list_analyses,
    upsert_user,
)

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
    db_ready = True
    db_error = ""
    try:
        init_db()
    except Exception as exc:
        db_ready = False
        db_error = str(exc)

    model = get_model(DEFAULT_MODEL_PATH)

    @app.get("/health")
    def health():
        db_connected = check_db_health() if db_ready else False
        return jsonify(
            {
                "status": "ok",
                "model_loaded": model is not None,
                "db_connected": db_connected,
                "db_error": db_error if not db_connected else "",
            }
        )

    @app.get("/db/health")
    def db_health():
        connected = check_db_health() if db_ready else False
        return jsonify({"status": "ok" if connected else "error", "connected": connected, "error": db_error})

    @app.post("/users/upsert")
    def user_upsert():
        if not db_ready:
            return jsonify({"error": f"Database is not available: {db_error}"}), 503
        payload = request.get_json(silent=True) or {}
        try:
            user = upsert_user(payload)
            return jsonify(user)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:
            return jsonify({"error": f"Failed to save user: {exc}"}), 500

    @app.get("/users/<path:email>")
    def get_user(email):
        if not db_ready:
            return jsonify({"error": f"Database is not available: {db_error}"}), 503
        try:
            user = get_user_by_email(email)
            if user is None:
                return jsonify({"error": "User not found"}), 404
            return jsonify(user)
        except Exception as exc:
            return jsonify({"error": f"Failed to fetch user: {exc}"}), 500

    @app.post("/analyses")
    def create_analysis_endpoint():
        if not db_ready:
            return jsonify({"error": f"Database is not available: {db_error}"}), 503
        payload = request.get_json(silent=True) or {}
        try:
            created = create_analysis(payload)
            return jsonify(created), 201
        except Exception as exc:
            return jsonify({"error": f"Failed to save analysis: {exc}"}), 500

    @app.get("/analyses")
    def list_analyses_endpoint():
        if not db_ready:
            return jsonify({"error": f"Database is not available: {db_error}"}), 503
        user_email = request.args.get("userEmail")
        try:
            data = list_analyses(user_email=user_email)
            return jsonify(data)
        except Exception as exc:
            return jsonify({"error": f"Failed to list analyses: {exc}"}), 500

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

        # Add image URL to the result so frontend can display it
        result["image_url"] = f"/image/{uploaded.filename}"
        
        print(f"[ANALYZE] Result keys: {list(result.keys())}")
        print(f"[ANALYZE] is_fractured: {result.get('is_fractured')}")
        print(f"[ANALYZE] heatmap_url: {result.get('heatmap_url')}")
        print(f"[ANALYZE] heatmap_generated: {result.get('heatmap_generated')}")
        print(f"[ANALYZE] image_url: {result.get('image_url')}")

        # Best-effort persistence when caller passes user email in form data.
        user_email = (request.form.get("userEmail") or "").strip().lower()
        try:
            create_analysis(
                {
                    "userEmail": user_email or None,
                    "prediction": result.get("predicted_class"),
                    "isFractured": result.get("is_fractured"),
                    "fractureType": result.get("fracture_type"),
                    "confidence": result.get("confidence"),
                    "bonePart": result.get("bone_part"),
                    "resultJson": json.dumps(result),
                }
            )
        except Exception:
            pass

        return jsonify(result)

    @app.get("/get_heatmap")
    def get_heatmap():
        if not os.path.exists(DEFAULT_HEATMAP_PATH):
            return jsonify({"error": "No heatmap available yet. POST /analyze first."}), 404
        resp = send_file(DEFAULT_HEATMAP_PATH, mimetype="image/jpeg")
        resp.headers["Cache-Control"] = "no-store, max-age=0"
        return resp

    @app.get("/image/<filename>")
    def get_image(filename):
        """Serve uploaded images to the frontend"""
        uploads_dir = os.path.join(os.path.dirname(__file__), "uploads")
        img_path = os.path.join(uploads_dir, filename)
        
        # Security: Only serve files from uploads directory
        if not os.path.abspath(img_path).startswith(os.path.abspath(uploads_dir)):
            return jsonify({"error": "Invalid file path"}), 403
        
        if not os.path.exists(img_path):
            return jsonify({"error": "Image not found"}), 404
        
        resp = send_file(img_path, mimetype="image/jpeg")
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