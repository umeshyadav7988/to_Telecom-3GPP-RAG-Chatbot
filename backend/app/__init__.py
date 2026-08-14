"""Flask application factory."""

from __future__ import annotations

import logging
import sys

from flask import Flask, jsonify
from flask_cors import CORS

from config import settings


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.DEBUG if settings.debug else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        stream=sys.stdout,
    )
    # These are chatty at INFO and drown out pipeline logs.
    for noisy in ("httpx", "urllib3", "sentence_transformers", "anthropic._base_client"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def create_app() -> Flask:
    _configure_logging()
    app = Flask(__name__)
    app.config["JSON_SORT_KEYS"] = False
    app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024

    CORS(
        app,
        resources={r"/api/*": {"origins": list(settings.cors_origins) or "*"}},
        supports_credentials=False,
    )

    from .api.chat import bp as chat_bp
    from .api.conversations import bp as conversations_bp
    from .api.documents import bp as documents_bp
    from .api.evaluation import bp as evaluation_bp
    from .api.health import bp as health_bp

    for blueprint in (health_bp, chat_bp, documents_bp, conversations_bp, evaluation_bp):
        app.register_blueprint(blueprint)

    @app.get("/")
    def root():
        return jsonify(
            {
                "service": "Telecom 3GPP RAG Chatbot API",
                "docs": "See README.md for the full endpoint reference.",
                "endpoints": [
                    "GET  /api/health",
                    "GET  /api/status",
                    "POST /api/chat",
                    "POST /api/chat/stream  (SSE)",
                    "POST /api/search",
                    "GET  /api/documents",
                    "POST /api/documents/upload",
                    "POST /api/documents/reindex",
                    "GET  /api/documents/chunk/<chunk_id>",
                    "GET  /api/conversations",
                    "POST /api/feedback",
                    "GET  /api/evaluation/golden-set",
                    "POST /api/evaluation/run  (SSE)",
                ],
            }
        )

    @app.errorhandler(404)
    def not_found(_):
        return jsonify({"error": "Not found."}), 404

    @app.errorhandler(413)
    def too_large(_):
        return jsonify({"error": "Payload too large."}), 413

    @app.errorhandler(500)
    def server_error(exc):
        logging.getLogger(__name__).exception("Unhandled error")
        return jsonify({"error": "Internal server error.", "detail": str(exc)}), 500

    # Warm the engine at boot so the first user request is not the one that
    # pays for model loading and index deserialisation.
    with app.app_context():
        from .services.engine import get_engine

        engine = get_engine()
        logging.getLogger(__name__).info(
            "Engine ready | chunks=%d | embedder=%s | reranker=%s | mode=%s",
            engine.index.size,
            engine.index.embedder.name,
            engine.reranker.name,
            "generative" if engine.client else "extractive",
        )

    return app
