"""Flask blueprint: POST /api/agriculture/ask and GET /api/agriculture/status. Independent of the crop ML model."""
import threading

from flask import Blueprint, jsonify, request

from . import config
from .pipeline import RagError, RagPipeline

bp = Blueprint("agriculture_rag", __name__, url_prefix="/api/agriculture")

_pipeline = None
_lock = threading.Lock()


def get_pipeline():
    global _pipeline
    with _lock:
        if _pipeline is None:
            _pipeline = RagPipeline()
        return _pipeline


@bp.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True)
    question = data.get("question") if isinstance(data, dict) else None
    if not isinstance(question, str) or len(question.strip()) < 3:
        return jsonify({"success": False, "message": "Provide a 'question' string (at least 3 characters)."}), 400
    question = question.strip()
    if len(question) > config.MAX_QUESTION_CHARS:
        return jsonify({"success": False, "message": f"Question is too long (max {config.MAX_QUESTION_CHARS} characters)."}), 400
    try:
        result = get_pipeline().ask(question)
    except RagError as exc:
        return jsonify({"success": False, "message": str(exc), "retrieved_context": exc.retrieved}), exc.status
    except Exception as exc:  # e.g. embedding model could not be loaded/downloaded
        print(f"[RAG] unexpected error: {exc!r}")
        return jsonify({"success": False, "message": "The assistant failed to process this question. "
                        "Check the server log (embedding model download, index files)."}), 500
    return jsonify({"success": True, "question": question, **result})


@bp.route("/status")
def status():
    return jsonify({"success": True, **get_pipeline().status()})
