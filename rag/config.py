"""RAG settings. Everything is read lazily from environment variables (.env) so tests can override them."""
import os

_BASE = os.path.dirname(os.path.abspath(__file__))


def docs_dir():
    return os.environ.get("RAG_DOCS_DIR", os.path.join(_BASE, "documents"))


def index_dir():
    return os.environ.get("RAG_INDEX_DIR", os.path.join(_BASE, "index"))


def embed_model():
    # ONNX model run locally via fastembed (no PyTorch). ~130 MB, downloaded on first use.
    return os.environ.get("RAG_EMBED_MODEL", "BAAI/bge-small-en-v1.5")


def llm_model():
    return os.environ.get("RAG_LLM_MODEL", "claude-sonnet-5-5")


def anthropic_api_key():
    return os.environ.get("ANTHROPIC_API_KEY")  # only ever read from the environment / .env


def top_k():
    return int(os.environ.get("RAG_TOP_K", "5"))


def min_score():
    # Cosine-similarity floor below which a chunk is considered irrelevant. Embedding models
    # differ in score range, so treat this as a tunable: check it against your own documents.
    return float(os.environ.get("RAG_MIN_SCORE", "0.45"))


CHUNK_CHARS = 900
CHUNK_OVERLAP_CHARS = 150
MIN_CHUNK_CHARS = 40
MAX_QUESTION_CHARS = 1000
