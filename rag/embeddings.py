"""Local embedding model (fastembed / ONNX). Returns L2-normalised float32 vectors, so cosine == dot product."""
import threading

import numpy as np

from . import config


def _normalise(mat):
    mat = np.asarray(mat, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.maximum(norms, 1e-12)


class FastEmbedder:
    def __init__(self, model_name=None):
        self.name = model_name or config.embed_model()
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            if self._model is None:
                from fastembed import TextEmbedding  # imported lazily: heavy, and only needed for RAG
                self._model = TextEmbedding(model_name=self.name)
        return self._model

    def embed_passages(self, texts):
        return _normalise(list(self._load().embed(list(texts))))

    def embed_query(self, text):
        # query_embed adds the model's query instruction prefix where the model defines one (e.g. BGE).
        return _normalise(list(self._load().query_embed([text])))[0]


_default = None


def get_embedder():
    global _default
    if _default is None:
        _default = FastEmbedder()
    return _default
