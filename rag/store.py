"""
Minimal persistent vector store: a float32 matrix (vectors.npy) + chunk metadata (chunks.json)
+ a manifest of indexed documents (manifest.json). Search is exact brute-force cosine similarity
(vectors are pre-normalised), which is fast and dependency-free for the few thousand chunks a
set of agricultural PDFs produces. If the corpus ever outgrows this, swap this class for
Chroma/FAISS - nothing else in rag/ depends on how vectors are stored.
"""
import json
import os

import numpy as np


class VectorStore:
    def __init__(self, index_dir):
        self.index_dir = index_dir
        self.vectors = np.zeros((0, 0), dtype=np.float32)
        self.chunks = []      # [{"chunk_id","doc","page","text", ...}]
        self.manifest = {"embed_model": None, "documents": {}}
        self._mtime = None
        self.load()

    def _p(self, name):
        return os.path.join(self.index_dir, name)

    def _stamp(self):
        try:
            return os.path.getmtime(self._p("manifest.json"))
        except OSError:
            return None

    def load(self):
        try:
            self.vectors = np.load(self._p("vectors.npy"))
            with open(self._p("chunks.json"), encoding="utf-8") as f:
                self.chunks = json.load(f)
            with open(self._p("manifest.json"), encoding="utf-8") as f:
                self.manifest = json.load(f)
            if len(self.chunks) != len(self.vectors):
                raise ValueError("index files out of sync")
        except (OSError, ValueError):
            self.vectors, self.chunks = np.zeros((0, 0), dtype=np.float32), []
            self.manifest = {"embed_model": None, "documents": {}}
        self._mtime = self._stamp()

    def refresh_if_changed(self):
        """Pick up a re-run of the ingest command without restarting the server."""
        if self._stamp() != self._mtime:
            self.load()

    def save(self, vectors, chunks, manifest):
        os.makedirs(self.index_dir, exist_ok=True)
        np.save(self._p("vectors.npy"), np.asarray(vectors, dtype=np.float32))
        with open(self._p("chunks.json"), "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False)
        with open(self._p("manifest.json"), "w", encoding="utf-8") as f:  # written last: it is the change stamp
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        self.load()

    def __len__(self):
        return len(self.chunks)

    def search(self, query_vec, k):
        """Top-k chunks as [(score, chunk)], best first."""
        if not len(self.chunks):
            return []
        scores = self.vectors @ np.asarray(query_vec, dtype=np.float32)
        top = np.argsort(scores)[::-1][:k]
        return [(float(scores[i]), self.chunks[i]) for i in top]
