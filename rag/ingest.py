"""
PDF -> text -> chunks -> embeddings -> vector store.

    python -m rag.ingest                # index every PDF in rag/documents/ (or RAG_DOCS_DIR)

Unchanged PDFs (same SHA-256, same embedding model) are not re-embedded; PDFs that
were removed from the folder are dropped from the index.
"""
import hashlib
import os
import re
import sys
from datetime import datetime, timezone

import numpy as np
from pypdf import PdfReader

from . import config
from .store import VectorStore

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def extract_pages(pdf_path):
    """Returns ([(page_number, text), ...], title_or_None). Raises ValueError for unreadable PDFs."""
    try:
        reader = PdfReader(pdf_path)
        if reader.is_encrypted and not reader.decrypt(""):
            raise ValueError("PDF is password-protected")
        title = None
        if reader.metadata and reader.metadata.title:
            title = str(reader.metadata.title).strip() or None
        pages = []
        for i, page in enumerate(reader.pages, start=1):
            pages.append((i, page.extract_text() or ""))
        return pages, title
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"could not read PDF ({exc})") from exc


def _clean(text):
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)      # re-join words hyphenated across lines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", " ", text)              # PDF line breaks are not sentence breaks
    return text.strip()


def chunk_text(text, max_chars=None, overlap=None, min_chars=None):
    """Sentence-packed chunks of ~max_chars with a sentence-level overlap. Never spans pages."""
    max_chars = max_chars or config.CHUNK_CHARS
    overlap = config.CHUNK_OVERLAP_CHARS if overlap is None else overlap
    min_chars = min_chars or config.MIN_CHUNK_CHARS
    sentences = [s for s in _SENT_SPLIT.split(_clean(text)) if s]
    chunks, current = [], []
    for sent in sentences:
        # a single enormous "sentence" (tables, garbled text) is hard-split
        pieces = [sent[i:i + max_chars] for i in range(0, len(sent), max_chars)] if len(sent) > max_chars else [sent]
        for piece in pieces:
            if current and sum(len(s) + 1 for s in current) + len(piece) > max_chars:
                chunks.append(" ".join(current))
                tail, size = [], 0
                for s in reversed(current):
                    if size >= overlap:
                        break
                    tail.insert(0, s)
                    size += len(s)
                current = tail
            current.append(piece)
    if current:
        chunks.append(" ".join(current))
    return [c for c in chunks if len(c) >= min_chars]


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ingest(docs_dir=None, index_dir=None, embedder=None, log=print):
    docs_dir = docs_dir or config.docs_dir()
    index_dir = index_dir or config.index_dir()
    if embedder is None:
        from .embeddings import get_embedder
        embedder = get_embedder()
    store = VectorStore(index_dir)

    same_model = store.manifest.get("embed_model") == embedder.name
    old_docs = store.manifest["documents"] if same_model else {}
    old_vectors = {}  # doc -> (vectors, chunks) for reuse
    if same_model:
        for doc in old_docs:
            idx = [i for i, c in enumerate(store.chunks) if c["doc"] == doc]
            old_vectors[doc] = (store.vectors[idx], [store.chunks[i] for i in idx])

    pdfs = sorted(f for f in os.listdir(docs_dir) if f.lower().endswith(".pdf")) if os.path.isdir(docs_dir) else []
    all_vectors, all_chunks, documents = [], [], {}
    stats = {"indexed": 0, "reused": 0, "skipped": 0, "removed": [d for d in old_docs if d not in pdfs]}

    for name in pdfs:
        path = os.path.join(docs_dir, name)
        sha = _sha256(path)
        if name in old_docs and old_docs[name]["sha256"] == sha:
            vecs, chunks = old_vectors[name]
            documents[name] = old_docs[name]
            stats["reused"] += 1
            log(f"  reused   {name} ({len(chunks)} chunks)")
        else:
            try:
                pages, title = extract_pages(path)
            except ValueError as exc:
                log(f"  SKIPPED  {name}: {exc}")
                stats["skipped"] += 1
                continue
            chunks = []
            for page_no, text in pages:
                for piece in chunk_text(text):
                    chunks.append({"doc": name, "title": title, "page": page_no, "text": piece})
            if not chunks:
                log(f"  SKIPPED  {name}: no extractable text (scanned PDF? OCR is not supported)")
                stats["skipped"] += 1
                continue
            for n, c in enumerate(chunks):
                c["chunk_id"] = f"{name}#{n}"
            vecs = embedder.embed_passages([c["text"] for c in chunks])
            documents[name] = {
                "sha256": sha, "title": title, "pages": len(pages), "chunks": len(chunks),
                "ingested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            stats["indexed"] += 1
            log(f"  indexed  {name}: {len(pages)} pages, {len(chunks)} chunks")
        all_vectors.append(np.asarray(vecs, dtype=np.float32))
        all_chunks.extend(chunks)

    matrix = np.vstack(all_vectors) if all_vectors else np.zeros((0, 0), dtype=np.float32)
    store.save(matrix, all_chunks, {"embed_model": embedder.name, "documents": documents})
    log(f"Index now holds {len(all_chunks)} chunks from {len(documents)} documents "
        f"({stats['indexed']} new/changed, {stats['reused']} unchanged, "
        f"{len(stats['removed'])} removed, {stats['skipped']} skipped).")
    return stats


if __name__ == "__main__":
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    if not os.path.isdir(config.docs_dir()) or not any(f.lower().endswith(".pdf") for f in os.listdir(config.docs_dir())):
        print(f"No PDFs found in {config.docs_dir()} - add agricultural PDFs there first (see rag/documents/README.md).")
        sys.exit(1)
    ingest()
