"""
RAG pipeline tests. Run from project root:  python -m unittest discover -s tests -v

TEST DOUBLES (not part of the app): a deterministic hashing "embedder" and a stub LLM, so the
pipeline can be exercised offline. PDF content is generated test text, not agricultural guidance.
"""
import os
import re
import sys
import tempfile
import unittest
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag import config  # noqa: E402
from rag.ingest import chunk_text, extract_pages, ingest  # noqa: E402
from rag.llm import LLMError  # noqa: E402
from rag.pipeline import INSUFFICIENT, RagError, RagPipeline  # noqa: E402
from rag.store import VectorStore  # noqa: E402


class HashEmbedder:
    """Bag-of-words hashing embedder: texts sharing words get high cosine similarity."""
    name = "test-hash-embedder"
    calls = 0

    def _vec(self, text):
        v = np.zeros(512, dtype=np.float32)
        for w in re.findall(r"[a-z]{3,}", text.lower()):
            v[zlib.crc32(w.encode()) % 512] += 1
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_passages(self, texts):
        HashEmbedder.calls += len(texts)
        return np.vstack([self._vec(t) for t in texts])

    def embed_query(self, text):
        return self._vec(text)


class StubLLM:
    def __init__(self, reply="Sow in June [1].", available=True, error=False):
        self.reply, self._available, self.error, self.prompts = reply, available, error, []

    def available(self):
        return self._available

    def generate(self, system, user):
        self.prompts.append((system, user))
        if self.error:
            raise LLMError("boom")
        return self.reply


def make_pdf(path, pages):
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    c = canvas.Canvas(path, pagesize=A4)
    c.setTitle("Test Document")
    for text in pages:
        y = 800
        for line in text.split("\n"):
            c.drawString(40, y, line)
            y -= 16
        c.showPage()
    c.save()


RICE = ["Transplanting of rice seedlings should be done when seedlings are twenty five days old. "
        "Maintain a shallow water layer in the field during the early growth stage of the rice crop.",
        "Nitrogen for rice should be applied in three split doses during the vegetative growth stage."]
WHEAT = ["Wheat is sown in the winter season and needs cool temperatures for good tillering. "
         "Irrigation of wheat at the crown root initiation stage improves wheat grain yield."]


class IngestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.docs, self.idx = os.path.join(self.tmp.name, "docs"), os.path.join(self.tmp.name, "idx")
        os.makedirs(self.docs)
        make_pdf(os.path.join(self.docs, "rice.pdf"), RICE)
        make_pdf(os.path.join(self.docs, "wheat.pdf"), WHEAT)
        self.log = []

    def tearDown(self):
        self.tmp.cleanup()

    def run_ingest(self):
        return ingest(self.docs, self.idx, HashEmbedder(), log=self.log.append)

    def test_extract_pages_and_title(self):
        pages, title = extract_pages(os.path.join(self.docs, "rice.pdf"))
        self.assertEqual([p for p, _ in pages], [1, 2])
        self.assertIn("rice seedlings", pages[0][1])
        self.assertEqual(title, "Test Document")

    def test_chunking_respects_size_and_overlap(self):
        sents = [f"Sentence number {i} talks about irrigation of the crop in some detail." for i in range(40)]
        chunks = chunk_text(" ".join(sents), max_chars=300, overlap=80)
        self.assertGreater(len(chunks), 3)
        self.assertTrue(all(len(c) <= 300 for c in chunks))
        last_sentence_of_first = re.findall(r"Sentence number \d+", chunks[0])[-1]
        self.assertIn(last_sentence_of_first, chunks[1])  # overlap: previous chunk's tail is repeated
        self.assertEqual(chunk_text("tiny"), [])

    def test_index_metadata_and_incremental(self):
        stats = self.run_ingest()
        self.assertEqual((stats["indexed"], stats["reused"]), (2, 0))
        store = VectorStore(self.idx)
        self.assertEqual({c["doc"] for c in store.chunks}, {"rice.pdf", "wheat.pdf"})
        self.assertEqual({c["page"] for c in store.chunks if c["doc"] == "rice.pdf"}, {1, 2})
        self.assertIn("sha256", store.manifest["documents"]["rice.pdf"])

        HashEmbedder.calls = 0
        stats = self.run_ingest()                       # nothing changed -> nothing re-embedded
        self.assertEqual((stats["indexed"], stats["reused"]), (0, 2))
        self.assertEqual(HashEmbedder.calls, 0)
        self.assertEqual(len(VectorStore(self.idx)), len(store))

        make_pdf(os.path.join(self.docs, "wheat.pdf"), WHEAT + ["Extra page about wheat harvest timing and storage."])
        stats = self.run_ingest()                       # only the changed file is re-embedded
        self.assertEqual((stats["indexed"], stats["reused"]), (1, 1))

        os.remove(os.path.join(self.docs, "rice.pdf"))
        stats = self.run_ingest()
        self.assertEqual(stats["removed"], ["rice.pdf"])
        self.assertEqual({c["doc"] for c in VectorStore(self.idx).chunks}, {"wheat.pdf"})

    def test_unreadable_and_textless_pdfs_skipped(self):
        open(os.path.join(self.docs, "broken.pdf"), "wb").write(b"not a pdf")
        make_pdf(os.path.join(self.docs, "blank.pdf"), [""])
        stats = self.run_ingest()
        self.assertEqual(stats["skipped"], 2)
        self.assertEqual(stats["indexed"], 2)


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        docs, cls.idx = os.path.join(cls.tmp.name, "docs"), os.path.join(cls.tmp.name, "idx")
        os.makedirs(docs)
        make_pdf(os.path.join(docs, "rice.pdf"), RICE)
        make_pdf(os.path.join(docs, "wheat.pdf"), WHEAT)
        ingest(docs, cls.idx, HashEmbedder(), log=lambda *_: None)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def pipe(self, llm):
        os.environ["RAG_MIN_SCORE"] = "0.2"
        return RagPipeline(embedder=HashEmbedder(), llm=llm, store=VectorStore(self.idx))

    def test_retrieves_right_document_and_answers_from_context_only(self):
        llm = StubLLM(reply="Transplant when seedlings are about twenty five days old [1].")
        out = self.pipe(llm).ask("When should rice seedlings be transplanted?")
        self.assertTrue(out["answerable"])
        self.assertEqual(out["sources"][0]["document"], "rice.pdf")
        self.assertEqual(out["sources"][0]["page"], 1)
        self.assertTrue(out["retrieved_context"][0]["used_in_answer"])
        system, user = llm.prompts[0]
        self.assertIn("ONLY", system)
        self.assertIn("twenty five days old", user)       # retrieved text is what the LLM sees
        self.assertIn("(source: rice.pdf, page 1)", user)  # passages are labelled with their source
        self.assertTrue(user.rstrip().endswith("When should rice seedlings be transplanted?"))
        self.assertTrue(out["citations_found"])

    def test_irrelevant_question_never_calls_llm(self):
        llm = StubLLM()
        out = self.pipe(llm).ask("Who won the cricket world cup in nineteen eighty three?")
        self.assertFalse(out["answerable"])
        self.assertEqual(out["answer"], INSUFFICIENT)
        self.assertEqual(llm.prompts, [])
        self.assertEqual(out["sources"], [])

    def test_llm_can_decline(self):
        out = self.pipe(StubLLM(reply="INSUFFICIENT_CONTEXT")).ask("rice transplanting seedlings water")
        self.assertFalse(out["answerable"])
        self.assertEqual(out["answer"], INSUFFICIENT)

    def test_no_citations_falls_back_to_all_used_passages(self):
        out = self.pipe(StubLLM(reply="Transplant at about 25 days.")).ask("rice seedlings transplanted")
        self.assertFalse(out["citations_found"])
        self.assertTrue(out["sources"])

    def test_missing_key_gives_503_with_retrieved_context(self):
        with self.assertRaises(RagError) as cm:
            self.pipe(StubLLM(available=False)).ask("rice seedlings transplanted")
        self.assertEqual(cm.exception.status, 503)
        self.assertTrue(cm.exception.retrieved)

    def test_llm_failure_gives_502(self):
        with self.assertRaises(RagError) as cm:
            self.pipe(StubLLM(error=True)).ask("rice seedlings transplanted")
        self.assertEqual(cm.exception.status, 502)

    def test_empty_index_message(self):
        empty = RagPipeline(embedder=HashEmbedder(), llm=StubLLM(), store=VectorStore(os.path.join(self.tmp.name, "none")))
        out = empty.ask("anything about rice")
        self.assertFalse(out["answerable"])
        self.assertIn("No documents are indexed", out["answer"])

    def test_embed_model_mismatch_is_an_error(self):
        class Other(HashEmbedder):
            name = "other-model"
        p = RagPipeline(embedder=Other(), llm=StubLLM(), store=VectorStore(self.idx))
        with self.assertRaises(RagError) as cm:
            p.ask("rice seedlings")
        self.assertEqual(cm.exception.status, 409)


class ApiTests(unittest.TestCase):
    def setUp(self):
        import rag.api as api
        from flask import Flask
        self.api = api
        app = Flask(__name__)
        app.register_blueprint(api.bp)
        self.client = app.test_client()
        self._old = api._pipeline

    def tearDown(self):
        self.api._pipeline = self._old

    def test_validation(self):
        self.assertEqual(self.client.post("/api/agriculture/ask", json={}).status_code, 400)
        self.assertEqual(self.client.post("/api/agriculture/ask", json={"question": "hi"}).status_code, 400)
        self.assertEqual(self.client.post("/api/agriculture/ask", json={"question": "x" * 2000}).status_code, 400)
        self.assertEqual(self.client.post("/api/agriculture/ask", data="nope").status_code, 400)

    def test_success_and_error_shapes(self):
        class P:
            def ask(self, q):
                return {"answer": "A [1]", "answerable": True, "sources": [{"document": "d.pdf", "page": 1}],
                        "retrieved_context": []}
        self.api._pipeline = P()
        body = self.client.post("/api/agriculture/ask", json={"question": "what is rice?"}).get_json()
        self.assertTrue(body["success"])
        self.assertEqual({"answer", "sources", "retrieved_context"} - set(body), set())

        class Bad:
            def ask(self, q):
                raise RagError("no key", status=503, retrieved=[{"document": "d.pdf"}])
        self.api._pipeline = Bad()
        r = self.client.post("/api/agriculture/ask", json={"question": "what is rice?"})
        self.assertEqual(r.status_code, 503)
        self.assertEqual(r.get_json()["retrieved_context"][0]["document"], "d.pdf")

    def test_no_key_hardcoded_in_source(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for fn in os.listdir(os.path.join(root, "rag")):
            if fn.endswith(".py"):
                self.assertNotRegex(open(os.path.join(root, "rag", fn), encoding="utf-8").read(), r"sk-ant-[A-Za-z0-9]")


if __name__ == "__main__":
    unittest.main()
