# Agricultural documents (PDF)

Put the PDFs you want the assistant to answer from in this folder, then run
`python -m rag.ingest`. Nothing is bundled: the assistant answers **only** from the
documents you add, and says so when they don't cover a question.

Choose authoritative, citable sources, for example (verify availability and licence yourself):
- ICAR / State Agricultural University "Package of Practices" for your crops and state
- FAO crop and soil management publications
- Government of India Ministry of Agriculture & Farmers Welfare advisories / Soil Health Card guides
- Extension bulletins from your state agriculture department

Text-based PDFs only - scanned images are skipped (OCR isn't supported). Re-run ingest after adding,
changing or removing files; unchanged files are not re-embedded.
