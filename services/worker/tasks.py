"""Ingestion pipeline (Celery task).

This is the Python worker's job description. It runs the stages we designed in
Phase 1:

    load -> clean -> hash/dedup -> chunk -> enrich metadata -> embed & store

DESIGN NOTE ON WHAT IS AND ISN'T IMPLEMENTED
--------------------------------------------
The orchestration and the stages whose design we have already locked are fully
implemented:
    - compute_content_hash  (SHA-256 on CLEANED text -- decided)
    - the dedup gate
    - build_chunk_metadata  (the metadata contract per chunk)

The stages that still depend on an open board decision are deliberately left as
stubs that raise NotImplementedError, so nothing is silently hardcoded:
    - load_document   -> loader library choice is not locked
    - clean_text      -> boilerplate/normalisation approach not locked
    - chunk_text      -> chunk size + overlap + strategy = M6S1, must be chosen
                         and defended to the mentor; NOT hardcoding it here
    - embed_and_store -> embedding model + ChromaDB collection details not locked

Fill these in stage by stage as each decision is made.
"""

import hashlib
import logging
import re
import html
import unicodedata
from dataclasses import dataclass, field
from typing import Any
from collections import Counter

from celery_app import app

log = logging.getLogger(__name__)


# --- Per-chunk metadata contract -------------------------------------------
# Every chunk carries this so that (a) citations can name the exact document and
# section (mission M6S4) and (b) retrieval can filter later. The embedding model
# name/version is stored so we never silently mix vectors from different models.
@dataclass
class ChunkMetadata:
    doc_id: str            # the job_id: identifies the source document
    original_filename: str
    content_type: str
    content_hash: str      # SHA-256 of the cleaned document text
    chunk_index: int       # position of this chunk within the document
    section: str | None = None      # section/heading, if the loader recovered it
    page: int | None = None         # page number, if applicable
    extraction_method: str | None = None  # e.g. "pdf-text", "ocr" -- feeds confidence later
    embedding_model: str | None = None    # model name+version used for this vector
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    text: str
    metadata: ChunkMetadata


# --- DECIDED: content hashing ----------------------------------------------
def compute_content_hash(cleaned_text: str) -> str:
    """SHA-256 over the CLEANED text.

    Hashing cleaned text (not raw bytes) means the same document uploaded as a
    PDF and again as a DOCX dedupes correctly, because both normalise to the
    same text. This is the dedup key.
    """
    return hashlib.sha256(cleaned_text.encode("utf-8")).hexdigest()


def is_duplicate(content_hash: str) -> bool:
    """Return True if a document with this content hash was already ingested.

    STUB (storage-backed): the lookup itself is trivial, but WHERE we record
    seen hashes depends on the vector-store decision (a ChromaDB metadata query
    vs. a small side index). Wire this up together with embed_and_store.
    """
    raise NotImplementedError(
        "is_duplicate: depends on the vector-store decision -- implement "
        "alongside embed_and_store (check for existing content_hash)."
    )


# --- OPEN DECISION: loading -------------------------------------------------
def _detect_boilerplate_lines(pages_text: list[str], threshold: float = 0.6) -> set[str]:
    """Return lines that appear on at least `threshold` fraction of pages.

    A line repeating across most pages (a company footer, a running header)
    is almost certainly boilerplate, not real content. Needs at least 2 pages
    to detect repetition at all.
    """
    if len(pages_text) < 2:
        return set()
    line_page_count = Counter()
    for page_text in pages_text:
        lines = set(l.strip() for l in page_text.split("\n") if l.strip())
        for line in lines:
            line_page_count[line] += 1
    num_pages = len(pages_text)
    return {line for line, count in line_page_count.items() if count / num_pages >= threshold}


_PAGE_NUMBER_PATTERNS = [
    re.compile(r"^page\s+\d+(\s+of\s+\d+)?$", re.IGNORECASE),  # "Page 3", "Page 3 of 47"
    re.compile(r"^\d{1,4}$"),                                    # a lone number on its own line
    re.compile(r"^-\s*\d{1,4}\s*-$"),                            # "- 3 -"
]


def _looks_like_page_number(line: str) -> bool:
    stripped = line.strip()
    return any(p.match(stripped) for p in _PAGE_NUMBER_PATTERNS)


def _strip_boilerplate_from_page(page_text: str, boilerplate_lines: set[str]) -> str:
    kept = []
    for line in page_text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped in boilerplate_lines:
            continue
        if _looks_like_page_number(stripped):
            continue
        kept.append(line)
    return "\n".join(kept)


def _load_pdf(storage_path: str) -> str:
    """Extract text from a text-based PDF using PyMuPDF, stripping repeated
    headers/footers and page numbers.

    Repetition is detected across pages (a line appearing on most pages is
    almost certainly boilerplate), which is why extraction happens per-page
    before being joined into one string -- this detection is impossible once
    pages are flattened into a single blob.

    Assumes the PDF has an extractable text layer (not a scanned image). OCR
    fallback for scanned PDFs is explicitly out of v1 scope (deferred to v2).
    """
    import pymupdf  # the `fitz` alias is deprecated as of recent PyMuPDF releases

    doc = pymupdf.open(storage_path)
    try:
        pages_text = [page.get_text() for page in doc]
    finally:
        doc.close()

    boilerplate = _detect_boilerplate_lines(pages_text)
    cleaned_pages = [_strip_boilerplate_from_page(p, boilerplate) for p in pages_text]
    return "\n".join(cleaned_pages)


def _load_docx(storage_path: str) -> str:
    """Extract text from a Word document using python-docx.

    Reads paragraph text in document order. Tables inside the docx are not
    specially handled in v1 -- their cell text is not extracted by this path.
    """
    import docx  # python-docx

    document = docx.Document(storage_path)
    paragraphs = [p.text for p in document.paragraphs]
    return "\n".join(paragraphs)


def _load_html(storage_path: str) -> str:
    """Strip an HTML file down to its visible text using BeautifulSoup.

    Drops script/style tags entirely (not visible content, would pollute
    chunks) before extracting text.
    """
    from bs4 import BeautifulSoup

    with open(storage_path, encoding="utf-8", errors="replace") as f:
        soup = BeautifulSoup(f, "html.parser")

    for tag in soup(["script", "style"]):
        tag.decompose()

    return soup.get_text(separator="\n")


# Dispatch table: content_type -> loader function. Adding a new format later
# means writing one _load_x function and adding one line here -- nothing else
# in the pipeline changes.
_LOADERS = {
    "application/pdf": _load_pdf,
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": _load_docx,
    "text/html": _load_html,
}


def load_document(storage_path: str, content_type: str) -> str:
    """Turn a raw file on the shared volume into plain text.

    Dispatches by content_type to a format-specific loader. v1 supports PDF
    (text-based), DOCX, and HTML. CSV is accepted at the Go gateway's intake
    for future use but intentionally not handled here yet -- see v2 scope.
    Scanned/OCR PDFs are also deferred to v2.
    """
    if content_type == "text/csv":
        raise NotImplementedError(
            "load_document: CSV loading is deferred to v2 (tabular data needs "
            "its own chunking strategy, not plain-text chunking)."
        )

    loader = _LOADERS.get(content_type)
    if loader is None:
        raise ValueError(f"load_document: unsupported content_type {content_type!r}")

    return loader(storage_path)


# --- OPEN DECISION: cleaning ------------------------------------------------
_ENCODING_REPLACEMENTS = {
    "\u2018": "'", "\u2019": "'",   # smart single quotes -> straight quote
    "\u201c": '"', "\u201d": '"',  # smart double quotes -> straight quote
    "\u2013": "-", "\u2014": "-",  # en-dash, em-dash -> hyphen
    "\u00a0": " ",                  # non-breaking space -> regular space
    "\u2026": "...",                 # ellipsis character -> three dots
}


def clean_text(raw_text: str) -> str:
    """Universal mechanical cleaning applied after any loader (PDF/DOCX/HTML).

    Format-specific cleaning (PDF header/footer stripping) already happened
    inside the loader, where page structure still existed. This step is
    safe and useful regardless of source format:
      - unescape leftover HTML entities (e.g. &nbsp; that survived a loader)
      - normalise smart quotes/dashes/nbsp/ellipsis to plain ASCII equivalents
      - normalise whitespace (collapse runs of spaces/tabs, collapse 3+ blank
        lines to one, strip trailing spaces per line)
    """
    text = html.unescape(raw_text)

    for bad, good in _ENCODING_REPLACEMENTS.items():
        text = text.replace(bad, good)
    text = unicodedata.normalize("NFKC", text)

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    return text.strip()

# --- OPEN DECISION: chunking (mission M6S1) --------------------------------
def chunk_text(cleaned_text: str) -> list[str]:
    """Split cleaned text into chunks.

    OPEN DECISION -- this is mission item M6S1 ("chunk size is intentional, not
    default") and must be chosen deliberately and defended to the mentor. Size,
    overlap, and splitter strategy (recursive vs semantic vs parent-child) are
    NOT hardcoded here on purpose.
    """
    raise NotImplementedError(
        "chunk_text: chunk size/overlap/strategy is M6S1 -- decide and defend "
        "before implementing."
    )


# --- DECIDED: metadata enrichment ------------------------------------------
def build_chunk_metadata(
    chunks: list[str],
    job: dict,
    content_hash: str,
    embedding_model: str | None = None,
) -> list[Chunk]:
    """Attach the metadata contract to each chunk.

    Section/page/extraction_method are filled with what the loader recovers;
    they stay None until load_document is implemented and reports them.
    """
    enriched: list[Chunk] = []
    for i, text in enumerate(chunks):
        meta = ChunkMetadata(
            doc_id=job["job_id"],
            original_filename=job["original_filename"],
            content_type=job["content_type"],
            content_hash=content_hash,
            chunk_index=i,
            embedding_model=embedding_model,
        )
        enriched.append(Chunk(text=text, metadata=meta))
    return enriched


# --- OPEN DECISION: embedding + storage ------------------------------------
def embed_and_store(chunks: list[Chunk]) -> None:
    """Embed each chunk and upsert into the persisted vector store.

    OPEN DECISION: embedding model (e.g. OpenAI text-embedding-3 vs a local
    sentence-transformers model) and the ChromaDB collection/persistence layout.
    Persistence itself is required (mission M6S2), the model choice is not locked.
    """
    raise NotImplementedError(
        "embed_and_store: embedding model + ChromaDB collection details not "
        "locked (persistence is required by M6S2)."
    )


# --- Orchestration (the Celery task) ---------------------------------------
@app.task(name="tasks.ingest_document", bind=True, max_retries=3)
def ingest_document(self, job: dict) -> dict:
    """Run the full ingestion pipeline for one document.

    `job` is the contract from shared/schemas/ingestion_job.schema.json.
    """
    job_id = job["job_id"]
    log.info("ingest start job=%s file=%s", job_id, job["original_filename"])

    raw = load_document(job["storage_path"], job["content_type"])
    cleaned = clean_text(raw)

    content_hash = compute_content_hash(cleaned)
    if is_duplicate(content_hash):
        log.info("ingest skip (duplicate) job=%s hash=%s", job_id, content_hash[:12])
        return {"job_id": job_id, "status": "skipped_duplicate", "content_hash": content_hash}

    chunks = chunk_text(cleaned)
    enriched = build_chunk_metadata(chunks, job, content_hash)
    embed_and_store(enriched)

    log.info("ingest done job=%s chunks=%d", job_id, len(enriched))
    return {"job_id": job_id, "status": "ingested", "chunks": len(enriched), "content_hash": content_hash}