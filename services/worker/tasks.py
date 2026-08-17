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

import os
import hashlib
import logging
import re
import html
import unicodedata
import chromadb
from dataclasses import dataclass, field
from dotenv import load_dotenv
from typing import Any
from collections import Counter

from celery_app import app
from langchain_text_splitters import RecursiveCharacterTextSplitter

log = logging.getLogger(__name__)
load_dotenv()

_CHUNK_SIZE_CHARS = 1400   # ~350 tokens
_CHUNK_OVERLAP_CHARS = 200  # ~50 tokens, ~15% of chunk size

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=_CHUNK_SIZE_CHARS,
    chunk_overlap=_CHUNK_OVERLAP_CHARS,
    separators=["\n\n", "\n", ". ", " ", ""],  # paragraph -> line -> sentence -> word -> char
)

_CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_data")
_COLLECTION_NAME = "athenaeum_chunks"
_EMBEDDING_MODEL = "text-embedding-3-small"

_chroma_client = None
_collection = None

# --- Per-chunk metadata contract -------------------------------------------
# Every chunk carries this so that (a) citations can name the exact document and
# section (mission M6S4) and (b) retrieval can filter later. The embedding model
# name/version is stored so we never silently mix vectors from different models.
@dataclass
class ChunkMetadata:
    doc_id: str            # the job_id: identifies the source document
    user_id:str
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

def _get_collection():
    """Lazily create the persisted ChromaDB client/collection (once per process)."""
    global _chroma_client, _collection
    if _collection is None:
        _chroma_client = chromadb.PersistentClient(path=_CHROMA_PATH)
        _collection = _chroma_client.get_or_create_collection(name=_COLLECTION_NAME)
    return _collection

def is_duplicate(user_id: str, content_hash: str) -> bool:
    """Return True if THIS USER already ingested a document with this content hash.

    Scoped per-user, not global: if two different users upload the same
    document, both get their own copy stored, each tagged with their own
    user_id. A global hash check would incorrectly skip storing it for the
    second user, making it invisible to their queries even though it is
    genuinely theirs. The lookup key is (user_id, content_hash), not
    content_hash alone.
    """
    collection = _get_collection()
    result = collection.get(
        where={"$and": [{"user_id": user_id}, {"content_hash": content_hash}]}
    )
    return len(result["ids"]) > 0

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

def _load_pdf(storage_path: str) -> list[tuple[int | None, str]]:
    """Extract text from a text-based PDF using PyMuPDF, stripping repeated
    headers/footers and page numbers.

    Returns one (page_number, page_text) tuple per page, 1-indexed. Page
    boundaries are preserved (rather than joined into one string) so that
    chunk_text can later attribute each chunk to the page it came from,
    which is needed for citations (M6S4).

    Assumes the PDF has an extractable text layer (not a scanned image). OCR
    fallback for scanned PDFs is explicitly out of v1 scope (deferred to v2).
    """
    import pymupdf

    doc = pymupdf.open(storage_path)
    try:
        pages_text = [page.get_text() for page in doc]
    finally:
        doc.close()

    boilerplate = _detect_boilerplate_lines(pages_text)
    cleaned_pages = [_strip_boilerplate_from_page(p, boilerplate) for p in pages_text]
    return [(i + 1, text) for i, text in enumerate(cleaned_pages)]


def _load_docx(storage_path: str) -> list[tuple[int | None, str]]:
    """Extract text from a Word document using python-docx.

    DOCX has no fixed "page" concept in the file itself (pagination is a
    rendering-time detail in Word) -- returns a single segment with page=None.
    """
    import docx

    document = docx.Document(storage_path)
    paragraphs = [p.text for p in document.paragraphs]
    return [(None, "\n".join(paragraphs))]


def _load_html(storage_path: str) -> list[tuple[int | None, str]]:
    """Strip an HTML file down to its visible text using BeautifulSoup.

    HTML has no pages at all -- returns a single segment with page=None.
    """
    from bs4 import BeautifulSoup

    with open(storage_path, encoding="utf-8", errors="replace") as f:
        soup = BeautifulSoup(f, "html.parser")

    for tag in soup(["script", "style"]):
        tag.decompose()

    return [(None, soup.get_text(separator="\n"))]

# Dispatch table: content_type -> loader function. Adding a new format later
# means writing one _load_x function and adding one line here -- nothing else
# in the pipeline changes.
_LOADERS = {
    "application/pdf": _load_pdf,
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": _load_docx,
    "text/html": _load_html,
}

def load_document(storage_path: str, content_type: str) -> list[tuple[int | None, str]]:
    """Turn a raw file on the shared volume into a list of (page, text) segments.

    Dispatches by content_type to a format-specific loader. PDF returns one
    segment per page (page=1-indexed); DOCX/HTML have no page concept and
    return a single segment with page=None.
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

def clean_document(segments: list[tuple[int | None, str]]) -> list[tuple[int | None, str]]:
    """Apply clean_text to each page segment independently, preserving page tags."""
    return [(page, clean_text(text)) for page, text in segments]

def _build_offset_map(segments: list[tuple[int | None, str]]):
    """Concatenate segment texts; return (full_text, ranges) where ranges is
    a list of (start_offset, end_offset, page) for mapping a chunk's position
    back to the page it came from."""
    parts, ranges, offset = [], [], 0
    for page, text in segments:
        start = offset
        parts.append(text)
        offset += len(text)
        ranges.append((start, offset, page))
        parts.append("\n")
        offset += 1
    return "".join(parts), ranges

def _page_for_offset(ranges, idx: int) -> int | None:
    candidate = None
    for start, end, page in ranges:
        if start <= idx:
            candidate = page
        if idx < end:
            return page
    return candidate

def chunk_text(segments: list[tuple[int | None, str]]) -> list[tuple[str, int | None]]:
    """Split cleaned, page-tagged segments into chunks using recursive
    character splitting, attributing each chunk to the page it starts on.

    See the module-level comment above _CHUNK_SIZE_CHARS for the reasoning
    behind the splitting strategy, size, and overlap (mission M6S1) -- that
    decision is unchanged; this only adds page attribution on top of it.

    A chunk spanning a page boundary is attributed to whichever page it
    STARTS on -- a reasonable approximation for citation purposes, since the
    alternative (tracking every page a chunk touches) adds complexity for
    marginal citation-accuracy benefit at chunk sizes this small relative to
    a typical page.
    """
    full_text, ranges = _build_offset_map(segments)
    chunks = _splitter.split_text(full_text)

    result = []
    search_from = 0
    for chunk in chunks:
        start_search = max(0, search_from - _CHUNK_OVERLAP_CHARS - 50)
        idx = full_text.find(chunk, start_search)
        if idx == -1:
            idx = full_text.find(chunk)  # fallback: search from the beginning
        page = _page_for_offset(ranges, idx if idx != -1 else search_from)
        result.append((chunk, page))
        if idx != -1:
            search_from = idx + len(chunk) - _CHUNK_OVERLAP_CHARS

    return result

def build_chunk_metadata(
    chunks_with_pages: list[tuple[str, int | None]],
    job: dict,
    content_hash: str,
    embedding_model: str | None = None,
) -> list[Chunk]:
    """Attach the metadata contract to each chunk, including the page it
    starts on (page=None for DOCX/HTML, which have no page concept).
    """
    enriched: list[Chunk] = []
    for i, (text, page) in enumerate(chunks_with_pages):
        meta = ChunkMetadata(
            doc_id=job["job_id"],
            user_id=job["user_id"],
            original_filename=job["original_filename"],
            content_type=job["content_type"],
            content_hash=content_hash,
            chunk_index=i,
            page=page,
            embedding_model=embedding_model,
        )
        enriched.append(Chunk(text=text, metadata=meta))
    return enriched

def _embed_batch(texts: list[str]) -> list[list[float]]:
    """Call the OpenAI embeddings API for a batch of chunk texts.

    Batching in one call (rather than one call per chunk) is both faster and
    cheaper than per-chunk calls.
    """
    from openai import OpenAI

    client = OpenAI()  # reads OPENAI_API_KEY from the environment
    response = client.embeddings.create(model=_EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]

def embed_and_store(chunks: list[Chunk]) -> None:
    """Embed each chunk and upsert into the persisted ChromaDB collection.

    Every chunk's metadata (including user_id) is stored alongside its vector,
    which is what makes per-user retrieval filtering and dedup possible later.
    The embedding_model name/version is stamped into each chunk's metadata so
    a future model change can never silently mix incompatible vectors in the
    same collection without it being visible in the data itself.
    """
    if not chunks:
        return

    texts = [c.text for c in chunks]
    embeddings = _embed_batch(texts)

    collection = _get_collection()
    ids = [f"{c.metadata.doc_id}_{c.metadata.chunk_index}" for c in chunks]
    metadatas = []
    for c in chunks:
        c.metadata.embedding_model = _EMBEDDING_MODEL
        meta = {k: v for k, v in c.metadata.__dict__.items() if k != "extra"}
        meta.update(c.metadata.extra)
        # Chroma metadata values must be str/int/float/bool -- drop any None values.
        metadatas.append({k: v for k, v in meta.items() if v is not None})

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )

# Orchestration (the Celery task) 
@app.task(name="tasks.ingest_document", bind=True, max_retries=3)
def ingest_document(self, job: dict) -> dict:
    """Run the full ingestion pipeline for one document.

    `job` is the contract from shared/schemas/ingestion_job.schema.json.
    """
    job_id = job["job_id"]
    log.info("ingest start job=%s file=%s", job_id, job["original_filename"])

    segments = load_document(job["storage_path"], job["content_type"])
    cleaned_segments = clean_document(segments)

    # Hash over the full joined cleaned text -- dedup behavior is unchanged,
    # it just now assembles the text from segments instead of one flat string.
    full_cleaned_text = "\n".join(text for _, text in cleaned_segments)
    content_hash = compute_content_hash(full_cleaned_text)

    if is_duplicate(job["user_id"], content_hash):
        log.info("ingest skip (duplicate) job=%s hash=%s", job_id, content_hash[:12])
        return {"job_id": job_id, "status": "skipped_duplicate", "content_hash": content_hash}

    chunks_with_pages = chunk_text(cleaned_segments)
    enriched = build_chunk_metadata(chunks_with_pages, job, content_hash)
    embed_and_store(enriched)

    log.info("ingest done job=%s chunks=%d", job_id, len(enriched))
    return {"job_id": job_id, "status": "ingested", "chunks": len(enriched), "content_hash": content_hash}