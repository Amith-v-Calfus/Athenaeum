# Athenaeum

A production-shaped, multi-tenant RAG (Retrieval-Augmented Generation) document Q&A system, built for Calfus Odyssey — Port 06, "The Oracle's Trial."

Upload internal documents (HR policies, SOPs, manuals, reference material) and ask questions in plain English. Answers are grounded in the uploaded documents, cited by source and page, and the system explicitly refuses to answer when it doesn't have the information — rather than guessing.

---

## Why this exists

Most RAG demos stop at "upload a PDF, embed it, retrieve a few chunks, get an answer." That's a prototype, not a system. Athenaeum is built to be defensible under questioning: every architectural decision below has a stated reason, a documented tradeoff, and — where it mattered — a real test proving it works, not just a claim that it does.

The two things this project treats as the actual hard problems, rather than afterthoughts:

1. **Handling documents at scale, correctly.** Different formats need different extraction. Real PDFs have repeated headers/footers, multi-column layouts, and inconsistent structure. A production system has to survive that, not just work on one clean example file.
2. **Knowing when the system is wrong.** The single most important property of a RAG system isn't "can it answer" — it's "does it know when it can't." A system that confidently hallucinates is worse than one that says "I don't know."

---

## Architecture

```
                                    INGESTION (async)
Upload ──▶ Go gateway ──▶ Redis queue ──▶ bridge.py ──▶ Celery worker
(file)     (validate,      (job          (adapts Go's    (tasks.py:
            save, queue)    metadata)     queue to        load → clean →
                                          Celery)          hash/dedup →
                                                            chunk → embed)
                                                                │
                                                                ▼
                                                            Weaviate
                                                      (hybrid dense+BM25
                                                       vector store,
                                                       per-user isolated)
                                                                │
                                    RETRIEVAL (sync)            │
Question ──▶ Streamlit ──▶ FastAPI ──▶ retrieval.py ◀───────────┘
              (UI)          (/query)   (hybrid search →
                                        rerank → assemble →
                                        generate w/ citations
                                        + refusal)
```

**Why two separate pipelines:** ingestion is an asynchronous background job triggered by a file upload; retrieval is a synchronous request triggered by a live question. Different trigger, different lifecycle — they're deliberately built as separate services (`services/worker` vs `services/api`), not bolted onto one file.

**Why Go + Python:** Go handles only the ingestion *intake* — accepting uploads, validating them, queueing a job — because that's the one place high-concurrency matters (many simultaneous uploads) and where Go's goroutines genuinely outperform Python. Everything else (parsing, chunking, embedding, retrieval, generation) is Python, because that's where the entire ML/RAG ecosystem lives. Go never parses a file's contents; it only ever touches file-level metadata. See `learnings/python_go.md` for the full reasoning, including why this wasn't built as a premature optimization.

---

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Ingestion gateway | Go (`net/http`, `go-redis`) | High-concurrency file intake; see above |
| Task queue | Redis + Celery | Async, retryable, horizontally scalable ingestion |
| Document loaders | PyMuPDF (PDF), python-docx (DOCX), BeautifulSoup (HTML) | Format-specific libraries chosen over `unstructured` for predictability and defensibility — see `learnings/` |
| Chunking | LangChain `RecursiveCharacterTextSplitter`, 1400 chars (~350 tokens), 200 char (~15%) overlap | Recursive splitting respects paragraph/sentence structure at near-zero cost; matches our document types (structured policies/FAQs, not free-flowing narrative). See M6S1 defense below. |
| Embeddings | OpenAI `text-embedding-3-small` | Strong general-English performance at low cost; `large` and multilingual (BGE-M3) models weren't justified by this corpus's domain or language |
| Vector store | Weaviate (self-hosted, Docker) | Native hybrid (dense + BM25) search — see "Why Weaviate, not ChromaDB" below |
| Reranker | Local cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) | No external API dependency; runs offline once cached |
| Generation | OpenAI `gpt-4o-mini` | Fast, cheap, strong enough for grounded Q&A; same provider as embeddings |
| Retrieval API | FastAPI | Wraps the retrieval pipeline as an HTTP service; preloads the reranker at startup |
| UI | Streamlit | Upload, chat-style Q&A, visible source citations |
| Containerization | Docker Compose | All 6 services (Redis, Weaviate, gateway, worker, bridge, api) start with one command |

---

## Design decisions worth defending

### Why Weaviate, not ChromaDB (or FAISS)

Started on ChromaDB (simple, embedded, persisted, native metadata filtering). Switched to Weaviate, with mentor approval, specifically to get **native hybrid dense+BM25 search** — ChromaDB is dense-vector-only, and dense embeddings alone miss exact keyword/ID matches (product codes, email addresses, specific policy terms) that BM25 catches directly. Hybrid retrieval combining both outperforms either alone.

FAISS was ruled out separately: it's a similarity-search *library*, not a database — no built-in persistence, no metadata storage, no filtering. Building those ourselves would mean reimplementing what a real vector database already provides.

**Cost of the switch:** Weaviate runs as its own server process (Docker), versus ChromaDB's embedded, no-server model — one more piece of infrastructure. Considered worth it for the retrieval-quality gain.

### Why chunk size = 1400 chars / 200 char overlap, recursive splitting (M6S1)

- **Recursive over semantic chunking:** our documents (HR FAQs, SOPs, architecture docs) already have clear paragraph/heading structure. Recursive splitting respects that structure directly, at effectively zero ingestion cost. Semantic chunking (embedding-similarity-based topic breaks) earns its extra compute cost mainly on unstructured narrative text — not what we're dealing with.
- **Size ~1400 characters (~350 tokens):** measured in characters rather than tokens deliberately — token-accurate sizing via `tiktoken` needs a network download of its encoding file on first use, an avoidable runtime dependency risk on demo day. ~4 characters per token is a standard English-text approximation.
- **Overlap ~200 characters (~15%):** preserves context across chunk boundaries so a sentence split between two chunks isn't orphaned.
- **Known limitation:** a fixed chunk size will occasionally cut mid-sentence when a paragraph exceeds the chunk size, and recursive splitting has no special handling for code blocks (verified against a code-heavy research paper — chunks can cut mid-function). See `learnings/limitations.md`.

### Multi-tenant isolation

Every document belongs to a `user_id`. Isolation is enforced by **mandatory metadata filtering on every read**, not physically separate storage per user — one shared Weaviate collection, every query and every duplicate-check filters by `user_id`. This was chosen over per-user collections because that doesn't scale operationally (imagine managing hundreds of separate collections) and is the same pattern used for any other metadata filter (date, department, etc.).

Deduplication is scoped **per-user, not global**: if two different users upload the same document, both get their own stored copy. A global hash check would silently skip storage for the second user, making content invisible to their queries even though it's genuinely theirs.

Real authentication (login/sessions) is deliberately **not implemented** in v1 — `user_id` is a plain field supplied by the client (curl, Streamlit). This was a conscious scope decision: the isolation *mechanism* is real and fully tested; swapping in real auth later only changes *where the `user_id` value comes from*, not any downstream logic.

### Citations and refusal (M6S4 / M6S5)

Every chunk carries its source filename and page number (for PDFs — tracked via per-page extraction preserved through cleaning and chunking; DOCX/HTML have no page concept and store `page: null`). The generation prompt requires the model to cite sources by bracketed number for every claim, and explicitly instructs it to say "I don't have that information in the provided documents" rather than answer from general knowledge when the retrieved context doesn't cover the question. Both behaviors are verified in the eval harness (see below), including on genuinely out-of-scope questions (e.g. "What is the capital of France?").

### v1 format scope

Supported: PDF (text-based), DOCX, HTML. Deliberately deferred to v2/exploratory: scanned PDFs requiring OCR, and CSV (tabular data needs its own chunking strategy, not plain-text chunking — the loader raises a clear error rather than silently mishandling it). See `learnings/not_covered.md`.

---

## Evaluation

`services/eval/` contains a golden-dataset regression harness: 23 real questions against an actual ingested document, each checked for the expected fact appearing in the answer and the expected source being cited — plus 3 deliberately out-of-scope questions verifying refusal behavior.

Current result: **22/23 (96%)**. The one failure is a genuine retrieval gap (a specific fact wasn't surfaced by hybrid search into the reranked context), not a hallucination — the system correctly refused rather than guessing. This is treated as a known, documented limitation rather than a silent failure, which is itself the point of building the eval harness in the first place.

Run it:
```bash
cd services/eval
python3 run_eval.py
```

---

## Running it

### Option A — Docker Compose (recommended, one command)

```bash
docker-compose up --build
```

Brings up all 6 services: Redis, Weaviate, the Go gateway, the Celery worker, the bridge, and the FastAPI retrieval service. Then use the UI or curl (below) against `localhost`.

### Option B — manual (for development)

Requires: Go, Python 3.x with a venv per Python service, Redis, and a running Weaviate instance.

```bash
# Terminal 1 — ingestion gateway
cd services/gateway && go run .

# Terminal 2 — Celery worker (macOS: --pool=solo avoids a fork-safety crash
# with native ML libraries)
cd services/worker && celery -A celery_app worker --loglevel=info --pool=solo

# Terminal 3 — bridge (adapts Go's plain Redis queue to Celery's protocol)
cd services/worker && python bridge.py

# Terminal 4 — retrieval API
cd services/api && uvicorn main:app --reload --port 8000

# Terminal 5 — UI
cd ui && streamlit run app.py
```

Each Python service needs its own `.env` (see `.env.example` in each folder) with `OPENAI_API_KEY` and Weaviate connection details.

### Uploading and querying via curl

```bash
curl -X POST http://localhost:8080/upload \
  -F "file=@data/inputs/your_document.pdf" \
  -F "user_id=your_name"

curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"your question here","user_id":"your_name"}'
```

---

## Repository structure

```
Athenaeum/
├── docker-compose.yml
├── shared/schemas/              # The Go↔Python job contract (JSON Schema) — source of truth for the handoff
├── services/
│   ├── gateway/                 # Go: upload intake, validation, queueing
│   ├── worker/                  # Python: Celery ingestion pipeline (load → clean → chunk → embed → store)
│   ├── api/                     # Python: retrieval pipeline + FastAPI wrapper
│   └── eval/                    # Python: golden-dataset regression harness
├── ui/                          # Streamlit app
├── data/inputs/                 # Test documents
└── learnings/                   # Design-decision notes, known limitations, out-of-scope items
```

---

## Known limitations (v1, by design)

- No OCR — scanned/image-only PDFs are out of scope
- No CSV ingestion — accepted at intake, explicitly rejected by the loader with a clear error
- No real authentication — `user_id` is a client-supplied field, not a verified session
- Recursive chunking can cut mid-sentence on long paragraphs, and has no special handling for code blocks
- PDF header/footer stripping uses cross-page exact-line-repetition detection, which can miss lines that vary slightly between pages (e.g. a running timestamp)