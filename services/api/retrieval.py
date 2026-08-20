"""Retrieval pipeline (query-time, synchronous).

This is a separate module/service from tasks.py deliberately: ingestion
(tasks.py) is an async background job triggered by file upload; retrieval is
a synchronous request triggered by a live user question. Different trigger,
different lifecycle -- they belong in different files, and eventually
different services (this lives behind the FastAPI backend, tasks.py behind
Celery).

Pipeline, matching the reference architecture:

    User Query -> Query Embedding -> Hybrid Retrieval (dense + BM25, Weaviate
    native) -> Rerank (local cross-encoder) -> Context Assembly (dedup, sort,
    trim) -> LLM Generation (with citations + refusal instructions)

DESIGN NOTE ON WHAT IS TESTED
-----------------------------
Tested for real, standalone, without needing a live server:
    - build_prompt          (prompt construction logic)
    - _assemble_context     (dedup/sort/trim logic)

NOT yet tested end-to-end against a live Weaviate server or the real
cross-encoder model (sandbox constraints) -- verify these on first real run:
    - retrieve() -- the hybrid() query call itself
    - _rerank()  -- needs sentence-transformers + model download
Both are built against the confirmed current weaviate-client v4 API surface
(hybrid()'s real parameter names were inspected directly, not guessed), but
"compiles against the right API" is not the same as "proven to run" -- run
the smoke test at the bottom of this file on first use.
"""

import os

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
_EMBEDDING_MODEL = "text-embedding-3-small"
_GENERATION_MODEL = "gpt-4o-mini"
_COLLECTION_NAME = "AthenaeumChunks"
_HYBRID_ALPHA = 0.5  # 0.0 = pure BM25, 1.0 = pure dense. 0.5 = even blend, a
                      # reasonable starting point before tuning against real
                      # eval data.
_RERANK_TOP_K = 10    # how many hybrid results to pull before reranking
_FINAL_TOP_K = 5       # how many results survive reranking, into the prompt
_MAX_CONTEXT_CHARS = 6000  # trim budget for assembled context
_reranker_model=None

def _embed_query(question: str) -> list[float]:
    """Embed the user's question with the SAME model used at ingestion time.
    Using a different model here would produce vectors that are not
    comparable to what's stored -- this must always match _EMBEDDING_MODEL
    in tasks.py.
    """
    client = OpenAI()
    response = client.embeddings.create(model=_EMBEDDING_MODEL, input=[question])
    return response.data[0].embedding


def retrieve(question: str, user_id: str, top_k: int = _RERANK_TOP_K) -> list[dict]:
    """Run Weaviate's native hybrid (dense + BM25) search, filtered by user_id.

    NOT YET LIVE-TESTED (needs a running Weaviate server) -- built against
    the real, currently-installed weaviate-client v4 hybrid() signature:
    query, vector, alpha, filters, limit, return_metadata, return_properties.
    """
    import weaviate
    from weaviate.classes.query import Filter, MetadataQuery

    client = weaviate.connect_to_local(
        host=os.getenv("WEAVIATE_HOST", "localhost"),
        port=int(os.getenv("WEAVIATE_HTTP_PORT", "8081")),
        grpc_port=int(os.getenv("WEAVIATE_GRPC_PORT", "50051")),
    )
    try:
        collection = client.collections.get(_COLLECTION_NAME)
        query_vector = _embed_query(question)

        response = collection.query.hybrid(
            query=question,
            vector=query_vector,
            alpha=_HYBRID_ALPHA,
            filters=Filter.by_property("user_id").equal(user_id),
            limit=top_k,
            return_metadata=MetadataQuery(score=True),
        )

        results = []
        for obj in response.objects:
            props = obj.properties
            results.append({
                "text": props.get("text", ""),
                "doc_id": props.get("doc_id"),
                "original_filename": props.get("original_filename"),
                "content_hash": props.get("content_hash"),
                "chunk_index": props.get("chunk_index"),
                "page": props.get("page"),
                "score": obj.metadata.score if obj.metadata else 0.0,
            })
        return results
    finally:
        client.close()

def _get_reranker():
    """Lazily load and cache the cross-encoder model (once per process)."""
    global _reranker_model
    if _reranker_model is None:
        from sentence_transformers import CrossEncoder
        _reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker_model

def _rerank(question: str, results: list[dict], top_k: int = _FINAL_TOP_K) -> list[dict]:
    """The cross-encoder takes the query and a single chunk together, as one combined input, and the model directly outputs a single relevance score for that specific pair. 
    Because it sees both texts simultaneously, it can pick up on nuanced relationships bi-encoders miss — but it's much more computationally expensive, since it can't be precomputed or batched the same way. 
    You cannot run a cross-encoder over your entire corpus for every query; it would be far too slow."""
    model = _get_reranker()
    pairs = [[question, r["text"]] for r in results]
    scores = model.predict(pairs)

    for r, score in zip(results, scores):
        r["rerank_score"] = float(score)

    results.sort(key=lambda r: r["rerank_score"], reverse=True)
    return results[:top_k]

def _assemble_context(results: list[dict], max_chars: int = _MAX_CONTEXT_CHARS) -> list[dict]:
    """Dedupe by (content_hash, chunk_index), sort by best available score,
    trim to a character budget. TESTED with mock data (see module docstring).
    """
    seen = set()
    deduped = []
    for r in results:
        key = (r.get("content_hash"), r.get("chunk_index"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)

    score_key = "rerank_score" if deduped and "rerank_score" in deduped[0] else "score"
    deduped.sort(key=lambda r: r.get(score_key, 0.0), reverse=True)

    assembled = []
    total_chars = 0
    for r in deduped:
        if total_chars + len(r["text"]) > max_chars:
            break
        assembled.append(r)
        total_chars += len(r["text"])

    return assembled


def build_prompt(question: str, context_chunks: list[dict]) -> tuple[str, str]:
    """Build the system + user prompt, with numbered citations and explicit
    grounding/refusal instructions. TESTED with mock data (see module
    docstring). This is the M6S4/M6S5 enforcement point.
    """
    context_blocks = []
    for i, c in enumerate(context_chunks):
        source = c["original_filename"]
        if c.get("page") is not None:
            source += f", page {c['page']}"
        context_blocks.append(f"[{i + 1}] (Source: {source})\n{c['text']}")

    context_text = "\n\n".join(context_blocks)

    system_prompt = (
        "You are a document Q&A assistant. Answer the user's question using "
        "ONLY the provided context below. For every factual claim, cite the "
        "source using the bracketed number, e.g. [1]. If the context does not "
        "contain enough information to answer the question, say clearly that "
        "you don't have that information in the provided documents -- do not "
        "guess or use outside knowledge."
    )
    user_prompt = f"Context:\n{context_text}\n\nQuestion: {question}"
    return system_prompt, user_prompt


_REFUSAL_PHRASES = [
    "don't have", "do not have", "does not contain", "no information",
    "doesn't contain", "not contain enough information",
]


def _is_refusal(answer: str) -> bool:
    """Heuristic check: does the answer look like a refusal, based on
    the same phrases already used in services/eval/golden_dataset.json's
    refusal test cases. Not perfectly precise (a real answer could
    coincidentally contain one of these phrases), but good enough for
    deciding whether to suppress a misleading sources list.
    """
    answer_lower = answer.lower()
    return any(phrase in answer_lower for phrase in _REFUSAL_PHRASES)


def generate_answer(question: str, context_chunks: list[dict]) -> str:
    """Call the generation model with the grounded, citation-instructed prompt."""
    system_prompt, user_prompt = build_prompt(question, context_chunks)
    client = OpenAI()
    response = client.chat.completions.create(
        model=_GENERATION_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content


def answer_question(question: str, user_id: str) -> dict:
    """Full pipeline: retrieve -> rerank -> assemble -> generate.

    This is the one function the FastAPI endpoint should call.
    """
    hybrid_results = retrieve(question, user_id, top_k=_RERANK_TOP_K)
    if not hybrid_results:
        return {
            "answer": "I don't have any documents to search for this user yet.",
            "sources": [],
        }

    reranked = _rerank(question, hybrid_results, top_k=_FINAL_TOP_K)
    context = _assemble_context(reranked)
    answer = generate_answer(question, context)

    if _is_refusal(answer):
        return {"answer": answer, "sources": []}

    sources = [
        {"filename": c["original_filename"], "page": c.get("page")}
        for c in context
    ]
    return {"answer": answer, "sources": sources}


if __name__ == "__main__":
    # Smoke test -- run this file directly once your Weaviate server and
    # ingested documents are up, to confirm the whole chain works end to end
    # before wiring it into FastAPI:
    #     python3 retrieval.py
    import sys

    user_id = sys.argv[1] if len(sys.argv) > 1 else "amith"
    question = sys.argv[2] if len(sys.argv) > 2 else "What are the working hours?"

    print(f"Asking (user={user_id!r}): {question!r}")
    result = answer_question(question, user_id)
    print()
    print("=== ANSWER ===")
    print(result["answer"])
    print()
    print("=== SOURCES ===")
    for s in result["sources"]:
        print(f"  {s}")