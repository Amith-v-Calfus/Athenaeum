import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from retrieval import answer_question, _get_reranker


@asynccontextmanager
async def lifespan(app: FastAPI):
    _get_reranker()
    yield


app = FastAPI(title="Athenaeum Retrieval API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str
    user_id: str


class Source(BaseModel):
    filename: str
    page: int | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> dict:
    return answer_question(request.question, request.user_id)


@app.get("/health")
def health():
    import weaviate

    try:
        client = weaviate.connect_to_local(
            host=os.getenv("WEAVIATE_HOST", "localhost"),
            port=int(os.getenv("WEAVIATE_HTTP_PORT", "8081")),
            grpc_port=int(os.getenv("WEAVIATE_GRPC_PORT", "50051")),
        )
        try:
            if not client.is_ready():
                raise RuntimeError("weaviate not ready")
        finally:
            client.close()
    except Exception:
        return JSONResponse(status_code=503, content={"status": "unavailable"})

    return {"status": "ok"}
