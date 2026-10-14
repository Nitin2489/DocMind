"""DocMind API: upload documents, ask questions, get cited answers."""
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config, llm
from .chunking import chunk_pages
from .embeddings import HashingEmbedder
from .ingest import UnsupportedFile, extract_pages
from .retrieval import HybridIndex
from .store import Store

FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=config.MAX_QUESTION_CHARS)
    doc_ids: list[int] | None = None
    top_k: int = Field(default=config.TOP_K, ge=1, le=10)


class State:
    """Holds the store and a lazily rebuilt search index."""

    def __init__(self) -> None:
        self.store = Store(str(config.data_dir() / "docmind.sqlite3"))
        self.embedder = HashingEmbedder(config.EMBED_DIM)
        self._index: HybridIndex | None = None
        self._built_version = -1
        self._lock = threading.Lock()

    def index(self) -> HybridIndex:
        with self._lock:
            if self._index is None or self._built_version != self.store.version:
                self._built_version = self.store.version
                self._index = HybridIndex(self.store.all_chunks(), self.embedder)
            return self._index


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.docmind = State()
    yield


app = FastAPI(title="DocMind", version="1.0.0", lifespan=lifespan)


def _state() -> State:
    return app.state.docmind


@app.get("/api/health")
def health():
    return {"status": "ok", "llm_mode": llm.provider(), "documents": len(_state().store.list_documents())}


@app.get("/api/documents")
def list_documents():
    return _state().store.list_documents()


@app.post("/api/documents", status_code=201)
async def upload_documents(files: list[UploadFile] = File(...)):
    state = _state()
    results, errors = [], []
    for upload in files:
        name = Path(upload.filename or "upload").name
        data = await upload.read(config.MAX_UPLOAD_BYTES + 1)
        if len(data) > config.MAX_UPLOAD_BYTES:
            errors.append({"name": name, "error": f"File exceeds {config.MAX_UPLOAD_BYTES // 1048576} MB limit"})
            continue
        try:
            pages = extract_pages(name, data)
        except UnsupportedFile as exc:
            errors.append({"name": name, "error": str(exc)})
            continue
        chunks = chunk_pages(pages, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        embeddings = state.embedder.embed([text for _, text in chunks])
        doc_id = state.store.add_document(name, chunks, embeddings)
        results.append({"id": doc_id, "name": name, "chunks": len(chunks)})
    if not results:
        raise HTTPException(status_code=422, detail=errors or "No files uploaded")
    return {"documents": results, "errors": errors}


@app.delete("/api/documents/{doc_id}", status_code=204)
def delete_document(doc_id: int):
    if not _state().store.delete_document(doc_id):
        raise HTTPException(status_code=404, detail="Document not found")


@app.post("/api/ask")
def ask(req: AskRequest):
    hits = _state().index().search(
        req.question, k=req.top_k, doc_ids=set(req.doc_ids) if req.doc_ids else None
    )
    answer = llm.generate(req.question, hits)
    return {
        "answer": answer.text,
        "mode": answer.mode,
        "warning": answer.warning,
        "sources": [
            {
                "n": i,
                "doc_id": h.chunk.doc_id,
                "doc_name": h.chunk.doc_name,
                "page": h.chunk.page,
                "text": h.chunk.text,
                "score": round(h.score, 5),
            }
            for i, h in enumerate(hits, 1)
        ],
    }


@app.get("/")
def index_page():
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
