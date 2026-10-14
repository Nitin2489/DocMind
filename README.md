# DocMind

Ask questions about your own documents and get answers that cite their sources.

Upload PDFs, text or Markdown files, ask a question in plain English, and DocMind retrieves the most relevant passages, answers from them, and shows exactly which document and page each part came from. If the documents don't contain the answer, it says so instead of guessing.

**The problem it solves:** people lose time searching through policy documents, manuals, notes and reports for one specific fact. DocMind turns any pile of documents into something you can ask.

## Features

- **Upload and manage documents** (PDF, TXT, MD) through a drag-and-drop web UI; filter a question to chosen files.
- **Hybrid retrieval:** BM25 keyword ranking and dense-vector similarity, combined with reciprocal rank fusion.
- **Grounded, cited answers** using Claude or OpenAI (retrieval-augmented generation). The prompt tells the model to answer only from the sources and to treat document text as untrusted data (prompt-injection mitigation).
- **Works with no API key:** an offline extractive mode returns the best-matching sentences with citations, so you can try it immediately.
- **Graceful degradation:** if the LLM request fails, the app falls back to extractive mode and shows a warning.
- **Persistent storage** in SQLite; Docker image and GitHub Actions CI included.

## Architecture

```
Browser (vanilla JS)  ->  FastAPI  ->  ingest (pypdf) -> chunking -> hashing embeddings -> SQLite
                                  \->  hybrid index (BM25 + cosine, RRF) -> LLM (Claude / OpenAI / extractive)
```

| File | Role |
| --- | --- |
| `backend/app/ingest.py` | Extract text per page from PDF/TXT/MD |
| `backend/app/chunking.py` | Sentence-aligned overlapping chunks that never cross pages |
| `backend/app/embeddings.py` | Dependency-free hashing-trick embeddings (unigrams + bigrams) |
| `backend/app/retrieval.py` | BM25 + vector search fused with RRF |
| `backend/app/llm.py` | Prompting, provider calls, extractive fallback |
| `backend/app/store.py` | SQLite persistence |
| `backend/app/main.py` | REST API and static frontend |

## Run it

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
# open http://localhost:8000
```

Optional, for LLM-written answers (otherwise offline extractive mode is used):

```bash
export ANTHROPIC_API_KEY=...   # or OPENAI_API_KEY=...
```

With Docker: `docker compose up --build` (put keys in a `.env` file, see `.env.example`).

Try it with the files in `samples/`.

## API

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/api/documents` | Upload one or more files (multipart `files`) |
| `GET` | `/api/documents` | List documents |
| `DELETE` | `/api/documents/{id}` | Delete a document |
| `POST` | `/api/ask` | `{"question": "...", "doc_ids": [1], "top_k": 5}` returns answer and cited sources |
| `GET` | `/api/health` | Status and active LLM mode |

## Tests and evaluation

```bash
python -m pytest backend -q     # 24 tests: chunking, retrieval, API flow, LLM paths (HTTP mocked)
python eval/run_eval.py         # retrieval comparison on the sample documents
```

Retrieval evaluation (15 hand-written questions over the 3 sample documents, split into 300-character chunks; a hit means the right document and the answer text appear in the top-k chunks):

| Method | hit@1 | hit@3 | hit@5 | MRR |
| --- | --- | --- | --- | --- |
| BM25 only | 73% | 93% | 93% | 0.822 |
| Vector only | 80% | 87% | 93% | 0.850 |
| Hybrid (RRF) | 80% | 93% | 93% | 0.844 |

This is a very small test, so the three methods are not meaningfully different here; treat it as a sanity check and a harness for experiments, not a benchmark.

## Limitations

- The default embedder is a **hashing-trick** model: it matches words and phrases, not meaning. It misses pure paraphrases (for example "2FA" vs "two-factor authentication", or "price" vs "costs"). Swapping in a neural embedding model only requires implementing `embed()` in `embeddings.py`.
- Scanned PDFs (images) are not supported; there is no OCR.
- The search index is held in memory and rebuilt after changes, which suits thousands of chunks, not millions.
- No authentication or per-user separation: it is a single-user tool. Add auth before exposing it publicly.
- The Claude and OpenAI paths are tested with mocked HTTP only; they were not run against the live APIs in development.
- Questions are answered one at a time, without chat memory.

## Roadmap

- Neural embeddings (sentence-transformers or an embeddings API) and a re-ranking step
- Streaming answers
- Conversation memory with query rewriting
- OCR for scanned PDFs
- User accounts and per-user document collections

## License

MIT
