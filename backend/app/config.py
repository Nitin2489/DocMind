"""Runtime configuration, read from environment variables."""
import os
from pathlib import Path


def data_dir() -> Path:
    path = Path(os.getenv("DOCMIND_DATA_DIR", "data"))
    path.mkdir(parents=True, exist_ok=True)
    return path


CHUNK_SIZE = int(os.getenv("DOCMIND_CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("DOCMIND_CHUNK_OVERLAP", "150"))
EMBED_DIM = int(os.getenv("DOCMIND_EMBED_DIM", "512"))
TOP_K = int(os.getenv("DOCMIND_TOP_K", "5"))
MAX_UPLOAD_BYTES = int(os.getenv("DOCMIND_MAX_UPLOAD_MB", "10")) * 1024 * 1024
MAX_QUESTION_CHARS = 2000
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}

ANTHROPIC_MODEL = os.getenv("DOCMIND_ANTHROPIC_MODEL", "claude-sonnet-5-5")
OPENAI_MODEL = os.getenv("DOCMIND_OPENAI_MODEL", "gpt-4o-mini")
LLM_TIMEOUT_SECONDS = float(os.getenv("DOCMIND_LLM_TIMEOUT", "60"))
