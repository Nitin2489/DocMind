"""Turn an uploaded file into (page_number, text) pairs."""
import io
from pathlib import PurePath

from pypdf import PdfReader
from pypdf.errors import PyPdfError

from . import config


class UnsupportedFile(ValueError):
    pass


def extract_pages(filename: str, data: bytes) -> list[tuple[int, str]]:
    ext = PurePath(filename).suffix.lower()
    if ext not in config.ALLOWED_EXTENSIONS:
        raise UnsupportedFile(
            f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(config.ALLOWED_EXTENSIONS))}"
        )
    if ext == ".pdf":
        try:
            reader = PdfReader(io.BytesIO(data))
            pages = [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]
        except (PyPdfError, ValueError, OSError) as exc:
            raise UnsupportedFile(f"Could not read PDF: {exc}") from exc
    else:
        pages = [(1, data.decode("utf-8", errors="replace"))]
    pages = [(n, t) for n, t in pages if t.strip()]
    if not pages:
        raise UnsupportedFile("No extractable text found (scanned PDFs need OCR, which is not supported).")
    return pages
