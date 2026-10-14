"""Split page text into overlapping, sentence-aligned chunks."""
import re

_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n{2,}")


def _tail(buf: str, overlap: int) -> str:
    """Last `overlap` characters of buf, snapped to a word boundary."""
    if overlap <= 0 or not buf:
        return ""
    tail = buf[-overlap:]
    space = tail.find(" ")
    return tail[space + 1 :] if space != -1 else tail


def chunk_pages(
    pages: list[tuple[int, str]], size: int = 900, overlap: int = 150
) -> list[tuple[int, str]]:
    """Return (page_number, chunk_text) pairs. Chunks never cross pages."""
    if size <= overlap:
        raise ValueError("size must be larger than overlap")
    chunks: list[tuple[int, str]] = []
    for page, text in pages:
        text = re.sub(r"[ \t]+", " ", text).strip()
        buf = ""
        for sentence in _SPLIT_RE.split(text):
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(buf) + len(sentence) + 1 <= size:
                buf = f"{buf} {sentence}".strip()
                continue
            if buf:
                chunks.append((page, buf))
            buf = f"{_tail(buf, overlap)} {sentence}".strip()
            while len(buf) > size * 2:  # a single enormous "sentence"
                chunks.append((page, buf[:size]))
                buf = buf[size - overlap :]
        if buf:
            chunks.append((page, buf))
    return chunks
