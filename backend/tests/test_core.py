import numpy as np
import pytest

from app.chunking import chunk_pages
from app.embeddings import HashingEmbedder
from app.ingest import UnsupportedFile, extract_pages
from app.retrieval import HybridIndex
from app.store import ChunkRow
from app.textutil import tokenize


def make_chunks(texts):
    emb = HashingEmbedder(128)
    vectors = emb.embed(texts)
    return emb, [ChunkRow(i, 1, "d.txt", i, 1, t, vectors[i]) for i, t in enumerate(texts)]


def test_tokenize_drops_stopwords():
    assert tokenize("What is the Refund Policy?") == ["refund", "policy"]


def test_chunks_respect_size_and_pages():
    page1 = " ".join(f"Sentence number {i} is here." for i in range(60))
    chunks = chunk_pages([(1, page1), (2, "Short page two.")], size=300, overlap=50)
    assert all(len(text) <= 600 for _, text in chunks)
    assert [p for p, _ in chunks][-1] == 2
    assert len({p for p, _ in chunks}) == 2


def test_chunk_overlap_repeats_text():
    text = " ".join(f"Word{i} appears in sentence {i}." for i in range(40))
    chunks = chunk_pages([(1, text)], size=200, overlap=60)
    assert len(chunks) > 2
    first_tail = chunks[0][1].split()[-2:]
    assert " ".join(first_tail) in chunks[1][1]


def test_chunk_rejects_bad_overlap():
    with pytest.raises(ValueError):
        chunk_pages([(1, "x")], size=100, overlap=100)


def test_huge_sentence_terminates():
    chunks = chunk_pages([(1, "a" * 5000)], size=300, overlap=50)
    assert len(chunks) > 5


def test_embeddings_are_normalised_and_similar_texts_closer():
    emb = HashingEmbedder(256)
    a, b, c = emb.embed(["annual leave days per year", "paid annual leave per year", "deploy to production friday"])
    assert abs(np.linalg.norm(a) - 1) < 1e-5
    assert a @ b > a @ c


def test_empty_embedding_batch():
    assert HashingEmbedder(64).embed([]).shape == (0, 64)


def test_hybrid_search_finds_relevant_chunk():
    emb, chunks = make_chunks(
        ["Employees get 24 days of annual leave.", "Deployments freeze on Fridays.", "Keys rotate every 90 days."]
    )
    hits = HybridIndex(chunks, emb).search("how many days of annual leave", k=2)
    assert hits[0].chunk.text.startswith("Employees get 24")


def test_search_respects_doc_filter_and_empty_query():
    emb, chunks = make_chunks(["alpha beta gamma", "alpha delta"])
    chunks[1].doc_id = 2
    index = HybridIndex(chunks, emb)
    assert {h.chunk.doc_id for h in index.search("alpha", doc_ids={2})} == {2}
    assert index.search("   ") == []
    assert HybridIndex([], emb).search("anything") == []


def test_extract_rejects_unknown_type_and_empty_text():
    with pytest.raises(UnsupportedFile):
        extract_pages("virus.exe", b"x")
    with pytest.raises(UnsupportedFile):
        extract_pages("empty.txt", b"   \n  ")
    with pytest.raises(UnsupportedFile):
        extract_pages("broken.pdf", b"not a pdf")
