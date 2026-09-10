import os

import pytest

from funicular import embeddings
from funicular.search import SearchIndex, chunk_text, make_snippet
from funicular.textnorm import normalize, phonetic_key, similarity, tokens

DOC_A = (
    "Riluzole Exposure and Survival in Motor Neuron Disease\n\n"
    "Chiò A et al. reported prognostic factors. "
    "The ALSFRS-R score fell by 0.9 points per month. "
    "Alzheimer’s disease was an exclusion criterion.\n\f"
    "Page two discusses non‑invasive ventilation (NIV) and forced vital capacity."
)
DOC_B = (
    "A cookbook of Mediterranean recipes.\n\n"
    "Olive oil, tomatoes and basil; nothing about neurology."
)


def test_normalize_unifies_diacritics_apostrophes_dashes_ligatures():
    assert normalize("Chiò") == "chio"
    assert normalize("Alzheimer’s") == normalize("Alzheimer's") == "alzheimer's"
    assert normalize("non‑invasive – test — x") == "non-invasive - test - x"
    assert normalize("ﬁbrosis ﬂow") == "fibrosis flow"
    assert normalize("neuro-\ndegenerative") == "neurodegenerative"
    assert normalize("  a b  ") == "a b"
    assert tokens("ALSFRS-R score, Chiò's data!") == ["alsfrs-r", "score", "chio's", "data"]


def test_phonetic_key_and_similarity():
    assert phonetic_key("riluzole") == phonetic_key("rilusole") != ""
    assert phonetic_key("123") == "" and phonetic_key("β") == ""
    assert similarity("Chiò", "Chio") == 1.0
    assert similarity("riluzole", "rilusole") > 0.9


def test_chunking_tracks_pages_and_overlap():
    chunks = chunk_text(DOC_A, target=120, overlap=20)
    assert chunks[0].page == 1 and chunks[-1].page == 2
    assert all(len(c.text) <= 300 for c in chunks)
    text = "para one\n\n" + ("x" * 5000)
    big = chunk_text(text, target=1000, overlap=100)
    assert len(big) >= 3
    assert chunk_text("") == []
    assert chunk_text("no page breaks here")[0].page is None


@pytest.fixture
def index(tmp_path):
    ix = SearchIndex(tmp_path / "search.sqlite3", embed="none")
    ix.index_document("a", "Riluzole paper", DOC_A, embed=False)
    ix.index_document("b", "Cookbook", DOC_B, embed=False)
    return ix


def test_keyword_search_is_diacritic_and_case_insensitive(index):
    hits, info = index.search("CHIO prognostic")
    assert hits and hits[0].doc_id == "a" and "keyword" in hits[0].routes
    assert "[Chiò]" in hits[0].snippet
    hits, _ = index.search("alzheimer's")
    assert hits and hits[0].doc_id == "a"
    hits, _ = index.search("Alzheimer’s")  # curly apostrophe
    assert hits and hits[0].doc_id == "a"


def test_phonetic_expansion_rescues_misspelt_drug(index):
    hits, info = index.search("rilusole survival")
    assert info.expansions.get("rilusole") == ["riluzole"]
    assert hits and hits[0].doc_id == "a"


def test_jargon_substring_via_trigram(index):
    hits, info = index.search("ALSFRS")
    assert hits and hits[0].doc_id == "a"
    assert "jargon" in info.routes
    hits, _ = index.search("ventilation")
    assert hits[0].page == 2


def test_no_match_and_filter(index):
    hits, info = index.search("zzzz qqqq")
    assert hits == []
    hits, _ = index.search("recipes", doc_ids=["a"])
    assert hits == []
    hits, _ = index.search("recipes", doc_ids=["b"])
    assert hits and hits[0].doc_id == "b"


def test_remove_and_stats(index):
    assert index.stats()["documents"] == 2
    index.remove_document("b")
    assert index.stats()["documents"] == 1
    assert index.search("recipes")[0] == []


def test_snippet_marks_first_term():
    s = make_snippet("alpha beta gamma delta", {"gamma"})
    assert "[gamma]" in s


@pytest.mark.skipif(
    os.environ.get("FUNICULAR_TEST_EMBED") != "1",
    reason="set FUNICULAR_TEST_EMBED=1 to run semantic search (downloads a small model once)",
)
def test_semantic_search_finds_paraphrase(tmp_path):
    emb = embeddings.get_embedder("auto")
    assert emb is not None, "no embedding backend available"
    ix = SearchIndex(tmp_path / "s.sqlite3", embed="auto")
    ix.index_document("a", "Riluzole paper", DOC_A)
    ix.index_document("b", "Cookbook", DOC_B)
    hits, info = ix.search("breathing support for ALS patients", mode="semantic")
    assert "semantic" in info.routes
    assert hits and hits[0].doc_id == "a" and hits[0].page == 2
    assert ix.stats()["embedded_documents"] == 2


def test_vocab_counts_track_documents_exactly(index):
    import sqlite3

    def n(term):
        with sqlite3.connect(index.path) as c:
            row = c.execute("SELECT n FROM vocab WHERE term=?", (term,)).fetchone()
        return row[0] if row else 0

    before = n("riluzole")
    assert before > 0 and n("recipes") > 0
    index.index_document("a", "Riluzole paper", DOC_A, embed=False)  # re-index: no inflation
    assert n("riluzole") == before
    index.remove_document("b")
    assert n("recipes") == 0  # removed with its document
    assert n("riluzole") == before
