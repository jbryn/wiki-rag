import zlib

import numpy as np
import pytest

from wiki_rag.chunking import Chunk
from wiki_rag.embeddings import normalize
from wiki_rag.index import Index


class SynonymEmbedder:
    """Bag-of-words embedder that maps listed synonyms to the same dimension."""

    name = "synonym-test"
    synonyms = {"morality": "ethics", "moral": "ethics"}

    def _embed(self, text):
        vector = np.zeros(256, dtype=np.float32)
        for word in text.lower().replace(">", " ").split():
            word = self.synonyms.get(word, word)
            vector[zlib.crc32(word.encode()) % 256] += 1
        return normalize(vector)

    def embed_documents(self, texts):
        return np.array([self._embed(text) for text in texts])

    def embed_query(self, text):
        return self._embed(text)


def chunk(n, title, text):
    return Chunk(id=f"{n}:0", pageid=n, title=title, url=f"https://x/{n}", path=[], text=text)


CHUNKS = [
    chunk(1, "Ethics", "the study of right and wrong conduct"),
    chunk(2, "Logic", "the study of valid inference and syllogism"),
    chunk(3, "Metaphysics", "the study of being and existence"),
]


@pytest.fixture
def index():
    return Index.build(CHUNKS, SynonymEmbedder())


def titles(results):
    return [result.chunk.title for result in results]


def test_bm25_mode_only_returns_lexical_matches(index):
    assert titles(index.search("syllogism", mode="bm25")) == ["Logic"]
    assert index.search("morality", mode="bm25") == []


def test_dense_mode_finds_matches_without_shared_words(index):
    assert titles(index.search("morality", mode="dense"))[0] == "Ethics"


def test_hybrid_mode_combines_both_retrievers(index):
    assert titles(index.search("morality", mode="hybrid"))[0] == "Ethics"
    assert titles(index.search("syllogism", mode="hybrid"))[0] == "Logic"


def test_hybrid_ranks_agreement_between_retrievers_first(index):
    results = index.search("study of existence", mode="hybrid")

    assert titles(results)[0] == "Metaphysics"
    assert results[0].score > results[1].score


def test_search_limits_results_to_k(index):
    assert len(index.search("study", k=2)) == 2


def test_search_rejects_unknown_mode(index):
    with pytest.raises(ValueError, match="unknown search mode"):
        index.search("study", mode="fuzzy")


def test_empty_index_returns_nothing():
    index = Index.build([], SynonymEmbedder())

    assert index.search("anything") == []


def test_save_and_load_round_trip(index, tmp_path):
    index.save(tmp_path / "index")
    loaded = Index.load(tmp_path / "index", SynonymEmbedder())

    assert loaded.chunks == CHUNKS
    for mode in ("hybrid", "dense", "bm25"):
        assert loaded.search("study of existence", mode=mode) == index.search(
            "study of existence", mode=mode
        )


def test_load_rejects_a_different_embedding_model(index, tmp_path):
    index.save(tmp_path)
    other = SynonymEmbedder()
    other.name = "another-model"

    with pytest.raises(ValueError, match="synonym-test"):
        Index.load(tmp_path, other)
