"""Searchable index over chunks: dense vectors, BM25, and a hybrid of the two."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from wiki_rag.bm25 import BM25
from wiki_rag.chunking import Chunk
from wiki_rag.embeddings import Embedder

Mode = Literal["hybrid", "dense", "bm25"]
MODES: tuple[Mode, ...] = ("hybrid", "dense", "bm25")

# Standard reciprocal rank fusion constant (Cormack et al., 2009).
RRF_K = 60
# How many results each retriever contributes to the fusion.
FUSION_POOL = 100


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float


def _ranked(scores: np.ndarray, limit: int) -> np.ndarray:
    """Document ids with a positive score, best first."""
    order = np.argsort(-scores, kind="stable")[:limit]
    return order[scores[order] > 0]


class Index:
    def __init__(self, chunks: list[Chunk], embeddings: np.ndarray, embedder: Embedder) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings differ in length")
        self.chunks = chunks
        self._embeddings = embeddings
        self._embedder = embedder
        self._bm25 = BM25([chunk.indexed_text for chunk in chunks])

    @classmethod
    def build(cls, chunks: list[Chunk], embedder: Embedder) -> Index:
        embeddings = embedder.embed_documents([chunk.indexed_text for chunk in chunks])
        return cls(chunks, embeddings, embedder)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "chunks.jsonl").open("w", encoding="utf-8") as f:
            for chunk in self.chunks:
                f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
        np.save(directory / "embeddings.npy", self._embeddings)
        meta = {"embedding_model": self._embedder.name, "chunks": len(self.chunks)}
        (directory / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    @classmethod
    def load(cls, directory: Path, embedder: Embedder) -> Index:
        meta = json.loads((directory / "meta.json").read_text())
        if meta["embedding_model"] != embedder.name:
            raise ValueError(
                f"index was built with {meta['embedding_model']!r}, not {embedder.name!r}"
            )
        with (directory / "chunks.jsonl").open(encoding="utf-8") as f:
            chunks = [Chunk(**json.loads(line)) for line in f]
        return cls(chunks, np.load(directory / "embeddings.npy"), embedder)

    def search(self, query: str, k: int = 5, mode: Mode = "hybrid") -> list[SearchResult]:
        if mode not in MODES:
            raise ValueError(f"unknown search mode: {mode!r}")
        if not self.chunks:
            return []
        if mode == "bm25":
            scores = self._bm25.scores(query)
        elif mode == "dense":
            scores = self._dense_scores(query)
        else:
            scores = np.zeros(len(self.chunks))
            for retriever_scores in (self._dense_scores(query), self._bm25.scores(query)):
                order = _ranked(retriever_scores, FUSION_POOL)
                scores[order] += 1 / (RRF_K + np.arange(1, len(order) + 1))
        return [SearchResult(self.chunks[i], float(scores[i])) for i in _ranked(scores, k)]

    def _dense_scores(self, query: str) -> np.ndarray:
        return self._embeddings @ self._embedder.embed_query(query)
