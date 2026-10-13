"""Text embedding models."""

from __future__ import annotations

from typing import Protocol

import numpy as np

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


class Embedder(Protocol):
    name: str

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.where(norms == 0, 1, norms)


class FastEmbedEmbedder:
    """ONNX embedding model run locally through fastembed (no PyTorch needed)."""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        from fastembed import TextEmbedding

        self.name = model_name
        self._model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return normalize(np.array(list(self._model.embed(texts)), dtype=np.float32))

    def embed_query(self, text: str) -> np.ndarray:
        return normalize(np.array(next(iter(self._model.query_embed(text))), dtype=np.float32))
