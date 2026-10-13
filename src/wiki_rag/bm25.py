"""Okapi BM25 lexical scoring."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

import numpy as np

TOKEN_RE = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class BM25:
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self._size = len(documents)
        self._k1 = k1
        postings: dict[str, tuple[list[int], list[int]]] = defaultdict(lambda: ([], []))
        lengths = np.zeros(self._size)
        for doc_id, document in enumerate(documents):
            counts = Counter(tokenize(document))
            lengths[doc_id] = sum(counts.values())
            for term, frequency in counts.items():
                doc_ids, frequencies = postings[term]
                doc_ids.append(doc_id)
                frequencies.append(frequency)

        average = lengths.mean() if self._size and lengths.any() else 1.0
        self._length_norm = k1 * (1 - b + b * lengths / average)
        self._postings = {
            term: (np.array(doc_ids), np.array(frequencies, dtype=float))
            for term, (doc_ids, frequencies) in postings.items()
        }

    def scores(self, query: str) -> np.ndarray:
        """Return one score per document; zero means no query term matched."""
        scores = np.zeros(self._size)
        for term in set(tokenize(query)):
            if term not in self._postings:
                continue
            doc_ids, frequencies = self._postings[term]
            idf = math.log(1 + (self._size - len(doc_ids) + 0.5) / (len(doc_ids) + 0.5))
            scores[doc_ids] += (
                idf * frequencies * (self._k1 + 1) / (frequencies + self._length_norm[doc_ids])
            )
        return scores
