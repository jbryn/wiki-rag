"""Split ingested articles into retrieval-sized chunks that stay within one section."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Chunk:
    id: str
    pageid: int
    title: str
    url: str
    path: list[str]
    text: str

    @property
    def heading(self) -> str:
        return " > ".join([self.title, *self.path])

    @property
    def source_url(self) -> str:
        """Link to the section of the article this chunk came from."""
        if not self.path:
            return self.url
        return f"{self.url}#{self.path[-1].replace(' ', '_')}"

    @property
    def indexed_text(self) -> str:
        # The heading gives short chunks the context they lose by being cut out of the article.
        return f"{self.heading}\n{self.text}"

    def to_dict(self) -> dict:
        return asdict(self)


def split_text(text: str, max_chars: int) -> list[str]:
    """Pack paragraphs into pieces of at most `max_chars`, splitting long ones at sentences."""
    units: list[str] = []
    for paragraph in filter(None, (p.strip() for p in text.split("\n"))):
        if len(paragraph) <= max_chars:
            units.append(paragraph)
        else:
            units.extend(SENTENCE_BOUNDARY_RE.split(paragraph))

    pieces: list[str] = []
    current = ""
    for unit in units:
        if current and len(current) + 1 + len(unit) > max_chars:
            pieces.append(current)
            current = unit
        else:
            current = f"{current} {unit}" if current else unit
    if current:
        pieces.append(current)
    return pieces


def chunk_article(article: dict, max_chars: int = 1200) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section in article["sections"]:
        for text in split_text(section["text"], max_chars):
            chunks.append(
                Chunk(
                    id=f"{article['pageid']}:{len(chunks)}",
                    pageid=article["pageid"],
                    title=article["title"],
                    url=article["url"],
                    path=section["path"],
                    text=text,
                )
            )
    return chunks
