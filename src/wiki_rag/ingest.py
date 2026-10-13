"""Download Wikipedia articles from a category tree via the MediaWiki API."""

from __future__ import annotations

import re
import time
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime

import httpx

USER_AGENT = "wiki-rag/0.1 (https://github.com/jbryn/wiki-rag)"
RETRY_STATUSES = {429, 500, 502, 503, 504}
MAX_ATTEMPTS = 4

PAGE_NAMESPACE = 0
CATEGORY_NAMESPACE = 14

# Top-level sections that hold link lists and citations rather than prose.
SKIPPED_SECTIONS = {
    "see also",
    "references",
    "external links",
    "further reading",
    "notes",
    "footnotes",
    "citations",
    "bibliography",
    "sources",
    "works cited",
}

HEADING_RE = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$", re.MULTILINE)


@dataclass(frozen=True)
class PageRef:
    pageid: int
    title: str


@dataclass
class Section:
    # Headings from the top level down to this section; empty for the lead.
    path: list[str]
    text: str


@dataclass
class Article:
    pageid: int
    revid: int
    title: str
    url: str
    sections: list[Section]
    fetched_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def char_count(self) -> int:
        return sum(len(section.text) for section in self.sections)

    def to_dict(self) -> dict:
        return asdict(self)


def parse_sections(extract: str) -> list[Section]:
    """Split a plain-text extract with wiki-style headings into prose sections."""
    sections: list[Section] = []
    path: list[str] = []
    position = 0

    def close(end: int) -> None:
        text = extract[position:end].strip()
        skipped = path and path[0].lower() in SKIPPED_SECTIONS
        if text and not skipped:
            sections.append(Section(path=list(path), text=text))

    for match in HEADING_RE.finditer(extract):
        close(match.start())
        depth = len(match.group(1)) - 2
        path = [*path[:depth], match.group(2)]
        position = match.end()
    close(len(extract))
    return sections


class WikipediaClient:
    def __init__(
        self,
        client: httpx.Client | None = None,
        lang: str = "en",
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._client = client or httpx.Client(timeout=30)
        self._url = f"https://{lang}.wikipedia.org/w/api.php"
        self._sleep = sleep

    def _get(self, **params: str | int) -> dict:
        query = {"action": "query", "format": "json", "formatversion": 2, **params}
        for attempt in range(MAX_ATTEMPTS):
            response = self._client.get(self._url, params=query, headers={"User-Agent": USER_AGENT})
            if response.status_code in RETRY_STATUSES and attempt < MAX_ATTEMPTS - 1:
                self._sleep(float(response.headers.get("Retry-After", 2**attempt)))
                continue
            response.raise_for_status()
            break
        data = response.json()
        if "error" in data:
            raise RuntimeError(f"MediaWiki API error: {data['error'].get('info', data['error'])}")
        return data

    def _category_members(self, category: str) -> Iterator[dict]:
        continuation: dict = {}
        while True:
            data = self._get(
                list="categorymembers",
                cmtitle=f"Category:{category}",
                cmtype="page|subcat",
                cmlimit="max",
                **continuation,
            )
            yield from data.get("query", {}).get("categorymembers", [])
            if "continue" not in data:
                return
            continuation = data["continue"]

    def walk_category(
        self, category: str, depth: int = 1, limit: int | None = None
    ) -> Iterator[PageRef]:
        """Yield unique articles breadth-first, descending `depth` levels of subcategories."""
        seen_pages: set[int] = set()
        seen_categories = {category}
        queue = deque([(category, 0)])
        while queue:
            current, level = queue.popleft()
            for member in self._category_members(current):
                if member["ns"] == CATEGORY_NAMESPACE:
                    name = member["title"].split(":", 1)[1]
                    if level < depth and name not in seen_categories:
                        seen_categories.add(name)
                        queue.append((name, level + 1))
                elif member["ns"] == PAGE_NAMESPACE and member["pageid"] not in seen_pages:
                    if limit is not None and len(seen_pages) >= limit:
                        return
                    seen_pages.add(member["pageid"])
                    yield PageRef(pageid=member["pageid"], title=member["title"])

    def fetch_article(self, pageid: int) -> Article | None:
        """Fetch one article as sections, or None if it is missing or has no prose."""
        data = self._get(
            pageids=pageid,
            prop="extracts|info|revisions",
            explaintext=1,
            exsectionformat="wiki",
            inprop="url",
            rvprop="ids",
        )
        pages = data.get("query", {}).get("pages", [])
        if not pages or pages[0].get("missing"):
            return None
        page = pages[0]
        sections = parse_sections(page.get("extract", ""))
        if not sections:
            return None
        return Article(
            pageid=page["pageid"],
            revid=page["revisions"][0]["revid"],
            title=page["title"],
            url=page["fullurl"],
            sections=sections,
        )
