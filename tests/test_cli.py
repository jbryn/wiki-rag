import json

from wiki_rag import cli
from wiki_rag.ingest import Article, PageRef, Section


class FakeClient:
    categories = {
        "Ethics": [PageRef(1, "Virtue"), PageRef(2, "Stub")],
        "Logic": [PageRef(1, "Virtue"), PageRef(3, "Syllogism"), PageRef(4, "Modus ponens")],
    }
    lengths = {1: 2000, 2: 10, 3: 2000, 4: 2000}

    def __init__(self, lang):
        self.fetched = []
        FakeClient.instance = self

    def walk_category(self, category, depth):
        yield from self.categories[category]

    def fetch_article(self, pageid):
        self.fetched.append(pageid)
        section = Section(path=[], text="x" * self.lengths[pageid])
        return Article(pageid=pageid, revid=1, title=str(pageid), url="u", sections=[section])


def run_ingest(monkeypatch, tmp_path, *extra):
    monkeypatch.setattr(cli, "WikipediaClient", FakeClient)
    out = tmp_path / "nested" / "articles.jsonl"
    args = ["ingest", "--category", "Ethics", "--category", "Logic", "--out", str(out), *extra]
    assert cli.main(args) == 0
    return [json.loads(line)["pageid"] for line in out.read_text().splitlines()]


def test_ingest_merges_categories_and_skips_short_articles(monkeypatch, tmp_path):
    assert run_ingest(monkeypatch, tmp_path) == [1, 3, 4]
    assert FakeClient.instance.fetched == [1, 2, 3, 4]


def test_ingest_limit_counts_written_articles(monkeypatch, tmp_path):
    assert run_ingest(monkeypatch, tmp_path, "--limit", "2") == [1, 3]
