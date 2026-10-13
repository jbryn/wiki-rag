import httpx
import pytest

from wiki_rag.ingest import PageRef, WikipediaClient, parse_sections

EXTRACT = """Stoicism is a school of Hellenistic philosophy.


== History ==
It was founded in Athens.


=== Roman period ===
Seneca and Epictetus wrote in this period.


== See also ==
Cynicism


=== Lists ===
List of Stoics


== Ethics ==
Virtue is the only good."""


def make_client(handler):
    transport = httpx.MockTransport(handler)
    return WikipediaClient(client=httpx.Client(transport=transport), sleep=lambda _: None)


def page(pageid, title):
    return {"pageid": pageid, "ns": 0, "title": title}


def subcat(name):
    return {"pageid": 0, "ns": 14, "title": f"Category:{name}"}


def category_handler(tree):
    """Serve categorymembers from a {category: [batch, ...]} mapping, one batch per request."""

    def handler(request):
        params = request.url.params
        batches = tree[params["cmtitle"].removeprefix("Category:")]
        index = int(params.get("cmcontinue", 0))
        body = {"query": {"categorymembers": batches[index]}}
        if index + 1 < len(batches):
            body["continue"] = {"cmcontinue": str(index + 1), "continue": "-||"}
        return httpx.Response(200, json=body)

    return handler


def test_parse_sections_tracks_heading_path():
    sections = parse_sections(EXTRACT)

    assert [s.path for s in sections] == [[], ["History"], ["History", "Roman period"], ["Ethics"]]
    assert sections[0].text == "Stoicism is a school of Hellenistic philosophy."
    assert sections[2].text == "Seneca and Epictetus wrote in this period."


def test_parse_sections_drops_reference_sections_and_their_children():
    texts = [s.text for s in parse_sections(EXTRACT)]

    assert "Cynicism" not in texts
    assert "List of Stoics" not in texts


def test_parse_sections_of_empty_extract():
    assert parse_sections("") == []


def test_walk_category_follows_continuation_and_deduplicates():
    tree = {
        "Philosophy": [[page(1, "Ethics"), subcat("Logic")], [page(2, "Metaphysics")]],
        "Logic": [[page(2, "Metaphysics"), page(3, "Syllogism")]],
    }
    client = make_client(category_handler(tree))

    refs = list(client.walk_category("Philosophy", depth=1))

    assert refs == [PageRef(1, "Ethics"), PageRef(2, "Metaphysics"), PageRef(3, "Syllogism")]


def test_walk_category_respects_depth():
    tree = {
        "Philosophy": [[page(1, "Ethics"), subcat("Logic")]],
        "Logic": [[page(3, "Syllogism"), subcat("Modal logic")]],
        "Modal logic": [[page(4, "Possible world")]],
    }

    def titles(depth):
        client = make_client(category_handler(tree))
        return [r.title for r in client.walk_category("Philosophy", depth=depth)]

    assert titles(0) == ["Ethics"]
    assert titles(1) == ["Ethics", "Syllogism"]
    assert titles(2) == ["Ethics", "Syllogism", "Possible world"]


def test_walk_category_survives_category_cycles():
    tree = {
        "Philosophy": [[page(1, "Ethics"), subcat("Logic")]],
        "Logic": [[page(3, "Syllogism"), subcat("Philosophy")]],
    }
    client = make_client(category_handler(tree))

    assert len(list(client.walk_category("Philosophy", depth=5))) == 2


def test_walk_category_stops_at_limit():
    tree = {"Philosophy": [[page(1, "Ethics"), page(2, "Metaphysics"), page(3, "Logic")]]}
    client = make_client(category_handler(tree))

    assert len(list(client.walk_category("Philosophy", limit=2))) == 2


def test_walk_category_ignores_other_namespaces():
    tree = {"Philosophy": [[{"pageid": 9, "ns": 100, "title": "Portal:Philosophy"}]]}
    client = make_client(category_handler(tree))

    assert list(client.walk_category("Philosophy")) == []


def test_fetch_article():
    def handler(request):
        assert request.url.params["pageids"] == "7"
        assert request.headers["User-Agent"].startswith("wiki-rag/")
        body = {
            "query": {
                "pages": [
                    {
                        "pageid": 7,
                        "title": "Stoicism",
                        "extract": EXTRACT,
                        "fullurl": "https://en.wikipedia.org/wiki/Stoicism",
                        "revisions": [{"revid": 42, "parentid": 41}],
                    }
                ]
            }
        }
        return httpx.Response(200, json=body)

    article = make_client(handler).fetch_article(7)

    assert article.title == "Stoicism"
    assert article.revid == 42
    assert article.url == "https://en.wikipedia.org/wiki/Stoicism"
    assert len(article.sections) == 4
    assert article.to_dict()["sections"][1] == {
        "path": ["History"],
        "text": "It was founded in Athens.",
    }


@pytest.mark.parametrize(
    "pages",
    [
        [{"pageid": 7, "missing": True}],
        [
            {
                "pageid": 7,
                "title": "Stub",
                "extract": "",
                "fullurl": "u",
                "revisions": [{"revid": 1}],
            }
        ],
    ],
)
def test_fetch_article_returns_none_without_prose(pages):
    client = make_client(lambda request: httpx.Response(200, json={"query": {"pages": pages}}))

    assert client.fetch_article(7) is None


def test_get_retries_transient_errors():
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) < 3:
            return httpx.Response(503, headers={"Retry-After": "1"})
        return httpx.Response(200, json={"query": {"categorymembers": [page(1, "Ethics")]}})

    refs = list(make_client(handler).walk_category("Philosophy"))

    assert len(calls) == 3
    assert refs == [PageRef(1, "Ethics")]


def test_get_gives_up_after_repeated_errors():
    client = make_client(lambda request: httpx.Response(503))

    with pytest.raises(httpx.HTTPStatusError):
        list(client.walk_category("Philosophy"))


def test_get_raises_on_api_error():
    body = {"error": {"code": "invalidcategory", "info": "Bad category"}}
    client = make_client(lambda request: httpx.Response(200, json=body))

    with pytest.raises(RuntimeError, match="Bad category"):
        list(client.walk_category("Philosophy"))
