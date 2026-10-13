from wiki_rag.chunking import chunk_article, split_text

ARTICLE = {
    "pageid": 7,
    "title": "Stoicism",
    "url": "https://en.wikipedia.org/wiki/Stoicism",
    "sections": [
        {"path": [], "text": "Stoicism is a school of philosophy."},
        {"path": ["History", "Roman period"], "text": "Seneca wrote letters.\nEpictetus taught."},
    ],
}


def test_split_text_keeps_short_text_whole():
    assert split_text("One paragraph.\nAnother one.", max_chars=100) == [
        "One paragraph. Another one."
    ]


def test_split_text_packs_paragraphs_up_to_the_limit():
    pieces = split_text("aaaa\nbbbb\ncccc", max_chars=9)

    assert pieces == ["aaaa bbbb", "cccc"]


def test_split_text_splits_long_paragraphs_at_sentences():
    pieces = split_text("First sentence here. Second sentence here. Third one.", max_chars=30)

    assert pieces == ["First sentence here.", "Second sentence here.", "Third one."]
    assert all(len(piece) <= 30 for piece in pieces)


def test_split_text_of_blank_text():
    assert split_text(" \n\n ", max_chars=10) == []


def test_chunk_article_keeps_chunks_inside_sections():
    chunks = chunk_article(ARTICLE)

    assert [c.id for c in chunks] == ["7:0", "7:1"]
    assert chunks[1].path == ["History", "Roman period"]
    assert chunks[1].text == "Seneca wrote letters. Epictetus taught."


def test_chunk_ids_are_unique_across_sections():
    chunks = chunk_article(ARTICLE, max_chars=25)

    assert len({c.id for c in chunks}) == len(chunks) == 3


def test_chunk_heading_and_source_url():
    lead, roman = chunk_article(ARTICLE)

    assert lead.heading == "Stoicism"
    assert lead.source_url == "https://en.wikipedia.org/wiki/Stoicism"
    assert roman.heading == "Stoicism > History > Roman period"
    assert roman.source_url == "https://en.wikipedia.org/wiki/Stoicism#Roman_period"
    assert roman.indexed_text.startswith("Stoicism > History > Roman period\n")
