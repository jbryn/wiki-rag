"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from wiki_rag.ingest import WikipediaClient

DEFAULT_CATEGORY = "Philosophy"


def _ingest(args: argparse.Namespace) -> int:
    client = WikipediaClient(lang=args.lang)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    seen: set[int] = set()
    written = 0
    with args.out.open("w", encoding="utf-8") as f:
        for category in args.category or [DEFAULT_CATEGORY]:
            if written >= args.limit:
                break
            for ref in client.walk_category(category, depth=args.depth):
                if written >= args.limit:
                    break
                if ref.pageid in seen:
                    continue
                seen.add(ref.pageid)
                article = client.fetch_article(ref.pageid)
                if article is None or article.char_count < args.min_chars:
                    print(f"skipped (too short): {ref.title}", file=sys.stderr)
                    continue
                f.write(json.dumps(article.to_dict(), ensure_ascii=False) + "\n")
                written += 1
                print(f"[{written}] {article.title}", file=sys.stderr)
    print(f"Wrote {written} articles to {args.out}", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wiki-rag", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    ingest = commands.add_parser(
        "ingest", help="download the articles of a Wikipedia category as JSON Lines"
    )
    ingest.add_argument(
        "--category",
        action="append",
        help=f"category name without prefix; repeatable (default: {DEFAULT_CATEGORY})",
    )
    ingest.add_argument("--depth", type=int, default=1, help="levels of subcategories to descend")
    ingest.add_argument("--limit", type=int, default=300, help="maximum number of articles")
    ingest.add_argument(
        "--min-chars", type=int, default=1000, help="skip articles with less prose than this"
    )
    ingest.add_argument("--lang", default="en", help="Wikipedia language edition")
    ingest.add_argument("--out", type=Path, default=Path("data/articles.jsonl"))
    ingest.set_defaults(run=_ingest)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
