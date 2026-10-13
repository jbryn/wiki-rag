# RAG System for Wikipedia Articles

A search and answer generation system based on Wikipedia articles from the Philosophy category.

## Requirements

- Python 3.x
- Groq API key

## Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or venv\Scripts\activate  # Windows

# Install PyTorch
pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu

# Install other packages
pip3 install streamlit langchain langchain-groq langchain-community faiss-cpu sentence-transformers
```

Create a `.env` file in the project root directory and add your Groq API key:

```
GROQ_API_KEY=your_api_key
```

## Project Files

- `app.py` - main application code
- `summaries_status.json` - article summaries cache
- `articles.json` - articles database
- `src/wiki_rag/` - package for the new pipeline (in progress)

## Running the Application

```bash
streamlit run app.py
```

The application will be available at http://localhost:8501

## Building the corpus

The `wiki-rag ingest` command downloads the articles of one or more English Wikipedia
categories through the MediaWiki API and writes them, split into sections, to a JSON Lines file:

```bash
uv run wiki-rag ingest \
  --category Philosophical_concepts --category Philosophical_theories \
  --depth 1 --limit 300 --out data/articles.jsonl
```

Each line holds one article with its URL, revision id and a list of sections. Every section
keeps its heading path (for example `History > Western`), so answers can cite the exact part
of the article they came from. Reference-only sections such as "See also" are dropped, and
articles with less than `--min-chars` of prose are skipped.

## Indexing and searching

```bash
uv run wiki-rag index                                   # data/articles.jsonl -> data/index
uv run wiki-rag search "what did Sartre mean by self-deception" --mode hybrid -k 5
```

`index` splits every section into chunks of at most `--max-chars` characters (never crossing a
section boundary), embeds them locally with `BAAI/bge-small-en-v1.5` through
[fastembed](https://github.com/qdrant/fastembed) and saves the result to disk.

`search` supports three modes:

- `bm25` - lexical matching with Okapi BM25
- `dense` - cosine similarity between embeddings
- `hybrid` (default) - both rankings merged with reciprocal rank fusion

Each result links to the section of the Wikipedia article it came from.

## Development

The project is managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync                 # install the package and dev tools
uv run ruff check .     # lint
uv run ruff format .    # format
uv run pytest           # tests
```

The same checks run in CI on every pull request.
