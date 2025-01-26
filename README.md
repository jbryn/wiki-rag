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
- `wiki_articles.json` - articles database

## Running the Application

```bash
streamlit run app.py
```

The application will be available at http://localhost:8501
