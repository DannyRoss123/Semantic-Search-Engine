# Semantic Search Engine

Semantic search over the AG News dataset built with Chroma for storage and a Streamlit UI for querying. The app can run locally or be deployed to Hugging Face Spaces.

## Setup

```bash
pip install -r requirements.txt
```

- Set `OPENAI_API_KEY` in your environment (used for embeddings).

## Build the vector store

```bash
python prepare_data.py --split train[:4000] --reset
```

- Downloads AG News, chunks each article, embeds with `text-embedding-3-small` (OpenAI), and persists to `chroma_db/`.
- Tweak `--split`, `--chunk-size`, or `--overlap` to adjust runtime and chunk granularity.
- Use `--reset` to rebuild from scratch.

## Run the app locally

```bash
streamlit run app.py
```

On first launch the app will auto-build a smaller DB (`train[:2500]`) if none exists.

## Deploy to Hugging Face Spaces

1. Create a new Space and choose **Streamlit** as the SDK.
2. Upload `app.py`, `prepare_data.py`, `requirements.txt`, and `.gitignore`.
3. In the Space secrets, add `OPENAI_API_KEY`.
4. The Space will install dependencies and run `app.py`; if the Chroma DB is missing, it builds automatically on first load.

## Project structure

- `app.py` — Streamlit UI for querying the vector store.
- `prepare_data.py` — dataset download, chunking, embedding, and Chroma persistence.
- `requirements.txt` — runtime dependencies (lighter: OpenAI embeddings, no local PyTorch).
- `chroma_db/` — generated Chroma database (ignored in git).
