"""
Utilities to download a text dataset, chunk it, embed it, and store it in a Chroma DB.
The Streamlit app imports the build helpers, so keep the defaults lightweight.
"""
from __future__ import annotations

import argparse
import shutil
import os
from pathlib import Path
from typing import List, Tuple
from dotenv import load_dotenv

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from datasets import load_dataset
from tqdm import tqdm

DATASET_NAME = "ag_news"
COLLECTION_NAME = "ag_news_chunks"
PERSIST_DIR = Path("chroma_db")
DEFAULT_MODEL = "text-embedding-3-small"
LABEL_NAMES = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}


def chunk_text(
    text: str, *, chunk_size: int = 600, overlap: int = 80
) -> List[str]:
    """Simple word-based chunking with configurable overlap."""
    words = text.split()
    if not words:
        return []

    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = max(end - overlap, 0)
    return chunks


def get_embedding_function(model_name: str = DEFAULT_MODEL):
    """Use OpenAI embeddings (requires OPENAI_API_KEY env var)."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for embedding calls (set it in .env or your env).")
    return OpenAIEmbeddingFunction(model_name=model_name, api_key=api_key)


def _format_article(row) -> Tuple[str, str, str]:
    """Normalize the article fields and return (title, body, label_name)."""
    title = row.get("title")
    body = row.get("description") or row.get("text") or ""
    if not title:
        title = " ".join(body.split()[:12]) or "Untitled"
    label_idx = row.get("label")
    label_name = LABEL_NAMES.get(label_idx, str(label_idx))
    return title.strip(), body.strip(), label_name


def build_chroma_collection(
    *,
    split: str = "train[:4000]",
    chunk_size: int = 600,
    overlap: int = 80,
    reset: bool = False,
    model_name: str = DEFAULT_MODEL,
) -> Tuple[int, Path]:
    """
    Load the dataset, chunk and embed, and persist to Chroma.

    Returns the number of chunks stored and the DB path.
    """
    if reset and PERSIST_DIR.exists():
        shutil.rmtree(PERSIST_DIR)

    dataset = load_dataset(DATASET_NAME, split=split)
    embedding_function = get_embedding_function(model_name)
    client = chromadb.PersistentClient(path=str(PERSIST_DIR))

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            # If the collection does not exist yet, ignore.
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME, embedding_function=embedding_function
    )

    documents: List[str] = []
    metadatas: List[dict] = []
    ids: List[str] = []

    for idx, row in enumerate(tqdm(dataset, desc="Chunking and embedding")):
        title, body, label_name = _format_article(row)
        text = f"Title: {title}\n\nArticle: {body}"
        for chunk_idx, chunk in enumerate(chunk_text(text, chunk_size=chunk_size, overlap=overlap)):
            documents.append(chunk)
            metadatas.append(
                {
                    "label": label_name,
                    "title": title,
                    "chunk_index": chunk_idx,
                }
            )
            ids.append(f"ag_{idx}_{chunk_idx}")

        # Avoid oversized batches; add in slices.
        if len(documents) >= 256:
            collection.add(documents=documents, metadatas=metadatas, ids=ids)
            documents, metadatas, ids = [], [], []

    if documents:
        collection.add(documents=documents, metadatas=metadatas, ids=ids)

    return collection.count(), PERSIST_DIR


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Chroma DB for semantic search.")
    parser.add_argument("--split", default="train[:4000]", help="Datasets split to use.")
    parser.add_argument("--chunk-size", type=int, default=600, help="Words per chunk.")
    parser.add_argument("--overlap", type=int, default=80, help="Word overlap between chunks.")
    parser.add_argument("--reset", action="store_true", help="Delete any existing DB before writing.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL, help="SentenceTransformer model name.")
    return parser.parse_args()


def main():
    args = _parse_args()
    count, db_path = build_chroma_collection(
        split=args.split,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        reset=args.reset,
        model_name=args.model_name,
    )
    print(f"Stored {count} chunks in {db_path}")


if __name__ == "__main__":
    main()
