from __future__ import annotations

import os
from typing import List

import chromadb
import streamlit as st
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv

from prepare_data import (
    COLLECTION_NAME,
    DEFAULT_MODEL,
    PERSIST_DIR,
    build_chroma_collection,
    get_embedding_function,
)


st.set_page_config(page_title="Semantic Search: AG News", page_icon="🔎", layout="wide")


@st.cache_resource(show_spinner=False)
def load_embedding_function(model_name: str = DEFAULT_MODEL):
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for embedding calls (set it in .env or your env).")
    return OpenAIEmbeddingFunction(model_name=model_name, api_key=api_key)


@st.cache_resource(show_spinner=True)
def load_collection():
    try:
        embedding_function = load_embedding_function()
    except Exception as exc:  # surface missing API key or config issues
        st.error(f"Failed to load embedding function: {exc}")
        st.stop()
    if not PERSIST_DIR.exists() or not any(PERSIST_DIR.iterdir()):
        with st.spinner("Building vector store (first run takes ~1 minute)..."):
            build_chroma_collection(
                split="train[:2500]",
                chunk_size=600,
                overlap=80,
                reset=True,
                model_name=DEFAULT_MODEL,
            )

    client = chromadb.PersistentClient(path=str(PERSIST_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME, embedding_function=embedding_function
    )
    return collection


def render_results(documents: List[str], distances: List[float], metadatas: List[dict]):
    st.subheader("Results")
    for doc, dist, meta in zip(documents, distances, metadatas):
        st.markdown(f"**Title:** {meta.get('title', 'N/A')}")
        st.markdown(f"**Label:** {meta.get('label', 'N/A')}")
        st.markdown(f"**Similarity:** {1 - dist:.3f}")
        st.markdown(f"**Chunk:** {meta.get('chunk_index', 0)}")
        st.write(doc)
        st.markdown("---")


def main():
    st.title("Semantic Search on AG News")
    st.write(
        "Ask a question about news articles. Queries are embedded with a transformer "
        "and matched against a Chroma vector store built from the AG News dataset."
    )

    collection = load_collection()

    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input("Enter a query", value="SpaceX launch success")
    with col2:
        top_k = st.slider("Results", min_value=3, max_value=15, value=6)

    st.markdown("Try searches like `election results`, `hurricane damage`, `stock market rally`.")

    if query:
        results = collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "distances", "metadatas"],
        )
        docs = results["documents"][0]
        dists = results["distances"][0]
        metas = results["metadatas"][0]
        render_results(docs, dists, metas)


if __name__ == "__main__":
    main()
