"""Embed chunked RBA minutes and index them into ChromaDB (dense) and BM25 (sparse).

Reads `data/processed/chunks.json` (produced by `scripts/parse_documents.py`),
embeds every chunk with a sentence-transformers model, upserts the vectors plus
metadata into a persistent ChromaDB collection, and builds a parallel BM25
sparse index over the same chunks for hybrid retrieval.

Usage:
    python scripts/embed_and_index.py
    python scripts/embed_and_index.py --chunks data/processed/chunks.json \
        --chroma-dir data/vectorstore --bm25-path data/bm25_index.pkl
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import re
import time
from datetime import datetime
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("embed_and_index")

# --- Embedding configuration ---
# BAAI/bge-m3 is the model locked in by CLAUDE.md. It is a ~2.2GB multilingual
# embedding model; on a modern laptop CPU it embeds this corpus (~2.3k chunks)
# in a few minutes with no GPU required.
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_BATCH_SIZE = 32
CHROMA_COLLECTION_NAME = "rba_minutes"

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def load_chunks(chunks_path: Path) -> list[dict]:
    """Load the chunk records produced by `parse_documents.py`.

    Args:
        chunks_path: Path to `chunks.json`.

    Returns:
        A list of chunk dicts.

    Raises:
        FileNotFoundError: If the chunks file does not exist.
    """
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"{chunks_path} not found. Run scripts/parse_documents.py first."
        )
    with chunks_path.open("r", encoding="utf-8") as f:
        chunks = json.load(f)
    logger.info("Loaded %d chunks from %s", len(chunks), chunks_path)
    return chunks


def contextualize(chunk: dict) -> str:
    """Prepend a human-readable date header to a chunk's text before indexing.

    Many RBA "hold rate unchanged" meetings are phrased almost identically, so
    a raw-text embedding or BM25 match on wording alone often can't tell two
    meetings apart. Putting the meeting date in the indexed text itself (both
    dense and sparse) gives both retrievers a direct textual signal to key on.

    Args:
        chunk: A chunk dict with "meeting_date" (ISO "YYYY-MM-DD") and "text".

    Returns:
        The chunk text prefixed with a "[RBA Minutes - D Month YYYY]" header.
    """
    dt = datetime.strptime(chunk["meeting_date"], "%Y-%m-%d")
    human_date = f"{dt.day} {dt.strftime('%B %Y')}"
    return f"[RBA Minutes - {human_date}]\n{chunk['text']}"


def tokenize(text: str) -> list[str]:
    """Lowercase, alphanumeric tokenizer used for the BM25 sparse index.

    Args:
        text: Raw chunk text.

    Returns:
        A list of lowercase alphanumeric tokens.
    """
    return _TOKEN_RE.findall(text.lower())


def embed_chunks(chunks: list[dict], model_name: str = EMBEDDING_MODEL) -> list[list[float]]:
    """Embed chunk text with a sentence-transformers model.

    Args:
        chunks: Chunk dicts, each with a "text" field.
        model_name: HuggingFace model id to load.

    Returns:
        A list of embedding vectors, one per chunk, in input order.
    """
    logger.info("Loading embedding model %s (first run downloads weights)...", model_name)
    t0 = time.time()
    model = SentenceTransformer(model_name)
    logger.info("Model loaded in %.1fs", time.time() - t0)

    texts = [contextualize(c) for c in chunks]
    t0 = time.time()
    embeddings = model.encode(
        texts,
        batch_size=EMBEDDING_BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    logger.info("Embedded %d chunks in %.1fs", len(texts), time.time() - t0)
    return embeddings.tolist()


def build_chroma_index(
    chunks: list[dict], embeddings: list[list[float]], chroma_dir: Path
) -> None:
    """Upsert chunks and their embeddings into a persistent ChromaDB collection.

    Args:
        chunks: Chunk dicts (provide ids, documents, and metadata).
        embeddings: Pre-computed embedding vectors, aligned with `chunks`.
        chroma_dir: Directory for ChromaDB's persistent storage.
    """
    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_dir))
    client.delete_collection(CHROMA_COLLECTION_NAME) if CHROMA_COLLECTION_NAME in {
        c.name for c in client.list_collections()
    } else None
    collection = client.create_collection(CHROMA_COLLECTION_NAME)

    ids = [c["chunk_id"] for c in chunks]
    documents = [contextualize(c) for c in chunks]
    metadatas = [
        {
            "source_file": c["source_file"],
            "document_title": c["document_title"],
            "meeting_date": c["meeting_date"],
            "section": c["section"],
            "chunk_index": c["chunk_index"],
        }
        for c in chunks
    ]

    batch_size = 500
    for start in range(0, len(ids), batch_size):
        end = start + batch_size
        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadatas[start:end],
        )
    logger.info("Indexed %d chunks into ChromaDB collection '%s' at %s", len(ids), CHROMA_COLLECTION_NAME, chroma_dir)


def build_bm25_index(chunks: list[dict], bm25_path: Path) -> None:
    """Build and pickle a BM25 sparse index over the chunk corpus.

    The pickle stores a dict with the fitted `BM25Okapi` index plus the
    aligned `chunk_ids` list, so retrieval code can map BM25 result positions
    back to chunk ids without re-tokenizing or re-loading `chunks.json`.

    Args:
        chunks: Chunk dicts.
        bm25_path: Output path for the pickled index.
    """
    tokenized_corpus = [tokenize(contextualize(c)) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    bm25_path.parent.mkdir(parents=True, exist_ok=True)
    with bm25_path.open("wb") as f:
        pickle.dump({"bm25": bm25, "chunk_ids": [c["chunk_id"] for c in chunks]}, f)
    logger.info("Wrote BM25 index over %d chunks to %s", len(chunks), bm25_path)


def main() -> None:
    """CLI entry point: embed all chunks and build the dense + sparse indexes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, default=Path("data/processed/chunks.json"))
    parser.add_argument("--chroma-dir", type=Path, default=Path("data/vectorstore"))
    parser.add_argument("--bm25-path", type=Path, default=Path("data/bm25_index.pkl"))
    parser.add_argument("--model", type=str, default=EMBEDDING_MODEL)
    args = parser.parse_args()

    chunks = load_chunks(args.chunks)
    if not chunks:
        raise ValueError(f"No chunks found in {args.chunks}")

    embeddings = embed_chunks(chunks, model_name=args.model)
    build_chroma_index(chunks, embeddings, args.chroma_dir)
    build_bm25_index(chunks, args.bm25_path)

    dates = sorted({c["meeting_date"] for c in chunks})
    logger.info(
        "Summary: %d chunks | %d documents | date range %s to %s | embedding model %s | dim %d",
        len(chunks),
        len(dates),
        dates[0],
        dates[-1],
        args.model,
        len(embeddings[0]),
    )


if __name__ == "__main__":
    main()
