"""Hybrid (dense + sparse) retrieval over the indexed RBA minutes corpus.

Loads the persistent ChromaDB collection and the pickled BM25 index built by
`scripts/embed_and_index.py`, and exposes a single `hybrid_search` function
that merges dense and sparse rankings with reciprocal rank fusion (RRF).
"""

from __future__ import annotations

import logging
import pickle
import re
from dataclasses import dataclass
from functools import lru_cache

import chromadb

from agent.config import (
    BM25_INDEX_PATH,
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
    EMBEDDING_MODEL,
)

logger = logging.getLogger("agent.retrieval")

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Reciprocal rank fusion constant: dampens the weight of low ranks. 60 is the
# standard default used in the original RRF paper and most hybrid-search
# implementations.
RRF_K = 60


@dataclass
class RetrievedChunk:
    """A single retrieved passage with provenance for citation."""

    chunk_id: str
    text: str
    source_file: str
    document_title: str
    meeting_date: str
    section: str
    score: float


def tokenize(text: str) -> list[str]:
    """Lowercase, alphanumeric tokenizer matching the one used at index time."""
    return _TOKEN_RE.findall(text.lower())


@lru_cache(maxsize=1)
def _get_embedding_model():
    """Lazily load and cache the sentence-transformers embedding model."""
    from sentence_transformers import SentenceTransformer

    logger.info("Loading embedding model %s for retrieval...", EMBEDDING_MODEL)
    return SentenceTransformer(EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def _get_chroma_collection():
    """Lazily open the persistent ChromaDB collection."""
    client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
    return client.get_collection(CHROMA_COLLECTION_NAME)


@lru_cache(maxsize=1)
def _get_bm25_index():
    """Lazily load the pickled BM25 index and its aligned chunk id list."""
    with BM25_INDEX_PATH.open("rb") as f:
        data = pickle.load(f)
    return data["bm25"], data["chunk_ids"]


@lru_cache(maxsize=1)
def _get_chunk_id_to_date() -> dict[str, str]:
    """Build and cache a chunk_id -> meeting_date map from the ChromaDB collection."""
    collection = _get_chroma_collection()
    all_records = collection.get(include=["metadatas"])
    return {
        chunk_id: meta["meeting_date"]
        for chunk_id, meta in zip(all_records["ids"], all_records["metadatas"])
    }


def _date_where_clause(meeting_date_range: tuple[str, str]) -> dict:
    """Build a ChromaDB `where` filter restricting results to a meeting-date window.

    ChromaDB's `$gte`/`$lte` operators require numeric operands, so date
    strings can't be range-compared directly. Instead, resolve the window to
    the exact set of meeting dates present in the corpus and filter with `$in`.
    """
    start, end = meeting_date_range
    dates_in_range = sorted({d for d in _get_chunk_id_to_date().values() if start <= d <= end})
    return {"meeting_date": {"$in": dates_in_range or ["__no_match__"]}}


def _dense_search(
    query: str, top_k: int, meeting_date_range: tuple[str, str] | None = None
) -> list[str]:
    """Run dense retrieval and return chunk ids ranked best-first."""
    model = _get_embedding_model()
    collection = _get_chroma_collection()
    query_embedding = model.encode([query], normalize_embeddings=True).tolist()
    where = _date_where_clause(meeting_date_range) if meeting_date_range else None
    results = collection.query(query_embeddings=query_embedding, n_results=top_k, where=where)
    return results["ids"][0] if results["ids"] else []


def _sparse_search(
    query: str, top_k: int, meeting_date_range: tuple[str, str] | None = None
) -> list[str]:
    """Run BM25 sparse retrieval and return chunk ids ranked best-first.

    When `meeting_date_range` is given, scores over the full corpus but only
    keeps candidates within the date window, pulling extra candidates first
    so the date-restricted result set isn't starved.
    """
    bm25, chunk_ids = _get_bm25_index()
    scores = bm25.get_scores(tokenize(query))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    if meeting_date_range is None:
        return [chunk_ids[i] for i in ranked[:top_k] if scores[i] > 0]

    start, end = meeting_date_range
    date_map = _get_chunk_id_to_date()
    matches = []
    for i in ranked:
        if scores[i] <= 0:
            break
        chunk_id = chunk_ids[i]
        if start <= date_map.get(chunk_id, "") <= end:
            matches.append(chunk_id)
        if len(matches) >= top_k:
            break
    return matches


def _reciprocal_rank_fusion(
    ranked_lists: list[list[str]], k: int = RRF_K
) -> list[tuple[str, float]]:
    """Merge multiple ranked id lists into one score-sorted list via RRF.

    Args:
        ranked_lists: One or more best-first lists of ids from different retrievers.
        k: RRF damping constant.

    Returns:
        (id, fused_score) pairs sorted by descending fused score.
    """
    fused_scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, item_id in enumerate(ranked):
            fused_scores[item_id] = fused_scores.get(item_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(fused_scores.items(), key=lambda pair: pair[1], reverse=True)


def hybrid_search(
    query: str,
    top_k: int = 5,
    candidate_k: int = 20,
    meeting_date_range: tuple[str, str] | None = None,
) -> list[RetrievedChunk]:
    """Retrieve the top passages for a query via hybrid dense + sparse search.

    Args:
        query: The natural-language search query.
        top_k: Number of fused results to return.
        candidate_k: Number of candidates each retriever contributes before fusion.
        meeting_date_range: Optional (start_date, end_date) ISO strings (inclusive)
            to restrict results to a meeting-date window.

    Returns:
        Up to `top_k` `RetrievedChunk` objects, best match first. Empty if no
        chunk is retrieved by either retriever.
    """
    dense_ids = _dense_search(query, candidate_k, meeting_date_range)
    sparse_ids = _sparse_search(query, candidate_k, meeting_date_range)
    fused = _reciprocal_rank_fusion([dense_ids, sparse_ids])

    if not fused:
        return []

    collection = _get_chroma_collection()
    ordered_ids = [chunk_id for chunk_id, _ in fused]
    records = collection.get(ids=ordered_ids, include=["documents", "metadatas"])
    by_id = {
        record_id: (doc, meta)
        for record_id, doc, meta in zip(records["ids"], records["documents"], records["metadatas"])
    }

    results: list[RetrievedChunk] = []
    for chunk_id, score in fused:
        if chunk_id not in by_id:
            continue
        doc, meta = by_id[chunk_id]
        if meeting_date_range is not None:
            start, end = meeting_date_range
            if not (start <= meta["meeting_date"] <= end):
                continue
        results.append(
            RetrievedChunk(
                chunk_id=chunk_id,
                text=doc,
                source_file=meta["source_file"],
                document_title=meta["document_title"],
                meeting_date=meta["meeting_date"],
                section=meta["section"],
                score=score,
            )
        )
        if len(results) >= top_k:
            break

    return results
