"""`search_rba_documents` tool: hybrid retrieval over the RBA minutes corpus."""

from __future__ import annotations

import logging

from langchain_core.tools import tool

from agent.tools.retrieval import hybrid_search

logger = logging.getLogger("agent.search")


def _format_results(chunks) -> str:
    """Format retrieved chunks into a citation-ready block of passages."""
    if not chunks:
        return "No relevant passages were found in the RBA minutes for this query."

    lines = []
    for i, chunk in enumerate(chunks, start=1):
        lines.append(
            f"[{i}] ({chunk.meeting_date}, {chunk.section}, source: {chunk.source_file})\n"
            f"{chunk.text}"
        )
    return "\n\n".join(lines)


@tool
def search_rba_documents(query: str, top_k: int = 5) -> str:
    """Search RBA Monetary Policy Board minutes for passages relevant to a query.

    Uses hybrid retrieval (dense embedding similarity + BM25 keyword search,
    merged with reciprocal rank fusion) over all indexed minutes documents.
    Always cite the meeting date and source file shown in brackets when using
    these passages in your answer.

    Args:
        query: The natural-language question or topic to search for.
        top_k: Number of passages to return (default 5).

    Returns:
        A formatted block of the most relevant passages with meeting date,
        section, and source file citations, or a message saying nothing
        relevant was found.
    """
    try:
        chunks = hybrid_search(query, top_k=top_k)
    except Exception:
        logger.exception("search_rba_documents failed for query=%r", query)
        return "Retrieval failed due to an internal error. Treat this as no results found."
    return _format_results(chunks)
