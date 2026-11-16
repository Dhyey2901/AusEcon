"""`summarise_policy_trend` tool: retrieve RBA policy-stance passages across a date range.

This tool does retrieval only, not summarisation. It returns the most
relevant passages from each meeting in the requested window so the agent's
own reasoning step can synthesise the trend narrative, citing each date.
Keeping the LLM call in the agent loop (instead of nesting a second LLM call
inside the tool) keeps tool execution cheap, deterministic, and testable.
"""

from __future__ import annotations

import logging

from langchain_core.tools import tool

from agent.tools.retrieval import hybrid_search

logger = logging.getLogger("agent.trend")

_DEFAULT_TREND_QUERY = (
    "cash rate decision policy stance outlook inflation unemployment economic conditions"
)


@tool
def summarise_policy_trend(start_date: str, end_date: str, focus: str = "") -> str:
    """Retrieve RBA policy-stance passages across a date range to analyse how the stance changed.

    Use this when a user asks how the RBA's position evolved over multiple
    meetings (e.g. "how did the RBA's stance change between 2022 and 2023").
    After calling this, synthesise the trend yourself from the returned
    passages, citing the meeting date for each point you make.

    Args:
        start_date: Start of the date range, ISO format "YYYY-MM-DD".
        end_date: End of the date range, ISO format "YYYY-MM-DD" (inclusive).
        focus: Optional topic to focus the retrieval on (e.g. "inflation
            outlook", "labour market"). Defaults to general policy stance.

    Returns:
        Passages grouped by meeting date, most relevant first within each
        date, or a message saying nothing relevant was found.
    """
    query = focus.strip() or _DEFAULT_TREND_QUERY
    try:
        chunks = hybrid_search(
            query, top_k=15, candidate_k=100, meeting_date_range=(start_date, end_date)
        )
    except Exception:
        logger.exception(
            "summarise_policy_trend failed for range=%s..%s", start_date, end_date
        )
        return "Retrieval failed due to an internal error. Treat this as no results found."

    if not chunks:
        return (
            f"No relevant passages were found in the RBA minutes between "
            f"{start_date} and {end_date}."
        )

    by_date: dict[str, list] = {}
    for chunk in chunks:
        by_date.setdefault(chunk.meeting_date, []).append(chunk)

    lines = []
    for date in sorted(by_date):
        lines.append(f"=== Meeting: {date} ===")
        for chunk in by_date[date]:
            lines.append(f"({chunk.section}, source: {chunk.source_file})\n{chunk.text}")
    return "\n\n".join(lines)
