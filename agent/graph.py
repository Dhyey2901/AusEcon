"""LangGraph ReAct agent for answering questions about RBA monetary policy."""

from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent

from agent.config import AGENT_MODEL, ANTHROPIC_API_KEY
from agent.tools.calculator import calculate_rate_impact
from agent.tools.search import search_rba_documents
from agent.tools.trend import summarise_policy_trend

SYSTEM_PROMPT = """You are AusEcon, an assistant answering questions about Reserve Bank of \
Australia (RBA) monetary policy using the Board's published minutes.

Rules you must always follow:
1. Use `search_rba_documents` to find relevant passages before answering any factual question \
about RBA policy, reasoning, or economic conditions discussed in the minutes.
2. Use `summarise_policy_trend` when asked how the RBA's stance evolved over a date range, then \
synthesise the trend yourself from the passages it returns.
3. Use `calculate_rate_impact` for mortgage repayment / rate-change arithmetic questions.
4. Every factual claim drawn from the minutes must cite the meeting date and source file shown \
in the retrieved passages, e.g. "(2023-06-06, Minutes of the Monetary Policy Meeting ...)".
5. If retrieval returns no relevant passages, say "I don't know" rather than guessing or relying \
on prior knowledge. Do not fabricate dates, figures, or quotes.
6. Be concise and precise. Do not speculate beyond what the retrieved passages support."""

TOOLS = [search_rba_documents, summarise_policy_trend, calculate_rate_impact]


def build_agent():
    """Construct the compiled LangGraph ReAct agent.

    Returns:
        A compiled LangGraph graph exposing `.invoke({"messages": [...]})`.

    Raises:
        ValueError: If `ANTHROPIC_API_KEY` is not set.
    """
    if not ANTHROPIC_API_KEY:
        raise ValueError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and fill it in."
        )

    model = ChatAnthropic(model=AGENT_MODEL, api_key=ANTHROPIC_API_KEY)
    return create_react_agent(model, TOOLS, prompt=SYSTEM_PROMPT)
