"""CLI runner for the AusEcon RBA policy agent.

Usage:
    python -m agent.run "What did the RBA say about inflation in June 2023?"
    python -m agent.run   # interactive loop
"""

from __future__ import annotations

import logging
import sys

from langchain_core.messages import HumanMessage

from agent.graph import build_agent

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("agent.run")


def ask(agent, question: str) -> str:
    """Run one question through the agent and return the final text answer.

    Args:
        agent: A compiled LangGraph agent from `build_agent()`.
        question: The user's natural-language question.

    Returns:
        The agent's final response text.
    """
    result = agent.invoke({"messages": [HumanMessage(content=question)]})
    return extract_text(result["messages"][-1])


def extract_text(message) -> str:
    """Extract plain answer text from a message whose `.content` may be a string
    or a list of content blocks (Claude's extended-thinking responses include
    non-text blocks such as `thinking` alongside the final `text` block).

    Args:
        message: A LangChain message object (e.g. the final AIMessage).

    Returns:
        The concatenated text of all "text"-type content blocks, or the raw
        content if it is already a plain string.
    """
    content = message.content
    if isinstance(content, str):
        return content
    return "\n".join(
        block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"
    ).strip()


class Conversation:
    """A multi-turn session that keeps message history across calls to `ask`.

    `agent.run.ask()` (used by the single-shot CLI and `eval/run_eval.py`) is
    intentionally stateless - each call is one independent question. This
    class is for the interactive REPL, where follow-up questions like "what
    about the meeting before that?" should resolve against prior turns.
    """

    def __init__(self, agent):
        self.agent = agent
        self.history: list = []

    def ask(self, question: str) -> str:
        """Run one turn, appending to and updating the conversation history.

        On failure, the history is left unchanged (the failed turn is not
        recorded), so a later retry doesn't replay a broken exchange.

        Args:
            question: The user's natural-language question.

        Returns:
            The agent's final response text for this turn.
        """
        turn_input = self.history + [HumanMessage(content=question)]
        result = self.agent.invoke({"messages": turn_input})
        self.history = result["messages"]
        return extract_text(self.history[-1])


def main() -> None:
    """CLI entry point: answer a single question from argv, or loop interactively."""
    try:
        agent = build_agent()
    except ValueError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        print(ask(agent, question))
        return

    print("AusEcon agent ready. Ask a question about RBA monetary policy (Ctrl+C to exit).")
    print("Follow-up questions can refer back to earlier turns in this session.")
    conversation = Conversation(agent)
    while True:
        try:
            question = input("\n> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question.strip():
            continue
        try:
            print(conversation.ask(question))
        except Exception:
            logger.exception("Agent failed to answer question: %r", question)
            print("Sorry, something went wrong answering that question.")


if __name__ == "__main__":
    main()
