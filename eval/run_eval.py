"""Evaluate the AusEcon RBA agent against eval/ground_truth.json.

For each question, this script:
1. Runs the hybrid retriever directly to measure retrieval hit rate (did any
   retrieved chunk come from a meeting date the question is actually about?).
2. Runs the full LangGraph agent to get its final answer.
3. For answerable questions, scores answer correctness with an LLM-as-judge
   (Claude Sonnet, via a single plain message, not via the agent loop).
4. For unanswerable questions, checks whether the agent abstained ("I don't
   know") instead of fabricating an answer.

This script makes real Anthropic API calls (one per question for the agent,
one more per answerable question for judging) and is therefore not run in CI.

Usage:
    python -m eval.run_eval
    python -m eval.run_eval --limit 5   # quick smoke test on the first 5 questions
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import anthropic

from agent.config import AGENT_MODEL, ANTHROPIC_API_KEY
from agent.graph import build_agent
from agent.run import ask
from agent.tools.retrieval import hybrid_search

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("eval.run_eval")

GROUND_TRUTH_PATH = Path("eval/ground_truth.json")
RESULTS_PATH = Path("eval/results.json")

ABSTENTION_PHRASES = ("i don't know", "i do not know", "don't have", "can't find", "no relevant")

JUDGE_PROMPT_TEMPLATE = """You are grading an AI assistant's answer about RBA monetary policy minutes.

Question: {question}

Reference answer (ground truth, verified against the source minutes):
{reference_answer}

Assistant's answer:
{agent_answer}

Judge whether the assistant's answer is factually consistent with the reference answer. Minor \
differences in wording, phrasing, or level of detail are fine - focus on whether the key facts \
(dates, rate decisions, basis point amounts, reasoning) match and whether the assistant \
contradicts or fabricates anything that isn't in the reference answer.

Respond with exactly one line in this format: VERDICT: CORRECT or VERDICT: INCORRECT
Then on a new line, give a one-sentence rationale."""


def check_retrieval_hit(question: str, relevant_dates: list[str], top_k: int = 5) -> bool:
    """Run hybrid retrieval for `question` and check if any hit's date is relevant.

    Args:
        question: The eval question text, used as the retrieval query.
        relevant_dates: Meeting dates (ISO strings) the question is grounded in.
        top_k: Number of passages to retrieve.

    Returns:
        True if at least one retrieved chunk's meeting_date is in `relevant_dates`.
    """
    if not relevant_dates:
        return False
    chunks = hybrid_search(question, top_k=top_k)
    retrieved_dates = {c.meeting_date for c in chunks}
    return bool(retrieved_dates & set(relevant_dates))


def judge_answer(client: anthropic.Anthropic, question: str, reference_answer: str, agent_answer: str) -> dict:
    """Use Claude as an LLM judge to score answer correctness against the reference.

    Args:
        client: An initialised Anthropic client.
        question: The eval question.
        reference_answer: The hand-verified ground-truth answer.
        agent_answer: The agent's actual answer.

    Returns:
        A dict with "correct" (bool) and "rationale" (str).
    """
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question, reference_answer=reference_answer, agent_answer=agent_answer
    )
    response = client.messages.create(
        model=AGENT_MODEL,
        max_tokens=1024,
        output_config={"effort": "low"},
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(block.text for block in response.content if block.type == "text")
    first_line = text.strip().splitlines()[0] if text.strip() else ""
    correct = "VERDICT: CORRECT" in first_line.upper()
    rationale = text.strip().splitlines()[1] if len(text.strip().splitlines()) > 1 else ""
    return {"correct": correct, "rationale": rationale, "raw_judge_response": text.strip()}


def is_abstention(answer_text: str) -> bool:
    """Heuristically check whether an answer is an abstention ("I don't know")."""
    lowered = answer_text.lower()
    return any(phrase in lowered for phrase in ABSTENTION_PHRASES)


def main() -> None:
    """CLI entry point: run the full eval suite and write eval/results.json."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N questions.")
    parser.add_argument("--ground-truth", type=Path, default=GROUND_TRUTH_PATH)
    parser.add_argument("--output", type=Path, default=RESULTS_PATH)
    args = parser.parse_args()

    if not ANTHROPIC_API_KEY:
        raise SystemExit("ANTHROPIC_API_KEY is not set. Copy .env.example to .env and fill it in.")

    with args.ground_truth.open("r", encoding="utf-8") as f:
        questions = json.load(f)
    if args.limit:
        questions = questions[: args.limit]

    agent = build_agent()
    judge_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    per_question_results = []
    for item in questions:
        logger.warning("Running %s: %s", item["id"], item["question"])
        t0 = time.time()
        try:
            agent_answer = ask(agent, item["question"])
        except Exception:
            logger.exception("Agent failed on %s", item["id"])
            agent_answer = ""
        elapsed = time.time() - t0

        result = {
            "id": item["id"],
            "question": item["question"],
            "category": item["category"],
            "answerable": item["answerable"],
            "agent_answer": agent_answer,
            "elapsed_seconds": round(elapsed, 1),
        }

        if item["answerable"]:
            result["retrieval_hit"] = check_retrieval_hit(item["question"], item["relevant_dates"])
            judge_result = judge_answer(
                judge_client, item["question"], item["reference_answer"], agent_answer
            )
            result["judge_correct"] = judge_result["correct"]
            result["judge_rationale"] = judge_result["rationale"]
        else:
            result["abstained"] = is_abstention(agent_answer)

        per_question_results.append(result)

    answerable_results = [r for r in per_question_results if r["answerable"]]
    unanswerable_results = [r for r in per_question_results if not r["answerable"]]

    summary = {
        "n_questions": len(per_question_results),
        "n_answerable": len(answerable_results),
        "n_unanswerable": len(unanswerable_results),
        "retrieval_hit_rate": (
            sum(r["retrieval_hit"] for r in answerable_results) / len(answerable_results)
            if answerable_results
            else None
        ),
        "answer_correctness_rate": (
            sum(r["judge_correct"] for r in answerable_results) / len(answerable_results)
            if answerable_results
            else None
        ),
        "abstention_rate_on_unanswerable": (
            sum(r["abstained"] for r in unanswerable_results) / len(unanswerable_results)
            if unanswerable_results
            else None
        ),
    }

    output = {"summary": summary, "results": per_question_results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2))
    print(f"Full results written to {args.output}")


if __name__ == "__main__":
    main()
