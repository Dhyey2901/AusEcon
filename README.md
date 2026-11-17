# AusEcon

An agentic RAG system over Reserve Bank of Australia (RBA) Monetary Policy Board minutes. Ask
questions about RBA cash rate decisions and policy reasoning and get answers grounded in, and
cited to, the real minutes — with an "I don't know" fallback when nothing relevant is retrieved.

This is an honest MVP covering ingestion, a working LangGraph ReAct agent, and a measured
evaluation pipeline (Phases 1-3 below). FastAPI/Docker/deployment (Phase 4) was not built — see
Known limitations.

## What actually works

- **Ingestion**: 53 RBA minutes PDFs (2021-02-02 to 2026-06-16) parsed, chunked, embedded, and
  indexed into ChromaDB (dense) + BM25 (sparse).
- **Agent**: a LangGraph ReAct agent (Claude Sonnet) with 3 tools - hybrid document search, a
  mortgage rate-impact calculator, and cross-meeting trend retrieval. Verified end-to-end: it
  answers with correct citations, does the arithmetic correctly, and says "I don't know" on
  out-of-scope questions instead of fabricating an answer.
- **Evaluation**: 30 hand-verified Q&A pairs, run through the live agent with an LLM-as-judge
  scoring correctness. Real results below - not tuned to look good.

## Corpus stats (Phase 1)

- 53 source PDFs parsed, 53 successfully chunked (0 failures)
- 2,258 chunks, average 631 chars/chunk
- Date range: 2021-02-02 to 2026-06-16
- Embeddings: `BAAI/bge-m3` (1024-dim) - ran fine on a 16GB M-series laptop (~2.5 min for the
  full corpus on MPS), no need to fall back to a smaller model
- Indexed in ChromaDB (persistent, local) + BM25 (`rank-bm25`)

## Eval results (Phase 3, real numbers from `eval/results.json`)

| Metric | Result | n |
|---|---|---|
| Answer correctness (LLM-as-judge vs. hand-verified reference) | **100%** (25/25) | 25 answerable questions |
| Abstention rate on unanswerable questions | **100%** (5/5) | 5 deliberately unanswerable questions |
| Standalone retrieval hit rate (single raw-question query, top-5) | **88%** (22/25) | 25 answerable questions |

The retrieval hit-rate number started at 36% - many RBA "hold rate unchanged" decisions are
phrased almost identically across different meetings, so a single naive top-5 query on the raw
question text often couldn't disambiguate the exact meeting date from wording alone. Two fixes
closed most of that gap: (1) a `[RBA Minutes - D Month YYYY]` header is prepended to each chunk's
text before both embedding and BM25 indexing, so the meeting date is a direct textual signal to
both retrievers, not just metadata; (2) `hybrid_search` now parses a month/year or bare year
mentioned in the query itself (e.g. "June 2023") and applies it as a metadata filter before
ranking, in `agent/tools/retrieval.py`'s `_extract_date_hint`. That took standalone hit rate from
36% to 88%. The remaining 3 misses are genuinely hard for a heuristic: one question names two
years ("2022-2023 tightening cycle", ambiguous on purpose), and two use indirect phrasing
("the December meeting", "first hike of the 2022 cycle") with no adjacent explicit month+year to
parse. The agent's actual answer correctness was 100% even before this fix, because its ReAct loop
reformulates queries and sometimes issues multiple tool calls rather than relying on one blind
retrieval pass - which this standalone diagnostic measures, but the agent itself doesn't rely on.
Re-run with `python -m eval.run_eval`.

## Tech stack

| Layer | Choice |
|---|---|
| PDF parsing | pdfplumber |
| Chunking | Custom — 800 char chunks, 150 char overlap, sentence-boundary aware |
| Embeddings | BAAI/bge-m3 |
| Vector store | ChromaDB (persistent, local) |
| Sparse index | BM25 (rank-bm25) — hybrid retrieval (reciprocal rank fusion) |
| Agent framework | LangGraph (`create_react_agent`, ReAct pattern) |
| LLM | Claude Sonnet (Anthropic API) |
| Evaluation | Custom: hit-rate + LLM-as-judge correctness + abstention rate (not RAGAS - see limitations) |
| Testing | pytest (38 tests: chunker, calculator, RRF fusion) |
| CI | GitHub Actions (runs pytest on push, no API key needed) |

## Project structure

```
ausecon/
├── data/
│   ├── raw/minutes/         # 53 RBA board minutes PDFs (2021-2026)
│   ├── processed/chunks.json   # gitignored, generated
│   ├── vectorstore/             # gitignored, generated (ChromaDB)
│   └── bm25_index.pkl            # gitignored, generated
├── scripts/
│   ├── parse_documents.py   # pdfplumber -> sentence-aware chunks
│   └── embed_and_index.py   # embed -> ChromaDB + BM25
├── agent/
│   ├── config.py             # env-driven config
│   ├── graph.py               # LangGraph ReAct agent + system prompt
│   ├── run.py                  # CLI runner
│   └── tools/
│       ├── retrieval.py        # hybrid dense+BM25 search with RRF
│       ├── search.py            # search_rba_documents tool
│       ├── trend.py              # summarise_policy_trend tool
│       └── calculator.py          # calculate_rate_impact tool
├── eval/
│   ├── ground_truth.json    # 30 hand-verified Q&A pairs (25 answerable + 5 not)
│   ├── run_eval.py             # runs the live agent + LLM-as-judge
│   └── results.json              # real measured results
├── tests/                    # pytest: chunker, calculator, RRF (no API calls)
├── .github/workflows/tests.yml
└── requirements.txt
```

## Agent tools

1. `search_rba_documents` — hybrid retrieval (ChromaDB dense + BM25 sparse, merged with
   reciprocal rank fusion), returns passages with meeting date + source file citations.
2. `calculate_rate_impact` — pure-Python monthly mortgage repayment delta for a given loan
   amount, term, and rate change. Unit tested, no LLM call.
3. `summarise_policy_trend` — retrieves passages across a date range (ChromaDB `$in` date filter
   + BM25 restricted to that window) for the agent to synthesise a trend narrative from, citing
   each meeting date.

A fourth tool (`fetch_live_macro_data`, pulling CPI/unemployment/GDP from the ABS API) was
planned as a stretch goal and was **not built** - see Known limitations.

## Setup

```bash
# This project is pinned to Python 3.12 (see .python-version) - torch/chromadb
# don't reliably ship wheels for bleeding-edge Python releases. If you don't
# have 3.12, install it with pyenv: `pyenv install 3.12.8`.
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in ANTHROPIC_API_KEY (get one at console.anthropic.com/settings/keys)
```

### Run it

```bash
# Phase 1: parse + chunk + embed + index (only needed once, or after changing data/raw/minutes)
python scripts/parse_documents.py
python scripts/embed_and_index.py

# Phase 2: talk to the agent
python -m agent.run "What did the RBA decide in June 2023 and why?"
python -m agent.run   # interactive mode

# Phase 3: run the eval suite (makes real Anthropic API calls)
python -m eval.run_eval
python -m eval.run_eval --limit 5   # quick smoke test

# Tests (no API calls)
pytest tests/ -v
```

## Known limitations

- **No Phase 4/5**: no FastAPI endpoint, no Docker, no Railway deployment, no MLflow tracking.
  The agent is CLI-only right now.
- **Evaluation isn't RAGAS**: the original plan called for RAGAS (faithfulness, answer relevance,
  context precision/recall). The actual eval is a simpler custom harness (retrieval hit-rate +
  LLM-as-judge correctness + abstention rate on 30 hand-written Q&A pairs) - sufficient to
  demonstrate the agent works, but not as rigorous as RAGAS would be.
- **Only 1 of 4 originally-planned tools omitted**: the ABS live macro data tool
  (`fetch_live_macro_data`) was never implemented.
- **Standalone retrieval hit-rate is a conservative, partial signal** (88%, see Eval results
  above) - it measures one blind top-5 query per question, not what the agent's actual
  multi-step retrieval achieves (100% answer correctness). The 3 remaining misses are queries
  with ambiguous or indirect date phrasing that the date-hint heuristic in `retrieval.py` can't
  resolve (see Eval results for detail).
- **Dependency versions were bumped off the original plan**: `langgraph`/`langchain-anthropic`/
  `langchain-core` were originally pinned to late-2024 releases that predate Claude's current
  "adaptive thinking" response format and corrupt multi-turn tool conversations when replaying
  those blocks. Bumped to the oldest versions that work correctly with the current API
  (see `requirements.txt` for exact versions and reasoning).
- **No reranking model**: retrieval fusion is RRF over dense + BM25 only; no cross-encoder
  reranker was added on top.
- **System prompt is not leak-proof**: the agent will disclose its system prompt if asked
  directly, and in at least one manual test, the "verbatim" text it returned included a
  hallucinated fragment unrelated to the actual prompt. Don't rely on this agent to keep its
  instructions confidential, and don't demo prompt-injection-hardening as a feature - it isn't
  one. (Multi-turn conversation memory - the other gap found during manual boundary testing -
  is implemented; see `agent/run.py`'s `Conversation` class.)
- **Single-threaded eval**: `run_eval.py` runs questions sequentially; a 30-question run takes
  a few minutes and a small amount of real API spend.
