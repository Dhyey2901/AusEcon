# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

AusEcon — an Agentic RAG system over Reserve Bank of Australia (RBA) monetary policy documents. Users ask questions about RBA cash rate decisions, policy reasoning, and Australian macro indicators and get accurate, cited answers grounded in real RBA documents. See [README.md](README.md) for the full stack table, structure, and build phases.

## Tech stack — locked in, do not substitute without asking

- PDF parsing: pdfplumber
- Chunking: custom (800 char chunks, 150 char overlap, sentence-boundary aware)
- Embeddings: BAAI/bge-m3
- Vector store: ChromaDB (persistent, local)
- Sparse index: BM25 (rank-bm25) — hybrid retrieval
- Agent framework: LangGraph (ReAct pattern)
- LLM: Claude Sonnet via the Anthropic API
- Evaluation: RAGAS (faithfulness, answer relevance, context precision, context recall)
- Experiment tracking: MLflow
- API: FastAPI
- Containerisation: Docker + docker-compose
- CI/CD: GitHub Actions
- Deployment: Railway

## Data

- `data/raw/minutes/` — 53 RBA board minutes PDFs (2021–2025), the primary corpus. Filenames are the raw scraped titles (e.g. `1 April 2025 _ Minutes of the Monetary Policy Board Meeting _ RBA.pdf`) — parse the date out of the filename or document text rather than renaming the source files.
- ABS API — live macro data (CPI, unemployment, GDP), used as an agent tool, not pre-indexed.
- RBA stats CSVs — cash rate history, inflation, used as an agent tool.

Generated artifacts (`data/processed/`, `data/vectorstore/`, `data/bm25_index.pkl`) are gitignored and rebuilt via the scripts in `scripts/`, never edited by hand.

## How to help on this project

- Write production-quality code, not tutorial code.
- Always include error handling and logging.
- Add docstrings to every function.
- When asked to build something, build the full thing — not a skeleton.
- If a better approach exists than what was described, say so before building it.
- Keep the codebase consistent — same patterns, same naming conventions throughout.
- After finishing each phase, state what the next phase/step is.

## Build phases

1. Parse PDFs → chunk → embed → index into ChromaDB + BM25 (`scripts/`)
2. LangGraph agent with 4 tools (`agent/`)
3. RAGAS eval pipeline with 80 Q&A ground truth set (`eval/`)
4. MLflow experiment tracking + query monitoring
5. FastAPI + Docker + Railway deploy + GitHub Actions CI/CD (`api/`)

## Environment

- Target Python 3.11 or 3.12 — pinned deps (torch, chromadb) may lag behind very new Python releases.
- Secrets live in `.env` (gitignored); `.env.example` documents required keys (`ANTHROPIC_API_KEY`, `ABS_API_KEY`).
