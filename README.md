# AusEcon

An Agentic RAG system over Reserve Bank of Australia (RBA) monetary policy documents. Ask questions about RBA cash rate decisions, economic policy reasoning, and Australian macro indicators, and get accurate, cited answers grounded in real RBA board minutes and live economic data.

## What it does

AusEcon combines hybrid retrieval over 53 RBA Monetary Policy Board minutes (2021–2025) with a LangGraph ReAct agent that can also pull live macro data (CPI, unemployment, GDP) from the ABS API and reason about rate-change impacts — all answers are evaluated for faithfulness and relevance with RAGAS.

## Tech stack

| Layer | Choice |
|---|---|
| PDF parsing | pdfplumber |
| Chunking | Custom — 800 char chunks, 150 char overlap, sentence-boundary aware |
| Embeddings | BAAI/bge-m3 |
| Vector store | ChromaDB (persistent, local) |
| Sparse index | BM25 (rank-bm25) — hybrid retrieval |
| Agent framework | LangGraph (ReAct pattern) |
| LLM | Claude Sonnet (Anthropic API) |
| Evaluation | RAGAS (faithfulness, answer relevance, context precision, context recall) |
| Experiment tracking | MLflow |
| API | FastAPI |
| Containerisation | Docker + docker-compose |
| CI/CD | GitHub Actions |
| Deployment | Railway |

## Project structure

```
ausecon/
├── data/
│   ├── raw/minutes/         # 53 RBA board minutes PDFs (2021-2025)
│   ├── processed/chunks.json
│   ├── vectorstore/         # ChromaDB persistent store
│   └── bm25_index.pkl
├── scripts/
│   ├── parse_documents.py   # pdfplumber -> chunks
│   └── embed_and_index.py   # embed -> ChromaDB + BM25
├── agent/
│   ├── tools/                # LangGraph tool definitions
│   ├── graph.py               # agent graph
│   └── run.py                  # CLI runner
├── eval/
│   ├── ground_truth.json      # 80 Q&A pairs
│   └── run_eval.py            # RAGAS pipeline
├── api/
│   ├── main.py                # FastAPI app
│   └── schemas.py
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Agent tools

1. `search_rba_documents` — hybrid retrieval (ChromaDB dense + BM25 sparse, reranked)
2. `fetch_live_macro_data` — ABS API for current CPI, unemployment, GDP
3. `calculate_rate_impact` — mortgage repayment impact given a rate delta
4. `summarise_policy_trend` — multi-document synthesis across a date range

## Evaluation targets

| Metric | Target |
|---|---|
| Faithfulness | > 0.85 |
| Answer Relevance | > 0.80 |
| Context Precision | > 0.75 |
| Context Recall | > 0.70 |

## Build phases

- **Phase 1** — Parse PDFs → chunk → embed → index into ChromaDB + BM25
- **Phase 2** — LangGraph agent with 4 tools
- **Phase 3** — RAGAS eval pipeline with 80 Q&A ground truth set
- **Phase 4** — MLflow experiment tracking + query monitoring
- **Phase 5** — FastAPI + Docker + Railway deploy + GitHub Actions CI/CD

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in ANTHROPIC_API_KEY, ABS_API_KEY
```

> **Note:** pin your interpreter to Python 3.11 or 3.12 for this project — several pinned dependencies (torch, chromadb) may not yet ship wheels for very new Python releases.
