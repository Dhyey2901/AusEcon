"""Shared configuration for the AusEcon agent, loaded from environment variables.

All paths are relative to the repository root unless overridden via `.env`.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
AGENT_MODEL = os.environ.get("AGENT_MODEL", "claude-sonnet-5-5")

CHROMA_PERSIST_DIR = Path(os.environ.get("CHROMA_PERSIST_DIR", "data/vectorstore"))
CHROMA_COLLECTION_NAME = "rba_minutes"
BM25_INDEX_PATH = Path(os.environ.get("BM25_INDEX_PATH", "data/bm25_index.pkl"))
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")
