"""Unit tests for the retrieval fusion logic in agent/tools/retrieval.py.

These test pure functions only (tokenization and reciprocal rank fusion) -
no ChromaDB collection, BM25 index, or embedding model is touched, so these
tests run fast with no external dependencies or API calls.
"""

from __future__ import annotations

from agent.tools.retrieval import RRF_K, _reciprocal_rank_fusion, tokenize


class TestTokenize:
    def test_lowercases_and_splits_on_non_alphanumeric(self):
        assert tokenize("Inflation Rose, by 2.5%!") == ["inflation", "rose", "by", "2", "5"]

    def test_empty_string_returns_empty_list(self):
        assert tokenize("") == []


class TestReciprocalRankFusion:
    def test_single_ranked_list_preserves_order(self):
        fused = _reciprocal_rank_fusion([["a", "b", "c"]])
        assert [item_id for item_id, _ in fused] == ["a", "b", "c"]

    def test_item_ranked_first_in_both_lists_wins(self):
        fused = _reciprocal_rank_fusion([["a", "b", "c"], ["a", "c", "b"]])
        assert fused[0][0] == "a"

    def test_item_appearing_in_both_lists_outranks_single_appearance(self):
        # "b" is ranked #1 in list 2 only; "a" is ranked #2 in both lists.
        # Appearing in both lists should give "a" a higher fused score.
        fused = _reciprocal_rank_fusion([["x", "a"], ["b", "a"]])
        scores = dict(fused)
        assert scores["a"] > scores["b"]
        assert scores["a"] > scores["x"]

    def test_disjoint_lists_include_all_items(self):
        fused = _reciprocal_rank_fusion([["a", "b"], ["c", "d"]])
        assert {item_id for item_id, _ in fused} == {"a", "b", "c", "d"}

    def test_empty_lists_produce_no_results(self):
        assert _reciprocal_rank_fusion([[], []]) == []

    def test_rrf_score_matches_formula(self):
        fused = _reciprocal_rank_fusion([["a", "b"]], k=RRF_K)
        scores = dict(fused)
        assert scores["a"] == 1.0 / (RRF_K + 1)
        assert scores["b"] == 1.0 / (RRF_K + 2)
