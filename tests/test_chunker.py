"""Unit tests for the PDF parsing and chunking logic in scripts/parse_documents.py."""

from __future__ import annotations

from datetime import date

import pytest

from scripts.parse_documents import (
    chunk_text,
    clean_page_text,
    extract_document_title,
    find_title_lines,
    parse_meeting_date,
    split_into_sections,
    split_sentences,
)


class TestParseMeetingDate:
    def test_single_day_meeting(self):
        name = "1 April 2025 _ Minutes of the Monetary Policy Board Meeting _ RBA.pdf"
        assert parse_meeting_date(name) == date(2025, 4, 1)

    def test_two_day_meeting_uses_first_day(self):
        name = "5-6 February 2024 _ Minutes of the Monetary Policy Meeting of the Board _ RBA.pdf"
        assert parse_meeting_date(name) == date(2024, 2, 5)

    def test_invalid_filename_raises(self):
        with pytest.raises(ValueError):
            parse_meeting_date("not_a_minutes_file.pdf")


class TestCleanPageText:
    def test_strips_browser_print_header(self):
        text = "7/17/26, 12:06 PM 1 April 2025 | Minutes | RBA\nActual content here."
        cleaned = clean_page_text(text)
        assert "12:06 PM" not in cleaned
        assert "Actual content here." in cleaned

    def test_strips_url_page_footer(self):
        text = "Some content.\nhttps://www.rba.gov.au/monetary-policy/rba-board-minutes-2025-04-01.html 1/5"
        cleaned = clean_page_text(text)
        assert "rba.gov.au" not in cleaned
        assert "Some content." in cleaned

    def test_strips_copyright_footer(self):
        text = "Real content.\nThe materials on this webpage are subject to copyright and more legal text."
        cleaned = clean_page_text(text)
        assert "copyright" not in cleaned
        assert "Real content." in cleaned


class TestTitleExtraction:
    def test_single_line_title(self):
        first_page = (
            "Minutes of the Monetary Policy Board Meeting\n"
            "Sydney – 31 March and 1 April 2025\n"
            "Members present"
        )
        lines = find_title_lines(first_page)
        assert lines == ["Minutes of the Monetary Policy Board Meeting"]
        assert extract_document_title(first_page) == "Minutes of the Monetary Policy Board Meeting"

    def test_multi_line_title(self):
        first_page = (
            "Minutes of the Monetary Policy Meeting of the\n"
            "Reserve Bank Board\n"
            "Sydney – 2 February 2021\n"
            "Members present"
        )
        lines = find_title_lines(first_page)
        assert lines == ["Minutes of the Monetary Policy Meeting of the", "Reserve Bank Board"]

    def test_missing_title_falls_back(self):
        assert extract_document_title("No title line here at all") == "Minutes of the Monetary Policy Meeting"


class TestSplitIntoSections:
    def test_preamble_and_headed_sections(self):
        text = (
            "Minutes of the Monetary Policy Board Meeting\n"
            "Members present\n"
            "Alice, Bob, Carol\n"
            "International Economic Conditions\n"
            "Global growth remained resilient."
        )
        sections = split_into_sections(text, title_lines=["Minutes of the Monetary Policy Board Meeting"])
        headings = [heading for heading, _ in sections]
        assert headings[0] == "Preamble"
        assert "International Economic Conditions" in headings

    def test_empty_sections_are_dropped(self):
        text = "Heading One\nSome body text."
        sections = split_into_sections(text, title_lines=[])
        assert all(body.strip() for _, body in sections)


class TestSplitSentences:
    def test_splits_on_sentence_boundaries(self):
        sentences = split_sentences("Inflation rose. The Board discussed options. Rates were held.")
        assert sentences == [
            "Inflation rose.",
            "The Board discussed options.",
            "Rates were held.",
        ]

    def test_empty_text_returns_empty_list(self):
        assert split_sentences("   ") == []

    def test_collapses_whitespace(self):
        sentences = split_sentences("Line one.\n\n  Line two.")
        assert sentences == ["Line one.", "Line two."]


class TestChunkText:
    def test_respects_chunk_size_target(self):
        text = " ".join([f"Sentence number {i} about monetary policy." for i in range(50)])
        chunks = chunk_text(text, chunk_size=200, overlap=50)
        assert len(chunks) > 1
        # Allow some slack since a single sentence is never split mid-word.
        assert all(len(c) <= 200 + 60 for c in chunks)

    def test_overlap_carries_context_between_chunks(self):
        text = " ".join([f"Sentence {i} about policy." for i in range(60)])
        chunks = chunk_text(text, chunk_size=200, overlap=100)
        assert len(chunks) > 1
        # At least one sentence from the end of chunk N should reappear at the
        # start of chunk N+1, so context carries across the chunk boundary.
        sentences_0 = {s.strip() for s in chunks[0].split(".") if s.strip()}
        sentences_1 = {s.strip() for s in chunks[1].split(".") if s.strip()}
        assert sentences_0 & sentences_1

    def test_long_single_sentence_is_its_own_chunk(self):
        long_sentence = "A" * 1000 + "."
        chunks = chunk_text(long_sentence, chunk_size=800, overlap=150)
        assert chunks == [long_sentence]

    def test_empty_text_returns_no_chunks(self):
        assert chunk_text("") == []
