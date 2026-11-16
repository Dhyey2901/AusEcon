"""Parse RBA Monetary Policy Board minutes PDFs into sentence-aware, metadata-tagged chunks.

Reads every PDF in `data/raw/minutes/`, strips the browser-print header/footer
boilerplate that RBA's "save as PDF" webpages embed on every page, splits the
body into sections using the documents' own heading structure, chunks each
section into overlapping ~800-character windows on sentence boundaries, and
writes the result to `data/processed/chunks.json` for downstream embedding.

Usage:
    python scripts/parse_documents.py
    python scripts/parse_documents.py --raw-dir data/raw/minutes --output data/processed/chunks.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path

import pdfplumber

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("parse_documents")

# --- Chunking configuration (locked in CLAUDE.md — do not change without asking) ---
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# Matches the browser "print to PDF" timestamp header repeated on every page, e.g.
# "7/17/26, 12:06 PM 1 April 2025 | Minutes of the Monetary Policy Board Meeting | RBA"
_PAGE_HEADER_RE = re.compile(
    r"^\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}\s*[AP]M\b.*$", re.MULTILINE
)

# The footer boilerplate that appears on the final page of every minutes document.
_FOOTER_MARKER = "The materials on this webpage are subject to copyright"

# Matches the "https://www.rba.gov.au/.../YYYY-MM-DD.html N/M" URL + page-number
# footer that the browser print job embeds at the bottom of *every* page.
_PAGE_URL_FOOTER_RE = re.compile(r"^https://www\.rba\.gov\.au/\S*\s+\d+/\d+\s*$", re.MULTILINE)

# Filenames look like "1 April 2025 _ Minutes of the Monetary Policy Board Meeting _ RBA.pdf"
# or "5-6 February 2024 _ Minutes of the Monetary Policy Meeting of the Board _ RBA.pdf".
# We take the first day when a meeting spans two days.
_FILENAME_DATE_RE = re.compile(
    r"^(?P<day>\d{1,2})(?:-\d{1,2})?\s+(?P<month>[A-Za-z]+)\s+(?P<year>\d{4})\s*_"
)

# A section heading is a short standalone line, title-cased at the start, with no
# terminal sentence punctuation, immediately followed by a new paragraph.
_HEADING_RE = re.compile(r"^[A-Z][A-Za-z0-9,'’&/\- ]{2,70}$")
_HEADING_MAX_WORDS = 8
_PREAMBLE_SECTION = "Preamble"

# Simple sentence splitter: break after ., !, or ? when followed by whitespace and
# a capital letter, digit, or opening quote. Good enough for RBA minutes prose,
# which does not use dotted abbreviations like "U.S." (it writes "US").
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


@dataclass
class Chunk:
    """A single retrievable unit of text with provenance metadata."""

    chunk_id: str
    text: str
    source_file: str
    document_title: str
    meeting_date: str
    section: str
    chunk_index: int
    char_count: int


def parse_meeting_date(filename: str) -> date:
    """Extract the meeting date from an RBA minutes filename.

    Args:
        filename: The PDF filename, e.g.
            "1 April 2025 _ Minutes of the Monetary Policy Board Meeting _ RBA.pdf".

    Returns:
        The meeting date (the first day, for two-day meetings like "5-6 February 2024").

    Raises:
        ValueError: If the filename does not match the expected RBA naming pattern.
    """
    match = _FILENAME_DATE_RE.match(filename)
    if not match:
        raise ValueError(f"Filename does not match expected RBA minutes pattern: {filename!r}")

    day, month, year = match.group("day"), match.group("month"), match.group("year")
    try:
        return datetime.strptime(f"{day} {month} {year}", "%d %B %Y").date()
    except ValueError as exc:
        raise ValueError(f"Could not parse date from filename {filename!r}: {exc}") from exc


def clean_page_text(text: str) -> str:
    """Strip the browser-print header line and the trailing copyright footer from page text.

    Args:
        text: Raw text extracted from a single PDF page.

    Returns:
        The page text with boilerplate removed. May be an empty string if the
        entire page was boilerplate (e.g. a footer-only final page).
    """
    text = _PAGE_HEADER_RE.sub("", text)
    text = _PAGE_URL_FOOTER_RE.sub("", text).strip()

    footer_idx = text.find(_FOOTER_MARKER)
    if footer_idx != -1:
        text = text[:footer_idx].strip()

    return text


def find_title_lines(first_page_text: str) -> list[str]:
    """Locate the raw lines making up the document title on the first page.

    The title starts on a line beginning "Minutes of ..." and, in older-format
    documents, wraps onto a second line (e.g. "Minutes of the Monetary Policy
    Meeting of the" / "Reserve Bank Board"). It ends just before the
    meeting location/date line, which always contains an en dash
    (e.g. "Sydney – 31 March and 1 April 2025").

    Args:
        first_page_text: Cleaned text of the PDF's first page.

    Returns:
        The list of raw title lines, in order. Empty if no title line is found.
    """
    lines = first_page_text.splitlines()
    title_lines: list[str] = []
    started = False

    for line in lines:
        stripped = line.strip()
        if not started:
            if stripped.lower().startswith("minutes of"):
                started = True
                title_lines.append(stripped)
            continue
        if not stripped or "–" in stripped or "-" in stripped:
            break
        title_lines.append(stripped)

    return title_lines


def extract_document_title(first_page_text: str) -> str:
    """Extract the (possibly multi-line) document title from the first page.

    Args:
        first_page_text: Cleaned text of the PDF's first page.

    Returns:
        The title as a single joined string, or a fallback if it cannot be located.
    """
    title_lines = find_title_lines(first_page_text)
    if not title_lines:
        logger.warning("Could not locate title line; using fallback title")
        return "Minutes of the Monetary Policy Meeting"
    return " ".join(title_lines)


def _looks_like_heading(line: str, next_line: str | None, excluded_lines: set[str]) -> bool:
    """Heuristic check for whether a line is a section heading.

    A heading is a short line with no terminal sentence punctuation that is
    immediately followed by the start of a new paragraph. Lines in
    `excluded_lines` (the document's own, possibly multi-line, title) are
    never treated as headings, so they stay folded into the Preamble section
    instead of duplicating `document_title` or fragmenting into a bogus one.
    """
    stripped = line.strip()
    if not stripped or stripped in excluded_lines:
        return False
    if not _HEADING_RE.match(stripped):
        return False
    if stripped[-1] in ".:;,":
        return False
    if len(stripped.split()) > _HEADING_MAX_WORDS:
        return False
    if next_line is None or not next_line.strip():
        return False
    return True


def split_into_sections(full_text: str, title_lines: list[str]) -> list[tuple[str, str]]:
    """Split a document's full cleaned text into (section_heading, body_text) pairs.

    Text preceding the first detected heading (title, meeting location/date,
    attendee lists) is tagged as the "Preamble" section.

    Args:
        full_text: The cleaned, concatenated text of an entire minutes document.
        title_lines: Raw title lines (from `find_title_lines`) to exclude from
            heading detection.

    Returns:
        An ordered list of (section_heading, body_text) tuples.
    """
    lines = full_text.splitlines()
    excluded_lines = set(title_lines)
    sections: list[tuple[str, list[str]]] = [(_PREAMBLE_SECTION, [])]

    for i, line in enumerate(lines):
        next_line = lines[i + 1] if i + 1 < len(lines) else None
        if _looks_like_heading(line, next_line, excluded_lines):
            sections.append((line.strip(), []))
        else:
            sections[-1][1].append(line)

    return [
        (heading, "\n".join(body_lines).strip())
        for heading, body_lines in sections
        if "\n".join(body_lines).strip()
    ]


def split_sentences(text: str) -> list[str]:
    """Split body text into sentences, collapsing internal whitespace.

    Args:
        text: A block of prose text.

    Returns:
        A list of sentence strings.
    """
    normalised = re.sub(r"\s+", " ", text).strip()
    if not normalised:
        return []
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(normalised) if s.strip()]


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Chunk text into overlapping windows on sentence boundaries.

    Sentences are greedily packed into a chunk until adding the next sentence
    would exceed `chunk_size`. The next chunk is seeded with trailing sentences
    from the current chunk totalling up to `overlap` characters, so context
    carries across chunk boundaries. A single sentence longer than `chunk_size`
    is emitted as its own chunk rather than being dropped or split mid-word.

    Args:
        text: The section body text to chunk.
        chunk_size: Target maximum characters per chunk.
        overlap: Target characters of trailing-sentence overlap between chunks.

    Returns:
        A list of chunk strings, in document order.
    """
    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        sentence_len = len(sentence) + 1

        if sentence_len > chunk_size:
            if current:
                chunks.append(" ".join(current).strip())
                current, current_len = [], 0
            chunks.append(sentence)
            continue

        if current and current_len + sentence_len > chunk_size:
            chunks.append(" ".join(current).strip())

            overlap_sentences: list[str] = []
            overlap_len = 0
            for s in reversed(current):
                if overlap_len + len(s) + 1 > overlap:
                    break
                overlap_sentences.insert(0, s)
                overlap_len += len(s) + 1
            current, current_len = overlap_sentences, overlap_len

        current.append(sentence)
        current_len += sentence_len

    if current:
        chunks.append(" ".join(current).strip())

    return chunks


def parse_pdf(pdf_path: Path) -> list[Chunk]:
    """Parse a single RBA minutes PDF into a list of metadata-tagged chunks.

    Args:
        pdf_path: Path to the source PDF.

    Returns:
        A list of `Chunk` objects covering the whole document.

    Raises:
        ValueError: If the meeting date cannot be parsed from the filename.
        pdfplumber.pdfminer.pdfparser.PDFSyntaxError: If the PDF is corrupt or unreadable.
    """
    meeting_date = parse_meeting_date(pdf_path.name)

    page_texts: list[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            raw_text = page.extract_text() or ""
            if not raw_text.strip():
                logger.warning("No extractable text on page %d of %s", page_num, pdf_path.name)
                continue
            cleaned = clean_page_text(raw_text)
            if cleaned:
                page_texts.append(cleaned)

    if not page_texts:
        raise ValueError(f"No extractable text found in {pdf_path.name}")

    full_text = "\n".join(page_texts)
    title_lines = find_title_lines(page_texts[0])
    document_title = extract_document_title(page_texts[0])
    sections = split_into_sections(full_text, title_lines)

    chunks: list[Chunk] = []
    doc_id = meeting_date.isoformat()
    chunk_index = 0
    for section_heading, body_text in sections:
        for chunk_body in chunk_text(body_text):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}_{chunk_index:04d}",
                    text=chunk_body,
                    source_file=pdf_path.name,
                    document_title=document_title,
                    meeting_date=doc_id,
                    section=section_heading,
                    chunk_index=chunk_index,
                    char_count=len(chunk_body),
                )
            )
            chunk_index += 1

    return chunks


def parse_all_documents(raw_dir: Path) -> list[Chunk]:
    """Parse every PDF in `raw_dir`, logging and skipping any that fail.

    Args:
        raw_dir: Directory containing RBA minutes PDFs.

    Returns:
        All chunks from all successfully parsed documents, sorted by meeting date.
    """
    pdf_paths = sorted(raw_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in {raw_dir}")

    logger.info("Found %d PDF files in %s", len(pdf_paths), raw_dir)

    all_chunks: list[Chunk] = []
    failures: list[str] = []

    for pdf_path in pdf_paths:
        try:
            doc_chunks = parse_pdf(pdf_path)
            all_chunks.extend(doc_chunks)
            logger.info(
                "Parsed %s -> %d chunks (meeting date %s)",
                pdf_path.name,
                len(doc_chunks),
                doc_chunks[0].meeting_date if doc_chunks else "unknown",
            )
        except Exception:
            logger.exception("Failed to parse %s; skipping", pdf_path.name)
            failures.append(pdf_path.name)

    if failures:
        logger.warning("Failed to parse %d/%d documents: %s", len(failures), len(pdf_paths), failures)

    all_chunks.sort(key=lambda c: (c.meeting_date, c.chunk_index))
    return all_chunks


def save_chunks(chunks: list[Chunk], output_path: Path) -> None:
    """Write chunks to a JSON file, creating parent directories as needed.

    Args:
        chunks: The chunks to serialise.
        output_path: Destination JSON file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in chunks], f, indent=2, ensure_ascii=False)
    logger.info("Wrote %d chunks to %s", len(chunks), output_path)


def main() -> None:
    """CLI entry point: parse all RBA minutes PDFs and write chunks.json."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw/minutes"),
        help="Directory containing source RBA minutes PDFs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/chunks.json"),
        help="Output path for the chunked JSON.",
    )
    args = parser.parse_args()

    chunks = parse_all_documents(args.raw_dir)
    save_chunks(chunks, args.output)

    if chunks:
        avg_len = sum(c.char_count for c in chunks) / len(chunks)
        logger.info(
            "Summary: %d chunks from %d unique documents, avg %.0f chars/chunk",
            len(chunks),
            len({c.meeting_date for c in chunks}),
            avg_len,
        )


if __name__ == "__main__":
    main()
