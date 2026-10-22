"""Chinese-friendly text normalization and structure-preserving chunking."""

import hashlib
import re
from typing import List, Optional, Tuple

from .models import Chunk


BREAK_MARKS = "。！？；\n.!?;"
HEADING_PATTERN = re.compile(
    r"^(?:#{1,6}\s+|第[一二三四五六七八九十百零〇\d]+[章节篇部分]|"
    r"[一二三四五六七八九十百零〇]+[、．.]|\d+(?:\.\d+)*[、．.\s])"
)


def normalize_text(text: str) -> str:
    """Normalize whitespace while retaining paragraph boundaries."""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _find_boundary(text: str, start: int, proposed_end: int, minimum_end: int) -> int:
    """Prefer ending a chunk at a sentence or paragraph boundary."""

    for index in range(proposed_end - 1, minimum_end - 1, -1):
        if text[index] in BREAK_MARKS:
            return index + 1
    return proposed_end


def _heading_text(line: str) -> Optional[str]:
    stripped = line.strip()
    if not stripped or len(stripped) > 80 or not HEADING_PATTERN.match(stripped):
        return None
    return re.sub(r"^#{1,6}\s+", "", stripped).strip()


def _split_sections(text: str) -> List[Tuple[Optional[str], str]]:
    """Split explicit Markdown/Chinese-numbered sections before size chunking."""

    sections: List[Tuple[Optional[str], str]] = []
    current_title: Optional[str] = None
    current_lines: List[str] = []
    for line in text.splitlines(keepends=True):
        heading = _heading_text(line)
        if heading and current_lines:
            body = "".join(current_lines).strip()
            if body:
                sections.append((current_title, body))
            current_lines = [line]
            current_title = heading
        else:
            current_lines.append(line)
            if heading:
                current_title = heading

    body = "".join(current_lines).strip()
    if body:
        sections.append((current_title, body))
    return sections


def chunk_text(
    text: str,
    source: str,
    page: Optional[int] = None,
    chunk_size: int = 480,
    overlap: int = 80,
    document_id: Optional[str] = None,
) -> List[Chunk]:
    """Split text into overlapping chunks without losing source metadata."""

    if chunk_size < 100:
        raise ValueError("chunk_size must be at least 100 characters")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")

    normalized = normalize_text(text)
    if not normalized:
        return []

    stable_document_id = document_id or hashlib.sha256(
        f"{source}\0{page}\0{normalized}".encode("utf-8")
    ).hexdigest()[:16]

    chunks: List[Chunk] = []
    chunk_index = 0
    page_part = str(page) if page is not None else "na"
    for section, section_text in _split_sections(normalized):
        start = 0
        while start < len(section_text):
            proposed_end = min(start + chunk_size, len(section_text))
            if proposed_end < len(section_text):
                minimum_end = start + int(chunk_size * 0.6)
                end = _find_boundary(section_text, start, proposed_end, minimum_end)
            else:
                end = proposed_end

            piece = section_text[start:end].strip()
            if piece:
                chunks.append(
                    Chunk(
                        chunk_id=f"{stable_document_id}-p{page_part}-c{chunk_index}",
                        text=piece,
                        source=source,
                        page=page,
                        chunk_index=chunk_index,
                        document_id=stable_document_id,
                        section=section,
                    )
                )
                chunk_index += 1

            if end >= len(section_text):
                break
            start = max(end - overlap, start + 1)

    return chunks
