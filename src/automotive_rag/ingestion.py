"""Document ingestion for text and page-aware PDFs."""

import hashlib
from io import BytesIO
from pathlib import Path
from typing import List

from .chunking import chunk_text
from .models import Chunk


MAX_DOCUMENT_BYTES = 25 * 1024 * 1024
MAX_PDF_PAGES = 500


def ingest_text(
    text: str,
    source: str,
    chunk_size: int = 480,
    overlap: int = 80,
) -> List[Chunk]:
    document_id = hashlib.sha256(f"{source}\0{text}".encode("utf-8")).hexdigest()[:16]
    return chunk_text(
        text,
        source=source,
        chunk_size=chunk_size,
        overlap=overlap,
        document_id=document_id,
    )


def ingest_pdf_bytes(
    data: bytes,
    source: str,
    chunk_size: int = 480,
    overlap: int = 80,
) -> List[Chunk]:
    """Extract every readable PDF page and retain physical page numbers."""

    if len(data) > MAX_DOCUMENT_BYTES:
        raise ValueError("PDF 超过 25 MB 限制")

    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("读取 PDF 需要安装 pypdf：pip install pypdf") from exc

    reader = PdfReader(BytesIO(data))
    if len(reader.pages) > MAX_PDF_PAGES:
        raise ValueError(f"PDF 超过 {MAX_PDF_PAGES} 页限制")
    document_id = hashlib.sha256(source.encode("utf-8") + b"\0" + data).hexdigest()[:16]
    chunks: List[Chunk] = []
    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        page_chunks = chunk_text(
            page_text,
            source=source,
            page=page_number,
            chunk_size=chunk_size,
            overlap=overlap,
            document_id=document_id,
        )
        chunks.extend(page_chunks)
    return chunks


def ingest_file(path: str | Path, chunk_size: int = 480, overlap: int = 80) -> List[Chunk]:
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return ingest_pdf_bytes(file_path.read_bytes(), file_path.name, chunk_size, overlap)
    if suffix in {".txt", ".md"}:
        return ingest_text(file_path.read_text(encoding="utf-8"), file_path.name, chunk_size, overlap)
    raise ValueError(f"暂不支持的文件格式：{suffix}")
