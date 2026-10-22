"""Data models for the standalone automotive RAG."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class Chunk:
    """A retrievable piece of one source document."""

    chunk_id: str
    text: str
    source: str
    page: Optional[int]
    chunk_index: int
    document_id: str = ""
    section: Optional[str] = None
    vehicle_model: Optional[str] = None
    model_year: Optional[str] = None
    market: Optional[str] = None
    powertrain: Optional[str] = None
    manual_version: Optional[str] = None

    @property
    def citation(self) -> str:
        page_text = f"，第{self.page}页" if self.page is not None else ""
        section_text = f"，{self.section}" if self.section else ""
        return f"{self.source}{page_text}{section_text}"

    @property
    def evidence_label(self) -> str:
        """Human-readable citation including the stable evidence identifier."""

        chunk_text = f"，块 {self.chunk_id}" if self.chunk_id else ""
        return f"{self.citation}{chunk_text}"


@dataclass(frozen=True)
class RetrievalHit:
    """One retrieved chunk and its explainable relevance signals."""

    chunk: Chunk
    score: float
    bm25_score: float
    query_coverage: float
    anchor_coverage: float = 0.0
    matched_anchors: Tuple[str, ...] = ()


@dataclass(frozen=True)
class DocumentInfo:
    """One imported source and the vehicle scope attached to it."""

    document_id: str
    content_hash: str
    source: str
    vehicle_model: Optional[str] = None
    model_year: Optional[str] = None
    market: Optional[str] = None
    powertrain: Optional[str] = None
    manual_version: Optional[str] = None

    @property
    def label(self) -> str:
        scope = " / ".join(
            value
            for value in (self.vehicle_model, self.model_year, self.powertrain, self.market)
            if value
        )
        return f"{self.source}（{scope}）" if scope else self.source


@dataclass
class RAGAnswer:
    """Answer returned to the UI or CLI."""

    answer: str
    sources: List[RetrievalHit] = field(default_factory=list)
    confidence: float = 0.0
    refused: bool = False
    mode: str = "extractive"
    notice: str = ""
