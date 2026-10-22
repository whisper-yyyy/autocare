"""End-to-end automotive RAG pipeline."""

import hashlib
from dataclasses import replace
from pathlib import Path
from threading import RLock
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .generation import extractive_answer, ollama_generate
from .ingestion import ingest_pdf_bytes, ingest_text
from .models import Chunk, DocumentInfo, RAGAnswer, RetrievalHit
from .retrieval import BM25Retriever
from .safety import classify_safety
from .storage import SQLiteRepository


METADATA_FIELDS = (
    "vehicle_model",
    "model_year",
    "market",
    "powertrain",
    "manual_version",
)


def _clean_metadata(metadata: Optional[Mapping[str, str]]) -> Dict[str, Optional[str]]:
    values: Dict[str, Optional[str]] = {}
    source = metadata or {}
    for field in METADATA_FIELDS:
        value = source.get(field)
        values[field] = str(value).strip() if value is not None and str(value).strip() else None
    return values


class AutomotiveRAG:
    """Evidence-first RAG with optional SQLite persistence and local generation."""

    def __init__(
        self,
        refusal_threshold: float = 0.14,
        db_path: str | Path | None = None,
    ):
        self.refusal_threshold = refusal_threshold
        self.chunks: List[Chunk] = []
        self.retriever = BM25Retriever()
        self.sources: List[str] = []
        self.documents: Dict[str, DocumentInfo] = {}
        self._lock = RLock()
        self.repository = SQLiteRepository(db_path) if db_path is not None else None

        if self.repository is not None:
            documents, chunks = self.repository.load()
            self.documents = {document.document_id: document for document in documents}
            self.chunks = chunks
            self.sources = [document.source for document in documents]
            self.retriever.build(self.chunks)

    @property
    def document_infos(self) -> List[DocumentInfo]:
        return list(self.documents.values())

    @property
    def content_hashes(self) -> set[str]:
        return {document.content_hash for document in self.documents.values()}

    def has_content_hash(self, content_hash: str) -> bool:
        return content_hash in self.content_hashes

    def _append_chunks(
        self,
        chunks: List[Chunk],
        source: str,
        content_hash: str,
        metadata: Optional[Mapping[str, str]] = None,
    ) -> int:
        if not chunks:
            return 0

        with self._lock:
            if self.has_content_hash(content_hash):
                return 0

            clean_metadata = _clean_metadata(metadata)
            document_id = hashlib.sha256(
                f"{source}\0{content_hash}".encode("utf-8")
            ).hexdigest()[:16]
            document = DocumentInfo(
                document_id=document_id,
                content_hash=content_hash,
                source=source,
                **clean_metadata,
            )
            enriched_chunks = []
            for chunk in chunks:
                page_part = str(chunk.page) if chunk.page is not None else "na"
                enriched_chunks.append(
                    replace(
                        chunk,
                        chunk_id=f"{document_id}-p{page_part}-c{chunk.chunk_index}",
                        document_id=document_id,
                        **clean_metadata,
                    )
                )

            if self.repository is not None:
                self.repository.save(document, enriched_chunks)
            self.documents[document_id] = document
            self.chunks.extend(enriched_chunks)
            self.sources.append(source)
            self.retriever.add(enriched_chunks)
            return len(enriched_chunks)

    def add_text(
        self,
        text: str,
        source: str,
        metadata: Optional[Mapping[str, str]] = None,
    ) -> int:
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return self._append_chunks(ingest_text(text, source), source, content_hash, metadata)

    def add_pdf_bytes(
        self,
        data: bytes,
        source: str,
        metadata: Optional[Mapping[str, str]] = None,
    ) -> int:
        content_hash = hashlib.sha256(data).hexdigest()
        return self._append_chunks(
            ingest_pdf_bytes(data, source), source, content_hash, metadata
        )

    def add_file(
        self,
        path: str | Path,
        metadata: Optional[Mapping[str, str]] = None,
    ) -> int:
        file_path = Path(path)
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            return self.add_pdf_bytes(file_path.read_bytes(), file_path.name, metadata)
        if suffix in {".txt", ".md"}:
            return self.add_text(
                file_path.read_text(encoding="utf-8-sig"), file_path.name, metadata
            )
        raise ValueError(f"暂不支持的文件格式：{suffix}")

    def delete_document(self, document_id: str) -> None:
        with self._lock:
            if document_id not in self.documents:
                return
            if self.repository is not None:
                self.repository.delete(document_id)
            self.documents.pop(document_id, None)
            self.chunks = [
                chunk for chunk in self.chunks if chunk.document_id != document_id
            ]
            self.sources = [document.source for document in self.documents.values()]
            self.retriever.build(self.chunks)

    def clear(self) -> None:
        with self._lock:
            if self.repository is not None:
                self.repository.clear()
            self.chunks = []
            self.sources = []
            self.documents = {}
            self.retriever.build([])

    def _search(
        self,
        question: str,
        top_k: int,
        document_ids: Optional[Sequence[str]],
    ) -> Tuple[List[RetrievalHit], float]:
        with self._lock:
            hits = self.retriever.search(
                question,
                top_k=top_k,
                document_ids=document_ids,
            )
        top_score = hits[0].score if hits else 0.0
        if not hits or top_score < self.refusal_threshold:
            return [], top_score
        evidence_floor = max(self.refusal_threshold, top_score * 0.5)
        return [hit for hit in hits if hit.score >= evidence_floor], top_score

    def ask(
        self,
        question: str,
        top_k: int = 4,
        use_ollama: bool = False,
        ollama_model: str = "qwen2.5:7b-instruct-q4_K_M",
        ollama_base_url: str = "http://localhost:11434",
        document_ids: Optional[Sequence[str]] = None,
    ) -> RAGAnswer:
        question = question.strip()
        if not question:
            return RAGAnswer("请输入一个车辆相关问题。", refused=True, mode="refusal")

        safety = classify_safety(question)
        if safety is not None:
            hits: List[RetrievalHit] = []
            confidence = 0.0
            if not safety.refused and self.chunks:
                hits, confidence = self._search(question, top_k, document_ids)
            return RAGAnswer(
                answer=safety.response,
                sources=hits,
                confidence=confidence,
                refused=safety.refused,
                mode=safety.mode,
                notice=safety.notice,
            )

        if not self.chunks:
            return RAGAnswer(
                "知识库为空，请先上传车辆手册或加载演示资料。",
                refused=True,
                mode="refusal",
            )

        if document_ids is None and len(self.documents) > 1:
            documents = list(self.documents.values())
            scopes = {
                (
                    document.vehicle_model,
                    document.model_year,
                    document.powertrain,
                    document.market,
                )
                for document in documents
            }
            if any(not document.vehicle_model for document in documents) or len(scopes) > 1:
                return RAGAnswer(
                    "知识库包含多份适用范围不明确或不同车型的资料，请先选择本次查询的资料范围。",
                    refused=True,
                    mode="scope-selection",
                )

        hits, top_score = self._search(question, top_k, document_ids)
        if not hits:
            return RAGAnswer(
                answer="当前资料不足，无法依据已上传的车辆手册回答这个问题。请换一种问法或补充对应车型资料。",
                sources=[],
                confidence=top_score,
                refused=True,
                mode="refusal",
            )

        notice = ""
        if use_ollama:
            try:
                answer = ollama_generate(
                    question,
                    hits,
                    model=ollama_model,
                    base_url=ollama_base_url,
                )
                mode = "ollama"
            except RuntimeError as exc:
                answer = extractive_answer(question, hits)
                mode = "extractive-fallback"
                notice = f"本地模型不可用或引用校验失败，已自动切换到证据摘录模式。{exc}"
        else:
            answer = extractive_answer(question, hits)
            mode = "extractive"

        return RAGAnswer(
            answer=answer,
            sources=hits,
            confidence=top_score,
            refused=False,
            mode=mode,
            notice=notice,
        )
