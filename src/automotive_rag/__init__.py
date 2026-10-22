"""Standalone automotive aftersales RAG."""

from .models import Chunk, DocumentInfo, RAGAnswer, RetrievalHit
from .pipeline import AutomotiveRAG

__all__ = ["AutomotiveRAG", "Chunk", "DocumentInfo", "RAGAnswer", "RetrievalHit"]
