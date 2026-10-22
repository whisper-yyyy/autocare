"""Small SQLite repository for documents and derived knowledge chunks."""

import sqlite3
from pathlib import Path
from typing import List, Tuple

from .models import Chunk, DocumentInfo


class SQLiteRepository:
    """Persist source metadata and chunks; the BM25 index remains derived data."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    content_hash TEXT NOT NULL UNIQUE,
                    source TEXT NOT NULL,
                    vehicle_model TEXT,
                    model_year TEXT,
                    market TEXT,
                    powertrain TEXT,
                    manual_version TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    page INTEGER,
                    chunk_index INTEGER NOT NULL,
                    section TEXT,
                    FOREIGN KEY(document_id) REFERENCES documents(document_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_chunks_document_id
                ON chunks(document_id);
                """
            )

    def load(self) -> Tuple[List[DocumentInfo], List[Chunk]]:
        with self._connect() as connection:
            document_rows = connection.execute(
                """
                SELECT document_id, content_hash, source, vehicle_model, model_year,
                       market, powertrain, manual_version
                FROM documents
                ORDER BY created_at, document_id
                """
            ).fetchall()
            chunk_rows = connection.execute(
                """
                SELECT c.chunk_id, c.document_id, c.text, c.page, c.chunk_index, c.section,
                       d.source, d.vehicle_model, d.model_year, d.market,
                       d.powertrain, d.manual_version
                FROM chunks AS c
                JOIN documents AS d ON d.document_id = c.document_id
                ORDER BY d.created_at, c.page, c.chunk_index
                """
            ).fetchall()

        documents = [
            DocumentInfo(
                document_id=row["document_id"],
                content_hash=row["content_hash"],
                source=row["source"],
                vehicle_model=row["vehicle_model"],
                model_year=row["model_year"],
                market=row["market"],
                powertrain=row["powertrain"],
                manual_version=row["manual_version"],
            )
            for row in document_rows
        ]
        chunks = [
            Chunk(
                chunk_id=row["chunk_id"],
                text=row["text"],
                source=row["source"],
                page=row["page"],
                chunk_index=row["chunk_index"],
                document_id=row["document_id"],
                section=row["section"],
                vehicle_model=row["vehicle_model"],
                model_year=row["model_year"],
                market=row["market"],
                powertrain=row["powertrain"],
                manual_version=row["manual_version"],
            )
            for row in chunk_rows
        ]
        return documents, chunks

    def save(self, document: DocumentInfo, chunks: List[Chunk]) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO documents (
                    document_id, content_hash, source, vehicle_model, model_year,
                    market, powertrain, manual_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document.document_id,
                    document.content_hash,
                    document.source,
                    document.vehicle_model,
                    document.model_year,
                    document.market,
                    document.powertrain,
                    document.manual_version,
                ),
            )
            connection.executemany(
                """
                INSERT INTO chunks (
                    chunk_id, document_id, text, page, chunk_index, section
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        chunk.chunk_id,
                        document.document_id,
                        chunk.text,
                        chunk.page,
                        chunk.chunk_index,
                        chunk.section,
                    )
                    for chunk in chunks
                ],
            )

    def delete(self, document_id: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))

    def clear(self) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM documents")
