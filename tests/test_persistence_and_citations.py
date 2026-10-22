"""Regression tests for persistence, document scoping, and citations."""

import gc
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from automotive_rag.generation import (  # noqa: E402
    ollama_generate,
    validate_and_render_citations,
)
from automotive_rag.models import Chunk, RetrievalHit  # noqa: E402
from automotive_rag.pipeline import AutomotiveRAG  # noqa: E402


class PersistenceRegressionTests(unittest.TestCase):
    def test_sqlite_restores_documents_and_answers_after_reopen(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            db_path = Path(temporary_directory) / "automotive-rag.sqlite3"
            first = AutomotiveRAG(db_path=db_path)
            imported = first.add_text(
                "蓄电池电量不足可能导致冷车启动困难。",
                "启动系统手册.txt",
                metadata={"vehicle_model": "星河 A"},
            )
            self.assertGreater(imported, 0)

            reopened = AutomotiveRAG(db_path=db_path)
            self.assertEqual(len(reopened.document_infos), 1)
            self.assertEqual(reopened.sources, ["启动系统手册.txt"])

            result = reopened.ask("冷车启动困难和蓄电池有关吗？")
            self.assertFalse(result.refused)
            self.assertTrue(result.sources)
            self.assertIn("启动系统手册.txt", result.answer)

            # sqlite3 connections may be finalized by cyclic GC on Windows;
            # collect before TemporaryDirectory attempts to remove the DB file.
            del result, reopened, first
            gc.collect()

    def test_duplicate_content_is_not_imported_twice(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            db_path = Path(temporary_directory) / "automotive-rag.sqlite3"
            rag = AutomotiveRAG(db_path=db_path)
            content = "制动液每两年或四万公里更换。"

            first_count = rag.add_text(content, "保养手册.txt")
            duplicate_count = rag.add_text(content, "保养手册-副本.txt")

            self.assertGreater(first_count, 0)
            self.assertEqual(duplicate_count, 0)
            self.assertEqual(len(rag.document_infos), 1)

            reopened = AutomotiveRAG(db_path=db_path)
            self.assertEqual(len(reopened.document_infos), 1)
            self.assertEqual(len(reopened.chunks), first_count)

            del reopened, rag
            gc.collect()

    def test_clear_persists_across_reopen(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            db_path = Path(temporary_directory) / "automotive-rag.sqlite3"
            rag = AutomotiveRAG(db_path=db_path)
            rag.add_text("胎压以车门框标牌为准。", "轮胎手册.txt")
            rag.clear()

            reopened = AutomotiveRAG(db_path=db_path)
            self.assertEqual(reopened.document_infos, [])
            self.assertEqual(reopened.chunks, [])
            self.assertEqual(reopened.sources, [])
            self.assertTrue(reopened.ask("胎压是多少？").refused)

            del reopened, rag
            gc.collect()


class DocumentScopeRegressionTests(unittest.TestCase):
    def setUp(self):
        self.rag = AutomotiveRAG()
        self.rag.add_text(
            "星河 A 的制动液每两年检查并更换。",
            "星河A保养手册.txt",
            metadata={"vehicle_model": "星河 A", "model_year": "2025"},
        )
        self.rag.add_text(
            "星河 B 的制动液每三年检查并更换。",
            "星河B保养手册.txt",
            metadata={"vehicle_model": "星河 B", "model_year": "2026"},
        )

    def test_multiple_vehicle_models_require_scope_selection(self):
        result = self.rag.ask("制动液多久更换？")

        self.assertTrue(result.refused)
        self.assertEqual(result.mode, "scope-selection")
        self.assertFalse(result.sources)

    def test_document_id_limits_answer_to_selected_source(self):
        document_id = next(
            document.document_id
            for document in self.rag.document_infos
            if document.vehicle_model == "星河 A"
        )

        result = self.rag.ask(
            "制动液多久更换？",
            document_ids=[document_id],
        )

        self.assertFalse(result.refused)
        self.assertTrue(result.sources)
        self.assertTrue(
            all(hit.chunk.document_id == document_id for hit in result.sources)
        )
        self.assertIn("每两年", result.answer)
        self.assertNotIn("每三年", result.answer)
        self.assertIn("星河A保养手册.txt", result.answer)
        self.assertNotIn("星河B保养手册.txt", result.answer)


class CitationValidationRegressionTests(unittest.TestCase):
    def setUp(self):
        chunk = Chunk(
            chunk_id="document-a-p7-c0",
            document_id="document-a",
            text="制动液每两年更换。",
            source="星河A保养手册.pdf",
            page=7,
            chunk_index=0,
            section="制动系统",
        )
        self.hits = [
            RetrievalHit(
                chunk=chunk,
                score=0.9,
                bm25_score=4.2,
                query_coverage=1.0,
            )
        ]

    def test_valid_evidence_id_is_rendered_as_real_source(self):
        rendered = validate_and_render_citations(
            "建议按手册周期更换制动液 [E1]。",
            self.hits,
        )

        self.assertNotIn("[E1]", rendered)
        self.assertIn("星河A保养手册.pdf", rendered)
        self.assertIn("第7页", rendered)
        self.assertIn("document-a-p7-c0", rendered)

    def test_answer_without_evidence_id_is_rejected(self):
        with self.assertRaises(RuntimeError):
            validate_and_render_citations("建议按手册周期更换制动液。", self.hits)

    def test_nonexistent_evidence_id_is_rejected(self):
        with self.assertRaises(RuntimeError):
            validate_and_render_citations("制动液更换周期见 [E9]。", self.hits)

    def test_ollama_rejects_non_local_service_url(self):
        with self.assertRaisesRegex(RuntimeError, "仅允许本机"):
            ollama_generate(
                "制动液多久更换？",
                self.hits,
                model="test-model",
                base_url="http://169.254.169.254",
            )


if __name__ == "__main__":
    unittest.main()
