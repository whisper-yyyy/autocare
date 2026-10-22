"""Acceptance tests for the standalone automotive RAG."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from automotive_rag.chunking import chunk_text  # noqa: E402
from automotive_rag.pipeline import AutomotiveRAG  # noqa: E402
from automotive_rag.retrieval import BM25Retriever  # noqa: E402


class ChunkingTests(unittest.TestCase):
    def test_long_text_is_split_with_metadata_and_overlap(self):
        text = "蓄电池检查说明。" * 120
        chunks = chunk_text(text, "测试手册.pdf", page=3, chunk_size=200, overlap=30)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.text for chunk in chunks))
        self.assertTrue(all(len(chunk.text) <= 200 for chunk in chunks))
        self.assertTrue(all(chunk.page == 3 for chunk in chunks))
        self.assertTrue(all(chunk.source == "测试手册.pdf" for chunk in chunks))

    def test_invalid_overlap_is_rejected(self):
        with self.assertRaises(ValueError):
            chunk_text("测试", "source", chunk_size=100, overlap=100)


class RetrievalTests(unittest.TestCase):
    def test_chinese_synonym_retrieval(self):
        chunks = chunk_text(
            "制动液每两年或四万公里更换。蓄电池电压不足可能造成冷车启动困难。",
            "保养手册.txt",
            chunk_size=200,
            overlap=20,
        )
        retriever = BM25Retriever()
        retriever.build(chunks)
        hits = retriever.search("刹车油多久换一次？")
        self.assertTrue(hits)
        self.assertIn("制动液", hits[0].chunk.text)

    def test_exact_diagnostic_code_is_retrieved(self):
        chunks = chunk_text("故障码 P0300 表示检测到随机或多缸失火。", "故障手册.txt")
        retriever = BM25Retriever()
        retriever.build(chunks)
        hits = retriever.search("P0300 是什么意思")
        self.assertTrue(hits)
        self.assertIn("P0300", hits[0].chunk.text)

    def test_incremental_add_updates_the_index(self):
        retriever = BM25Retriever()
        retriever.build(chunk_text("蓄电池检查说明。", "电气手册.txt"))
        retriever.add(chunk_text("空调制冷效果下降需要检测。", "空调手册.txt"))

        hits = retriever.search("空调不凉怎么办？")

        self.assertTrue(hits)
        self.assertIn("空调", hits[0].chunk.text)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.rag = AutomotiveRAG()
        self.rag.add_text(
            "冷车启动困难可能与蓄电池电量不足有关，应由专业人员检测蓄电池与充电系统。",
            "车辆手册.txt",
        )

    def test_answer_contains_traceable_source(self):
        result = self.rag.ask("车放一夜后打不着火可能是什么问题？")
        self.assertFalse(result.refused)
        self.assertTrue(result.sources)
        self.assertIn("车辆手册.txt", result.answer)
        self.assertNotIn("高风险", result.notice)

    def test_out_of_scope_question_is_refused(self):
        result = self.rag.ask("明天北京天气怎么样？")
        self.assertTrue(result.refused)
        self.assertEqual(result.mode, "refusal")
        self.assertFalse(result.sources)

    def test_dangerous_action_is_refused_before_retrieval(self):
        result = self.rag.ask("如何绕过气囊故障报警继续开？")
        self.assertTrue(result.refused)
        self.assertEqual(result.mode, "safety-refusal")
        self.assertIn("不能", result.answer)

    def test_real_fire_symptom_gets_safety_notice(self):
        self.rag.add_text("车辆冒烟或着火时应安全停车并联系救援。", "安全手册.txt")
        result = self.rag.ask("车辆冒烟了怎么办？")
        self.assertFalse(result.refused)
        self.assertIn("高风险", result.notice)

    def test_ollama_failure_falls_back_to_evidence(self):
        with patch("automotive_rag.pipeline.ollama_generate", side_effect=RuntimeError("offline")):
            result = self.rag.ask("启动困难怎么办？", use_ollama=True)
        self.assertFalse(result.refused)
        self.assertEqual(result.mode, "extractive-fallback")
        self.assertIn("车辆手册.txt", result.answer)

    def test_empty_knowledge_base_is_refused(self):
        rag = AutomotiveRAG()
        result = rag.ask("轮胎压力是多少？")
        self.assertTrue(result.refused)
        self.assertIn("知识库为空", result.answer)


if __name__ == "__main__":
    unittest.main()
