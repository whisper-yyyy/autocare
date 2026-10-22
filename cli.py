"""Small command-line demo that works without Streamlit."""

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from automotive_rag import AutomotiveRAG  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="车维智答：汽车售后知识 RAG 命令行演示")
    parser.add_argument("--file", action="append", help="PDF/TXT/MD 文件，可重复传入")
    parser.add_argument("--question", help="直接提问；省略则进入交互模式")
    parser.add_argument("--ollama", action="store_true", help="调用本地 Ollama")
    parser.add_argument("--model", default="qwen2.5:7b-instruct-q4_K_M")
    parser.add_argument(
        "--source",
        action="append",
        help="限定查询的文件名，可重复传入；多份不同车型资料时建议使用",
    )
    args = parser.parse_args()

    rag = AutomotiveRAG()
    paths = args.file or [str(PROJECT_ROOT / "data" / "demo_vehicle_manual.txt")]
    for value in paths:
        count = rag.add_file(value)
        print(f"已导入 {Path(value).name}：{count} 个片段")

    selected_document_ids = None
    if args.source:
        requested_sources = set(args.source)
        selected_document_ids = [
            document.document_id
            for document in rag.document_infos
            if document.source in requested_sources
        ]
        missing_sources = requested_sources.difference(
            document.source for document in rag.document_infos
        )
        if missing_sources:
            parser.error(f"未找到资料：{', '.join(sorted(missing_sources))}")

    def ask(question: str) -> None:
        result = rag.ask(
            question,
            use_ollama=args.ollama,
            ollama_model=args.model,
            document_ids=selected_document_ids,
        )
        print(f"\n[{result.mode}｜置信度 {result.confidence:.0%}]\n{result.answer}")
        if result.notice:
            print(f"提示：{result.notice}")

    if args.question:
        ask(args.question)
        return
    while True:
        question = input("\n请输入车辆问题（输入 q 退出）：").strip()
        if question.lower() in {"q", "quit", "exit"}:
            break
        ask(question)


if __name__ == "__main__":
    main()
