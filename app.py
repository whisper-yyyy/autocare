"""Streamlit UI for the standalone automotive aftersales RAG."""

import hashlib
import os
import sys
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
STORAGE_PATH = Path(
    os.environ.get("AUTOMOTIVE_RAG_DB_PATH", PROJECT_ROOT / "storage" / "knowledge.db")
)
MAX_FILE_BYTES = 25 * 1024 * 1024
MAX_FILES_PER_IMPORT = 10
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from automotive_rag import AutomotiveRAG  # noqa: E402


st.set_page_config(page_title="车维智答", page_icon="🚗", layout="wide")


@st.cache_resource
def get_rag() -> AutomotiveRAG:
    """Share one thread-safe persistent engine across browser sessions."""

    return AutomotiveRAG(db_path=STORAGE_PATH)


def decode_text_file(data: bytes) -> str:
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("文本编码无法识别，请另存为 UTF-8 或 GB18030 后重试。")


rag = get_rag()

st.title("🚗 车维智答")
st.caption("安全可追溯的汽车售后知识 RAG")
st.caption("车型范围隔离 · 可追溯引用 · 安全优先 · 无依据拒答 · 本地持久化")

with st.sidebar:
    st.header("运行设置")
    use_ollama = st.toggle("使用本地 Ollama 生成答案", value=False)
    ollama_model = st.text_input("Ollama 模型", value="qwen2.5:7b-instruct-q4_K_M")
    ollama_url = st.text_input("Ollama 地址", value="http://127.0.0.1:11434")
    st.caption("为避免访问非预期网络地址，仅允许 localhost/127.0.0.1/::1。")
    top_k = st.slider("检索片段数", 1, 8, 4)
    st.divider()
    document_metric = st.empty()
    chunk_metric = st.empty()
    document_metric.metric("知识文件", len(rag.document_infos))
    chunk_metric.metric("知识片段", len(rag.chunks))
    if st.button("清空知识库", use_container_width=True):
        rag.clear()
        st.session_state.pop("last_answer", None)
        st.session_state.pop("query_document_ids", None)
        st.success("已清空持久化知识库。")
        st.rerun()

left, right = st.columns([1, 1.35], gap="large")

with left:
    st.subheader("1. 建立车辆知识库")
    with st.expander("车辆适用范围（建议填写）", expanded=not rag.document_infos):
        vehicle_model = st.text_input("车型", placeholder="例如：2025款 某车型")
        model_year = st.text_input("年款", placeholder="例如：2025")
        powertrain = st.text_input("动力类型", placeholder="例如：纯电 / 插混 / 燃油")
        market = st.text_input("市场版本", placeholder="例如：中国大陆")
        manual_version = st.text_input("资料版本", placeholder="例如：V2.1 / 2026-03")

    metadata = {
        "vehicle_model": vehicle_model,
        "model_year": model_year,
        "powertrain": powertrain,
        "market": market,
        "manual_version": manual_version,
    }
    uploads = st.file_uploader(
        "上传车辆手册或服务资料",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        help="单文件最大 25 MB；扫描版 PDF 仍需先做 OCR。",
    )
    if uploads and len(uploads) > MAX_FILES_PER_IMPORT:
        st.warning(f"一次最多导入 {MAX_FILES_PER_IMPORT} 个文件，请分批上传。")

    if st.button("导入所选资料", type="primary", disabled=not uploads):
        total = 0
        for upload in (uploads or [])[:MAX_FILES_PER_IMPORT]:
            data = upload.getvalue()
            if len(data) > MAX_FILE_BYTES:
                st.error(f"{upload.name} 超过 25 MB，已跳过。")
                continue
            try:
                if upload.name.lower().endswith(".pdf"):
                    fingerprint = hashlib.sha256(data).hexdigest()
                    decoded_text = None
                else:
                    decoded_text = decode_text_file(data)
                    fingerprint = hashlib.sha256(decoded_text.encode("utf-8")).hexdigest()
                if rag.has_content_hash(fingerprint):
                    st.info(f"已跳过重复内容：{upload.name}")
                    continue

                if decoded_text is None:
                    count = rag.add_pdf_bytes(data, upload.name, metadata=metadata)
                else:
                    count = rag.add_text(decoded_text, upload.name, metadata=metadata)
                if count == 0:
                    st.warning(
                        f"未从 {upload.name} 提取到可检索文字。扫描版 PDF 需要先做 OCR。"
                    )
                    continue
                total += count
                st.success(f"{upload.name}：导入 {count} 个知识片段")
            except Exception as exc:
                st.error(f"{upload.name} 导入失败：{exc}")
        if total:
            st.caption(f"本次共新增 {total} 个片段；重启应用后仍会保留。")
            document_metric.metric("知识文件", len(rag.document_infos))
            chunk_metric.metric("知识片段", len(rag.chunks))

    demo_path = PROJECT_ROOT / "data" / "demo_vehicle_manual.txt"
    if st.button("加载内置演示资料", use_container_width=True):
        fingerprint = hashlib.sha256(demo_path.read_bytes()).hexdigest()
        if rag.has_content_hash(fingerprint):
            st.info("演示资料已经加载。")
        else:
            count = rag.add_file(
                demo_path,
                metadata={"vehicle_model": "虚构演示车型", "manual_version": "演示版"},
            )
            st.success(f"已加载 {count} 个演示知识片段。")
            document_metric.metric("知识文件", len(rag.document_infos))
            chunk_metric.metric("知识片段", len(rag.chunks))

    if rag.document_infos:
        st.write("**已导入资料：**")
        for document in rag.document_infos:
            name_column, delete_column = st.columns([5, 1])
            name_column.write(f"- {document.label} · `{document.document_id[:8]}`")
            if delete_column.button(
                "删除",
                key=f"delete-{document.document_id}",
                help=f"删除 {document.source}",
            ):
                rag.delete_document(document.document_id)
                st.session_state.pop("last_answer", None)
                st.session_state.pop("query_document_ids", None)
                st.rerun()

with right:
    st.subheader("2. 提问并核对证据")
    documents = rag.document_infos
    document_lookup = {document.document_id: document for document in documents}
    default_scope = [documents[0].document_id] if len(documents) == 1 else []
    selected_document_ids = st.multiselect(
        "本次查询资料范围",
        options=list(document_lookup),
        default=default_scope,
        format_func=lambda document_id: document_lookup[document_id].label,
        placeholder="多份资料时请选择对应车型或手册",
    )
    if len(documents) > 1 and not selected_document_ids:
        st.info("知识库包含多份资料。请选择查询范围，系统不会默认混合不同车型答案。")

    question = st.text_area(
        "车辆问题",
        placeholder="例如：冷车启动困难可能和电瓶有关吗？",
        height=110,
    )
    if st.button("检索并回答", type="primary", disabled=not question.strip()):
        with st.spinner("正在检索车辆资料…"):
            result = rag.ask(
                question,
                top_k=top_k,
                use_ollama=use_ollama,
                ollama_model=ollama_model,
                ollama_base_url=ollama_url,
                document_ids=selected_document_ids or None,
            )
        st.session_state.last_answer = result

    result = st.session_state.get("last_answer")
    if result:
        if result.refused:
            st.warning(result.answer)
        else:
            st.markdown(result.answer)
        if result.notice:
            if "高风险" in result.notice:
                st.error(result.notice)
            else:
                st.info(result.notice)

        m1, m2 = st.columns(2)
        m1.metric("回答模式", result.mode)
        m2.metric("检索相关度", f"{result.confidence:.0%}")

        if result.sources:
            st.subheader("证据片段")
            for index, hit in enumerate(result.sources, start=1):
                with st.expander(
                    f"{index}. {hit.chunk.citation} · 相关度 {hit.score:.0%}",
                    expanded=index == 1,
                ):
                    st.write(hit.chunk.text)
                    anchors = "、".join(hit.matched_anchors) or "—"
                    st.caption(
                        f"BM25={hit.bm25_score:.3f}｜锚点覆盖={hit.anchor_coverage:.0%}｜"
                        f"命中={anchors}｜块={hit.chunk.chunk_id}"
                    )

st.divider()
st.caption(
    "知识库保存在本机 SQLite，不连接真实4S店或CRM。答案仅依据选定资料；涉及安全、维修和保养时，以厂商官方手册和专业检查为准。"
)
