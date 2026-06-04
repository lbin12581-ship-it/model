from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

import streamlit as st

from summary_system.agents import AgentOrchestrator
from summary_system.archive import archive_result, list_archives, search_archives
from summary_system.asr_client import ASRClient, asr_status_text
from summary_system.exporters import result_to_markdown, save_docx, save_markdown, save_pdf
from summary_system.llm_client import llm_status_text
from summary_system.models import InputDocument
from summary_system.text_utils import read_docx, read_txt


st.set_page_config(page_title="语音文本结构化总结系统", page_icon="📝", layout="wide")


def load_uploaded_text() -> tuple[str, str]:
    uploaded = st.file_uploader("上传 TXT 或 DOCX 文件", type=["txt", "docx"])
    if not uploaded:
        return "", "manual-input"
    data = uploaded.getvalue()
    if uploaded.name.lower().endswith(".txt"):
        return read_txt(data), uploaded.name
    with NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)
    try:
        return read_docx(tmp_path), uploaded.name
    finally:
        tmp_path.unlink(missing_ok=True)


def file_download(path: Path, label: str, mime: str) -> None:
    st.download_button(label, path.read_bytes(), file_name=path.name, mime=mime)


st.title("基于智能体的语音文本结构化总结系统")

with st.sidebar:
    st.subheader("处理设置")
    st.caption(llm_status_text())
    st.caption(asr_status_text())
    mode = st.selectbox(
        "场景模式",
        [
            ("auto", "自动识别"),
            ("meeting", "会议总结"),
            ("classroom", "课堂知识"),
            ("mixed", "混合场景"),
            ("general", "通用摘要"),
        ],
        format_func=lambda item: item[1],
    )[0]
    title = st.text_input("结果标题", value="结构化总结")
    st.divider()
    st.subheader("历史检索")
    keyword = st.text_input("关键词")
    if keyword:
        for item in search_archives(keyword):
            st.caption(f"{item['created_at']} · {item['scene']} · {item['title']}")

left, right = st.columns([0.95, 1.05])

with left:
    with st.expander("语音转文字（阿里云 Paraformer）"):
        st.caption("使用 paraformer-8k-v1 文件转写。支持本地音频上传，也支持公网 HTTP/HTTPS 或 OSS URL。")
        audio_file = st.file_uploader(
            "上传本地音频文件",
            type=["wav", "mp3", "m4a", "aac", "flac", "ogg", "opus", "amr", "mp4"],
        )
        audio_url = st.text_input("音频文件 URL", placeholder="https://example.com/audio.wav")
        if st.button("识别音频", use_container_width=True):
            asr = ASRClient.from_env()
            if not asr:
                st.error("请先在 .env 中配置 DASHSCOPE_API_KEY 或 ASR_API_KEY。")
            elif audio_file is not None:
                with st.spinner("正在上传本地音频并调用阿里云语音识别，请稍等..."):
                    try:
                        st.session_state["asr_text"] = asr.transcribe_file_bytes(audio_file.name, audio_file.getvalue())
                        st.success("语音识别完成，转写文本已填入下方文本框。")
                    except Exception as exc:
                        st.error(f"语音识别失败：{exc}")
            elif audio_url.strip():
                with st.spinner("正在调用阿里云语音识别，请稍等..."):
                    try:
                        st.session_state["asr_text"] = asr.transcribe_url(audio_url.strip())
                        st.success("语音识别完成，转写文本已填入下方文本框。")
                    except Exception as exc:
                        st.error(f"语音识别失败：{exc}")
            else:
                st.error("请先上传本地音频文件或填写音频文件 URL。")
    file_text, source_name = load_uploaded_text()
    default_text = file_text or st.session_state.get("asr_text", "")
    pasted_text = st.text_area("或粘贴语音转写文本", value=default_text, height=420)
    run = st.button("生成结构化总结", type="primary", use_container_width=True)

with right:
    if run:
        if not pasted_text.strip():
            st.error("请先上传文件或粘贴文本。")
            st.stop()
        orchestrator = AgentOrchestrator()
        result = orchestrator.run(InputDocument(title=title, raw_text=pasted_text, mode=mode, source_name=source_name))
        archive_path = archive_result(result)
        st.session_state["result"] = result
        st.session_state["archive_path"] = archive_path

    result = st.session_state.get("result")
    if result:
        st.success(f"已生成：{result.scene.recommended_template}，置信度 {result.scene.confidence}")
        tabs = st.tabs(["总结结果", "清洗文本", "处理链路", "导出"])
        with tabs[0]:
            st.markdown(result_to_markdown(result))
        with tabs[1]:
            st.text_area("清洗后的文本", result.cleaned_text, height=420)
        with tabs[2]:
            st.write("场景识别依据")
            for reason in result.scene.reasons:
                st.write(f"- {reason}")
            st.write(f"长文本分块数量：{len(result.chunks)}")
            st.write("质量校验")
            if result.quality_issues:
                for issue in result.quality_issues:
                    st.warning(issue.message) if issue.level == "warning" else st.info(issue.message)
            else:
                st.info("未发现明显质量问题。")
        with tabs[3]:
            export_dir = Path("data/exports")
            export_dir.mkdir(parents=True, exist_ok=True)
            stem = result.created_at.replace(":", "-") + "_" + "".join(ch if ch.isalnum() else "_" for ch in result.title)[:40]
            md_path = save_markdown(result, export_dir / f"{stem}.md")
            docx_path = save_docx(result, export_dir / f"{stem}.docx")
            pdf_path = save_pdf(result, export_dir / f"{stem}.pdf")
            col1, col2, col3 = st.columns(3)
            with col1:
                file_download(md_path, "下载 Markdown", "text/markdown")
            with col2:
                file_download(docx_path, "下载 Word", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
            with col3:
                file_download(pdf_path, "下载 PDF", "application/pdf")
            st.caption(f"已归档：{st.session_state.get('archive_path')}")
    else:
        st.info("上传或粘贴转写文本后，点击生成即可查看结构化总结。")

with st.expander("最近归档"):
    archives = list_archives()[:8]
    if archives:
        for path in archives:
            st.caption(str(path))
    else:
        st.caption("暂无归档记录。")
