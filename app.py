"""Run with: streamlit run app.py"""
import os
from contextlib import contextmanager

import pandas as pd
import streamlit as st

from intelliresearch.ingestion import corpus_id, preview_page, validate_uploads
from intelliresearch.models import ResearchError, Upload
from intelliresearch.provider import Gemini
from intelliresearch.retrieval import E5Encoder
from intelliresearch.service import ask, make_chart, prepare, summarize


@st.cache_resource
def get_encoder():
    # Only public model weights are shared. Uploaded documents and indexes are session-local.
    return E5Encoder()


@contextmanager
def provider(api_key, model):
    instance = Gemini(api_key, model)
    try:
        yield instance
    finally:
        instance.close()


def show_sources(sources, uploads):
    if not sources:
        return
    with st.expander("Inspect source passages"):
        for source in sources:
            st.text(f"{source.filename} · page {source.page} · {source.kind}")
            st.text(source.text)
        pages = list(dict.fromkeys((c.document_id, c.page) for c in sources))
        # Render only the cited pages, without shared screenshot directories.
        for identity, number in pages:
            st.image(preview_page(uploads, identity, number), caption=f"Source {identity[:8]} · page {number}")


def submit_question(workspace, question, api_key, model):
    with st.spinner("Searching your documents…"):
        with provider(api_key, model) as client:
            ask(workspace, question, client)


def render_chat(workspace, api_key, model):
    for message in workspace.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            show_sources(message.get("sources", ()), workspace.uploads)
    question = st.chat_input("Ask a question about your papers", max_chars=2000)
    if question:
        submit_question(workspace, question, api_key, model)
        st.rerun()


def main():
    st.set_page_config(page_title="IntelliResearch", page_icon="📚", layout="wide")
    st.title("IntelliResearch")
    st.caption("Read across papers. Ask questions. Inspect the evidence.")

    with st.sidebar:
        st.header("Your research sources")
        uploaded = st.file_uploader("PDF documents", type=["pdf"], accept_multiple_files=True)
        st.caption("Up to 5 PDFs, 20 MB each, 100 pages total.")
        api_key = st.text_input("Gemini API key", type="password", value=os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", "")))
        model = st.text_input("Gemini model", value=os.getenv("GEMINI_MODEL", "gemini-flash-latest"))
        ocr = st.checkbox("OCR scanned pages", value=False)
        languages = st.selectbox("OCR languages", ["eng+ara", "eng", "ara"], disabled=not ocr)
        visual = st.checkbox("Analyze figures and tables", value=False)
        st.caption("Answers and summaries send selected document text to Gemini. Figure analysis also sends page images. OCR and embeddings run locally.")
        process = st.button("Process documents", type="primary", disabled=not uploaded)
        clear = st.button("Clear conversation")

    uploads = [Upload(u.name, u.getvalue()) for u in uploaded or []]
    options = {"ocr": ocr, "ocr_languages": languages, "visual": visual,
               "visual_model": model if visual else ""}
    identity = corpus_id(uploads, options)
    workspace = st.session_state.get("workspace")
    if workspace and workspace.fingerprint != identity:
        # Changing the corpus/options immediately removes stale answers and pending charts.
        st.session_state.pop("workspace", None)
        st.session_state.pop("chart", None)
        workspace = None
    if clear and workspace:
        workspace.messages.clear()

    if process and not workspace:
        with st.spinner("Extracting pages and building your search index…"):
            validate_uploads(uploads)
            encoder = get_encoder()
            if visual:
                with provider(api_key, model) as client:
                    workspace = prepare(uploads, options, encoder, client)
            else:
                workspace = prepare(uploads, options, encoder)
            st.session_state.workspace = workspace
    if not workspace:
        st.info("Upload your papers and choose Process documents to begin.")
        st.markdown("**Explore your papers**\n\n- Get a combined summary with source passages.\n- Ask follow-up questions in English or Arabic.\n- Compare printed values in a chart you can inspect.")
        return

    st.success(f"Ready · {len(workspace.uploads)} documents · {workspace.page_count} pages · {len(workspace.index.chunks)} searchable passages")
    for warning in workspace.warnings:
        st.warning(warning)
    st.caption("Source excerpts are checked for exact matches. This does not guarantee that every interpretation is correct; inspect the cited pages.")
    chat, summary_tab, chart_tab = st.tabs(["Ask your papers", "Research summary", "Charts"])
    with chat:
        render_chat(workspace, api_key, model)
    with summary_tab:
        language = st.selectbox("Summary language", ["English", "Arabic"])
        if st.button("Generate summary and questions"):
            with st.spinner("Reading all indexed passages, then combining their evidence…"):
                with provider(api_key, model) as client:
                    summarize(workspace, client, language)
        if workspace.summary:
            st.markdown(workspace.summary.markdown)
            show_sources(workspace.summary.sources, workspace.uploads)
            st.download_button("Download summary", workspace.summary.markdown, file_name="research-summary.md")
            for index, question in enumerate(workspace.questions):
                if st.button(question, key=f"question-{index}"):
                    submit_question(workspace, question, api_key, model)
                    st.rerun()
    with chart_tab:
        st.write("Create a chart from explicitly printed values. Values inferred from a plotted curve are not used.")
        request = st.text_input("Describe the comparison", max_chars=2000)
        if st.button("Extract chart data"):
            st.session_state.pop("chart", None)
            with st.spinner("Finding printed values and their source excerpts…"):
                with provider(api_key, model) as client:
                    st.session_state.chart = make_chart(workspace, request, client)
        if st.session_state.get("chart"):
            spec, sources = st.session_state.chart
            frame = pd.DataFrame([{"Label": p.label, "Value": p.value, "Source excerpt": p.evidence.quote} for p in spec.points])
            st.subheader(spec.title)
            st.caption(f"Unit: {spec.unit}. Review the values and excerpts before plotting.")
            st.dataframe(frame, hide_index=True)
            if st.button("Plot these values"):
                plot = frame.set_index("Label")[["Value"]]
                if spec.kind == "bar":
                    st.bar_chart(plot)
                else:
                    st.line_chart(plot)
            st.download_button("Download chart data", frame.to_csv(index=False), file_name="chart-data.csv", mime="text/csv")
            cited = {p.evidence.source_id for p in spec.points}
            show_sources([c for c in sources if c.id in cited], workspace.uploads)


if __name__ == "__main__":
    try:
        main()
    except ResearchError as error:
        st.error(str(error))
    except Exception:
        # Provider errors can contain request details. Do not display them or log documents/keys.
        st.error("Processing failed. Check your installation, network connection, and PDF files, then retry.")
