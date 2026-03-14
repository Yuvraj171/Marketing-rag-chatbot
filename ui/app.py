from __future__ import annotations

import re
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UI_DIR = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.memory.conversation_memory import build_conversation_history
from src.rag.answer_generator import AnswerGenerator, AnswerGeneratorError
from ui.pdf_export import build_answer_pdf


VALID_UNIT_FILTERS = ["All Units", "bestec", "unit1", "unit3", "unit5"]
VALID_RETRIEVAL_MODES = ["mmr", "similarity"]

QUICK_PROMPTS = [
    "Choose a suggested prompt...",
    "Robo2Go specs",
    "What machines support Robo2Go?",
    "Max payload of Robo2Go?",
    "Does the PLC support Modbus TCP?",
    "Compatible DMG MORI machines?",
]


st.set_page_config(
    page_title="Marketing RAG Chatbot",
    page_icon="💬",
    layout="wide",
)


@st.cache_resource
def get_answer_generator() -> AnswerGenerator:
    return AnswerGenerator()


def inject_css() -> None:
    css_file = UI_DIR / "styles.css"
    with open(css_file, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def init_session_state() -> None:
    defaults = {
        "question_input": "",
        "unit_filter": "All Units",
        "retrieval_mode": "mmr",
        "top_k": 4,
        "error_message": None,
        "query_history": [],
        "conversations": [],
        "quick_prompt_selection": QUICK_PROMPTS[0],
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def normalize_unit_filter(unit_value: str):
    return None if unit_value == "All Units" else unit_value


def extract_unique_sources(retrieved_chunks: list[dict]) -> list[dict]:
    seen = set()
    unique_sources = []

    for chunk in retrieved_chunks:
        metadata = chunk.get("metadata", {})
        unit = metadata.get("unit", "unknown")
        file_name = metadata.get("file_name", "unknown")
        page = metadata.get("page", "unknown")
        source_number = chunk.get("source_number")

        key = (source_number, unit, file_name, page)
        if key in seen:
            continue

        seen.add(key)
        unique_sources.append(
            {
                "source_number": source_number,
                "unit": unit,
                "file_name": file_name,
                "page": page,
            }
        )

    unique_sources.sort(
        key=lambda x: (
            x["source_number"] if x["source_number"] is not None else 9999
        )
    )
    return unique_sources


def clean_section_text(text: str) -> str:
    if not text:
        return ""

    cleaned = text.strip()
    cleaned = re.sub(r"^\*\*\s*$", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\n\s*\*\*\s*\n", "\n\n", cleaned)
    cleaned = re.sub(r"^\s*\*\*\s*", "", cleaned)
    cleaned = re.sub(r"\s*\*\*\s*$", "", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned.strip()


def parse_answer_sections(answer_text: str) -> dict:
    markers = [
        ("direct_answer", r"Direct Answer\s*:"),
        ("relevant_details", r"Relevant Product Details\s*:"),
        ("gaps_validation", r"Gaps\s*/\s*Validation Needed\s*:"),
        ("simple_explanation", r"Simple Explanation\s*:"),
    ]

    positions = []
    for key, pattern in markers:
        match = re.search(pattern, answer_text, flags=re.IGNORECASE)
        if match:
            positions.append((key, match.start(), match.end()))

    if not positions:
        return {
            "direct_answer": clean_section_text(answer_text),
            "relevant_details": "",
            "gaps_validation": "",
            "simple_explanation": "",
        }

    positions.sort(key=lambda x: x[1])
    sections = {
        "direct_answer": "",
        "relevant_details": "",
        "gaps_validation": "",
        "simple_explanation": "",
    }

    for i, (key, start, end) in enumerate(positions):
        next_start = positions[i + 1][1] if i + 1 < len(positions) else len(answer_text)
        content = answer_text[end:next_start].strip()
        sections[key] = clean_section_text(content)

    return sections


def handle_quick_prompt_change() -> None:
    selected_prompt = st.session_state.quick_prompt_selection
    if selected_prompt != QUICK_PROMPTS[0]:
        st.session_state.question_input = selected_prompt
        st.session_state.error_message = None


def render_header() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-badge">Internal AI Assistant</div>
            <div class="hero-title">💬 Product Assistant for Sales & Marketing</div>
            <div class="hero-subtitle">
                Ask technical product questions using internal manufacturing documents
                and get clear, business-friendly answers with source grounding.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_query_history() -> None:
    st.sidebar.markdown('<div class="sidebar-title">Recent Questions</div>', unsafe_allow_html=True)

    if not st.session_state.query_history:
        st.sidebar.markdown('<div class="sidebar-empty">No queries yet</div>', unsafe_allow_html=True)
        return

    for idx, q in enumerate(st.session_state.query_history[:10], start=1):
        if st.sidebar.button(q, key=f"history_{idx}", use_container_width=True):
            st.session_state.question_input = q
            run_query(q)


def render_question_panel() -> None:
    st.markdown('<div class="panel-title">Ask Your Question</div>', unsafe_allow_html=True)

    st.markdown("**Quick Prompt**")
    st.selectbox(
        "Quick Prompt",
        options=QUICK_PROMPTS,
        key="quick_prompt_selection",
        on_change=handle_quick_prompt_change,
        label_visibility="collapsed",
    )

    st.text_area(
        "Question",
        placeholder="Example: Does the PLC support Modbus TCP?",
        height=120,
        key="question_input",
    )

    col1, col2 = st.columns([2.2, 1.8])

    with col1:
        st.markdown("**Unit Filter**")
    with col2:
        st.markdown("&nbsp;", unsafe_allow_html=True)

    with col1:
        st.session_state.unit_filter = st.selectbox(
            "Unit Filter",
            options=VALID_UNIT_FILTERS,
            index=VALID_UNIT_FILTERS.index(st.session_state.unit_filter),
            label_visibility="collapsed",
        )

    with col2:
        ask_clicked = st.button("Ask Assistant", use_container_width=True)

    with st.expander("Advanced Options"):
        st.session_state.retrieval_mode = st.selectbox(
            "Retrieval Mode",
            options=VALID_RETRIEVAL_MODES,
            index=VALID_RETRIEVAL_MODES.index(st.session_state.retrieval_mode),
        )

        st.session_state.top_k = st.slider(
            "Number of retrieved chunks",
            min_value=2,
            max_value=8,
            value=st.session_state.top_k,
            step=1,
        )

    if ask_clicked:
        run_query()


def render_answer_sections(answer_text: str) -> dict:
    sections = parse_answer_sections(answer_text)

    if sections["direct_answer"]:
        st.markdown('<div class="answer-section-heading">🟢 Direct Answer</div>', unsafe_allow_html=True)
        st.markdown(sections["direct_answer"])

    if sections["relevant_details"]:
        st.markdown('<div class="answer-section-heading">📋 Relevant Product Details</div>', unsafe_allow_html=True)
        st.markdown(sections["relevant_details"])

    if sections["gaps_validation"]:
        st.markdown('<div class="answer-section-heading">⚠️ Gaps / Validation Needed</div>', unsafe_allow_html=True)
        st.markdown(sections["gaps_validation"])

    if sections["simple_explanation"]:
        st.markdown('<div class="answer-section-heading">💡 Simple Explanation</div>', unsafe_allow_html=True)
        st.markdown(sections["simple_explanation"])

    return sections


def render_sources_for_conversation(conversation: dict) -> None:
    unique_sources = extract_unique_sources(conversation["retrieved_chunks"])
    if not unique_sources:
        return

    st.markdown('<div class="sources-title">📄 Sources Used</div>', unsafe_allow_html=True)

    cols = st.columns(2)
    for idx, source in enumerate(unique_sources[:4]):
        with cols[idx % 2]:
            st.markdown(
                f"""
                <div class="source-card">
                    <div class="source-file">[{source["source_number"]}] {source["file_name"]}</div>
                    <div class="source-meta">📁 Unit: <strong>{source["unit"]}</strong></div>
                    <div class="source-meta">📄 Page: <strong>{source["page"]}</strong></div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_pdf_export_button(conversation: dict, sections: dict) -> None:
    sources = extract_unique_sources(conversation["retrieved_chunks"])
    pdf_bytes = build_answer_pdf(
        question=conversation["question"],
        sections=sections,
        sources=sources,
    )

    safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", conversation["question"]).strip("_")
    filename = f"{safe_name[:60] or 'chatbot_answer'}.pdf"

    st.download_button(
        label="Download as PDF",
        data=pdf_bytes,
        file_name=filename,
        mime="application/pdf",
        use_container_width=False,
        key=f"pdf_download_{filename}",
    )


def render_conversations() -> None:
    if not st.session_state.conversations:
        return

    st.markdown('<div class="conversation-title">Latest Conversation</div>', unsafe_allow_html=True)

    for idx, conversation in enumerate(st.session_state.conversations):
        question = conversation["question"]
        expanded = idx == 0

        with st.expander(f"💬 {question}", expanded=expanded):
            st.markdown('<div class="conversation-block">', unsafe_allow_html=True)
            sections = render_answer_sections(conversation["answer"])
            st.markdown("<br/>", unsafe_allow_html=True)
            render_pdf_export_button(conversation, sections)
            render_sources_for_conversation(conversation)
            st.markdown('</div>', unsafe_allow_html=True)


def run_query(forced_query: str | None = None) -> None:
    st.session_state.error_message = None

    query = forced_query if forced_query is not None else st.session_state.question_input.strip()

    if not query:
        st.session_state.error_message = "Please enter a question"
        return

    if query in st.session_state.query_history:
        st.session_state.query_history.remove(query)
    st.session_state.query_history.insert(0, query)

    generator = get_answer_generator()
    normalized_unit_filter = normalize_unit_filter(st.session_state.unit_filter)
    conversation_history = build_conversation_history(st.session_state.conversations)

    try:
        with st.spinner("Analyzing documents and generating answer..."):
            result = generator.generate_answer(
                query=query,
                retrieval_mode=st.session_state.retrieval_mode,
                unit_filter=normalized_unit_filter,
                top_k=st.session_state.top_k,
                conversation_history=conversation_history,
            )

        conversation_entry = {
            "question": query,
            "answer": result.answer,
            "retrieved_chunks": result.retrieved_chunks,
            "model": result.model,
            "metadata": result.metadata,
        }

        st.session_state.conversations.insert(0, conversation_entry)
        st.session_state.error_message = None

    except AnswerGeneratorError as exc:
        st.session_state.error_message = str(exc)

    except Exception as exc:
        st.session_state.error_message = f"Unexpected error: {exc}"


def main() -> None:
    inject_css()
    init_session_state()
    render_query_history()
    render_header()
    render_question_panel()
    render_conversations()

    if st.session_state.error_message:
        st.error(st.session_state.error_message)


if __name__ == "__main__":
    main()