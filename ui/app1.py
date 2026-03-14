from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.answer_generator import AnswerGenerator, AnswerGeneratorError


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
    st.markdown(
        """
        <style>
        .main {
            background: linear-gradient(180deg, #f5f7fb 0%, #eef3ff 100%);
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1100px;
        }

        .hero-card {
            background: rgba(255, 255, 255, 0.72);
            border-radius: 24px;
            padding: 28px 32px;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
            margin-bottom: 1.25rem;
        }

        .hero-title {
            font-size: 2.2rem;
            font-weight: 700;
        }

        .hero-subtitle {
            font-size: 1rem;
            color: #4b5563;
        }

        textarea {
            border-radius: 14px !important;
        }

        div[data-testid="stButton"] > button {
            border-radius: 12px;
        }

        div[data-baseweb="select"] > div {
            border-radius: 12px !important;
        }

        .source-card {
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 16px;
            padding: 14px;
            margin-bottom: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_session_state() -> None:
    defaults = {
        "question_input": "",
        "unit_filter": "All Units",
        "retrieval_mode": "mmr",
        "top_k": 4,
        "error_message": None,
        "query_history": [],
        "chat_history": [],
        "quick_prompt_selection": QUICK_PROMPTS[0],
        "quick_prompt_pending_submission": False,
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

        key = (unit, file_name, page)
        if key in seen:
            continue

        seen.add(key)
        unique_sources.append(
            {
                "unit": unit,
                "file_name": file_name,
                "page": page,
            }
        )

    return unique_sources


def handle_quick_prompt_change() -> None:
    selected_prompt = st.session_state.quick_prompt_selection
    if selected_prompt != QUICK_PROMPTS[0]:
        st.session_state.question_input = selected_prompt
        st.session_state.quick_prompt_pending_submission = True
        st.session_state.error_message = None


def render_header() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">💬 Product Assistant for Sales & Marketing</div>
            <div class="hero-subtitle">
                Ask technical product questions using internal documents.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_query_history() -> None:
    st.sidebar.title("Recent Questions")

    if not st.session_state.query_history:
        st.sidebar.write("No queries yet")
        return

    for idx, q in enumerate(reversed(st.session_state.query_history[-10:]), start=1):
        if st.sidebar.button(q, key=f"history_{idx}"):
            st.session_state.question_input = q
            run_query(q)


def render_sources_for_result(result, suffix: str = "") -> None:
    if not hasattr(result, "retrieved_chunks") or not result.retrieved_chunks:
        return

    unique_sources = extract_unique_sources(result.retrieved_chunks)
    if not unique_sources:
        return

    st.markdown("**Sources Used**")

    cols = st.columns(2)
    for idx, source in enumerate(unique_sources[:4]):
        with cols[idx % 2]:
            st.markdown(
                f"""
                <div class="source-card">
                    <div style="font-weight:700; margin-bottom:8px;">{source["file_name"]}</div>
                    <div style="color:#cbd5e1;">Unit: {source["unit"]}</div>
                    <div style="color:#cbd5e1;">Page: {source["page"]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def build_chat_pairs() -> list[dict]:
    chat_history = st.session_state.chat_history
    pairs = []

    i = 0
    while i < len(chat_history):
        current = chat_history[i]

        if current["role"] == "user":
            pair = {
                "user": current["content"],
                "assistant": None,
                "result_object": None,
            }

            if i + 1 < len(chat_history) and chat_history[i + 1]["role"] == "assistant":
                pair["assistant"] = chat_history[i + 1]["content"]
                pair["result_object"] = chat_history[i + 1].get("result_object")

            pairs.append(pair)
            i += 2
        else:
            i += 1

    return pairs


def render_chat() -> None:
    if not st.session_state.chat_history:
        return

    chat_pairs = build_chat_pairs()

    for idx, pair in enumerate(reversed(chat_pairs), start=1):
        with st.chat_message("user"):
            st.markdown(pair["user"])

        if pair["assistant"] is not None:
            with st.chat_message("assistant"):
                st.markdown(pair["assistant"])
                if pair["result_object"] is not None:
                    render_sources_for_result(pair["result_object"], suffix=f"pair_{idx}")


def render_question_panel() -> None:
    st.markdown("### Ask Your Question")

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

    col1, col2 = st.columns(2)

    with col1:
        st.session_state.unit_filter = st.selectbox(
            "Unit Filter",
            options=VALID_UNIT_FILTERS,
            index=VALID_UNIT_FILTERS.index(st.session_state.unit_filter),
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

    if st.session_state.quick_prompt_pending_submission:
        st.session_state.quick_prompt_pending_submission = False
        run_query(st.session_state.question_input.strip())
    elif ask_clicked:
        run_query()


def run_query(forced_query: str | None = None) -> None:
    st.session_state.error_message = None

    query = forced_query if forced_query is not None else st.session_state.question_input.strip()

    if not query:
        st.session_state.error_message = "Please enter a question"
        return

    if query not in st.session_state.query_history:
        st.session_state.query_history.append(query)

    generator = get_answer_generator()
    normalized_unit_filter = normalize_unit_filter(st.session_state.unit_filter)

    conversation_context = ""
    history = st.session_state.chat_history[-6:]

    for msg in history:
        role = msg["role"]
        text = msg["content"]
        conversation_context += f"{role.upper()}: {text}\n"

    query_with_context = f"""Conversation History:
{conversation_context}

Current Question:
{query}
"""

    try:
        with st.spinner("Searching documents..."):
            result = generator.generate_answer(
                query=query_with_context,
                retrieval_mode=st.session_state.retrieval_mode,
                unit_filter=normalized_unit_filter,
                top_k=st.session_state.top_k,
            )

        st.session_state.chat_history.append(
            {"role": "user", "content": query}
        )

        st.session_state.chat_history.append(
            {"role": "assistant", "content": result.answer, "result_object": result}
        )

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
    render_chat()

    if st.session_state.error_message:
        st.error(st.session_state.error_message)


if __name__ == "__main__":
    main()