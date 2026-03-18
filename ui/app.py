from __future__ import annotations

import base64
import mimetypes
import os
import re
import sys
import tempfile
from pathlib import Path

import streamlit as st
from streamlit_mic_recorder import mic_recorder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UI_DIR = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.memory.conversation_memory import build_conversation_history
from src.rag.answer_generator import AnswerGenerator, AnswerGeneratorError
from src.voice.transcriber import AudioTranscriber, AudioTranscriptionError
from ui.pdf_export import build_answer_pdf


VALID_UNIT_FILTERS = ["All Units", "bestec", "unit1", "unit3", "unit5"]
VALID_RETRIEVAL_MODES = ["mmr", "similarity"]
SUPPORTED_AUDIO_TYPES = ["wav", "mp3", "m4a", "ogg", "webm"]

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


@st.cache_resource
def get_audio_transcriber() -> AudioTranscriber:
    return AudioTranscriber(
        model_size="base",
        device="cpu",
        compute_type="int8",
    )


def inject_css() -> None:
    css_file = UI_DIR / "styles.css"
    with open(css_file, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def get_base64_image(image_path: Path) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def init_session_state() -> None:
    defaults = {
        "question_input": "",
        "transcript_preview_input": "",
        "pending_question_input": None,
        "pending_forced_query": None,
        "unit_filter": "All Units",
        "retrieval_mode": "mmr",
        "top_k": 4,
        "error_message": None,
        "query_history": [],
        "conversations": [],
        "quick_prompt_selection": QUICK_PROMPTS[0],
        "latest_transcript": "",
        "latest_transcript_language": "",
        "latest_transcript_probability": None,
        "voice_auto_run": True,
        "show_transcript_preview": False,
        "mic_status_text": "Idle",
        "mic_last_processed_id": None,
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


def clear_transcript_state() -> None:
    st.session_state.latest_transcript = ""
    st.session_state.latest_transcript_language = ""
    st.session_state.latest_transcript_probability = None
    st.session_state.transcript_preview_input = ""
    st.session_state.mic_status_text = "Idle"


def queue_question_update(question: str, auto_run: bool = False) -> None:
    normalized_question = question.strip()
    if not normalized_question:
        return

    st.session_state.pending_question_input = normalized_question
    st.session_state.pending_forced_query = (
        normalized_question if auto_run else None
    )
    st.session_state.error_message = None


def apply_pending_question_update() -> None:
    pending_question = st.session_state.pop("pending_question_input", None)
    pending_forced_query = st.session_state.pop("pending_forced_query", None)

    if pending_question:
        st.session_state.question_input = pending_question

    if pending_forced_query:
        run_query(pending_forced_query)


def apply_transcript_to_question() -> None:
    transcript_text = st.session_state.transcript_preview_input.strip()
    if transcript_text:
        queue_question_update(transcript_text, auto_run=True)
        st.rerun()


def process_audio_bytes(audio_bytes: bytes, suffix: str = ".wav") -> bool:
    temp_path = None
    should_rerun = False

    try:
        st.session_state.error_message = None
        st.session_state.mic_status_text = "Processing voice..."

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio:
            temp_audio.write(audio_bytes)
            temp_path = temp_audio.name

        transcriber = get_audio_transcriber()

        with st.spinner(
            "Transcribing and generating answer..."
            if st.session_state.voice_auto_run
            else "Transcribing audio..."
        ):
            result = transcriber.transcribe(temp_path)

        if not result.transcript:
            st.session_state.error_message = "No speech detected."
            st.session_state.mic_status_text = "Idle"
            return

        st.session_state.latest_transcript = result.transcript
        st.session_state.latest_transcript_language = result.language
        st.session_state.latest_transcript_probability = result.language_probability
        st.session_state.transcript_preview_input = result.transcript
        st.session_state.mic_status_text = "Transcript ready"

        if st.session_state.voice_auto_run:
            queue_question_update(result.transcript, auto_run=True)
            should_rerun = True

    except AudioTranscriptionError as exc:
        st.session_state.error_message = str(exc)
        st.session_state.mic_status_text = "Idle"

    except Exception as exc:
        st.session_state.error_message = f"Voice processing error: {exc}"
        st.session_state.mic_status_text = "Idle"

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

    return should_rerun


def render_header() -> None:
    logo_path = UI_DIR / "assets" / "best_group_logo.png"

    logo_html = ""
    if logo_path.exists():
        logo_base64 = get_base64_image(logo_path)
        mime_type, _ = mimetypes.guess_type(str(logo_path))
        logo_html = (
            '<div class="hero-logo-wrap">'
            f'<img src="data:{mime_type or "image/png"};base64,{logo_base64}" '
            'class="hero-logo" alt="Best Group Logo">'
            "</div>"
        )

    header_html = (
        '<div class="hero-card compact-hero">'
        '<div class="hero-header-row">'
        f"{logo_html}"
        '<div class="hero-text-wrap">'
        '<div class="hero-badge">Internal AI Assistant</div>'
        '<div class="hero-title">💬 Product Assistant for Sales & Marketing</div>'
        '<div class="hero-subtitle">'
        "Ask technical product questions using internal manufacturing documents "
        "and get clear, business-friendly answers with source grounding."
        "</div>"
        "</div>"
        "</div>"
        "</div>"
    )

    if hasattr(st, "html"):
        st.html(header_html)
    else:
        st.markdown(header_html, unsafe_allow_html=True)


def render_query_history() -> None:
    st.sidebar.markdown('<div class="sidebar-title">Recent Questions</div>', unsafe_allow_html=True)

    if not st.session_state.query_history:
        st.sidebar.markdown('<div class="sidebar-empty">No queries yet</div>', unsafe_allow_html=True)
        return

    for idx, q in enumerate(st.session_state.query_history[:10], start=1):
        if st.sidebar.button(q, key=f"history_{idx}", use_container_width=True):
            st.session_state.question_input = q
            run_query(q)


def render_transcript_preview() -> None:
    if not st.session_state.show_transcript_preview:
        return

    if not st.session_state.latest_transcript:
        return

    st.markdown("**Transcript Preview**")
    st.text_area(
        "Transcript Preview",
        key="transcript_preview_input",
        height=90,
        label_visibility="collapsed",
    )

    if not st.session_state.voice_auto_run:
        if st.button("Use Transcript as Question", use_container_width=True):
            apply_transcript_to_question()


def render_voice_options() -> None:
    st.markdown("**Voice Input**")

    with st.expander("Voice Options", expanded=False):
        st.toggle(
            "Auto-run query after transcription",
            key="voice_auto_run",
            help="If enabled, the chatbot answers immediately after transcription.",
        )

        st.toggle(
            "Show transcript preview",
            key="show_transcript_preview",
            help="Show the recognized transcript so you can review or edit it.",
        )

    if st.session_state.mic_status_text == "Processing voice...":
        st.info("🌀 Processing your voice input...")
    elif st.session_state.mic_status_text == "Transcript ready":
        st.success("✅ Transcript ready")


def render_live_microphone() -> None:
    st.markdown("**🎙 Record a Voice Question**")

    audio = mic_recorder(
        start_prompt="Start Recording",
        stop_prompt="Stop Recording",
        just_once=True,
        use_container_width=True,
        key="live_mic_recorder",
    )

    if audio:
        audio_id = audio.get("id")

        if audio_id != st.session_state.mic_last_processed_id:
            st.session_state.mic_last_processed_id = audio_id
            if process_audio_bytes(audio["bytes"], suffix=".wav"):
                st.rerun()


def render_voice_input_panel() -> None:
    st.markdown("**📁 Upload a Voice File**")

    uploaded_audio = st.file_uploader(
        "Upload your spoken question",
        type=SUPPORTED_AUDIO_TYPES,
        accept_multiple_files=False,
        help="Supported formats: wav, mp3, m4a, ogg, webm",
    )

    if uploaded_audio is not None:
        col1, col2 = st.columns([1.3, 2.7])

        with col1:
            ask_from_audio_clicked = st.button(
                "Use Uploaded Audio",
                use_container_width=True,
            )

        with col2:
            st.caption(f"Selected file: {uploaded_audio.name}")

        if ask_from_audio_clicked:
            suffix = Path(uploaded_audio.name).suffix or ".tmp"
            if process_audio_bytes(uploaded_audio.getbuffer(), suffix=suffix):
                st.rerun()


def render_question_panel() -> None:
    st.markdown('<div class="panel-title">Ask Your Question</div>', unsafe_allow_html=True)

    st.markdown("**Type Your Question**")
    st.text_area(
        "Question",
        placeholder="Example: Does the PLC support Modbus TCP?",
        height=120,
        key="question_input",
    )

    st.markdown("**Quick Prompt**")
    st.selectbox(
        "Quick Prompt",
        options=QUICK_PROMPTS,
        key="quick_prompt_selection",
        on_change=handle_quick_prompt_change,
        label_visibility="collapsed",
    )

    render_voice_options()
    render_live_microphone()
    render_voice_input_panel()
    render_transcript_preview()

    st.markdown("**Unit Filter**")

    unit_col, button_col = st.columns([1.4, 1.2])

    with unit_col:
        st.session_state.unit_filter = st.selectbox(
            "Unit Filter",
            options=VALID_UNIT_FILTERS,
            index=VALID_UNIT_FILTERS.index(st.session_state.unit_filter),
            label_visibility="collapsed",
        )

    with button_col:
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
        st.session_state.mic_status_text = "Idle"

        clear_transcript_state()

    except AnswerGeneratorError as exc:
        st.session_state.error_message = str(exc)
        st.session_state.mic_status_text = "Idle"

    except Exception as exc:
        st.session_state.error_message = f"Unexpected error: {exc}"
        st.session_state.mic_status_text = "Idle"


def main() -> None:
    inject_css()
    init_session_state()
    apply_pending_question_update()
    render_query_history()
    render_header()
    render_question_panel()
    render_conversations()

    if st.session_state.error_message:
        st.error(st.session_state.error_message)


if __name__ == "__main__":
    main()
