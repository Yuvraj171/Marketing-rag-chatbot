from __future__ import annotations

from typing import List, Dict


DEFAULT_MEMORY_TURNS = 3
MAX_ANSWER_CHARS_PER_TURN = 700
MAX_TOTAL_HISTORY_CHARS = 2200


def _clean_text(text: str) -> str:
    return " ".join(text.strip().split())


def _truncate_text(text: str, max_chars: int) -> str:
    cleaned = _clean_text(text)

    if len(cleaned) <= max_chars:
        return cleaned

    return cleaned[:max_chars].rstrip() + "..."


def build_conversation_history(
    conversations: List[Dict],
    max_turns: int = DEFAULT_MEMORY_TURNS,
    max_answer_chars_per_turn: int = MAX_ANSWER_CHARS_PER_TURN,
    max_total_chars: int = MAX_TOTAL_HISTORY_CHARS,
) -> str:
    if not conversations:
        return "No prior conversation history."

    recent_conversations = list(reversed(conversations[:max_turns]))
    history_blocks = []
    current_total = 0

    for conversation in recent_conversations:
        question = _truncate_text(conversation.get("question", ""), 300)
        answer = _truncate_text(
            conversation.get("answer", ""),
            max_answer_chars_per_turn,
        )

        block = (
            f"User: {question}\n"
            f"Assistant: {answer}"
        )

        projected_total = current_total + len(block) + 2

        if projected_total > max_total_chars:
            remaining = max_total_chars - current_total

            if remaining > 120:
                trimmed_block = block[:remaining].rstrip() + "..."
                history_blocks.append(trimmed_block)
            break

        history_blocks.append(block)
        current_total += len(block) + 2

    if not history_blocks:
        return "No prior conversation history."

    return "\n\n".join(history_blocks)