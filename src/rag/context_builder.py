MAX_CONTEXT_CHARS_PER_SOURCE = 1700
MAX_TOTAL_CONTEXT_CHARS = 6000


def truncate_text(text: str, max_chars: int = MAX_CONTEXT_CHARS_PER_SOURCE) -> str:
    cleaned_text = text.strip()

    if len(cleaned_text) <= max_chars:
        return cleaned_text

    return cleaned_text[:max_chars].rstrip() + "..."


def format_source_block(chunk: dict, source_number: int) -> str:
    metadata = chunk.get("metadata", {})
    raw_text = chunk.get("text", "")

    text = truncate_text(raw_text)

    unit = metadata.get("unit", "Unknown Unit")
    file_name = metadata.get("file_name", "Unknown File")
    page = metadata.get("page", "Unknown Page")
    source_id = chunk.get("source_number", source_number)

    source_block = (
        f"Source [{source_id}]\n"
        f"Unit: {unit}\n"
        f"File: {file_name}\n"
        f"Page: {page}\n"
        f"Content:\n{text}\n"
    )

    return source_block


def build_context(chunks: list[dict]) -> str:
    if not chunks:
        return "No relevant internal document context was found."

    context_blocks = []
    current_total = 0

    for idx, chunk in enumerate(chunks, start=1):
        block = format_source_block(chunk, idx)

        if current_total + len(block) > MAX_TOTAL_CONTEXT_CHARS:
            remaining = MAX_TOTAL_CONTEXT_CHARS - current_total

            if remaining > 120:
                trimmed_block = block[:remaining].rstrip() + "\n..."
                context_blocks.append(trimmed_block)
            break

        context_blocks.append(block)
        current_total += len(block)

    final_context = "\n".join(context_blocks)
    return final_context