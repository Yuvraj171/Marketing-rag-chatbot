MAX_CONTEXT_CHARS_PER_SOURCE = 1700


def truncate_text(text: str, max_chars: int = MAX_CONTEXT_CHARS_PER_SOURCE) -> str:
    """
    Truncates text to a maximum character length for cleaner LLM context.
    """
    cleaned_text = text.strip()

    if len(cleaned_text) <= max_chars:
        return cleaned_text

    return cleaned_text[:max_chars].rstrip() + "..."


def format_source_block(chunk: dict, source_number: int) -> str:
    """
    Formats a single retrieved chunk into a readable source block.
    """
    metadata = chunk.get("metadata", {})
    raw_text = chunk.get("text", "")

    text = truncate_text(raw_text)

    unit = metadata.get("unit", "Unknown Unit")
    file_name = metadata.get("file_name", "Unknown File")
    page = metadata.get("page", "Unknown Page")

    source_block = (
        f"Source {source_number}\n"
        f"Unit: {unit}\n"
        f"File: {file_name}\n"
        f"Page: {page}\n"
        f"Content:\n{text}\n"
    )

    return source_block


def build_context(chunks: list[dict]) -> str:
    """
    Builds the full LLM context from retrieved chunks.
    """
    if not chunks:
        return "No relevant internal document context was found."

    context_blocks = []

    for idx, chunk in enumerate(chunks, start=1):
        block = format_source_block(chunk, idx)
        context_blocks.append(block)

    context_header = "INTERNAL DOCUMENT CONTEXT\n\n"
    final_context = context_header + "\n".join(context_blocks)

    return final_context