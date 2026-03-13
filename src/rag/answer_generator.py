from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.llm.gemini_client import GeminiClient, GeminiClientError
from src.prompts.rag_prompt import build_rag_prompt
from src.rag.context_builder import build_context
from src.retrieval.retriever import (
    DEFAULT_MMR_FETCH_K,
    DEFAULT_MMR_LAMBDA,
    Retriever,
)


DEFAULT_TOP_K = 4
DEFAULT_RETRIEVAL_MODE = "mmr"


@dataclass
class GeneratedAnswer:
    answer: str
    model: str
    retrieved_chunks: List[dict]
    context: str
    metadata: Dict[str, Any]


class AnswerGeneratorError(Exception):
    pass


class AnswerGenerator:
    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        gemini_client: Optional[GeminiClient] = None,
    ) -> None:
        self.retriever = retriever or Retriever()
        self.gemini_client = gemini_client or GeminiClient()

    def generate_answer(
        self,
        query: str,
        retrieval_mode: str = DEFAULT_RETRIEVAL_MODE,
        unit_filter: Optional[str] = None,
        top_k: int = DEFAULT_TOP_K,
    ) -> GeneratedAnswer:
        if not query or not query.strip():
            raise AnswerGeneratorError("Query is empty.")

        try:
            chunks = self.retriever.retrieve(
                query=query,
                top_k=top_k,
                search_type=retrieval_mode,
                fetch_k=DEFAULT_MMR_FETCH_K,
                lambda_mult=DEFAULT_MMR_LAMBDA,
                unit_filter=unit_filter,
            )

            if not chunks:
                return GeneratedAnswer(
                    answer=(
                        "Direct Answer:\n"
                        "I could not find relevant information in the internal documents for this question.\n\n"
                        "Relevant Product Details:\n"
                        "None.\n\n"
                        "Gaps / Validation Needed:\n"
                        "The available documents do not provide enough information to answer this question.\n\n"
                        "Simple Explanation:\n"
                        "Not needed."
                    ),
                    model="none",
                    retrieved_chunks=[],
                    context="",
                    metadata={
                        "query": query,
                        "retrieval_mode": retrieval_mode,
                        "unit_filter": unit_filter,
                        "top_k": top_k,
                        "reason": "no_retrieved_documents",
                    },
                )

            context = build_context(chunks)
            prompt = build_rag_prompt(query=query, context=context)
            response = self.gemini_client.generate_text(prompt)

            return GeneratedAnswer(
                answer=response.text,
                model=response.model,
                retrieved_chunks=chunks,
                context=context,
                metadata={
                    "query": query,
                    "retrieval_mode": retrieval_mode,
                    "unit_filter": unit_filter,
                    "top_k": top_k,
                    "retrieved_count": len(chunks),
                },
            )

        except GeminiClientError as exc:
            raise AnswerGeneratorError(f"Gemini error: {exc}") from exc
        except Exception as exc:
            raise AnswerGeneratorError(f"Answer generation failed: {exc}") from exc