import sys
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_chroma import Chroma
from src.embeddings.embedding_model import get_embedding_model
from src.rag.context_builder import build_context


VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store"
COLLECTION_NAME = "marketing_rag_chunks"

DEFAULT_TOP_K = 4
DEFAULT_MMR_FETCH_K = 15
DEFAULT_MMR_LAMBDA = 0.4

ALLOWED_UNITS = {"bestec", "unit1", "unit3", "unit5"}


class Retriever:
    def __init__(self) -> None:
        if not VECTOR_STORE_DIR.exists():
            raise FileNotFoundError(
                f"Vector store not found at: {VECTOR_STORE_DIR}\n"
                "Run Phase 3 vector build first."
            )

        self.embedding_model = get_embedding_model()

        self.vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=str(VECTOR_STORE_DIR),
            embedding_function=self.embedding_model,
        )

        collection_count = self.vector_store._collection.count()
        if collection_count == 0:
            raise ValueError(
                f"Collection '{COLLECTION_NAME}' is empty inside: {VECTOR_STORE_DIR}"
            )

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        search_type: str = "similarity",
        fetch_k: int = DEFAULT_MMR_FETCH_K,
        lambda_mult: float = DEFAULT_MMR_LAMBDA,
        unit_filter: Optional[str] = None,
    ) -> List[dict]:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        if fetch_k <= 0:
            raise ValueError("fetch_k must be greater than 0.")

        if fetch_k < top_k:
            raise ValueError("fetch_k must be greater than or equal to top_k.")

        if not 0.0 <= lambda_mult <= 1.0:
            raise ValueError("lambda_mult must be between 0.0 and 1.0.")

        if search_type not in ["similarity", "mmr"]:
            raise ValueError("search_type must be either 'similarity' or 'mmr'.")

        if unit_filter is not None:
            unit_filter = unit_filter.strip().lower()

            if unit_filter not in ALLOWED_UNITS:
                raise ValueError(
                    f"unit_filter must be one of: {', '.join(sorted(ALLOWED_UNITS))}"
                )

        metadata_filter = {"unit": unit_filter} if unit_filter else None

        if search_type == "similarity":
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=top_k,
                filter=metadata_filter,
            )

            formatted_results = []

            for doc, score in results:
                formatted_results.append(
                    {
                        "text": doc.page_content,
                        "metadata": doc.metadata,
                        "score": float(score),
                        "search_type": "similarity",
                        "fetch_k": None,
                        "lambda_mult": None,
                        "unit_filter": unit_filter,
                    }
                )

            return formatted_results

        mmr_docs = self.vector_store.max_marginal_relevance_search(
            query=query,
            k=top_k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=metadata_filter,
        )

        return [
            {
                "text": doc.page_content,
                "metadata": doc.metadata,
                "score": None,
                "search_type": "mmr",
                "fetch_k": fetch_k,
                "lambda_mult": lambda_mult,
                "unit_filter": unit_filter,
            }
            for doc in mmr_docs
        ]


def print_results(chunks: List[dict]) -> None:
    print(f"\nRetrieved {len(chunks)} chunk(s)\n")

    if chunks:
        unit_filter = chunks[0]["unit_filter"]
        if unit_filter:
            print(f"Unit filter applied: {unit_filter}")
        else:
            print("Unit filter applied: none (searching across all units)")

    if chunks and chunks[0]["search_type"] == "similarity":
        print("Mode: similarity")
        print("Note: For Chroma distance scores, lower is generally better.\n")
    elif chunks and chunks[0]["search_type"] == "mmr":
        print("Mode: mmr")
        print(
            f"MMR settings used: fetch_k={chunks[0]['fetch_k']}, "
            f"lambda_mult={chunks[0]['lambda_mult']}"
        )
        print("Note: MMR focuses on relevance plus diversity, so no score is shown.\n")

    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk["metadata"]

        print("=" * 80)
        print(f"Result #{index}")

        if chunk["score"] is not None:
            print(f"Score: {chunk['score']:.6f}")

        print(f"Unit: {metadata.get('unit', 'N/A')}")
        print(f"File: {metadata.get('file_name', 'N/A')}")
        print(f"Page: {metadata.get('page', 'N/A')}")
        print(f"Extraction Method: {metadata.get('extraction_method', 'N/A')}")
        print("\nText Preview:\n")
        print(chunk["text"][:500])
        print("=" * 80)
        print()


def get_unit_filter_from_user() -> Optional[str]:
    print("\nAvailable unit filters: bestec, unit1, unit3, unit5")
    unit_raw = input(
        "Enter unit filter (or press Enter to search all units): "
    ).strip().lower()

    return unit_raw if unit_raw else None


def main() -> None:
    print("\nInitializing Retriever...\n")

    retriever = Retriever()

    print("Retriever ready.")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            query = input("Enter your query: ").strip()
        except KeyboardInterrupt:
            print("\n\nExiting retriever test.\n")
            break

        if query.lower() in ["exit", "quit"]:
            print("\nExiting retriever test.\n")
            break

        mode = input(
            "Choose retrieval mode ('similarity' or 'mmr', default: similarity): "
        ).strip().lower()

        if not mode:
            mode = "similarity"

        unit_filter = get_unit_filter_from_user()

        try:
            chunks = retriever.retrieve(
                query=query,
                top_k=DEFAULT_TOP_K,
                search_type=mode,
                fetch_k=DEFAULT_MMR_FETCH_K,
                lambda_mult=DEFAULT_MMR_LAMBDA,
                unit_filter=unit_filter,
            )

            print_results(chunks)

            context = build_context(chunks)

            print("\n\n============= LLM CONTEXT =============\n")
            print(context)

        except Exception as error:
            print(f"\nError: {error}\n")


if __name__ == "__main__":
    main()