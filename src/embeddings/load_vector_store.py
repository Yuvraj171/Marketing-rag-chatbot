import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document
from langchain_chroma import Chroma

from src.embeddings.embedding_model import get_embedding_model


VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store"
COLLECTION_NAME = "marketing_rag_chunks"


def load_vector_store() -> Chroma:
    if not VECTOR_STORE_DIR.exists():
        raise FileNotFoundError(
            f"Vector store not found at: {VECTOR_STORE_DIR}\n"
            f"Run build_vector_store.py first."
        )

    embedding_model = get_embedding_model()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(VECTOR_STORE_DIR),
        embedding_function=embedding_model,
    )

    return vector_store


def similarity_search(query: str, k: int = 4) -> List[Document]:
    vector_store = load_vector_store()
    return vector_store.similarity_search(query=query, k=k)


def main() -> None:
    print("=" * 80)
    print("PHASE 3: LOAD VECTOR STORE + TEST SEARCH")
    print("=" * 80)

    query = input("Enter a test query: ").strip()
    if not query:
        print("Query cannot be empty.")
        return

    k_raw = input("Enter top-k value (default 4): ").strip()
    k = int(k_raw) if k_raw else 4

    results = similarity_search(query=query, k=k)

    print(f"\nRetrieved {len(results)} result(s)\n")

    for i, doc in enumerate(results, start=1):
        print("-" * 80)
        print(f"Result #{i}")
        print("Metadata:")
        print(doc.metadata)
        print("\nChunk preview:")
        print(doc.page_content[:700])
        print("-" * 80)


if __name__ == "__main__":
    main()