import json
import shutil
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document
from langchain_chroma import Chroma
from tqdm import tqdm

from src.embeddings.embedding_model import get_embedding_model


CHUNK_FILE_CANDIDATES = [
    PROJECT_ROOT / "data" / "processed" / "chunks" / "chunked_documents.json",
    PROJECT_ROOT / "data" / "processed" / "chunked_documents.json",
    PROJECT_ROOT / "artifacts" / "chunked_documents.json",
]

VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store"
COLLECTION_NAME = "marketing_rag_chunks"


def resolve_chunk_file() -> Path:
    for path in CHUNK_FILE_CANDIDATES:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find chunked documents file.\n"
        "Checked these paths:\n"
        + "\n".join(str(p) for p in CHUNK_FILE_CANDIDATES)
        + "\n\nPlease place your chunk output in one of these paths "
        "or update CHUNK_FILE_CANDIDATES in build_vector_store.py"
    )


def load_chunked_documents(chunk_file: Path) -> List[Document]:
    with open(chunk_file, "r", encoding="utf-8") as f:
        raw_items = json.load(f)

    documents = []
    for item in raw_items:
        page_content = item.get("page_content", "").strip()
        metadata = item.get("metadata", {})

        if not page_content:
            continue

        documents.append(
            Document(
                page_content=page_content,
                metadata=metadata
            )
        )

    return documents


def build_vector_store(documents: List[Document], reset: bool = True) -> Chroma:
    if reset and VECTOR_STORE_DIR.exists():
        shutil.rmtree(VECTOR_STORE_DIR)

    embedding_model = get_embedding_model()

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=str(VECTOR_STORE_DIR),
        collection_name=COLLECTION_NAME,
    )

    return vector_store


def main() -> None:
    print("=" * 80)
    print("PHASE 3: BUILD VECTOR STORE")
    print("=" * 80)

    chunk_file = resolve_chunk_file()
    print(f"Chunk file found: {chunk_file}")

    documents = load_chunked_documents(chunk_file)
    print(f"Loaded chunk documents: {len(documents)}")

    if not documents:
        raise ValueError("No valid chunk documents found.")

    print("\nSample metadata from first chunk:")
    print(documents[0].metadata)

    print("\nBuilding Chroma vector store...")
    build_vector_store(documents)

    print(f"\nVector store saved to: {VECTOR_STORE_DIR}")
    print(f"Collection name: {COLLECTION_NAME}")
    print("Phase 3 vector build complete.")


if __name__ == "__main__":
    main()