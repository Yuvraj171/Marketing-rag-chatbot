import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.embeddings.load_vector_store import similarity_search


TEST_QUERIES = [
    "motor specification and technical details",
    "machine dimensions and product features",
    "industrial component application and usage",
]


def main() -> None:
    print("=" * 80)
    print("PHASE 3 TEST: VECTOR SEARCH VALIDATION")
    print("=" * 80)

    for query in TEST_QUERIES:
        print(f"\nQuery: {query}")
        results = similarity_search(query=query, k=3)

        if not results:
            raise AssertionError(f"No results returned for query: {query}")

        for idx, doc in enumerate(results, start=1):
            print(f"\nResult {idx}")
            print("Metadata:", doc.metadata)

            required_keys = ["unit", "file_name", "page"]
            for key in required_keys:
                if key not in doc.metadata:
                    raise AssertionError(
                        f"Missing required metadata key '{key}' in result metadata."
                    )

            if not doc.page_content.strip():
                raise AssertionError("Retrieved document has empty page_content.")

            print("Preview:", doc.page_content[:300])

    print("\nAll vector search tests passed successfully.")


if __name__ == "__main__":
    main()