from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.answer_generator import AnswerGenerator, AnswerGeneratorError


VALID_UNIT_FILTERS = {"bestec", "unit1", "unit3", "unit5"}
VALID_MODES = {"similarity", "mmr"}


def print_divider() -> None:
    print("\n" + "=" * 100)


def normalize_unit_filter(raw_value: str):
    value = raw_value.strip().lower()
    if not value:
        return None
    if value not in VALID_UNIT_FILTERS:
        raise ValueError(
            f"Invalid unit filter '{raw_value}'. Valid options: {', '.join(sorted(VALID_UNIT_FILTERS))}"
        )
    return value


def normalize_mode(raw_value: str):
    value = raw_value.strip().lower()
    if not value:
        return "mmr"
    if value not in VALID_MODES:
        raise ValueError(
            f"Invalid retrieval mode '{raw_value}'. Valid options: {', '.join(sorted(VALID_MODES))}"
        )
    return value


def main() -> None:
    print("\nInitializing end-to-end RAG chat...")
    generator = AnswerGenerator()
    print("System ready.")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            query = input("Enter your question: ").strip()
            if query.lower() == "exit":
                print("Exiting chat.")
                break

            if not query:
                print("Please enter a question.")
                continue

            mode = normalize_mode(
                input("Choose retrieval mode ('similarity' or 'mmr', default: mmr): ")
            )

            print("\nAvailable unit filters: bestec, unit1, unit3, unit5")
            raw_unit = input("Enter unit filter (or press Enter to search all units): ")
            unit_filter = normalize_unit_filter(raw_unit)

            print("\nGenerating answer...\n")
            result = generator.generate_answer(
                query=query,
                retrieval_mode=mode,
                unit_filter=unit_filter,
                top_k=4,
            )

            print_divider()
            print("FINAL ANSWER")
            print_divider()
            print(result.answer)

            print_divider()
            print("DEBUG SUMMARY")
            print_divider()
            print(f"Model used: {result.model}")
            print(f"Retrieval mode: {result.metadata.get('retrieval_mode')}")
            print(f"Unit filter: {result.metadata.get('unit_filter') or 'none'}")
            print(f"Retrieved chunks: {len(result.retrieved_chunks)}")

            if result.retrieved_chunks:
                print_divider()
                print("RETRIEVED SOURCES")
                print_divider()
                for idx, chunk in enumerate(result.retrieved_chunks, start=1):
                    metadata = chunk.get("metadata", {})
                    unit = metadata.get("unit", "unknown")
                    file_name = metadata.get("file_name", "unknown")
                    page = metadata.get("page", "unknown")
                    score = chunk.get("score")
                    search_type = chunk.get("search_type", "unknown")

                    if score is not None:
                        print(
                            f"{idx}. unit={unit} | file={file_name} | page={page} | "
                            f"score={score:.4f} | mode={search_type}"
                        )
                    else:
                        print(
                            f"{idx}. unit={unit} | file={file_name} | page={page} | mode={search_type}"
                        )

            print_divider()
            print("BUILT CONTEXT SENT TO GEMINI")
            print_divider()
            print(result.context)
            print()

        except ValueError as exc:
            print(f"\nInput error: {exc}\n")
        except AnswerGeneratorError as exc:
            print(f"\nSystem error: {exc}\n")
        except KeyboardInterrupt:
            print("\nInterrupted. Exiting chat.")
            break
        except Exception as exc:
            print(f"\nUnexpected error: {exc}\n")


if __name__ == "__main__":
    main()