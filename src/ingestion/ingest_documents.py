import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import io
from typing import List

import fitz
import pytesseract
from PIL import Image
from langchain_core.documents import Document

from src.ingestion.chunk_documents import chunk_documents
from src.utils.helpers import (
    clean_text,
    get_knowledge_base_path,
    list_pdf_files,
    validate_unit_folders,
)

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


OCR_TEXT_THRESHOLD = 20
OCR_RENDER_DPI = 300


def scan_knowledge_base() -> List[Path]:
    knowledge_base_path = get_knowledge_base_path()
    unit_paths = validate_unit_folders(knowledge_base_path)
    pdf_files = list_pdf_files(unit_paths)
    return pdf_files


def detect_unit_from_path(pdf_path: Path, knowledge_base_path: Path) -> str:
    relative_parts = pdf_path.relative_to(knowledge_base_path).parts
    return relative_parts[0]


def extract_text_from_page(page: fitz.Page) -> str:
    text = page.get_text("text")
    return text.strip()


def render_page_to_image(page: fitz.Page, dpi: int = OCR_RENDER_DPI) -> Image.Image:
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix)
    image_bytes = pix.tobytes("png")
    return Image.open(io.BytesIO(image_bytes))


def ocr_page_image(image: Image.Image) -> str:
    text = pytesseract.image_to_string(image, lang="eng")
    return text.strip()


def should_use_ocr(text: str) -> bool:
    return len(text.strip()) < OCR_TEXT_THRESHOLD


def load_single_pdf(pdf_path: Path, knowledge_base_path: Path) -> List[Document]:
    pdf_document = fitz.open(pdf_path)
    page_documents = []
    unit_name = detect_unit_from_path(pdf_path, knowledge_base_path)

    for page_index in range(len(pdf_document)):
        page = pdf_document[page_index]
        extracted_text = extract_text_from_page(page)
        extraction_method = "native"

        if should_use_ocr(extracted_text):
            try:
                page_image = render_page_to_image(page)
                ocr_text = ocr_page_image(page_image)

                if ocr_text:
                    extracted_text = ocr_text
                    extraction_method = "ocr"
                    print(f"[OCR] {pdf_path.name} -> page {page_index}")
            except Exception as error:
                print(f"[OCR-ERROR] {pdf_path.name} -> page {page_index}: {error}")

        cleaned_text = clean_text(extracted_text)

        if not cleaned_text:
            continue

        metadata = {
            "unit": unit_name,
            "source": pdf_path.name,
            "file_name": pdf_path.name,
            "file_path": str(pdf_path),
            "page": page_index,
            "extraction_method": extraction_method,
        }

        page_documents.append(
            Document(
                page_content=cleaned_text,
                metadata=metadata
            )
        )

    pdf_document.close()
    return page_documents


def load_all_pdfs() -> List[Document]:
    knowledge_base_path = get_knowledge_base_path()
    pdf_files = scan_knowledge_base()

    if not pdf_files:
        raise FileNotFoundError("No PDF files found inside knowledge_base folders.")

    all_page_documents = []

    for pdf_file in pdf_files:
        try:
            pdf_page_docs = load_single_pdf(pdf_file, knowledge_base_path)
            all_page_documents.extend(pdf_page_docs)
            print(f"[LOADED] {pdf_file.name} -> {len(pdf_page_docs)} non-empty pages")
        except Exception as error:
            print(f"[ERROR] Failed to load {pdf_file.name}: {error}")

    return all_page_documents


def print_page_summary(documents: List[Document]) -> None:
    print("\n" + "=" * 60)
    print("PDF PAGE LOADING SUMMARY")
    print("=" * 60)
    print(f"Total page-level documents created: {len(documents)}")

    unit_counts = {}
    extraction_counts = {}

    for doc in documents:
        unit = doc.metadata.get("unit", "unknown")
        method = doc.metadata.get("extraction_method", "unknown")

        unit_counts[unit] = unit_counts.get(unit, 0) + 1
        extraction_counts[method] = extraction_counts.get(method, 0) + 1

    print("\nPages by unit:")
    for unit, count in sorted(unit_counts.items()):
        print(f"- {unit}: {count}")

    print("\nPages by extraction method:")
    for method, count in sorted(extraction_counts.items()):
        print(f"- {method}: {count}")

    print("\nSample page documents:")
    for index, doc in enumerate(documents[:3], start=1):
        preview = doc.page_content[:250].replace("\n", " ")
        print(f"\nSample {index}")
        print(f"Metadata: {doc.metadata}")
        print(f"Preview: {preview}...")


if __name__ == "__main__":

    documents = load_all_pdfs()
    print_page_summary(documents)

    chunked_docs = chunk_documents(documents)

    print("\n" + "=" * 60)
    print("CHUNKING SUMMARY")
    print("=" * 60)

    print("Original page documents:", len(documents))
    print("Total chunked documents:", len(chunked_docs))

    print("\nSample chunk:")

    sample = chunked_docs[0]

    print("Metadata:", sample.metadata)
    print("Chunk length:", len(sample.page_content))
    print("Preview:", sample.page_content[:300])