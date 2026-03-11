from pathlib import Path
import re
from typing import List


SUPPORTED_UNITS = ["bestec", "unit1", "unit3", "unit5"]


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_knowledge_base_path() -> Path:
    return get_project_root() / "knowledge_base"


def validate_unit_folders(base_path: Path) -> List[Path]:
    unit_paths = []

    for unit in SUPPORTED_UNITS:
        unit_path = base_path / unit

        if not unit_path.exists():
            raise FileNotFoundError(f"Missing unit folder: {unit_path}")

        if not unit_path.is_dir():
            raise NotADirectoryError(f"Expected a folder but found something else: {unit_path}")

        unit_paths.append(unit_path)

    return unit_paths


def list_pdf_files(unit_paths: List[Path]) -> List[Path]:
    unique_pdf_files = {}

    for unit_path in unit_paths:
        for file_path in unit_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() == ".pdf":
                normalized_path = str(file_path.resolve()).lower()
                unique_pdf_files[normalized_path] = file_path.resolve()

    return sorted(unique_pdf_files.values(), key=lambda path: str(path).lower())


def clean_text(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\x00", " ")
    text = text.replace("\t", " ")
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"[ ]+", " ", text)
    text = re.sub(r"\n[ ]+", "\n", text)
    text = re.sub(r"[ ]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()