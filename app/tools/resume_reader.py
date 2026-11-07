from pathlib import Path

from app.tools.docx_reader import read_docx
from app.tools.pdf_reader import read_pdf

READERS = {".pdf": read_pdf, ".docx": read_docx}


def extract_resume_text(file_path: str, high_quality: bool = False) -> str:
    extension = Path(file_path).suffix.lower()
    reader = READERS.get(extension)
    if reader is None:
        raise ValueError(f"Unsupported resume format: {extension}")
    if extension == ".pdf" and high_quality:
        return read_pdf(file_path, dpi=300)
    return reader(file_path)
