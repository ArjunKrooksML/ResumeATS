from pathlib import Path

from app.tools.docx_reader import read_docx
from app.tools.pdf_reader import read_pdf

READERS = {".pdf": read_pdf, ".docx": read_docx}


def extract_resume_text(file_path: str) -> str:
    extension = Path(file_path).suffix.lower()
    reader = READERS.get(extension)
    if reader is None:
        raise ValueError(f"Unsupported resume format: {extension}")
    return reader(file_path)
