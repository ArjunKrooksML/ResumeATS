from pathlib import Path

from crewai.tools import tool

from app.tools.docx_reader import read_docx
from app.tools.pdf_reader import read_pdf

READERS = {".pdf": read_pdf, ".docx": read_docx}


@tool("Read Resume File")
def read_resume(file_path: str) -> str:
    """Reads a PDF or DOCX resume and returns transcribed text plus layout risk notes."""
    extension = Path(file_path).suffix.lower()
    reader = READERS.get(extension)
    if reader is None:
        raise ValueError(f"Unsupported resume format: {extension}")
    return reader(file_path)
