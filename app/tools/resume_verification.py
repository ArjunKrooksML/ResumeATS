from contextvars import ContextVar

from crewai.tools import tool

_resume_text: ContextVar[str] = ContextVar("resume_text_for_verification", default="")


def set_resume_text(text: str) -> None:
    _resume_text.set(text)


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


@tool("Verify Text In Resume")
def verify_in_resume(snippet: str) -> str:
    """Check whether a snippet of text actually appears in the candidate's resume."""
    text = _resume_text.get()
    if snippet in text:
        return "Found verbatim in the resume."
    if normalize_whitespace(snippet) in normalize_whitespace(text):
        return "Found in the resume, though whitespace differs slightly."
    return "NOT found in the resume text, verbatim or otherwise."
