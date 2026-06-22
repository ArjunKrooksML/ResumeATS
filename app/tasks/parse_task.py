from crewai import Task

from app.agents.parser_agent import parser_agent
from app.crew.guardrails import require_schema
from app.models.schemas import ParsedResume

parse_task = Task(
    description=(
        "This resume has already been transcribed:\n\n{resume_text}\n\n"
        "From the transcribed text, extract all skills mentioned, the highest "
        "education level, and total years of professional experience. "
        "From any STRUCTURE notes in the text, build a list of structural "
        "risks, judging severity (low, medium, high) by how much resume content "
        "each risk could hide from a real ATS parser.\n\n"
        "Also assess the transcription itself: set is_reliable to false if the "
        "text above looks empty, garbled, cut off mid-sentence, or otherwise "
        "untrustworthy as a real resume, and explain why in reliability_reason. "
        "Otherwise set is_reliable to true."
    ),
    expected_output=(
        "A ParsedResume with skills, education, years_experience, "
        "structural_risks, is_reliable, and reliability_reason filled in. Leave "
        "raw_text as an empty string — it is filled in separately."
    ),
    agent=parser_agent,
    output_pydantic=ParsedResume,
    guardrail=require_schema(ParsedResume),
)
