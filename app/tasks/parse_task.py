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
        "each risk could hide from a real ATS parser."
    ),
    expected_output=(
        "A ParsedResume with skills, education, years_experience, and "
        "structural_risks filled in. Leave raw_text as an empty string — "
        "it is filled in separately."
    ),
    agent=parser_agent,
    output_pydantic=ParsedResume,
    guardrail=require_schema(ParsedResume),
)
