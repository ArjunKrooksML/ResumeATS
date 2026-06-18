from crewai import Task

from app.agents.parser_agent import parser_agent
from app.models.schemas import ParsedResume

parse_task = Task(
    description=(
        "Read the resume file at {file_path} using your tool. "
        "From the transcribed text, extract all skills mentioned, the highest "
        "education level, and total years of professional experience. "
        "From any STRUCTURE notes in the tool output, build a list of structural "
        "risks, judging severity (low, medium, high) by how much resume content "
        "each risk could hide from a real ATS parser."
    ),
    expected_output=(
        "A ParsedResume with raw_text, skills, education, years_experience, "
        "and structural_risks all filled in."
    ),
    agent=parser_agent,
    output_pydantic=ParsedResume,
)
