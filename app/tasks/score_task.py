from crewai import Task

from app.agents.scorer_agent import scorer_agent
from app.crew.guardrails import require_schema
from app.models.schemas import ScoreResult
from app.tasks.parse_task import parse_task

SCORING_INSTRUCTIONS = (
    "Identify which keywords and skills from the job description appear in the "
    "resume, and produce a match score from 0 to 100 reflecting overall fit."
)

score_task = Task(
    description=(
        "Compare this resume text against the job description: {job_description}. "
        "Resume text: {resume_text}. "
        + SCORING_INSTRUCTIONS
    ),
    expected_output="A ScoreResult with match_score, matched_keywords, and reasoning filled in.",
    agent=scorer_agent,
    context=[parse_task],
    output_pydantic=ScoreResult,
    guardrail=require_schema(ScoreResult),
)
