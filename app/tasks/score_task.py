from crewai import Task

from app.agents.scorer_agent import scorer_agent
from app.crew.guardrails import require_schema
from app.models.schemas import ScoreResult

SCORING_INSTRUCTIONS = (
    "Identify which keywords and skills from the job description appear in the "
    "resume, and produce a match score from 0 to 100 reflecting overall fit."
)

COVERAGE_INSTRUCTIONS = (
    "\n\nAlso produce a coverage list for the key requirements in the job description. "
    "For each specific skill, technology, tool, or methodology the JD requires, include it with:\n"
    "- status: 'covered' (clearly demonstrated), 'partial' (implied or tangential), or 'missing' (absent)\n"
    "- covered_by: the exact resume sentence or phrase that demonstrates it — empty string if missing\n"
    "Focus on concrete requirements only. Skip soft skills and generic descriptors."
)


def make_score_task() -> Task:
    return Task(
        description=(
            "Compare this resume text against the job description: {job_description}. "
            "Resume text: {resume_text}. "
            + SCORING_INSTRUCTIONS
            + COVERAGE_INSTRUCTIONS
        ),
        expected_output=(
            "A ScoreResult with match_score, matched_keywords, reasoning, and coverage filled in. "
            "coverage must list every key JD requirement with its status and the resume evidence."
        ),
        agent=scorer_agent,
        output_pydantic=ScoreResult,
        guardrail=require_schema(ScoreResult),
    )
