from crewai import Task

from app.agents.scorer_agent import scorer_agent
from app.crew.guardrails import require_schema
from app.models.schemas import ScoreResult
from app.tasks.score_task import COVERAGE_INSTRUCTIONS, SCORING_INSTRUCTIONS


def make_rescore_task() -> Task:
    return Task(
        description=(
            "Score this revised resume text against the job description: "
            "{job_description}. Revised resume text: {resume_text}. "
            + SCORING_INSTRUCTIONS
            + COVERAGE_INSTRUCTIONS
        ),
        expected_output=(
            "A ScoreResult with match_score, matched_keywords, reasoning, and coverage filled in. "
            "coverage must reflect the revised resume — update statuses where the revisions now address requirements."
        ),
        agent=scorer_agent,
        output_pydantic=ScoreResult,
        guardrail=require_schema(ScoreResult),
    )
