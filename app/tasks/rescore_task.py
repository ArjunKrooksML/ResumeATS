from crewai import Task

from app.agents.scorer_agent import scorer_agent
from app.models.schemas import ScoreResult
from app.tasks.score_task import SCORING_INSTRUCTIONS

rescore_task = Task(
    description=(
        "Score this revised resume text against the job description: "
        "{job_description}. Revised resume text: {resume_text}. "
        + SCORING_INSTRUCTIONS
    ),
    expected_output="A ScoreResult with match_score, matched_keywords, and reasoning filled in.",
    agent=scorer_agent,
    output_pydantic=ScoreResult,
)
