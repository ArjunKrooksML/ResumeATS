from crewai import Task

from app.agents.gap_analyst_agent import gap_analyst_agent
from app.crew.guardrails import require_schema
from app.models.schemas import GapAnalysis
from app.tasks.parse_task import parse_task
from app.tasks.score_task import score_task

gap_task = Task(
    description=(
        "Using the parsed resume and the match score reasoning, list every "
        "keyword and skill from this job description that the resume is "
        "missing: {job_description}. Rank the missing items by how much they "
        "likely impact the match score, most impactful first.\n\n"
        "Your output is a new GapAnalysis object, not the ScoreResult you were "
        "given as context — it must use exactly these field names: "
        "missing_keywords, missing_skills, priority_gaps. Do not reuse "
        "match_score, matched_keywords, or reasoning."
    ),
    expected_output="A GapAnalysis with missing_keywords, missing_skills, and priority_gaps filled in.",
    agent=gap_analyst_agent,
    context=[parse_task, score_task],
    output_pydantic=GapAnalysis,
    guardrail=require_schema(GapAnalysis),
)
