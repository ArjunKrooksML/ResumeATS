from crewai import Task

from app.agents.gap_analyst_agent import gap_analyst_agent
from app.crew.guardrails import require_schema
from app.models.schemas import GapAnalysis

regap_task = Task(
    description=(
        "Re-examine the missing keywords and skills for this job description: "
        "{job_description}\n\nResume text:\n{resume_text}\n\n"
        "The Strategist reviewed the previous gap analysis and gave this "
        "feedback: {strategist_feedback}\n\n"
        "Produce a revised, sharper list of missing keywords, missing skills, "
        "and priority gaps that directly addresses that feedback.\n\n"
        "Your output is a GapAnalysis with missing_keywords, missing_skills, "
        "and priority_gaps filled in."
    ),
    expected_output="A GapAnalysis with missing_keywords, missing_skills, and priority_gaps filled in.",
    agent=gap_analyst_agent,
    output_pydantic=GapAnalysis,
    guardrail=require_schema(GapAnalysis),
)
