from crewai import Task

from app.agents.jd_analyst_agent import jd_analyst_agent
from app.crew.guardrails import require_schema
from app.models.schemas import JDAnalysis

jd_task = Task(
    description=(
        "Job description:\n{job_description}\n\n"
        "Assess this job description for quality and realism before the resume is analyzed against it.\n\n"
        "Flag any of the following as red flags:\n"
        "- Unrealistic experience demands: entry/mid-level title with senior requirements (e.g. '5 YoE for junior role')\n"
        "- Keyword stuffing: 15+ required skills that no single candidate could realistically have\n"
        "- Vagueness: descriptions so generic that any candidate could qualify ('strong communication skills', 'ability to learn')\n"
        "- Ghost job signals: overly generic copy, no specific team/product context, identical re-post language\n\n"
        "credibility_score: 0-100 (0 = useless or fake, 100 = specific, realistic, clearly genuine)\n"
        "is_analyzable: False only if the JD is so vague that gap analysis will produce meaningless results\n"
        "summary: one sentence on the overall quality of this JD\n\n"
        "Return a JDAnalysis."
    ),
    expected_output="A JDAnalysis with credibility_score, red_flags, is_analyzable, and summary.",
    agent=jd_analyst_agent,
    output_pydantic=JDAnalysis,
    guardrail=require_schema(JDAnalysis),
)
