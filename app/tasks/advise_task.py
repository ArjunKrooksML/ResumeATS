from crewai import Task

from app.agents.advisor_agent import advisor_agent
from app.crew.guardrails import require_schema
from app.models.schemas import AdvisorOutput
from app.tasks.gap_task import gap_task
from app.tasks.parse_task import parse_task

advise_task = Task(
    description=(
        "Using the priority gaps identified, propose rewrites for resume bullet "
        "points that better reflect the candidate's existing experience in terms "
        "matching the job description. Do not invent skills, titles, or "
        "achievements not already present in the resume.\n\n"
        "Resume text:\n{resume_text}\n\n"
        "For each suggestion, original_bullet must be copied character-for-character "
        "from the resume text above, with no paraphrasing, retyping, or whitespace "
        "changes — it will be matched against the resume text verbatim.\n\n"
        "Your output is a new AdvisorOutput object, not the GapAnalysis you were "
        "given as context — it must be a JSON object with a single field "
        "'suggestions', a list of objects each with original_bullet, "
        "revised_bullet, and addresses_gap. Do not reuse missing_keywords, "
        "missing_skills, or priority_gaps as field names.\n\n"
        "{critic_feedback}"
    ),
    expected_output="An AdvisorOutput with a list of Suggestion objects.",
    agent=advisor_agent,
    context=[parse_task, gap_task],
    output_pydantic=AdvisorOutput,
    guardrail=require_schema(AdvisorOutput),
)
