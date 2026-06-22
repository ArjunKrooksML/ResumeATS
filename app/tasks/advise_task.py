from crewai import Task

from app.agents.advisor_agent import advisor_agent
from app.crew.guardrails import require_schema
from app.models.schemas import AdvisorOutput

advise_task = Task(
    description=(
        "Priority gaps to address, ranked most impactful first: {priority_gaps}\n\n"
        "Propose rewrites for resume bullet points that better reflect the "
        "candidate's existing experience in terms matching the job description. "
        "Do not invent skills, titles, or achievements not already present in "
        "the resume.\n\n"
        "Resume text:\n{resume_text}\n\n"
        "For each suggestion, original_bullet must be copied character-for-character "
        "from the resume text above, with no paraphrasing, retyping, or whitespace "
        "changes — it will be matched against the resume text verbatim.\n\n"
        "Your output is an AdvisorOutput — a JSON object with a single field "
        "'suggestions', a list of objects each with original_bullet, "
        "revised_bullet, and addresses_gap.\n\n"
        "{critic_feedback}"
    ),
    expected_output="An AdvisorOutput with a list of Suggestion objects.",
    agent=advisor_agent,
    output_pydantic=AdvisorOutput,
    guardrail=require_schema(AdvisorOutput),
)
