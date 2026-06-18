from crewai import Task

from app.agents.critic_agent import critic_agent
from app.crew.guardrails import require_schema
from app.models.schemas import CriticOutput
from app.tasks.advise_task import advise_task
from app.tasks.parse_task import parse_task

critic_task = Task(
    description=(
        "Original resume text:\n{resume_text}\n\n"
        "Compare each suggested rewrite against the original resume text above. For "
        "each suggestion, decide if it is grounded in the original text or if "
        "it invents unverifiable claims. Approve only the suggestions that are "
        "grounded.\n\n"
        "Your output is a new CriticOutput object, not the AdvisorOutput you were "
        "given as context — it must be a JSON object with two fields: "
        "'verdicts' (a list of objects with suggestion_index, is_grounded, "
        "explanation) and 'approved_suggestions' (a list of the Suggestion "
        "objects that passed). Do not reuse 'suggestions' as a top-level field name."
    ),
    expected_output="A CriticOutput with verdicts for every suggestion and a list of approved_suggestions.",
    agent=critic_agent,
    context=[parse_task, advise_task],
    output_pydantic=CriticOutput,
    guardrail=require_schema(CriticOutput),
)
