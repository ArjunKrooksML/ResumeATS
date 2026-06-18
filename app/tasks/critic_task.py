from crewai import Task

from app.agents.critic_agent import critic_agent
from app.models.schemas import CriticOutput
from app.tasks.advise_task import advise_task
from app.tasks.parse_task import parse_task

critic_task = Task(
    description=(
        "Compare each suggested rewrite against the original resume text. For "
        "each suggestion, decide if it is grounded in the original text or if "
        "it invents unverifiable claims. Approve only the suggestions that are "
        "grounded."
    ),
    expected_output="A CriticOutput with verdicts for every suggestion and a list of approved_suggestions.",
    agent=critic_agent,
    context=[parse_task, advise_task],
    output_pydantic=CriticOutput,
)
