from crewai import Task

from app.agents.advisor_agent import advisor_agent
from app.models.schemas import AdvisorOutput
from app.tasks.gap_task import gap_task
from app.tasks.parse_task import parse_task

advise_task = Task(
    description=(
        "Using the priority gaps identified, propose rewrites for resume bullet "
        "points that better reflect the candidate's existing experience in terms "
        "matching the job description. Do not invent skills, titles, or "
        "achievements not already present in the resume. {critic_feedback}"
    ),
    expected_output="An AdvisorOutput with a list of Suggestion objects.",
    agent=advisor_agent,
    context=[parse_task, gap_task],
    output_pydantic=AdvisorOutput,
)
