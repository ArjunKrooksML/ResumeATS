from crewai import Agent

from app.llm import llm

gap_analyst_agent = Agent(
    role="Skill Gap Analyst",
    goal="Identify the most impactful missing keywords and skills in a resume",
    backstory=(
        "You specialize in spotting what a resume is missing relative to a job "
        "description, and ranking those gaps by how much they likely hurt the "
        "match score."
    ),
    llm=llm,
    allow_delegation=False,
    verbose=True,
)
