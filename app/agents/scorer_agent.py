from crewai import Agent

from app.llm import llm

scorer_agent = Agent(
    role="ATS Match Scorer",
    goal="Score how well a resume matches a job description the way real ATS software does",
    backstory=(
        "You simulate the keyword and skill matching algorithms used by ATS "
        "platforms like Workday and Greenhouse. You are strict and consistent, "
        "never generous with the score."
    ),
    llm=llm,
    allow_delegation=False,
    verbose=True,
)
