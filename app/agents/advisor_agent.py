from crewai import Agent

from app.llm import llm
from app.tools.current_date import get_current_date

advisor_agent = Agent(
    role="Resume Advisor",
    goal="Rewrite resume bullet points to close identified skill gaps without inventing experience",
    backstory=(
        "You rewrite resume bullets so they better reflect a candidate's real "
        "experience in language that matches the job description. You never "
        "add claims the candidate did not make."
    ),
    tools=[get_current_date],
    llm=llm,
    allow_delegation=False,
    verbose=True,
)
