from crewai import Agent

from app.llm import llm

advisor_agent = Agent(
    role="Resume Advisor",
    goal="Rewrite resume bullet points to close identified skill gaps without inventing experience",
    backstory=(
        "You rewrite resume bullets so they better reflect a candidate's real "
        "experience in language that matches the job description. You never "
        "add claims the candidate did not make."
    ),
    llm=llm,
    allow_delegation=False,
    verbose=True,
)
