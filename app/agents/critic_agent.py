from crewai import Agent

from app.llm import llm
from app.tools.resume_verification import verify_in_resume

critic_agent = Agent(
    role="Integrity Fact-Checker",
    goal="Verify that suggested resume rewrites are grounded in the candidate's actual experience",
    backstory=(
        "You compare every suggested resume rewrite against the original resume "
        "text. You reject any rewrite that introduces a claim, title, or "
        "achievement not supported by the original text, even if it would score "
        "better."
    ),
    tools=[verify_in_resume],
    llm=llm,
    allow_delegation=False,
    verbose=True,
)
