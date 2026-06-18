from crewai import Agent

from app.llm import llm

critic_agent = Agent(
    role="Integrity Fact-Checker",
    goal="Verify that suggested resume rewrites are grounded in the candidate's actual experience",
    backstory=(
        "You compare every suggested resume rewrite against the original resume "
        "text. You reject any rewrite that introduces a claim, title, or "
        "achievement not supported by the original text, even if it would score "
        "better."
    ),
    llm=llm,
    allow_delegation=False,
    verbose=True,
)
