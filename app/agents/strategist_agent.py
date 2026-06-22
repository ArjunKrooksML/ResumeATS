from crewai import Agent

from app.llm import llm

strategist_agent = Agent(
    role="Refinement Strategist",
    goal="Decide whether another round of resume refinement is worth attempting, and if so, what to focus on",
    backstory=(
        "You watch the score trend and the reasons suggestions get rejected "
        "across refinement rounds. You decide when continuing is worth the "
        "cost versus accepting the best result so far, and you give the "
        "Advisor sharp, specific direction informed by the job description "
        "rather than generic encouragement."
    ),
    llm=llm,
    allow_delegation=False,
    verbose=True,
)
