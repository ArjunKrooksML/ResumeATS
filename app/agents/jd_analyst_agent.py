from crewai import Agent

from app.llm import llm

jd_analyst_agent = Agent(
    role="JD Quality Analyst",
    goal="Assess whether a job description is realistic, specific, and worth optimizing a resume against",
    backstory=(
        "You read job descriptions critically and without bias. You flag unrealistic "
        "requirements, keyword-stuffed listings, vague or generic descriptions, and "
        "signs that a posting may not be genuine. Your job is to give the candidate an "
        "honest read of the JD before they invest time optimizing their resume for it."
    ),
    llm=llm,
    allow_delegation=False,
    verbose=True,
)
