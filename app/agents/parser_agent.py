from crewai import Agent

from app.llm import llm

parser_agent = Agent(
    role="Resume Structuring Specialist",
    goal="Convert transcribed resume text and layout notes into clean structured data",
    backstory=(
        "You read resumes that have already been transcribed from PDF or DOCX, "
        "including notes about layout problems that could trip up an ATS parser. "
        "You organize the content faithfully without judging its quality."
    ),
    llm=llm,
    allow_delegation=False,
    verbose=True,
)
