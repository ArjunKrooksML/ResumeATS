from crewai import Task

from app.agents.gap_analyst_agent import gap_analyst_agent
from app.crew.guardrails import require_schema
from app.models.schemas import GapAnalysis


def make_gap_task() -> Task:
    return Task(
        description=(
            "Job description: {job_description}\n\nResume text: {resume_text}\n\n"
            "Identify the genuine technical gaps between this resume and the job description. "
            "A genuine gap is a specific skill, technology, methodology, or domain "
            "the job requires that the resume does not demonstrate — even conceptually, "
            "even under different terminology.\n\n"
            "Rules:\n"
            "- Only flag concrete, teachable technical skills, tools, or methodologies — "
            "things a candidate could actually put on a resume.\n"
            "- Product team names, business domains, and org areas are NEVER gaps. "
            "They describe where you work, not what you know. Examples of things to ignore: "
            "'Recommendations', 'Trust & Safety', 'Profile', 'Chat', 'Growth', 'Revenue', "
            "'Search', 'Discovery', 'Monetization', 'Platform'.\n"
            "- Event names and programs are not gaps (e.g. 'Hackathon', 'Bootcamp', 'Internship').\n"
            "- Marketing copy and generic descriptors are not gaps "
            "(e.g. 'real-world experience', 'cutting-edge technology', 'aspiring engineer').\n"
            "- If the resume demonstrates a concept under different terminology "
            "(e.g. 'production deployment' instead of 'deploying ML models in production'), "
            "do not flag it as missing.\n"
            "- Be conservative. Return 3 to 7 high-impact gaps only. More noise makes "
            "rewrites worse, not better.\n\n"
            "Your output is a GapAnalysis with missing_keywords, missing_skills, "
            "and priority_gaps filled in."
        ),
        expected_output="A GapAnalysis with 3-7 genuine technical gaps in missing_keywords, missing_skills, and priority_gaps.",
        agent=gap_analyst_agent,
        output_pydantic=GapAnalysis,
        guardrail=require_schema(GapAnalysis),
    )
