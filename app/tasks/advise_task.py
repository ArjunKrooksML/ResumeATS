from crewai import Task

from app.agents.advisor_agent import advisor_agent
from app.crew.guardrails import require_schema
from app.models.schemas import AdvisorOutput

advise_task = Task(
    description=(
        "Job description context: {job_description}\n\n"
        "Priority gaps to close, ranked by impact: {priority_gaps}\n\n"
        "Resume text:\n{resume_text}\n\n"
        "Rewrite specific resume bullets so they address the gaps above using "
        "the candidate's real, existing experience — not invented claims.\n\n"
        "Standards for every revised bullet:\n"
        "- Open with a strong, specific action verb (Engineered, Deployed, Architected, "
        "Trained, Optimized, Scaled, Shipped, Benchmarked, Automated, Reduced, Built). "
        "Avoid weak openers: Worked on, Helped, Assisted, Was responsible for.\n"
        "- Gap keywords must appear as natural technical descriptors inside the sentence — "
        "as the name of the tool used, the method applied, or the system built. "
        "If the keyword can only fit by appending a clause at the end that explains "
        "why the bullet is relevant (e.g. '...demonstrating X', '...illustrating Y', "
        "'...showcasing Z', '...reflecting familiarity with W'), that is a failure. "
        "The relevance must be self-evident from the technical claim. If you cannot "
        "embed the keyword naturally into the action itself, skip that keyword entirely.\n"
        "- Every bullet ends with a concrete metric or specific outcome — a number, "
        "latency, throughput, percentage, or named result. Not a justification clause.\n"
        "- One bullet = one technical claim + one outcome. No compound sentences joined "
        "by 'and' that pad length without adding substance.\n"
        "- Never add qualifiers: 'foundational knowledge of', 'familiarity with', "
        "'exposure to', 'experience in'. The bullet either describes something the "
        "candidate built/shipped/measured or it doesn't.\n\n"
        "For each suggestion, original_bullet must be copied character-for-character "
        "from the resume text above — it will be matched verbatim.\n\n"
        "Your output is an AdvisorOutput — a JSON object with a single field "
        "'suggestions', each entry having original_bullet, revised_bullet, and addresses_gap.\n\n"
        "{critic_feedback}"
    ),
    expected_output="An AdvisorOutput with tight, ATS-ready revised bullets that address the gaps naturally.",
    agent=advisor_agent,
    output_pydantic=AdvisorOutput,
    guardrail=require_schema(AdvisorOutput),
)
