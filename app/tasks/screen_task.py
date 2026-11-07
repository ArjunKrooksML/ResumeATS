from crewai import Task

from app.agents.advisor_agent import advisor_agent
from app.crew.guardrails import require_schema
from app.models.schemas import GapScreenResult

screen_task = Task(
    description=(
        "Resume text:\n{resume_text}\n\n"
        "Gaps to assess: {priority_gaps}\n\n"
        "For each gap, decide: does this resume contain existing experience, a project, "
        "or an achievement that could be reframed to address it — even under different terminology?\n\n"
        "Rules:\n"
        "- A gap is actionable if something real in the resume can be rewritten to cover it.\n"
        "- A gap is NOT actionable if the resume has zero relevant experience — "
        "no reframing can manufacture experience that isn't there.\n"
        "- Be strict. If you're uncertain, mark it skipped.\n"
        "- Copy each gap text verbatim from the input list into actionable_gaps or skipped_gaps — do not paraphrase.\n\n"
        "Return a GapScreenResult with actionable_gaps and skipped_gaps."
    ),
    expected_output="A GapScreenResult listing which gaps are actionable and which have no basis in the resume.",
    agent=advisor_agent,
    output_pydantic=GapScreenResult,
    guardrail=require_schema(GapScreenResult),
)
