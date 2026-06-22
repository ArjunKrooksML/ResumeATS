from crewai import Task

from app.agents.strategist_agent import strategist_agent
from app.crew.guardrails import require_schema
from app.models.schemas import RefinementDecision

review_task = Task(
    description=(
        "Job description for this role: {job_description}\n\n"
        "Refinement round {round_number} of {max_rounds}.\n"
        "Current best score: {base_score} ({base_reasoning}).\n"
        "This round's score after applying its approved suggestions: {new_score} ({new_reasoning}).\n"
        "Suggestions rejected this round and why: {rejection_summary}.\n\n"
        "First decide accept_this_round: this is purely about whether this "
        "round's suggestions should be added on top of what's already been "
        "accepted, not about whether the resume is good enough overall. "
        "Accepted suggestions accumulate across rounds — accepting this "
        "round does not discard earlier accepted rounds, and rejecting it "
        "does not undo them either. A genuine improvement should normally "
        "be accepted even if the result is still far from perfect — "
        "rejecting real progress just because the overall score remains "
        "mediocre throws away work for no reason. Set it to false only if "
        "this round actually regressed, made no meaningful difference, or "
        "the approved suggestions are weaker or riskier than what's already "
        "been accepted despite the score. Whether to keep refining further "
        "is a separate question, covered by next_step below — you can "
        "accept this round's improvement and still continue if there's more "
        "ground to gain.\n\n"
        "Then decide the next step:\n"
        "- 'stop' if the remaining round budget isn't worth spending, or the score is already strong.\n"
        "- 'retry_advisor' if the targeted gaps are right, but the rewrites need to be sharper or more grounded.\n"
        "- 'rerun_gap_analysis' if the rejections suggest the targeted gaps themselves "
        "were the wrong focus, not just the rewrites.\n\n"
        "If continuing, write specific, actionable feedback for whichever agent runs "
        "next, informed by the job description and the rejection reasons above — not "
        "generic encouragement. If stopping, leave feedback empty and explain why in "
        "the reason field.\n\n"
        "Your output is a RefinementDecision object with accept_this_round, "
        "next_step, reason, and feedback filled in."
    ),
    expected_output="A RefinementDecision with accept_this_round, next_step, reason, and feedback.",
    agent=strategist_agent,
    output_pydantic=RefinementDecision,
    guardrail=require_schema(RefinementDecision),
)
