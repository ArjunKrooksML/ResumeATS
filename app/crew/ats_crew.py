from crewai import Crew, Process

from app.agents.advisor_agent import advisor_agent
from app.agents.critic_agent import critic_agent
from app.agents.gap_analyst_agent import gap_analyst_agent
from app.agents.parser_agent import parser_agent
from app.agents.scorer_agent import scorer_agent
from app.debug_log import log_step
from app.models.schemas import AnalysisReport
from app.tasks.advise_task import advise_task
from app.tasks.critic_task import critic_task
from app.tasks.gap_task import gap_task
from app.tasks.parse_task import parse_task
from app.tasks.rescore_task import rescore_task
from app.tasks.score_task import score_task

MAX_REFINEMENT_ROUNDS = 3
FIRST_ATTEMPT_FEEDBACK = "This is the first attempt, no prior feedback to address."


def run_main_pipeline(file_path: str, job_description: str):
    log_step(f"Starting main pipeline for {file_path}")
    crew = Crew(
        agents=[parser_agent, scorer_agent, gap_analyst_agent],
        tasks=[parse_task, score_task, gap_task],
        process=Process.sequential,
        verbose=True,
    )
    crew.kickoff(inputs={"file_path": file_path, "job_description": job_description})
    parsed_resume = parse_task.output.pydantic
    score = score_task.output.pydantic
    gaps = gap_task.output.pydantic
    log_step(f"Parsed resume: {len(parsed_resume.skills)} skills, {len(parsed_resume.structural_risks)} structural risks")
    log_step(f"Initial score: {score.match_score}/100")
    log_step(f"Gap analysis: {len(gaps.priority_gaps)} priority gaps identified")
    return parsed_resume, score, gaps


def run_refinement_round(job_description: str, feedback: str):
    log_step(f"Running advisor with feedback: {feedback}")
    crew = Crew(
        agents=[advisor_agent, critic_agent],
        tasks=[advise_task, critic_task],
        process=Process.sequential,
        verbose=True,
    )
    crew.kickoff(inputs={"job_description": job_description, "critic_feedback": feedback})
    critic_output = critic_task.output.pydantic
    log_step(f"Critic approved {len(critic_output.approved_suggestions)} of {len(critic_output.verdicts)} suggestions")
    return critic_output


def rescore_resume(job_description: str, resume_text: str):
    crew = Crew(agents=[scorer_agent], tasks=[rescore_task], process=Process.sequential)
    crew.kickoff(inputs={"job_description": job_description, "resume_text": resume_text})
    new_score = rescore_task.output.pydantic
    log_step(f"Re-score after applying approved suggestions: {new_score.match_score}/100")
    return new_score


def apply_suggestions(raw_text: str, suggestions) -> str:
    revised = raw_text
    for suggestion in suggestions:
        revised = revised.replace(suggestion.original_bullet, suggestion.revised_bullet)
    return revised


def describe_rejections(verdicts) -> str:
    reasons = [v.explanation for v in verdicts if not v.is_grounded]
    return "; ".join(reasons) if reasons else "none of the suggestions were grounded in the original resume"


def build_no_improvement_feedback(new_score, best_score) -> str:
    return (
        f"Your suggestions were grounded but only scored {new_score.match_score} "
        f"versus the original {best_score.match_score}. Target the highest "
        "priority gaps more directly."
    )


def refine_until_improved(job_description: str, raw_text: str, base_score):
    feedback = FIRST_ATTEMPT_FEEDBACK
    approved = []
    for round_number in range(1, MAX_REFINEMENT_ROUNDS + 1):
        log_step(f"--- Refinement round {round_number}/{MAX_REFINEMENT_ROUNDS} ---")
        critic_output = run_refinement_round(job_description, feedback)
        approved = critic_output.approved_suggestions
        if not approved:
            feedback = (
                f"Every suggestion was rejected: {describe_rejections(critic_output.verdicts)}. "
                "Stay strictly grounded in the original resume."
            )
            log_step("No suggestions approved this round, retrying with rejection feedback")
            continue
        revised_text = apply_suggestions(raw_text, approved)
        new_score = rescore_resume(job_description, revised_text)
        if new_score.match_score > base_score.match_score:
            log_step(f"Score improved {base_score.match_score} -> {new_score.match_score}, stopping at round {round_number}")
            return approved, new_score, round_number
        feedback = build_no_improvement_feedback(new_score, base_score)
        log_step(f"Score did not improve ({new_score.match_score} vs {base_score.match_score}), retrying")
    log_step(f"Hit max refinement rounds ({MAX_REFINEMENT_ROUNDS}) without improvement, returning best effort")
    return approved, base_score, MAX_REFINEMENT_ROUNDS


def analyze_resume(file_path: str, job_description: str) -> AnalysisReport:
    log_step("=== Starting resume analysis ===")
    parsed_resume, base_score, gaps = run_main_pipeline(file_path, job_description)
    approved_suggestions, final_score, rounds_used = refine_until_improved(
        job_description, parsed_resume.raw_text, base_score
    )
    log_step(f"=== Analysis complete: final score {final_score.match_score}/100 after {rounds_used} round(s) ===")
    return AnalysisReport(
        parsed_resume=parsed_resume,
        score=final_score,
        gaps=gaps,
        final_suggestions=approved_suggestions,
        iterations_used=rounds_used,
    )
