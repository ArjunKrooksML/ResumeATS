from typing import Optional
from uuid import uuid4

from crewai import Crew, Process
from crewai.flow.flow import Flow, listen, router, start
from pydantic import BaseModel, Field

from app.agents.advisor_agent import advisor_agent
from app.agents.consult_wiring import wire_consult_tools
from app.agents.critic_agent import critic_agent
from app.agents.gap_analyst_agent import gap_analyst_agent
from app.agents.parser_agent import parser_agent
from app.agents.scorer_agent import scorer_agent
from app.agents.strategist_agent import strategist_agent
from app.debug_log import log_step
from app.models.schemas import AnalysisReport, GapAnalysis, ParsedResume, ScoreResult, Suggestion
from app.tasks.advise_task import advise_task
from app.tasks.critic_task import critic_task
from app.tasks.gap_task import gap_task
from app.tasks.parse_task import parse_task
from app.tasks.regap_task import regap_task
from app.tasks.rescore_task import rescore_task
from app.tasks.review_task import review_task
from app.tasks.score_task import score_task
from app.tools.consult import reset_consult_tools, track_active_agents
from app.tools.resume_reader import extract_resume_text
from app.tools.resume_verification import set_resume_text

wire_consult_tools()

MAX_REFINEMENT_ROUNDS = 3
FIRST_ATTEMPT_FEEDBACK = "This is the first attempt, no prior feedback to address."


def run_step(agents: list, tasks: list, inputs: dict, verbose: bool = False) -> None:
    reset_consult_tools()
    with track_active_agents(*(agent.role for agent in agents)):
        Crew(agents=agents, tasks=tasks, process=Process.sequential, verbose=verbose).kickoff(inputs=inputs)


def apply_suggestions(raw_text: str, suggestions) -> str:
    revised = raw_text
    for suggestion in suggestions:
        if suggestion.original_bullet not in revised:
            log_step(f"Suggestion skipped, original_bullet not found verbatim in resume text: {suggestion.original_bullet!r}")
            continue
        revised = revised.replace(suggestion.original_bullet, suggestion.revised_bullet)
    return revised


def describe_rejections(verdicts) -> str:
    reasons = [v.explanation for v in verdicts if not v.is_grounded]
    return "; ".join(reasons) if reasons else "none of the suggestions were grounded in the original resume"


class AnalysisState(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    file_path: str = ""
    job_description: str = ""
    resume_text: str = ""
    retried_parse: bool = False
    parsed_resume: Optional[ParsedResume] = None
    base_score: Optional[ScoreResult] = None
    best_score: Optional[ScoreResult] = None
    revised_text: str = ""
    accumulated_suggestions: list[Suggestion] = Field(default_factory=list)
    candidate_text: str = ""
    gaps: Optional[GapAnalysis] = None
    current_gaps: Optional[GapAnalysis] = None
    round_number: int = 0
    feedback: str = FIRST_ATTEMPT_FEEDBACK
    gap_feedback: str = ""
    approved: list[Suggestion] = Field(default_factory=list)
    rejection_summary: str = ""
    new_score: Optional[ScoreResult] = None


class AnalysisFlow(Flow[AnalysisState]):
    initial_state: type[AnalysisState] = AnalysisState

    @start()
    def transcribe(self):
        log_step(f"Starting main pipeline for {self.state.file_path}")
        self.state.resume_text = extract_resume_text(self.state.file_path)

    @listen(transcribe)
    def parse(self):
        run_step([parser_agent], [parse_task], {"resume_text": self.state.resume_text}, verbose=True)
        parsed = parse_task.output.pydantic
        parsed.raw_text = self.state.resume_text
        self.state.parsed_resume = parsed
        log_step(f"Parsed resume: {len(parsed.skills)} skills, {len(parsed.structural_risks)} structural risks")

    @router(parse)
    def check_parse_reliability(self) -> str:
        if not self.state.parsed_resume.is_reliable and not self.state.retried_parse:
            log_step(f"Parser flagged low confidence ({self.state.parsed_resume.reliability_reason}), retrying OCR once")
            return "retry_parse"
        return "score"

    @listen("retry_parse")
    def retry_transcription(self):
        self.state.retried_parse = True
        self.state.resume_text = extract_resume_text(self.state.file_path)
        run_step([parser_agent], [parse_task], {"resume_text": self.state.resume_text}, verbose=True)
        parsed = parse_task.output.pydantic
        parsed.raw_text = self.state.resume_text
        self.state.parsed_resume = parsed
        log_step(f"Re-parsed resume: {len(parsed.skills)} skills, reliable={parsed.is_reliable}")

    @router(retry_transcription)
    def after_retry(self) -> str:
        return "score"

    @listen("score")
    def run_score(self):
        run_step(
            [scorer_agent], [score_task],
            {"resume_text": self.state.resume_text, "job_description": self.state.job_description},
            verbose=True,
        )
        self.state.base_score = score_task.output.pydantic
        self.state.best_score = self.state.base_score
        log_step(f"Initial score: {self.state.base_score.match_score}/100")

    @listen(run_score)
    def gap_analysis(self):
        run_step(
            [gap_analyst_agent], [gap_task],
            {"resume_text": self.state.resume_text, "job_description": self.state.job_description},
            verbose=True,
        )
        self.state.gaps = gap_task.output.pydantic
        self.state.current_gaps = self.state.gaps
        self.state.revised_text = self.state.resume_text
        log_step(f"Gap analysis: {len(self.state.gaps.priority_gaps)} priority gaps identified")

    @router(gap_analysis)
    def begin_refinement(self) -> str:
        return "advise"

    @listen("advise")
    def run_advisor_round(self):
        self.state.round_number += 1
        log_step(f"--- Refinement round {self.state.round_number}/{MAX_REFINEMENT_ROUNDS} ---")
        log_step(f"Running advisor with feedback: {self.state.feedback}")
        set_resume_text(self.state.revised_text)
        run_step(
            [advisor_agent, critic_agent], [advise_task, critic_task],
            {
                "job_description": self.state.job_description,
                "resume_text": self.state.revised_text,
                "critic_feedback": self.state.feedback,
                "priority_gaps": "; ".join(self.state.current_gaps.priority_gaps),
            },
            verbose=True,
        )
        critic_output = critic_task.output.pydantic
        self.state.approved = critic_output.approved_suggestions
        self.state.rejection_summary = describe_rejections(critic_output.verdicts)
        log_step(f"Critic approved {len(critic_output.approved_suggestions)} of {len(critic_output.verdicts)} suggestions")

    @listen(run_advisor_round)
    def rescore_after_advisor(self):
        if self.state.approved:
            # any critic-approved suggestion is grounded advice worth showing, regardless
            # of whether the strategist later judges it worth tracking as the official score
            self.state.accumulated_suggestions = self.state.accumulated_suggestions + self.state.approved
            self.state.candidate_text = apply_suggestions(self.state.revised_text, self.state.approved)
            run_step([scorer_agent], [rescore_task], {"job_description": self.state.job_description, "resume_text": self.state.candidate_text})
            self.state.new_score = rescore_task.output.pydantic
            log_step(f"Re-score after applying approved suggestions: {self.state.new_score.match_score}/100")
        else:
            self.state.candidate_text = self.state.revised_text
            self.state.new_score = self.state.best_score
            log_step("No suggestions approved this round")

    @router(rescore_after_advisor)
    def strategist_decision(self) -> str:
        run_step([strategist_agent], [review_task], {
            "job_description": self.state.job_description,
            "round_number": self.state.round_number,
            "max_rounds": MAX_REFINEMENT_ROUNDS,
            "base_score": self.state.best_score.match_score,
            "base_reasoning": self.state.best_score.reasoning,
            "new_score": self.state.new_score.match_score,
            "new_reasoning": self.state.new_score.reasoning,
            "rejection_summary": self.state.rejection_summary,
        })
        decision = review_task.output.pydantic
        log_step(f"Strategist: accept_this_round={decision.accept_this_round}, next_step={decision.next_step} — {decision.reason}")

        if decision.accept_this_round:
            self.state.best_score = self.state.new_score
            self.state.revised_text = self.state.candidate_text

        if self.state.round_number >= MAX_REFINEMENT_ROUNDS:
            log_step(f"Hit max refinement rounds ({MAX_REFINEMENT_ROUNDS}), returning best effort")
            return "stop"
        if decision.next_step == "stop":
            return "stop"
        if decision.next_step == "rerun_gap_analysis":
            self.state.gap_feedback = decision.feedback
            return "gap"
        self.state.feedback = decision.feedback
        return "advise"

    @listen("gap")
    def run_gap_round(self):
        run_step([gap_analyst_agent], [regap_task], {
            "job_description": self.state.job_description,
            "resume_text": self.state.resume_text,
            "strategist_feedback": self.state.gap_feedback,
        })
        self.state.current_gaps = regap_task.output.pydantic
        self.state.feedback = FIRST_ATTEMPT_FEEDBACK
        log_step(f"Gap analysis revised: {len(self.state.current_gaps.priority_gaps)} priority gaps")

    @router(run_gap_round)
    def after_gap(self) -> str:
        return "advise"

    @listen("stop")
    def finalize(self) -> AnalysisReport:
        log_step(f"=== Analysis complete: final score {self.state.best_score.match_score}/100 after {self.state.round_number} round(s) ===")
        # Flow wraps list-typed state fields in LockedListProxy for thread safety across
        # steps; handing that directly to another model's list[Suggestion] field silently
        # validates to empty instead of erroring, so materialize a plain list first.
        return AnalysisReport(
            parsed_resume=self.state.parsed_resume,
            score=self.state.best_score,
            gaps=self.state.gaps,
            final_suggestions=list(self.state.accumulated_suggestions),
            iterations_used=self.state.round_number,
        )


def analyze_resume(file_path: str, job_description: str) -> AnalysisReport:
    log_step("=== Starting resume analysis ===")
    flow = AnalysisFlow()
    return flow.kickoff(inputs={"file_path": file_path, "job_description": job_description})
