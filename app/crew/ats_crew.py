import asyncio
import time
from typing import Optional
from uuid import uuid4

from crewai import Crew, Process
from crewai.flow.flow import Flow, listen, router, start
from pydantic import BaseModel, Field

from app.agents.advisor_agent import advisor_agent
from app.agents.consult_wiring import wire_consult_tools
from app.agents.critic_agent import critic_agent
from app.agents.gap_analyst_agent import gap_analyst_agent
from app.agents.jd_analyst_agent import jd_analyst_agent
from app.agents.parser_agent import parser_agent
from app.agents.scorer_agent import scorer_agent
from app.agents.strategist_agent import strategist_agent
from app.debug_log import capture_crew_output, log_step, log_verbose, run_log_file
from app.models.schemas import (
    AnalysisReport, GapAnalysis, GapClosure, GapScreenResult,
    JDAnalysis, ParsedResume, RequirementCoverage, ScoreResult, Suggestion,
)
from app.tasks.advise_task import advise_task
from app.tasks.critic_task import critic_task
from app.tasks.gap_task import make_gap_task
from app.tasks.jd_task import jd_task
from app.tasks.parse_task import parse_task
from app.tasks.regap_task import regap_task
from app.tasks.rescore_task import rescore_task
from app.tasks.review_task import review_task
from app.tasks.score_task import make_score_task
from app.tasks.screen_task import screen_task
from app.tools.consult import reset_consult_tools, track_active_agents
from app.tools.resume_reader import extract_resume_text
from app.tools.resume_verification import set_resume_text

wire_consult_tools()

MAX_REFINEMENT_ROUNDS = 3
FIRST_ATTEMPT_FEEDBACK = "This is the first attempt, no prior feedback to address."


def run_step(agents: list, tasks: list, inputs: dict) -> None:
    reset_consult_tools()
    with track_active_agents(*(agent.role for agent in agents)):
        with capture_crew_output():
            Crew(agents=agents, tasks=tasks, process=Process.sequential, verbose=True).kickoff(inputs=inputs)


def apply_suggestions(raw_text: str, suggestions) -> str:
    revised = raw_text
    for suggestion in suggestions:
        if suggestion.original_bullet not in revised:
            log_step(f"Suggestion skipped, original_bullet not found verbatim: {suggestion.original_bullet!r}")
            continue
        revised = revised.replace(suggestion.original_bullet, suggestion.revised_bullet)
    return revised


def describe_rejections(verdicts) -> str:
    reasons = [v.explanation for v in verdicts if not v.is_grounded]
    return "; ".join(reasons) if reasons else "none of the suggestions were grounded in the original resume"


def _deduplicate_gaps(gaps: GapAnalysis, matched_keywords: list[str]) -> GapAnalysis:
    matched_lower = {kw.lower() for kw in matched_keywords}

    def already_matched(term: str) -> bool:
        t = term.lower()
        return any(t in m or m in t for m in matched_lower)

    return GapAnalysis(
        missing_keywords=[k for k in gaps.missing_keywords if not already_matched(k)],
        missing_skills=[s for s in gaps.missing_skills if not already_matched(s)],
        priority_gaps=[g for g in gaps.priority_gaps if not already_matched(g)],
    )


def _match_gap(ref: str, lifecycle: dict) -> Optional[str]:
    # exact match first, then substring fallback for LLM paraphrasing
    ref_lower = ref.lower().strip()
    for key in lifecycle:
        if key.lower().strip() == ref_lower:
            return key
    for key in lifecycle:
        if ref_lower in key.lower() or key.lower() in ref_lower:
            return key
    return None


class AnalysisState(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    file_path: str = ""
    job_description: str = ""
    resume_text: str = ""
    original_resume_text: str = ""
    retried_parse: bool = False
    parsed_resume: Optional[ParsedResume] = None
    jd_analysis: Optional[JDAnalysis] = None
    base_score: Optional[ScoreResult] = None
    best_score: Optional[ScoreResult] = None
    revised_text: str = ""
    accumulated_suggestions: list[Suggestion] = Field(default_factory=list)
    candidate_text: str = ""
    gaps: Optional[GapAnalysis] = None
    current_gaps: Optional[GapAnalysis] = None
    gap_lifecycle: dict[str, str] = Field(default_factory=dict)
    gap_closed_by: dict[str, str] = Field(default_factory=dict)
    gap_closed_round: dict[str, int] = Field(default_factory=dict)
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
        self.state.original_resume_text = self.state.resume_text

    @listen(transcribe)
    def parse(self):
        run_step([parser_agent], [parse_task], {"resume_text": self.state.resume_text})
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
        self.state.resume_text = extract_resume_text(self.state.file_path, high_quality=True)
        self.state.original_resume_text = self.state.resume_text
        run_step([parser_agent], [parse_task], {"resume_text": self.state.resume_text})
        parsed = parse_task.output.pydantic
        parsed.raw_text = self.state.resume_text
        self.state.parsed_resume = parsed
        log_step(f"Re-parsed resume: {len(parsed.skills)} skills, reliable={parsed.is_reliable}")

    @router(retry_transcription)
    def after_retry(self) -> str:
        return "score"

    @listen("score")
    def analyze_jd(self):
        run_step([jd_analyst_agent], [jd_task], {"job_description": self.state.job_description})
        self.state.jd_analysis = jd_task.output.pydantic
        jd = self.state.jd_analysis
        if jd.red_flags:
            log_step(
                f"JD credibility: {jd.credibility_score}/100 — "
                f"{len(jd.red_flags)} red flag(s): {'; '.join(jd.red_flags[:2])}"
            )
        else:
            log_step(f"JD credibility: {jd.credibility_score}/100 — no red flags")

    @listen(analyze_jd)
    async def score_and_gap(self):
        score_t = make_score_task()
        gap_t = make_gap_task()
        inputs = {"resume_text": self.state.resume_text, "job_description": self.state.job_description}

        t0 = time.perf_counter()
        await asyncio.gather(
            Crew(agents=[scorer_agent], tasks=[score_t], process=Process.sequential).akickoff(inputs=inputs),
            Crew(agents=[gap_analyst_agent], tasks=[gap_t], process=Process.sequential).akickoff(inputs=inputs),
        )
        elapsed = time.perf_counter() - t0

        self.state.base_score = score_t.output.pydantic
        self.state.best_score = self.state.base_score
        raw_gaps = gap_t.output.pydantic
        self.state.gaps = _deduplicate_gaps(raw_gaps, self.state.base_score.matched_keywords)
        self.state.current_gaps = self.state.gaps
        self.state.revised_text = self.state.resume_text
        self.state.gap_lifecycle = {g: "open" for g in self.state.gaps.priority_gaps}

        removed = len(raw_gaps.priority_gaps) - len(self.state.gaps.priority_gaps)

        log_verbose(f"[Scorer raw output]\n{score_t.output.raw}")
        log_verbose(f"[Gap Analyst raw output]\n{gap_t.output.raw}")
        log_step(
            f"Score: {self.state.base_score.match_score}/100 | "
            f"{len(self.state.gaps.priority_gaps)} priority gaps"
            + (f" ({removed} deduped)" if removed else "")
            + f" | {len(self.state.base_score.coverage)} requirements mapped"
            + f" | parallel in {elapsed:.1f}s"
        )

    @router(score_and_gap)
    def begin_refinement(self) -> str:
        return "advise"

    @listen("advise")
    def run_advisor_round(self):
        self.state.round_number += 1
        log_step(f"--- Refinement round {self.state.round_number}/{MAX_REFINEMENT_ROUNDS} ---")

        run_step(
            [advisor_agent], [screen_task],
            {
                "resume_text": self.state.revised_text,
                "priority_gaps": "; ".join(self.state.current_gaps.priority_gaps),
            },
        )
        screened: GapScreenResult = screen_task.output.pydantic
        actionable = screened.actionable_gaps
        log_step(
            f"Gap screening: {len(actionable)} actionable, "
            f"{len(screened.skipped_gaps)} skipped"
            + (f" ({', '.join(screened.skipped_gaps)})" if screened.skipped_gaps else "")
        )

        for gap_text in screened.skipped_gaps:
            matched = _match_gap(gap_text, self.state.gap_lifecycle)
            if matched and self.state.gap_lifecycle[matched] == "open":
                self.state.gap_lifecycle[matched] = "screened"

        if not actionable:
            log_step("No actionable gaps — skipping advisor this round")
            self.state.approved = []
            self.state.rejection_summary = "no actionable gaps found"
            return

        set_resume_text(self.state.original_resume_text)
        run_step(
            [advisor_agent, critic_agent], [advise_task, critic_task],
            {
                "job_description": self.state.job_description,
                "resume_text": self.state.revised_text,
                "critic_feedback": self.state.feedback,
                "priority_gaps": "; ".join(actionable),
            },
        )
        critic_output = critic_task.output.pydantic
        self.state.approved = critic_output.approved_suggestions
        self.state.rejection_summary = describe_rejections(critic_output.verdicts)
        log_step(
            f"Critic approved {len(critic_output.approved_suggestions)} of {len(critic_output.verdicts)} suggestions"
        )

        # track gap lifecycle: closed vs attempted-but-rejected
        all_attempted = advise_task.output.pydantic.suggestions if advise_task.output else []
        closed_gaps = {s.addresses_gap for s in self.state.approved if s.addresses_gap}
        for s in all_attempted:
            if not s.addresses_gap:
                continue
            matched = _match_gap(s.addresses_gap, self.state.gap_lifecycle)
            if not matched:
                continue
            if s.addresses_gap in closed_gaps:
                self.state.gap_lifecycle[matched] = "closed"
                self.state.gap_closed_round[matched] = self.state.round_number
                if matched not in self.state.gap_closed_by:
                    for approved in self.state.approved:
                        if approved.addresses_gap == s.addresses_gap:
                            self.state.gap_closed_by[matched] = approved.revised_bullet
                            break
            elif self.state.gap_lifecycle.get(matched) not in ("closed", "screened"):
                self.state.gap_lifecycle[matched] = "attempted"

    @listen(run_advisor_round)
    def rescore_after_advisor(self):
        if self.state.approved:
            self.state.accumulated_suggestions = self.state.accumulated_suggestions + self.state.approved
            self.state.candidate_text = apply_suggestions(self.state.revised_text, self.state.approved)
            run_step(
                [scorer_agent], [rescore_task],
                {"job_description": self.state.job_description, "resume_text": self.state.candidate_text},
            )
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
        log_step(
            f"Strategist: accept_this_round={decision.accept_this_round}, "
            f"next_step={decision.next_step} — {decision.reason}"
        )

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
            "resume_text": self.state.revised_text,
            "strategist_feedback": self.state.gap_feedback,
        })
        self.state.current_gaps = _deduplicate_gaps(
            regap_task.output.pydantic,
            self.state.base_score.matched_keywords if self.state.base_score else [],
        )
        self.state.feedback = FIRST_ATTEMPT_FEEDBACK
        # add any new gaps from the rerun to lifecycle tracking
        for g in self.state.current_gaps.priority_gaps:
            if _match_gap(g, self.state.gap_lifecycle) is None:
                self.state.gap_lifecycle[g] = "open"
        log_step(f"Gap analysis revised: {len(self.state.current_gaps.priority_gaps)} priority gaps")

    @router(run_gap_round)
    def after_gap(self) -> str:
        return "advise"

    @listen("stop")
    def finalize(self) -> AnalysisReport:
        log_step(
            f"=== Analysis complete: final score {self.state.best_score.match_score}/100 "
            f"after {self.state.round_number} round(s) ==="
        )
        gap_closures = [
            GapClosure(
                gap=gap,
                status=status,
                closed_in_round=self.state.gap_closed_round.get(gap),
                closed_by=self.state.gap_closed_by.get(gap),
            )
            for gap, status in self.state.gap_lifecycle.items()
        ]
        return AnalysisReport(
            parsed_resume=self.state.parsed_resume,
            score=self.state.best_score,
            gaps=self.state.current_gaps,
            final_suggestions=list(self.state.accumulated_suggestions),
            iterations_used=self.state.round_number,
            jd_analysis=self.state.jd_analysis,
            gap_closures=gap_closures,
            coverage=list(self.state.base_score.coverage) if self.state.base_score else [],
        )


def analyze_resume(file_path: str, job_description: str) -> AnalysisReport:
    from datetime import datetime
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with run_log_file(run_id):
        log_step("=== Starting resume analysis ===")
        flow = AnalysisFlow()
        return flow.kickoff(inputs={"file_path": file_path, "job_description": job_description})
