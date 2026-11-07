from typing import Optional

from pydantic import BaseModel, Field


class StructuralRisk(BaseModel):
    issue: str = Field(description="The layout problem found, e.g. multi-column layout")
    explanation: str = Field(description="Why this confuses real ATS parsers")
    severity: str = Field(description="low, medium, or high")


class ParsedResume(BaseModel):
    raw_text: str = ""
    skills: list[str]
    education: str
    years_experience: int
    structural_risks: list[StructuralRisk]
    is_reliable: bool = Field(description="False if the transcription/parse itself looks unreliable, garbled, or empty")
    reliability_reason: str = Field(description="Why the parse is or isn't trustworthy")


class RequirementCoverage(BaseModel):
    requirement: str
    status: str = Field(description="covered, partial, or missing")
    covered_by: str = ""


class ScoreResult(BaseModel):
    match_score: int = Field(ge=0, le=100)
    matched_keywords: list[str]
    reasoning: str
    coverage: list[RequirementCoverage] = Field(default_factory=list)


class JDAnalysis(BaseModel):
    credibility_score: int = Field(ge=0, le=100, description="0=junk/vague, 100=clear and realistic")
    red_flags: list[str]
    is_analyzable: bool = Field(description="False if the JD is too vague to produce meaningful analysis")
    summary: str


class GapAnalysis(BaseModel):
    missing_keywords: list[str]
    missing_skills: list[str]
    priority_gaps: list[str] = Field(description="Gaps ranked by impact on the score")


class Suggestion(BaseModel):
    original_bullet: str
    revised_bullet: str
    addresses_gap: str


class GapScreenResult(BaseModel):
    actionable_gaps: list[str] = Field(description="Gaps that can be addressed with existing resume experience")
    skipped_gaps: list[str] = Field(description="Gaps with no basis in the resume — cannot be reframed")


class AdvisorOutput(BaseModel):
    suggestions: list[Suggestion]


class FactCheckVerdict(BaseModel):
    suggestion_index: int
    is_grounded: bool = Field(description="False if the rewrite invents experience not in the original resume")
    explanation: str


class CriticOutput(BaseModel):
    verdicts: list[FactCheckVerdict]
    approved_suggestions: list[Suggestion]


class RefinementDecision(BaseModel):
    accept_this_round: bool = Field(description="True if this round's score/suggestions should replace the current best result")
    next_step: str = Field(description="One of: stop, retry_advisor, rerun_gap_analysis")
    reason: str
    feedback: str = Field(description="Direction for whichever agent runs next, empty if stopping")


class GapClosure(BaseModel):
    gap: str
    status: str = Field(description="closed, attempted, screened, or open")
    closed_in_round: Optional[int] = None
    closed_by: Optional[str] = None


class AnalyzeRequest(BaseModel):
    job_description: str


class AnalysisReport(BaseModel):
    parsed_resume: ParsedResume
    score: ScoreResult
    gaps: GapAnalysis
    final_suggestions: list[Suggestion]
    iterations_used: int
    jd_analysis: Optional[JDAnalysis] = None
    gap_closures: list[GapClosure] = Field(default_factory=list)
    coverage: list[RequirementCoverage] = Field(default_factory=list)
