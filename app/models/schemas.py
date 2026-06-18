from pydantic import BaseModel, Field


class StructuralRisk(BaseModel):
    issue: str = Field(description="The layout problem found, e.g. multi-column layout")
    explanation: str = Field(description="Why this confuses real ATS parsers")
    severity: str = Field(description="low, medium, or high")


class ParsedResume(BaseModel):
    raw_text: str
    skills: list[str]
    education: str
    years_experience: int
    structural_risks: list[StructuralRisk]


class ScoreResult(BaseModel):
    match_score: int = Field(ge=0, le=100)
    matched_keywords: list[str]
    reasoning: str


class GapAnalysis(BaseModel):
    missing_keywords: list[str]
    missing_skills: list[str]
    priority_gaps: list[str] = Field(description="Gaps ranked by impact on the score")


class Suggestion(BaseModel):
    original_bullet: str
    revised_bullet: str
    addresses_gap: str


class AdvisorOutput(BaseModel):
    suggestions: list[Suggestion]


class FactCheckVerdict(BaseModel):
    suggestion_index: int
    is_grounded: bool = Field(description="False if the rewrite invents experience not in the original resume")
    explanation: str


class CriticOutput(BaseModel):
    verdicts: list[FactCheckVerdict]
    approved_suggestions: list[Suggestion]


class AnalyzeRequest(BaseModel):
    job_description: str


class AnalysisReport(BaseModel):
    parsed_resume: ParsedResume
    score: ScoreResult
    gaps: GapAnalysis
    final_suggestions: list[Suggestion]
    iterations_used: int
