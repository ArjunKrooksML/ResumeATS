from crewai import LLM

from app.config import settings

llm = LLM(
    model=settings.openai_model,
    api_key=settings.openai_api_key,
    max_tokens=settings.max_output_tokens,
    temperature=0.2,
)
