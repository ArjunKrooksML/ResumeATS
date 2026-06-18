from crewai import LLM

from app.config import settings

llm = LLM(
    model=settings.ollama_model,
    extra_body={"options": {"num_ctx": settings.context_window}},
)
