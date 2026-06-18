import litellm
from crewai import LLM

from app.config import settings

# instructor's structured-output path calls litellm.completion() directly with
# only model/messages, skipping any per-call kwargs below — so these class-level
# defaults are the only way to reach that path too.
litellm.OllamaChatConfig(num_ctx=settings.context_window, num_predict=8192, temperature=0.2)
litellm.OllamaConfig(num_ctx=settings.context_window, num_predict=8192, temperature=0.2)
# "think" isn't a constructor field on these configs, but get_config() picks up
# any class attribute, so setting it directly still reaches instructor's calls.
litellm.OllamaChatConfig.think = False
litellm.OllamaConfig.think = False

llm = LLM(
    model=settings.ollama_model,
    is_litellm=True,
    num_ctx=settings.context_window,
    num_predict=8192,
    think=False,
    temperature=0.2,
)
