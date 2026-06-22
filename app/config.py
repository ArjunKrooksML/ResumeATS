from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")
    openai_api_key: str
    openai_model: str = "gpt-4o-mini"
    max_output_tokens: int = 4096


settings = Settings()
