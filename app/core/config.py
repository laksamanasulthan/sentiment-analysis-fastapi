from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

PROVIDER_DEFAULTS = {
    "openai": {"base_url": "https://api.openai.com/v1", "model": "gpt-4o-mini"},
    "groq": {"base_url": "https://api.groq.com/openai/v1", "model": "llama-3.3-70b-versatile"},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1", "model": "openai/gpt-4o-mini"},
    "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai", "model": "gemini-2.5-flash"},
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Sentiment Analysis API"
    llm_provider: str = "openai"
    llm_api_key: str = ""
    llm_base_url: str = ""
    llm_model: str = ""
    llm_timeout_seconds: float = 30.0
    local_model_name: str = "Horizon-Labs/multilingual-sentiment-small"
    local_onnx_file: str = "onnx/model_quantized.onnx"
    local_max_length: int = 1024

    @property
    def llm_ready(self) -> bool:
        return bool(self.llm_api_key)

    @property
    def resolved_base_url(self) -> str:
        defaults = PROVIDER_DEFAULTS.get(self.llm_provider, PROVIDER_DEFAULTS["openai"])
        return self.llm_base_url or defaults["base_url"]

    @property
    def resolved_model(self) -> str:
        defaults = PROVIDER_DEFAULTS.get(self.llm_provider, PROVIDER_DEFAULTS["openai"])
        return self.llm_model or defaults["model"]


@lru_cache
def get_settings() -> Settings:
    return Settings()
