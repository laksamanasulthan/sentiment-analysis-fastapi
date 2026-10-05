import time

from app.core.errors import LLMUnavailableError
from app.schemas.analyze import AnalyzeRequest, AnalyzeResponse
from app.services.llm_client import LLMClient


class SentimentService:
    def __init__(self, client: LLMClient | None = None):
        self.client = client or LLMClient()

    def analyze(self, request: AnalyzeRequest) -> AnalyzeResponse:
        if not self.client.settings.llm_ready:
            raise LLMUnavailableError("LLM API key is not configured, set LLM_API_KEY in .env")
        started = time.perf_counter()
        result = self.client.analyze(request.text)
        elapsed = (time.perf_counter() - started) * 1000
        return AnalyzeResponse(
            sentiment=result["sentiment"],
            confidence_score=round(result["confidence"], 4),
            category=result["category"],
            processing_time_ms=round(elapsed, 1),
        )
