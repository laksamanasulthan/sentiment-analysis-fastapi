import json
import re

import httpx

from app.core.config import Settings, get_settings
from app.core.errors import LLMBadResponseError, LLMTimeoutError, LLMUnavailableError
from app.services.categories import CATEGORIES

SYSTEM_PROMPT = (
    "You are a sentiment analysis engine. Classify the sentiment of the user text and reply with JSON only. "
    'Format: {"sentiment": "positive|negative|neutral", "confidence": <float between 0 and 1>, '
    '"category": "Customer Support|Product Quality|Delivery & Shipping|Pricing & Billing|General"}. '
    "confidence reflects how certain you are about the label. Reply with the JSON object and nothing else."
)


class LLMClient:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

    def analyze(self, text: str) -> dict:
        if not self.settings.llm_ready:
            raise LLMUnavailableError("LLM API key is not configured, set LLM_API_KEY in .env")
        payload = {
            "model": self.settings.resolved_model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            "temperature": 0,
        }
        headers = {"Authorization": f"Bearer {self.settings.llm_api_key}"}
        try:
            response = httpx.post(
                f"{self.settings.resolved_base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=self.settings.llm_timeout_seconds,
            )
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError("LLM request timed out") from exc
        except httpx.HTTPError as exc:
            raise LLMUnavailableError("LLM provider is unreachable") from exc
        if response.status_code in (401, 403):
            raise LLMUnavailableError("LLM provider rejected the API key")
        if response.status_code == 429:
            raise LLMUnavailableError("LLM provider rate limit exceeded")
        if response.status_code >= 400:
            raise LLMBadResponseError(f"LLM provider returned status {response.status_code}")
        try:
            content = response.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, ValueError) as exc:
            raise LLMBadResponseError("LLM response has an unexpected structure") from exc
        return self._parse(content)

    def _parse(self, content: str) -> dict:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            raise LLMBadResponseError("LLM reply does not contain a JSON object")
        try:
            data = json.loads(match.group(0))
        except ValueError as exc:
            raise LLMBadResponseError("LLM reply is not valid JSON") from exc
        sentiment = str(data.get("sentiment", "")).lower()
        if sentiment not in ("positive", "negative", "neutral"):
            raise LLMBadResponseError("LLM returned an unknown sentiment label")
        try:
            confidence = max(0.0, min(1.0, float(data.get("confidence", 0.0))))
        except (TypeError, ValueError) as exc:
            raise LLMBadResponseError("LLM returned a non numeric confidence") from exc
        category = data.get("category") if data.get("category") in CATEGORIES else "General"
        return {"sentiment": sentiment, "confidence": confidence, "category": category}
