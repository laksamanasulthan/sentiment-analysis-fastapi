import pytest
from fastapi.testclient import TestClient

from app.api.v1.analyze import get_sentiment_service
from app.api.v1.local import get_local_sentiment_service
from app.core.config import Settings
from app.main import create_app
from app.services.llm_client import LLMClient
from app.services.local_sentiment import LocalSentimentService
from app.services.sentiment import SentimentService


class FakeLLMClient:
    def __init__(self):
        self.settings = Settings(llm_api_key="test-key")

    def analyze(self, text: str) -> dict:
        return {"sentiment": "positive", "confidence": 0.94, "category": "Customer Support"}


class FakeEngine:
    def predict(self, text: str) -> tuple[str, float]:
        return "positive", 0.91


@pytest.fixture
def app():
    application = create_app()
    application.dependency_overrides[get_sentiment_service] = lambda: SentimentService(FakeLLMClient())
    application.dependency_overrides[get_local_sentiment_service] = lambda: LocalSentimentService(FakeEngine())
    return application


@pytest.fixture
def client(app):
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def client_no_key(app):
    app.dependency_overrides[get_sentiment_service] = lambda: SentimentService(LLMClient(Settings(llm_api_key="")))
    with TestClient(app) as test_client:
        yield test_client
